import math
import os
import time
import numpy as np
from bosdyn.api.graph_nav.graph_nav_pb2 import TravelParams
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2, nav_pb2
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient, map_pb2
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose

class RecordingInterface(object):
    def __init__(self, robot, download_filepath, client_metadata):
        # Store base path for creating unique download folders
        self._base_download_filepath = download_filepath
        self._download_filepath = os.path.join(download_filepath, 'downloaded_graph')

        self._recording_client = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(client_metadata=client_metadata)
        )

        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        self._map_processing_client = robot.ensure_client(MapProcessingServiceClient.default_service_name)
        self._current_graph = None
        self._current_edges = dict()
        self._current_waypoint_snapshots = dict()
        self._current_edge_snapshots = dict()
        self._current_annotation_name_to_wp_id = dict()
        self.robot = robot

        # Store waypoint poses: {waypoint_name: {'x': x, 'y': y, 'z': z, 'yaw': yaw}}
        self.waypoint_poses = {}

    def _generate_unique_map_folder(self, base_name='downloaded_graph'):
        """
        Generate a unique folder name for map download.
        If the folder already exists, append a number (e.g., downloaded_graph_1, downloaded_graph_2).

        Args:
            base_name: Base name for the folder (default: 'downloaded_graph')

        Returns:
            str: Full path to unique folder
        """
        from datetime import datetime

        # Option 1: Use timestamp (recommended - always unique)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_folder = f"{base_name}_{timestamp}"
        full_path = os.path.join(self._base_download_filepath, unique_folder)

        # If somehow it still exists (very unlikely), add a counter
        counter = 1
        original_path = full_path
        while os.path.exists(full_path):
            full_path = f"{original_path}_{counter}"
            counter += 1

        print(f"[MAP DOWNLOAD] New map will be saved to: {unique_folder}")
        return full_path

    def initialize_with_fiducial(self, robot_state_client, fiducial_id=None):
        """
        Initialize localization and map using a visible Fiducial.
        This sets the fiducial as the origin (0,0,0) of the GraphNav map.

        Args:
            robot_state_client: Client to read robot state
            fiducial_id (int, optional): If you want to use a specific tag (e.g. 305).
                                         If None, uses the closest one.
        Returns:
            bool: True if initialized successfully, False if it fails.

        Note:
            - The robot must be approximately 1m from the fiducial and looking at it
            - The fiducial must be visible in the robot's cameras
            - Call clear_map() before this to start fresh
        """
        import time

        print(f"\n[{'=' * 40}]")
        print(f"[FIDUCIAL INIT] Starting initialization via FIDUCIAL...")

        try:
            robot_state = robot_state_client.get_robot_state()
            current_odom_tform_body = get_odom_tform_body(
                robot_state.kinematic_state.transforms_snapshot).to_proto()
        except Exception as e:
            print(f"[FIDUCIAL INIT] ✗ Error reading robot state: {e}")
            return False

        if fiducial_id is not None:
            print(f"[FIDUCIAL INIT] Target: Fiducial ID {fiducial_id}")
            init_type = graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC
        else:
            print(f"[FIDUCIAL INIT] Target: Nearest fiducial (any ID)")
            init_type = graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST

        localization = nav_pb2.Localization()

        try:
            self._graph_nav_client.set_localization(
                initial_guess_localization=localization,
                ko_tform_body=current_odom_tform_body,

                fiducial_init=init_type,
                use_fiducial_id=fiducial_id if fiducial_id else 0,

                refine_fiducial_result_with_icp=True,
                do_ambiguity_check=True
            )

            # Wait a moment for the system to process the change
            time.sleep(0.5)
            print(f"[FIDUCIAL INIT] ✓ SUCCESS: Robot localized. Map origin is now the Fiducial.")
            print(f"[{'=' * 40}]\n")
            return True

        except Exception as e:
            print(f"[FIDUCIAL INIT] ✗ FAILED: {e}")
            print(f"[FIDUCIAL INIT] Possible reasons:")
            print(f"  1. Fiducial {fiducial_id if fiducial_id else 'any'} is not visible")
            print(f"  2. Robot is too far from fiducial (should be ~1m)")
            print(f"  3. Robot is not facing the fiducial")
            print(f"  4. Lighting conditions are poor")
            print(f"  5. Map not cleared before initialization (call clear_map() first)")
            print(f"[FIDUCIAL INIT] The system can continue without fiducial - first waypoint will be origin.")
            print(f"[{'=' * 40}]\n")
            return False

    def force_localization_to_waypoint(self, robot_state_client, waypoint_id):
        """
        Forza la localizzazione del robot su un waypoint specifico.
        Da usare se il robot si perde (STATUS_LOST).
        """
        print(f"\n[RECOVERY] Tentativo di ripristino localizzazione su Waypoint ID: {waypoint_id}")

        try:
            # 1. Prepare the guess (Guess)
            localization = nav_pb2.Localization()
            localization.waypoint_id = waypoint_id
            # Assume identity rotation (w=1) as base, then vision will correct
            localization.waypoint_tform_body.rotation.w = 1.0

            # 2. Get kinematic state
            robot_state = robot_state_client.get_robot_state()
            current_odom_tform_body = get_odom_tform_body(
                robot_state.kinematic_state.transforms_snapshot).to_proto()

            # 3. Send SetLocalization command
            self._graph_nav_client.set_localization(
                initial_guess_localization=localization,
                ko_tform_body=current_odom_tform_body,

                # Wide tolerance parameters for recovery
                max_distance=1.0,  # Search within 1 meter
                max_yaw=1.0,  # Search within ~57 degrees

                # Non usare fiducial, usa il waypoint ID
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,

                # CRUCIALE: Usa la visione per raffinare la posizione
                refine_with_visual_features=True,
                verify_visual_features_quality=True,
                do_ambiguity_check=True
            )

            print(f"[RECOVERY] ✓ Localizzazione forzata con successo!")
            return True

        except Exception as e:
            print(f"[RECOVERY] ✗ Fallimento localizzazione forzata: {e}")
            return False

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def create_new_edge(self):
        graph = self._graph_nav_client.download_graph()
        if len(graph.waypoints) < 2:
            print(f'Graph contains {len(graph.waypoints)} waypoints -- at least two are needed to create loop.')
            return False

        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "wp_0":
                first_waypoint = waypoint
        if first_waypoint is None:
            print('No wp_0 (first manual waypoint) found in the graph.')
            return False

        from_wp = first_waypoint
        if from_wp is None:
            return

        to_wp = max(graph.waypoints, key=lambda wp: int(wp.annotations.name.split('_')[-1]))
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_wp.id
        new_edge.id.to_waypoint = to_wp.id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f'edge transform = {new_edge.from_tform_to}')

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def create_edge_between_waypoints(self, from_waypoint_name, to_waypoint_name):
        """
        Create an edge between two specific waypoints by name.

        Args:
            from_waypoint_name: Name of the starting waypoint (e.g., 'wp_5')
            to_waypoint_name: Name of the destination waypoint (e.g., 'wp_8')

        Returns:
            bool: True if edge was created successfully, False otherwise
        """
        graph = self._graph_nav_client.download_graph()
        if len(graph.waypoints) < 2:
            print(f'[EDGE] Graph contains {len(graph.waypoints)} waypoints -- at least two are needed.')
            return False

        # Find the two waypoints
        from_wp = None
        to_wp = None

        for waypoint in graph.waypoints:
            if waypoint.annotations.name == from_waypoint_name:
                from_wp = waypoint
            elif waypoint.annotations.name == to_waypoint_name:
                to_wp = waypoint

        if from_wp is None:
            print(f'[EDGE] ✗ Waypoint {from_waypoint_name} not found in graph.')
            return False
        if to_wp is None:
            print(f'[EDGE] ✗ Waypoint {to_waypoint_name} not found in graph.')
            return False

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_wp.id
        new_edge.id.to_waypoint = to_wp.id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        try:
            # Send request to add edge to map
            self._recording_client.create_edge(edge=new_edge)
            print(f'[EDGE] ✓ Created edge: {from_waypoint_name} → {to_waypoint_name}')
            return True
        except Exception as e:
            print(f'[EDGE] ✗ Failed to create edge: {e}')
            return False

    def get_graph_structure(self):
        """
        Get the current graph structure with waypoints and edges.

        Returns:
            dict: Dictionary with 'waypoints' (list of waypoint objects) and 'edges' (list of edge objects)
        """
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            return {'waypoints': [], 'edges': []}

        return {
            'waypoints': list(graph.waypoints),
            'edges': list(graph.edges)
        }

    def find_shortest_path(self, from_waypoint_name, to_waypoint_name, env_map):
        """
        Find the shortest path between two waypoints using Dijkstra's algorithm.
        If missing edges are found between adjacent visited cells, they are automatically created.

        Args:
            from_waypoint_name: Name of starting waypoint (e.g., 'wp_5')
            to_waypoint_name: Name of destination waypoint (e.g., 'wp_12')
            env_map: EnvironmentMap object for checking visited cells

        Returns:
            list: List of waypoint names representing the shortest path, or None if no path exists
        """
        print(f'\n[SHORTEST PATH] {"="*70}')
        print(f'[SHORTEST PATH] Finding optimal path: {from_waypoint_name} → {to_waypoint_name}')

        # Get current graph structure
        graph_data = self.get_graph_structure()
        waypoints = graph_data['waypoints']
        edges = graph_data['edges']

        if not waypoints:
            print(f'[SHORTEST PATH] ✗ No waypoints in graph')
            return None

        # Build waypoint name to ID mapping
        wp_name_to_id = {}
        wp_id_to_name = {}
        for wp in waypoints:
            name = wp.annotations.name
            wp_name_to_id[name] = wp.id
            wp_id_to_name[wp.id] = name

        # Check if waypoints exist
        if from_waypoint_name not in wp_name_to_id:
            print(f'[SHORTEST PATH] ✗ Starting waypoint {from_waypoint_name} not found')
            return None
        if to_waypoint_name not in wp_name_to_id:
            print(f'[SHORTEST PATH] ✗ Destination waypoint {to_waypoint_name} not found')
            return None

        # Build adjacency list from existing edges
        adjacency = {name: [] for name in wp_name_to_id.keys()}
        for edge in edges:
            from_id = edge.id.from_waypoint
            to_id = edge.id.to_waypoint
            if from_id in wp_id_to_name and to_id in wp_id_to_name:
                from_name = wp_id_to_name[from_id]
                to_name = wp_id_to_name[to_id]
                adjacency[from_name].append(to_name)

        print(f'[SHORTEST PATH] Graph has {len(waypoints)} waypoints and {len(edges)} edges')

        # Try to find path with existing edges first
        path = self._dijkstra(from_waypoint_name, to_waypoint_name, adjacency)

        if path:
            print(f'[SHORTEST PATH] ✓ Path found with existing edges: {" → ".join(path)}')
            print(f'[SHORTEST PATH] Path length: {len(path) - 1} hops')
            print(f'[SHORTEST PATH] {"="*70}\n')
            return path

        # No path found - try to create missing edges between adjacent visited cells
        print(f'[SHORTEST PATH] No direct path found with existing edges')
        print(f'[SHORTEST PATH] Analyzing potential shortcuts through adjacent visited cells...')

        # Find and create missing edges
        edges_created = self._create_missing_edges_for_path(from_waypoint_name, to_waypoint_name, env_map)

        if edges_created > 0:
            # Rebuild graph and try again
            print(f'[SHORTEST PATH] Retrying pathfinding with {edges_created} new edges...')
            graph_data = self.get_graph_structure()
            edges = graph_data['edges']

            # Rebuild adjacency list
            adjacency = {name: [] for name in wp_name_to_id.keys()}
            for edge in edges:
                from_id = edge.id.from_waypoint
                to_id = edge.id.to_waypoint
                if from_id in wp_id_to_name and to_id in wp_id_to_name:
                    from_name = wp_id_to_name[from_id]
                    to_name = wp_id_to_name[to_id]
                    adjacency[from_name].append(to_name)

            path = self._dijkstra(from_waypoint_name, to_waypoint_name, adjacency)

            if path:
                print(f'[SHORTEST PATH] ✓ Path found after creating shortcuts: {" → ".join(path)}')
                print(f'[SHORTEST PATH] Path length: {len(path) - 1} hops')
                print(f'[SHORTEST PATH] {"="*70}\n')
                return path

        print(f'[SHORTEST PATH] ✗ No path exists between waypoints (cells may not be connected)')
        print(f'[SHORTEST PATH] {"="*70}\n')
        return None

    def _create_missing_edges_for_path(self, from_waypoint_name, to_waypoint_name, env_map):
        """
        Create missing edges between waypoints in adjacent visited cells to enable shortest path.

        Args:
            from_waypoint_name: Starting waypoint name
            to_waypoint_name: Destination waypoint name
            env_map: EnvironmentMap object

        Returns:
            int: Number of edges created
        """
        edges_created = 0

        # Get all manual waypoints with cell information
        waypoints_with_cells = {}
        for wp_name, wp_data in self.waypoint_poses.items():
            if wp_data.get('cell_row') is not None and wp_data.get('cell_col') is not None:
                cell = (wp_data['cell_row'], wp_data['cell_col'])
                waypoints_with_cells[wp_name] = cell

        if len(waypoints_with_cells) < 2:
            print(f'[EDGE CREATION] Not enough waypoints with cell data ({len(waypoints_with_cells)})')
            return 0

        # Find all pairs of waypoints in adjacent visited cells
        waypoint_names = list(waypoints_with_cells.keys())
        for i in range(len(waypoint_names)):
            for j in range(i + 1, len(waypoint_names)):
                wp1_name = waypoint_names[i]
                wp2_name = waypoint_names[j]

                cell1 = waypoints_with_cells[wp1_name]
                cell2 = waypoints_with_cells[wp2_name]

                # Check if cells are adjacent (horizontally or vertically)
                row_diff = abs(cell1[0] - cell2[0])
                col_diff = abs(cell1[1] - cell2[1])

                # Adjacent if exactly one cell apart in one direction (no diagonals)
                if (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1):
                    # Check if both cells are visited
                    if env_map.is_visited(cell1[0], cell1[1]) and env_map.is_visited(cell2[0], cell2[1]):
                        print(f'[EDGE CREATION] Found adjacent visited cells: {wp1_name} {cell1} ↔ {wp2_name} {cell2}')

                        # Create bidirectional edges
                        if self.create_edge_between_waypoints(wp1_name, wp2_name):
                            edges_created += 1
                        if self.create_edge_between_waypoints(wp2_name, wp1_name):
                            edges_created += 1

        return edges_created

    def _dijkstra(self, start, end, adjacency):
        """
        Dijkstra's algorithm for finding shortest path in unweighted graph.

        Args:
            start: Starting waypoint name
            end: Destination waypoint name
            adjacency: Dictionary mapping waypoint names to list of neighbors

        Returns:
            list: Path as list of waypoint names, or None if no path exists
        """
        import heapq

        # Priority queue: (distance, waypoint_name)
        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            # Found destination
            if current == end:
                # Reconstruct path
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = previous.get(node)
                return list(reversed(path))

            # Explore neighbors
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue

                # In unweighted graph, distance is just hop count
                new_dist = current_dist + 1

                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # No path found
        return None

    def navigate_shortest_path(self, from_waypoint_name, to_waypoint_name, env_map, robot_state_client):
        """
        Navigate from one waypoint to another using the shortest path.
        Automatically creates missing edges if needed.

        Args:
            from_waypoint_name: Starting waypoint name (e.g., 'wp_5')
            to_waypoint_name: Destination waypoint name (e.g., 'wp_12')
            env_map: EnvironmentMap object
            robot_state_client: Robot state client for realignment

        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        path = self.find_shortest_path(from_waypoint_name, to_waypoint_name, env_map)

        if not path:
            print(f'[NAV SHORTEST] ✗ Cannot navigate - no path exists')
            return False

        if len(path) == 1:
            print(f'[NAV SHORTEST] Already at destination {to_waypoint_name}')
            return True

        print(f'\n[NAV SHORTEST] {"="*70}')
        print(f'[NAV SHORTEST] Navigating shortest path: {" → ".join(path)}')
        print(f'[NAV SHORTEST] Total hops: {len(path) - 1}')
        print(f'[NAV SHORTEST] {"="*70}\n')

        # Navigate through each waypoint in the path
        for i in range(1, len(path)):
            current_wp = path[i]
            hop_num = i
            total_hops = len(path) - 1

            print(f'[NAV SHORTEST] Hop {hop_num}/{total_hops}: Navigating to {current_wp}...')

            # Get waypoint ID
            wp_data = self.waypoint_poses.get(current_wp)
            if not wp_data or 'waypoint_id' not in wp_data:
                print(f'[NAV SHORTEST] ✗ Waypoint {current_wp} not found in poses')
                return False

            wp_id = wp_data['waypoint_id']

            # Navigate to waypoint
            success = self.navigate_to_waypoint(wp_id, robot_state_client)

            if not success:
                print(f'[NAV SHORTEST] ✗ Failed to reach {current_wp}')
                return False

            # Realign to waypoint orientation
            self.realign_robot_to_waypoint_orientation(current_wp)
            print(f'[NAV SHORTEST] ✓ Reached {current_wp}')

        print(f'\n[NAV SHORTEST] {"="*70}')
        print(f'[NAV SHORTEST] ✓ Successfully navigated to {to_waypoint_name}')
        print(f'[NAV SHORTEST] {"="*70}\n')
        return True

    def should_we_start_recording(self):
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    return False
        return True

    def get_localization_state(self):
        """
        Ottiene lo stato di localizzazione corrente del robot nel grafo.

        Questo metodo restituisce informazioni dettagliate su:
        - Se il robot è localizzato nel grafo
        - A quale waypoint è localizzato
        - La trasformazione tra waypoint e corpo del robot
        - Il livello di confidenza della localizzazione

        Returns:
            dict: Dizionario con informazioni sulla localizzazione:
                - 'is_localized': bool - True se il robot è localizzato
                - 'waypoint_id': str - ID del waypoint corrente (se localizzato)
                - 'waypoint_name': str - Nome del waypoint corrente (se disponibile)
                - 'waypoint_tform_body': SE3Pose - Trasformazione waypoint->body
                - 'localization_state': obj - Oggetto LocalizationState completo
                - 'seed_tform_body': SE3Pose - Seed transform (se disponibile)
                Oppure None se si verifica un errore

        Example:
            # Controlla se il robot è localizzato
            loc_state = recording.get_localization_state()
            if loc_state and loc_state['is_localized']:
                print(f"Robot localizzato a: {loc_state['waypoint_name']}")
                print(f"Waypoint ID: {loc_state['waypoint_id']}")
            else:
                print("Robot NON localizzato nel grafo")
        """
        try:
            # Get localization state from GraphNav client
            localization_state = self._graph_nav_client.get_localization_state()

            # Check if robot is localized (has a valid waypoint_id)
            is_localized = bool(localization_state.localization.waypoint_id)
            waypoint_id = localization_state.localization.waypoint_id if is_localized else None

            # Search for waypoint name if localized
            waypoint_name = None
            if is_localized:
                graph = self._graph_nav_client.download_graph()
                for waypoint in graph.waypoints:
                    if waypoint.id == waypoint_id:
                        waypoint_name = waypoint.annotations.name if waypoint.annotations.name else 'unnamed'
                        break

            # Prepare return dictionary
            result = {
                'is_localized': is_localized,
                'waypoint_id': waypoint_id,
                'waypoint_name': waypoint_name,
                'waypoint_tform_body': localization_state.localization.waypoint_tform_body,
                'localization_state': localization_state,
                'seed_tform_body': localization_state.localization.seed_tform_body if hasattr(localization_state.localization, 'seed_tform_body') else None
            }

            # Print debug information
            if is_localized:
                print(f"\n[LOCALIZATION STATE] ✓ Robot is LOCALIZED")
                print(f"  Waypoint ID: {waypoint_id}")
                print(f"  Waypoint Name: {waypoint_name}")

                # Print relative position if available
                if localization_state.localization.waypoint_tform_body:
                    pos = localization_state.localization.waypoint_tform_body.position
                    print(f"  Position relative to waypoint: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
            else:
                print(f"\n[LOCALIZATION STATE] ✗ Robot is NOT LOCALIZED")
                print(f"  No waypoint_id found in localization state")

            return result

        except Exception as e:
            print(f"\n[LOCALIZATION STATE] ✗ Error getting localization state: {e}")
            return None

    def create_default_waypoint(self, cell_row=None, cell_col=None):
        """Create a waypoint with an incremental ID (e.g., wp_0, wp_1).
        Now also saves the robot's pose and optionally the cell position.

        Args:
            cell_row: Row index of the cell where waypoint is created (optional)
            cell_col: Column index of the cell where waypoint is created (optional)
        """

        # Import spotUtils to get position
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import spotUtils
        import numpy as np

        # Get current robot position
        robot_state_client = self.robot.ensure_client('robot-state')
        x, y, z, quat = spotUtils.getPosition(robot_state_client)

        # Calculate yaw from quaternion
        yaw = np.arctan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y**2 + quat.z**2)
        )

        graph = self._graph_nav_client.download_graph()
        if not graph.waypoints:
            next_number = 0
        else:
            # Find highest number among manual waypoints (wp_N format)
            max_number = -1
            for wp in graph.waypoints:
                name = wp.annotations.name
                if name and name.startswith('wp_'):
                    try:
                        number = int(name.split('_')[-1])
                        max_number = max(max_number, number)
                    except (ValueError, IndexError):
                        continue
            next_number = max_number + 1

        new_name = f'wp_{next_number}'
        print(f"\n[WAYPOINT CREATION] Creating new MANUAL waypoint:")
        print(f"  Name: {new_name} (manual waypoint format)")
        print(f"  Number: {next_number}")
        print(f"  Position: ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"  Orientation (yaw): {np.degrees(yaw):.1f}°")
        if cell_row is not None and cell_col is not None:
            print(f"  Cell: ({cell_row}, {cell_col})")

        resp = self._recording_client.create_waypoint(waypoint_name=new_name)

        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            # Waypoint created, get ID from response
            created_waypoint_id = resp.created_waypoint.id if hasattr(resp, 'created_waypoint') else 'ID_not_available'

            # Save waypoint pose with cell (if available)
            self.waypoint_poses[new_name] = {
                'x': x,
                'y': y,
                'z': z,
                'yaw': yaw,
                'waypoint_id': created_waypoint_id,
                'cell_row': cell_row,
                'cell_col': cell_col
            }

            print(f"[WAYPOINT CREATION] ✓ Successfully created waypoint:")
            print(f"  Name: {new_name}")
            print(f"  ID: {created_waypoint_id}")
            print(f"  Pose saved for future realignment")
            if cell_row is not None and cell_col is not None:
                print(f"  Cell saved: ({cell_row}, {cell_col})")
            print(f"  Status: {resp.status}\n")

            return resp
        else:
            print(f"[WAYPOINT CREATION] ✗ Could not create waypoint {new_name}")
            print(f"  Status: {resp.status}\n")
            return False

    def get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
            print(status)
        else:
            print('The recording service is off.')

    def clear_map(self, *args):
        return self._graph_nav_client.clear_graph()

    def start_recording(self, *args):
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print(
                'The system is not in the proper state to start recording.'
                'Try using the graph_nav_command_line to either clear the map or'
                'attempt to localize to the map.'
            )
            return
        try:
            status = self._recording_client.start_recording(recording_environment=self._recording_environment)
            print('Successfully started recording a map.')
        except Exception as err:
            print(f'Start recording failed: {err}')

    def stop_recording(self, *args):
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print('Successfully stopped recording a map.')
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                if first_iter:
                    print('Cleaning up recording...')
                first_iter = False
                time.sleep(0.5)
                continue
            except Exception as err:
                print(f'Stop recording failed: {err}')
                break

    def online_full_graph_download(self):
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        for edge in graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            try:
                self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
        for waypoint in graph.waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                self._graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
            except Exception:
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue

    def download_full_graph_with_name(self, map_name, include_snapshots=True):
        """
        Download the full graph and save to a folder with a custom name.
        If a folder with the same name exists, appends a number (e.g., map_1, map_2).

        Args:
            map_name: Custom name for the map folder
            include_snapshots: If True, downloads waypoint and edge snapshots (heavy data).
                             If False, downloads only the graph structure (lightweight).

        Returns:
            str: Path to the downloaded map folder, or None if download failed
        """
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('[MAP DOWNLOAD] ✗ Failed to download the graph.')
            return None

        # Generate unique folder with custom name
        base_path = os.path.join(self._base_download_filepath, map_name)
        unique_path = base_path

        # If folder exists, append number
        if os.path.exists(unique_path):
            counter = 1
            while os.path.exists(f"{base_path}_{counter}"):
                counter += 1
            unique_path = f"{base_path}_{counter}"
            print(f"[MAP DOWNLOAD] Folder '{map_name}' exists, using '{os.path.basename(unique_path)}' instead")

        # Temporarily change download path
        old_path = self._download_filepath
        self._download_filepath = unique_path

        try:
            mode = "FULL" if include_snapshots else "LIGHTWEIGHT"
            print(f'\n[MAP DOWNLOAD - {mode}] Starting download...')
            print(f'[MAP DOWNLOAD - {mode}] Destination: {os.path.basename(unique_path)}')

            self._write_full_graph(graph)

            if include_snapshots:
                print(f'[MAP DOWNLOAD - {mode}] Graph structure saved: {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')
                print(f'[MAP DOWNLOAD - {mode}] Downloading waypoint snapshots...')
                self._download_and_write_waypoint_snapshots(graph.waypoints)
                print(f'[MAP DOWNLOAD - {mode}] Downloading edge snapshots...')
                self._download_and_write_edge_snapshots(graph.edges)
                print(f'[MAP DOWNLOAD - {mode}] ✓ Complete map saved to: {unique_path}')
            else:
                # Calculate size saved by skipping snapshots
                waypoints_skipped = sum(1 for wp in graph.waypoints if len(wp.snapshot_id) > 0)
                edges_skipped = sum(1 for edge in graph.edges if len(edge.snapshot_id) > 0)

                print(f'[MAP DOWNLOAD - {mode}] Graph structure saved:')
                print(f'  • {len(graph.waypoints)} waypoints')
                print(f'  • {len(graph.edges)} edges')
                print(f'[MAP DOWNLOAD - {mode}] Snapshots SKIPPED:')
                print(f'  • {waypoints_skipped} waypoint snapshots')
                print(f'  • {edges_skipped} edge snapshots')
                print(f'[MAP DOWNLOAD - {mode}] ✓ Lightweight graph saved to: {unique_path}')

            return unique_path

        finally:
            # Restore original path
            self._download_filepath = old_path

    def download_graph_lightweight(self, *args):
        """
        Download ONLY the graph structure (waypoints and edges) WITHOUT snapshots.
        This is much faster and uses less storage - perfect for navigation purposes.

        The graph structure contains:
        - Waypoint IDs, names, and poses (positions/orientations)
        - Edge connections and transformations
        - Annotations

        NOT included (snapshots contain heavy data like images and point clouds):
        - Waypoint snapshots (visual features, images)
        - Edge snapshots (visual features from edge traversal)

        Returns:
            str: Path to the downloaded graph file, or None if download failed
        """
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('[MAP DOWNLOAD] ✗ Failed to download the graph.')
            return None

        # Generate unique folder for this download
        unique_path = self._generate_unique_map_folder('graph_lightweight')

        # Temporarily change download path
        old_path = self._download_filepath
        self._download_filepath = unique_path

        try:
            print(f'\n[MAP DOWNLOAD - LIGHTWEIGHT] Starting download...')
            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] Destination: {os.path.basename(unique_path)}')
            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] Mode: Structure only (NO snapshots)')

            self._write_full_graph(graph)

            # Calculate size saved by skipping snapshots
            waypoints_skipped = sum(1 for wp in graph.waypoints if len(wp.snapshot_id) > 0)
            edges_skipped = sum(1 for edge in graph.edges if len(edge.snapshot_id) > 0)

            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] Graph structure saved:')
            print(f'  • {len(graph.waypoints)} waypoints')
            print(f'  • {len(graph.edges)} edges')
            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] Snapshots SKIPPED:')
            print(f'  • {waypoints_skipped} waypoint snapshots')
            print(f'  • {edges_skipped} edge snapshots')
            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] ✓ Lightweight graph saved to: {unique_path}')
            print(f'[MAP DOWNLOAD - LIGHTWEIGHT] This graph can be used for navigation and edge creation.')

            return unique_path

        finally:
            # Restore original path
            self._download_filepath = old_path

    def download_full_graph(self, include_snapshots=True, *args):
        """
        Download the full graph and optionally include snapshots.
        Each download creates a new folder with timestamp to avoid overwriting previous maps.

        Args:
            include_snapshots: If True, downloads waypoint and edge snapshots (heavy data).
                             If False, downloads only the graph structure (lightweight).

        Returns:
            str: Path to the downloaded map folder, or None if download failed
        """
        # If user explicitly wants lightweight version
        if not include_snapshots:
            return self.download_graph_lightweight()

        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('[MAP DOWNLOAD] ✗ Failed to download the graph.')
            return None

        # Generate unique folder for this download
        unique_path = self._generate_unique_map_folder()

        # Temporarily change download path
        old_path = self._download_filepath
        self._download_filepath = unique_path

        try:
            print(f'\n[MAP DOWNLOAD - FULL] Starting download...')
            print(f'[MAP DOWNLOAD - FULL] Destination: {os.path.basename(unique_path)}')
            print(f'[MAP DOWNLOAD - FULL] Mode: Full (with snapshots)')

            self._write_full_graph(graph)
            print(f'[MAP DOWNLOAD - FULL] Graph structure saved: {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')

            print(f'[MAP DOWNLOAD - FULL] Downloading waypoint snapshots...')
            self._download_and_write_waypoint_snapshots(graph.waypoints)

            print(f'[MAP DOWNLOAD - FULL] Downloading edge snapshots...')
            self._download_and_write_edge_snapshots(graph.edges)

            print(f'[MAP DOWNLOAD - FULL] ✓ Complete map saved to: {unique_path}')
            return unique_path

        finally:
            # Restore original path
            self._download_filepath = old_path

    def _write_full_graph(self, graph):
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
            except Exception:
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue
            self._write_bytes(
                os.path.join(self._download_filepath, 'waypoint_snapshots'),
                str(waypoint.snapshot_id),
                waypoint_snapshot.SerializeToString(),
            )
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
            self._write_bytes(
                os.path.join(self._download_filepath, 'edge_snapshots'),
                str(edge.snapshot_id),
                edge_snapshot.SerializeToString(),
            )
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print('Robot is impaired.')
            return True
        elif status.status == 7:
            return True
        else:
            return False

    def navigate_to_first_waypoint(self, robot_state_client):
        """
        Naviga verso il primo waypoint (wp_0) in modalità standard.
        Il robot arriverà vicino al punto e si fermerà con l'orientamento attuale.
        """
        # 1. Trova l'ID di wp_0
        graph = self._graph_nav_client.download_graph()
        first_waypoint = None
        for waypoint in graph.waypoints:
            if waypoint.annotations.name == "wp_0":
                first_waypoint = waypoint
                break

        if first_waypoint is None:
            print('[ERROR] Nessun "wp_0" trovato nel grafo.')
            return False

        print(f"\n[RETURN] Ritorno alla base (wp_0)...")

        nav_to_cmd_id = None
        is_finished = False

        # Parametri standard
        travel_params = TravelParams()

        while not is_finished:
            try:
                # Navigazione SEMPLICE (senza destination_waypoint_tform_body_goal)
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    first_waypoint.id,
                    1.0,
                    command_id=nav_to_cmd_id,
                    travel_params=travel_params
                )
            except Exception as e:
                print(f"[RETURN] Errore invio comando: {e}")
                time.sleep(0.5)
                continue

            time.sleep(0.5)

            # Controlla feedback
            try:
                feedback = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)

                if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    print("[RETURN] ✓ Arrivato a wp_0.")
                    return True

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    print("[RETURN] ⚠️ STATUS_LOST mentre tornavo a casa!")
                    # Tentativo di recupero (sempre meglio averlo)
                    if robot_state_client:
                        recovered = self.force_localization_to_waypoint(robot_state_client, first_waypoint.id)
                        if recovered: continue  # Riprova il loop
                    return False

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    print("[RETURN] ⚠️ Robot bloccato (STUCK).")
                    return False

            except Exception as e:
                print(f"[RETURN] Errore feedback: {e}")
                return False

        return False

    def get_waypoint_list(self):
        """
        Get the list of all waypoints in the current graph.

        Returns:
            list: List of waypoint objects from the graph
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            return []
        return list(graph.waypoints)

    def get_waypoint_details_list(self, only_manual=True):
        """
        Get detailed information about waypoints in the current graph.

        Args:
            only_manual: Se True, restituisce solo i waypoint manuali (formato 'wp_N').
                        Se False, restituisce tutti i waypoint del grafo.

        Returns:
            list: List of dictionaries with waypoint details:
                - 'id': Waypoint unique ID
                - 'name': Waypoint name (e.g., 'wp_0')
                - 'x': X position in world coordinates
                - 'y': Y position in world coordinates
                - 'z': Z position in world coordinates
                - 'waypoint_obj': Full waypoint object

        Example:
            # Solo waypoint manuali
            waypoints = recording.get_waypoint_details_list(only_manual=True)
            for wp in waypoints:
                print(f"Waypoint {wp['name']} at ({wp['x']:.2f}, {wp['y']:.2f})")
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            print("[WAYPOINTS] No waypoints found in graph")
            return []

        waypoint_details = []
        skipped_count = 0

        for waypoint in graph.waypoints:
            name = waypoint.annotations.name if waypoint.annotations.name else 'unnamed'

            # Filter only manual waypoints (wp_N format)
            if only_manual:
                if not (name.startswith('wp_') and len(name.split('_')) == 2 and name.split('_')[-1].isdigit()):
                    skipped_count += 1
                    continue

            # Extract position from waypoint_tform_ko (waypoint transform from kinematic odometry)
            transform = waypoint.waypoint_tform_ko

            details = {
                'id': waypoint.id,
                'name': name,
                'x': transform.position.x,
                'y': transform.position.y,
                'z': transform.position.z,
                'waypoint_obj': waypoint
            }
            waypoint_details.append(details)

        if only_manual:
            print(f"[WAYPOINTS] Found {len(waypoint_details)} MANUAL waypoints (wp_N format) in graph (skipped {skipped_count} automatic):")
        else:
            print(f"[WAYPOINTS] Found {len(waypoint_details)} waypoints in graph:")

        for wp in waypoint_details:
            print(f"  - {wp['name']}: ID={wp['id']}, pos=({wp['x']:.3f}, {wp['y']:.3f}, {wp['z']:.3f})")

        return waypoint_details

    def get_last_waypoint(self):
        """
        Trova l'ultimo waypoint MANUALE creato (quello con il numero più alto nel nome wp_N).

        Analizza tutti i waypoint manuali nel grafo e restituisce quello con il numero più alto
        nel nome (es. wp_5 è l'ultimo se ci sono wp_0, wp_1, ..., wp_5).

        Returns:
            dict or None: Dizionario con dettagli dell'ultimo waypoint manuale:
                - 'id': Waypoint unique ID
                - 'name': Waypoint name (e.g., 'wp_5')
                - 'number': Numero del waypoint (es. 5)
                - 'x': X position in world coordinates
                - 'y': Y position in world coordinates
                - 'z': Z position in world coordinates
                - 'waypoint_obj': Full waypoint object
                Oppure None se non ci sono waypoint manuali nel grafo.

        Example:
            last_wp = recording.get_last_waypoint()
            if last_wp:
                print(f"Ultimo waypoint: {last_wp['name']} (#{last_wp['number']})")
                print(f"Posizione: ({last_wp['x']:.2f}, {last_wp['y']:.2f})")
        """
        graph = self._graph_nav_client.download_graph()
        if not graph or len(graph.waypoints) == 0:
            print("[LAST_WP] No waypoints found in graph")
            return None

        last_waypoint = None
        max_number = -1

        for waypoint in graph.waypoints:
            # Extract number from manual waypoint name (e.g. "wp_5" -> 5)
            name = waypoint.annotations.name if waypoint.annotations.name else 'unnamed'

            try:
                # Try to extract number after "wp_" (manual waypoints only)
                if name.startswith('wp_') and len(name.split('_')) == 2:
                    number = int(name.split('_')[-1])

                    if number > max_number:
                        max_number = number
                        transform = waypoint.waypoint_tform_ko

                        last_waypoint = {
                            'id': waypoint.id,
                            'name': name,
                            'number': number,
                            'x': transform.position.x,
                            'y': transform.position.y,
                            'z': transform.position.z,
                            'waypoint_obj': waypoint
                        }
            except (ValueError, IndexError):
                # Skip waypoints with non-standard names
                continue

        if last_waypoint:
            print(f"[LAST_WP] ✓ Ultimo waypoint manuale: {last_waypoint['name']} (#{last_waypoint['number']})")
            print(f"[LAST_WP] Posizione: ({last_waypoint['x']:.3f}, {last_waypoint['y']:.3f}, {last_waypoint['z']:.3f})")
            print(f"[LAST_WP] ID: {last_waypoint['id']}")
        else:
            print(f"[LAST_WP] ⚠️ Nessun waypoint manuale con formato 'wp_N' trovato")

        return last_waypoint

    def find_nearest_waypoint_to_cell(self, target_row, target_col):
        """
        Find the nearest manual waypoint to a target cell using distance between cells.

        This method searches among manual waypoints (wp_N) that have saved cell
        and calculates Manhattan distance (|row1-row2| + |col1-col2|) instead of euclidean distance.

        Args:
            target_row: Row index of target cell
            target_col: Column index of target cell

        Returns:
            dict or None: Nearest waypoint with:
                - 'id': Waypoint ID
                - 'name': Waypoint name (e.g. 'wp_5')
                - 'x', 'y', 'z': World position
                - 'cell_row', 'cell_col': Waypoint cell
                - 'distance': Manhattan distance in cells
                Or None if no waypoints with saved cells

        Example:
            # Find nearest waypoint to cell (3, 2)
            nearest = recording.find_nearest_waypoint_to_cell(3, 2)
            if nearest:
                print(f"Nearest waypoint: {nearest['name']} in cell ({nearest['cell_row']}, {nearest['cell_col']})")
                print(f"Distance: {nearest['distance']} cells")
        """
        print(f"\n[NEAREST_WP_CELL] Searching nearest waypoint to cell ({target_row}, {target_col})")
        print(f"[NEAREST_WP_CELL] 🔍 Using Manhattan distance between cells")

        # Filter waypoints that have saved cell
        valid_waypoints = []
        for wp_name, wp_data in self.waypoint_poses.items():
            if wp_data.get('cell_row') is not None and wp_data.get('cell_col') is not None:
                # Calculate Manhattan distance
                distance = abs(wp_data['cell_row'] - target_row) + abs(wp_data['cell_col'] - target_col)

                valid_waypoints.append({
                    'name': wp_name,
                    'id': wp_data['waypoint_id'],
                    'x': wp_data['x'],
                    'y': wp_data['y'],
                    'z': wp_data['z'],
                    'yaw': wp_data['yaw'],
                    'cell_row': wp_data['cell_row'],
                    'cell_col': wp_data['cell_col'],
                    'distance': distance
                })

                print(f"  {wp_name}: cella ({wp_data['cell_row']}, {wp_data['cell_col']}) - distanza: {distance} celle")

        if not valid_waypoints:
            print(f"[NEAREST_WP_CELL] ✗ No waypoint with saved cell found")
            print(f"[NEAREST_WP_CELL] Suggestion: pass cell_row and cell_col to create_default_waypoint()")
            return None

        # Find waypoint with minimum distance
        nearest = min(valid_waypoints, key=lambda x: x['distance'])

        print(f"[NEAREST_WP_CELL] ✓ Nearest waypoint: {nearest['name']}")
        print(f"[NEAREST_WP_CELL]   Waypoint cell: ({nearest['cell_row']}, {nearest['cell_col']})")
        print(f"[NEAREST_WP_CELL]   Target cell: ({target_row}, {target_col})")
        print(f"[NEAREST_WP_CELL]   Distance: {nearest['distance']} cells (Manhattan)")

        return nearest

    def find_nearest_waypoint_to_position(self, target_x, target_y, return_all_distances=False, only_manual=True):
        """
        Find the nearest waypoint to a specified world position (x, y).

        Args:
            target_x: World X coordinate of target position
            target_y: World Y coordinate of target position
            return_all_distances: If True, print all distances
            only_manual: If True, consider only manual waypoints (format 'wp_N')

        Returns:
            dict or None: Dictionary with nearest waypoint details:
                - 'id': Waypoint ID
                - 'name': Waypoint name
                - 'x', 'y', 'z': Position
                - 'distance': Euclidean distance from target position
                - 'waypoint_obj': Complete waypoint object
                Or None if no waypoints

        Example:
            # Find nearest manual waypoint to cell center
            cell_center_x, cell_center_y = env.get_world_position_from_cell(row, col)
            nearest = recording.find_nearest_waypoint_to_position(cell_center_x, cell_center_y, only_manual=True)
            if nearest:
                print(f"Nearest waypoint: {nearest['name']} at {nearest['distance']:.2f}m")
        """
        import math

        waypoints = self.get_waypoint_details_list(only_manual=only_manual)
        if not waypoints:
            filter_type = "manual (wp_N)" if only_manual else "any"
            print(f"[NEAREST_WP] No {filter_type} waypoints available in graph")
            return None

        print(f"\n[NEAREST_WP] Searching nearest waypoint to ({target_x:.3f}, {target_y:.3f})")
        if only_manual:
            print(f"[NEAREST_WP] 🔍 Active filter: MANUAL waypoints only (format 'wp_N')")

        min_distance = float('inf')
        nearest_waypoint = None

        for wp in waypoints:
            # Calculate euclidean distance (ignoring Z for simplicity)
            distance = math.sqrt((wp['x'] - target_x)**2 + (wp['y'] - target_y)**2)

            print(f"  {wp['name']}: ({wp['x']:.3f}, {wp['y']:.3f}) - distanza: {distance:.3f}m")

            if distance < min_distance:
                min_distance = distance
                nearest_waypoint = wp.copy()
                nearest_waypoint['distance'] = distance

        if nearest_waypoint:
            print(f"[NEAREST_WP] ✓ Nearest waypoint: {nearest_waypoint['name']} "
                  f"at {nearest_waypoint['distance']:.3f}m")

        return nearest_waypoint

    def navigate_to_waypoint(self, waypoint_id, robot_state_client):
        """
        Naviga a un waypoint. Se il robot si perde, tenta di forzare la localizzazione
        sul waypoint target (assumendo di esserci vicino).
        """
        # Scarica il grafo per avere i nomi aggiornati
        graph = self._graph_nav_client.download_graph()
        target_waypoint_name = "unknown"
        for wp in graph.waypoints:
            if wp.id == waypoint_id:
                target_waypoint_name = wp.annotations.name
                break

        print(f"\n[NAV] Navigazione verso {target_waypoint_name} (ID: {waypoint_id})...")

        nav_to_cmd_id = None

        # Parametri di viaggio (opzionali)
        travel_params = TravelParams()

        while True:
            # Invia comando navigazione
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    waypoint_id,
                    1.0,
                    command_id=nav_to_cmd_id,
                    travel_params=travel_params
                )
            except Exception as e:
                print(f"[NAV] Errore invio comando: {e}")
                break

            time.sleep(0.5)

            # Controlla stato
            try:
                feedback = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)

                if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    print(f"[NAV] ✓ Arrivato a {target_waypoint_name}")
                    return True

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    print(f"[NAV] ⚠️ STATUS_LOST rilevato durante la navigazione!")

                    # --- LOGICA DI RECUPERO ---
                    print(f"[NAV] Tento di forzare la localizzazione su {target_waypoint_name}...")
                    recovered = self.force_localization_to_waypoint(robot_state_client, waypoint_id)

                    if recovered:
                        print(f"[NAV] Recupero riuscito. Considero il robot arrivato (o pronto per riprovare).")
                        # Opzione: Ritorna True perché ci siamo localizzati "sopra"
                        return True
                    else:
                        print(f"[NAV] ✗ Recupero fallito. Robot perso definitivamente.")
                        return False

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    print(f"[NAV] ⚠️ Robot STUCK (bloccato).")
                    return False

            except Exception as e:
                print(f"[NAV] Errore feedback: {e}")
                return False

        return False

    def realign_robot_to_waypoint_orientation(self, waypoint_name):
        """
        Re-orient the robot to align it with the orientation it had when creating a waypoint.

        This method:
        1. Retrieves the saved orientation of the waypoint
        2. Gets the current orientation of the robot
        3. Calculates the angular difference
        4. Commands the robot to rotate to align

        Args:
            waypoint_name: Waypoint name (e.g. 'waypoint_5')

        Returns:
            bool: True if realignment succeeded
        """
        import numpy as np
        import spotUtils
        from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand

        # Check that we have saved pose for this waypoint
        if waypoint_name not in self.waypoint_poses:
            print(f"[REALIGN] ✗ No saved pose for waypoint '{waypoint_name}'")
            print(f"[REALIGN] Available waypoint poses: {list(self.waypoint_poses.keys())}")
            return False

        saved_pose = self.waypoint_poses[waypoint_name]
        target_yaw = saved_pose['yaw']

        print(f"\n[REALIGN] Realigning to waypoint '{waypoint_name}' orientation")
        print(f"[REALIGN] Target orientation (yaw): {np.degrees(target_yaw):.1f}°")

        # Get current robot orientation
        robot_state_client = self.robot.ensure_client('robot-state')
        x_current, y_current, z_current, quat_current = spotUtils.getPosition(robot_state_client)

        # Calculate current yaw
        current_yaw = np.arctan2(
            2.0 * (quat_current.w * quat_current.z + quat_current.x * quat_current.y),
            1.0 - 2.0 * (quat_current.y**2 + quat_current.z**2)
        )

        print(f"[REALIGN] Current orientation (yaw): {np.degrees(current_yaw):.1f}°")

        # Calculate required rotation
        delta_yaw = target_yaw - current_yaw

        # Normalize angle between -π and π
        while delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        while delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi

        print(f"[REALIGN] Rotation needed: {np.degrees(delta_yaw):.1f}°")

        # If difference is small, no need to rotate
        if abs(delta_yaw) < np.radians(5):  # 5 degree tolerance
            print(f"[REALIGN] ✓ Robot already aligned (diff < 5°)")
            return True

        # Execute rotation
        try:
            command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

            # Create rotation command in place
            footprint_R_body = bosdyn.geometry.EulerZXY(yaw=delta_yaw, roll=0, pitch=0)
            cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)

            print(f"[REALIGN] Executing rotation of {np.degrees(delta_yaw):.1f}°...")
            command_client.robot_command(cmd)

            # Wait for completion
            time.sleep(2.0)

            # Check final orientation
            x_final, y_final, z_final, quat_final = spotUtils.getPosition(robot_state_client)
            final_yaw = np.arctan2(
                2.0 * (quat_final.w * quat_final.z + quat_final.x * quat_final.y),
                1.0 - 2.0 * (quat_final.y**2 + quat_final.z**2)
            )

            final_diff = abs(final_yaw - target_yaw)
            if final_diff > np.pi:
                final_diff = 2 * np.pi - final_diff

            print(f"[REALIGN] Final orientation (yaw): {np.degrees(final_yaw):.1f}°")
            print(f"[REALIGN] Final difference: {np.degrees(final_diff):.1f}°")

            if final_diff < np.radians(10):  # 10 degree tolerance
                print(f"[REALIGN] ✓ Robot successfully realigned!")
                return True
            else:
                print(f"[REALIGN] ⚠️ Partial realignment (diff: {np.degrees(final_diff):.1f}°)")
                return True  # Accept partial realignment too

        except Exception as e:
            print(f"[REALIGN] ✗ Error during rotation: {e}")
            return False

    def resume_recording_from_waypoint(self, waypoint_name):
        """
        Riprende la registrazione da un waypoint specifico, allineando prima il robot.

        IMPORTANTE: Questo metodo assume che il robot sia già stato navigato con successo
        al waypoint specificato usando navigate_to_waypoint(). Non viene fatto alcun
        controllo di localizzazione - ci si fida che navigate_to_waypoint() abbia avuto successo.

        Questo metodo:
        1. Ri-orienta il robot per allinearlo all'orientazione originale del waypoint
        2. Riprende la registrazione in modo continuo con la mappa precedente

        Args:
            waypoint_name: Nome del waypoint da cui riprendere (es. 'wp_5')

        Returns:
            bool: True se la ripresa è riuscita

        Example:
            # Naviga a un waypoint
            success = recording.navigate_to_waypoint(waypoint_id, robot_state_client)

            if success:
                # Riprendi la registrazione da quel waypoint (senza controlli aggiuntivi)
                recording.resume_recording_from_waypoint('wp_5')
        """
        print(f"\n{'='*70}")
        print(f"[RESUME] Resuming recording from waypoint '{waypoint_name}'")
        print(f"[RESUME] Assuming robot is already at this waypoint")
        print(f"{'='*70}")

        # Step 1: Realign orientation
        print(f"\n[RESUME] Step 1/2: Realigning orientation with '{waypoint_name}'...")
        if not self.realign_robot_to_waypoint_orientation(waypoint_name):
            print(f"[RESUME] ⚠️ Realignment failed, continuing anyway...")

        # Step 2: Resume recording
        print(f"\n[RESUME] Step 2/2: Resuming recording...")

        should_start = self.should_we_start_recording()
        if not should_start:
            print(f"[RESUME] ✗ System not ready for recording")
            print(f"[RESUME] Probably already recording or map not loaded")
            return False

        try:
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment
            )
            print(f"[RESUME] ✓ Recording resumed successfully")
            print(f"[RESUME] New session is aligned with previous map")
            print(f"{'='*70}\n")
            return True

        except Exception as err:
            print(f"[RESUME] ✗ Error during resume: {err}")
            print(f"{'='*70}\n")
            return False
