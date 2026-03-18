import math
import os
import time
import numpy as np
from bosdyn.api.graph_nav.graph_nav_pb2 import TravelParams
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.api.graph_nav import graph_nav_pb2, recording_pb2, nav_pb2, map_processing_pb2
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient, map_pb2
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from google.protobuf import wrappers_pb2 as wrappers

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

        # Graph caching system
        self._cached_graph = None
        self._graph_cache_valid = False

    def optimize_anchoring(self):
        """
        Ottimizza l'ancoraggio della mappa direttamente sul server del robot.
        Corregge il drift odometrico e rende il navgraph metricamente coerente,
        senza usare planimetrie esterne.
        """
        print(f"\n[{'=' * 40}]")
        print(f"[ANCHORING] Avvio ottimizzazione globale dell'ancoraggio...")

        try:
            # Importa wrappers se non è già importato all'inizio del file
            from google.protobuf import wrappers_pb2 as wrappers
            from bosdyn.api.graph_nav import map_processing_pb2

            # Chiamiamo il servizio senza alcun "initial_hint"
            response = self._map_processing_client.process_anchoring(
                params=map_processing_pb2.ProcessAnchoringRequest.Params(
                    # Impostato a False, il robot scarta l'ancoraggio grezzo corrente
                    # e ne genera uno nuovo globale e ottimizzato da zero
                    optimize_existing_anchoring=wrappers.BoolValue(value=False)
                ),
                modify_anchoring_on_server=True,  # Applica le modifiche alla mappa in memoria
                stream_intermediate_results=False
            )

            print(f"[ANCHORING] ✓ Ottimizzazione completata!")
            print(
                f"[ANCHORING] Status: {response.status}, Iterazioni: {response.iteration}, Costo finale: {response.cost:.4f}")
            print(f"[{'=' * 40}]\n")

            # Invalida la cache per forzare il download della mappa aggiornata
            self.invalidate_graph_cache()
            return True

        except Exception as e:
            print(f"[ANCHORING] ✗ Ottimizzazione fallita: {e}")
            print(f"[{'=' * 40}]\n")
            return False

    def auto_close_loops(self, close_fiducial_loops, close_odometry_loops, *args):
        """Automatically find and close all loops in the graph."""
        response = self._map_processing_client.process_topology(
            params=map_processing_pb2.ProcessTopologyRequest.Params(
                do_fiducial_loop_closure=wrappers.BoolValue(value=close_fiducial_loops),
                do_odometry_loop_closure=wrappers.BoolValue(value=close_odometry_loops)),
            modify_map_on_server=True)
        print(f'Created {len(response.new_subgraph.edges)} new edge(s).')


    def _get_graph(self, force_refresh=False):
        """
        Get the graph, using cache if available.

        Args:
            force_refresh: If True, always download fresh graph

        Returns:
            The downloaded graph
        """
        if force_refresh or not self._graph_cache_valid or self._cached_graph is None:
            print("[GRAPH_CACHE] Downloading fresh graph from robot...")
            self._cached_graph = self._graph_nav_client.download_graph()
            self._graph_cache_valid = True
        else:
            print("[GRAPH_CACHE] Using cached graph")

        return self._cached_graph

    def invalidate_graph_cache(self):
        """
        Invalidate the graph cache.
        Call this after operations that modify the graph (creating waypoints, edges, etc.)
        or when starting recording.
        """
        self._graph_cache_valid = False
        self._cached_graph = None
        print("[GRAPH_CACHE] Cache invalidated")

    def set_download_filepath(self, filepath):
        """
        Set the download filepath for saving graphs.
        All subsequent graph downloads will be saved as subfolders inside this path.

        Args:
            filepath: Path where graphs will be saved (mission graph folder)
        """
        self._base_download_filepath = filepath
        self._download_filepath = filepath
        os.makedirs(filepath, exist_ok=True)
        print(f"[INFO] Graph download path set to: {filepath}")

    def _generate_unique_map_folder(self, base_name='Grafo'):
        """
        Generate a unique folder name for map download.
        Creates folders with readable names like "Grafo_10-02-2026_14-30-00".

        Args:
            base_name: Base name for the folder (default: 'Grafo')

        Returns:
            str: Full path to unique folder
        """
        from datetime import datetime

        # Use Italian date format: DD-MM-YYYY_HH-MM-SS
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
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
        Force robot localization to a specific waypoint.
        To be used if the robot gets lost (STATUS_LOST).
        """
        print(f"\n[RECOVERY] Attempting to restore localization to Waypoint ID: {waypoint_id}")

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

                # Don't use fiducial, use waypoint ID
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,

                # CRUCIAL: Use vision to refine position
                refine_with_visual_features=True,
                verify_visual_features_quality=True,
                do_ambiguity_check=True
            )

            print(f"[RECOVERY] ✓ Forced localization successful!")
            return True

        except Exception as e:
            print(f"[RECOVERY] ✗ Forced localization failed: {e}")
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


    def should_we_start_recording(self):
        graph = self._get_graph()
        if graph is not None:
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    return False
        return True

    def get_localization_state(self):
        """
        Gets the current localization state of the robot in the graph.

        This method returns detailed information about:
        - Whether the robot is localized in the graph
        - Which waypoint it is localized to
        - The transformation between waypoint and robot body
        - The confidence level of the localization

        Returns:
            dict: Dictionary with localization information:
                - 'is_localized': bool - True if robot is localized
                - 'waypoint_id': str - ID of current waypoint (if localized)
                - 'waypoint_name': str - Name of current waypoint (if available)
                - 'waypoint_tform_body': SE3Pose - Waypoint->body transformation
                - 'localization_state': obj - Complete LocalizationState object
                - 'seed_tform_body': SE3Pose - Seed transform (if available)
                Or None if an error occurs

        Example:
            # Check if robot is localized
            loc_state = recording.get_localization_state()
            if loc_state and loc_state['is_localized']:
                print(f"Robot localized at: {loc_state['waypoint_name']}")
                print(f"Waypoint ID: {loc_state['waypoint_id']}")
            else:
                print("Robot NOT localized in graph")
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
                graph = self._get_graph()
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

        graph = self._get_graph()
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

            self.invalidate_graph_cache()

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
            self.invalidate_graph_cache()  # Cache will be stale when recording adds new data
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


    def download_full_graph(self, *args):
        """
        Download the full graph and save to a unique folder.
        Each download creates a new folder with timestamp to avoid overwriting previous maps.

        Returns:
            str: Path to the downloaded map folder, or None if download failed
        """
        graph = self._get_graph(force_refresh=True)  # Force refresh when saving
        if graph is None:
            print('[MAP DOWNLOAD] ✗ Failed to download the graph.')
            return None

        # Generate unique folder for this download
        unique_path = self._generate_unique_map_folder()

        # Temporarily change download path
        old_path = self._download_filepath
        self._download_filepath = unique_path

        try:
            print(f'\n[MAP DOWNLOAD] Starting download...')
            print(f'[MAP DOWNLOAD] Destination: {os.path.basename(unique_path)}')

            self._write_full_graph(graph)
            print(f'[MAP DOWNLOAD] Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')

            self._download_and_write_waypoint_snapshots(graph.waypoints)
            self._download_and_write_edge_snapshots(graph.edges)

            print(f'[MAP DOWNLOAD] ✓ Map successfully saved to: {unique_path}')
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
        Navigate to the first waypoint (wp_0) in standard mode.
        The robot will arrive near the point and stop with its current orientation.
        """
        # 1. Find the ID of wp_0
        graph = self._get_graph()
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
                print(f"[RETURN] Error sending command: {e}")
                time.sleep(0.5)
                continue

            time.sleep(0.5)

            # Check feedback
            try:
                feedback = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)

                if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    print("[RETURN] ✓ Arrived at wp_0.")
                    return True

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    print("[RETURN] ⚠️ STATUS_LOST mentre tornavo a casa!")
                    # Tentativo di recupero (sempre meglio averlo)
                    if robot_state_client:
                        recovered = self.force_localization_to_waypoint(robot_state_client, first_waypoint.id)
                        if recovered: continue  # Riprova il loop
                    return False

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    print("[RETURN] ⚠️ Robot blocked (STUCK).")
                    return False

            except Exception as e:
                print(f"[RETURN] Feedback error: {e}")
                return False

        return False

    def get_waypoint_list(self):
        """
        Get the list of all waypoints in the current graph.

        Returns:
            list: List of waypoint objects from the graph
        """
        graph = self._get_graph()
        if not graph or len(graph.waypoints) == 0:
            return []
        return list(graph.waypoints)


    def get_waypoint_details_list(self, only_manual=True):
        """
        Get detailed information about waypoints in the current graph.

        Args:
            only_manual: If True, returns only manual waypoints (format 'wp_N').
                        If False, returns all waypoints in the graph.

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
        graph = self._get_graph()
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
        Navigate to a waypoint. If the robot gets lost, attempts to force localization
        to the target waypoint (assuming being close to it).
        """
        # Download graph to have updated names
        graph = self._get_graph()
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
                print(f"[NAV] Error sending command: {e}")
                break

            time.sleep(0.5)

            try:
                feedback = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)

                if feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    print(f"[NAV] ✓ Arrived at {target_waypoint_name}")
                    return True

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    print(f"[NAV]️ STATUS_LOST detected during navigation!")

                    # --- RECOVERY LOGIC ---
                    print(f"[NAV] Attempting to force localization at {target_waypoint_name}...")
                    recovered = self.force_localization_to_waypoint(robot_state_client, waypoint_id)

                    if recovered:
                        print(f"[NAV] Recovery successful. Consider robot arrived (or ready to retry).")
                        # Option: Return True because we localized "on top"
                        return True
                    else:
                        print(f"[NAV] ✗ Recovery failed. Robot definitively lost.")
                        return False

                elif feedback.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    print(f"[NAV]️ Robot STUCK (blocked).")
                    return False

            except Exception as e:
                print(f"[NAV] Feedback error: {e}")
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
            time.sleep(0.5)

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

    def get_all_manual_waypoints_with_cells(self):
        """
        Get all manual waypoints that have cell information saved.

        Returns:
            dict: Dictionary mapping (cell_row, cell_col) -> waypoint_data
                  Each waypoint_data contains: 'name', 'id', 'x', 'y', 'z', 'yaw'
        """
        waypoints_by_cell = {}

        for wp_name, wp_data in self.waypoint_poses.items():
            # Only manual waypoints (wp_N format)
            if not (wp_name.startswith('wp_') and len(wp_name.split('_')) == 2):
                continue

            # Must have cell information
            if wp_data.get('cell_row') is not None and wp_data.get('cell_col') is not None:
                cell_key = (wp_data['cell_row'], wp_data['cell_col'])
                waypoints_by_cell[cell_key] = {
                    'name': wp_name,
                    'id': wp_data['waypoint_id'],
                    'x': wp_data['x'],
                    'y': wp_data['y'],
                    'z': wp_data['z'],
                    'yaw': wp_data['yaw'],
                    'cell_row': wp_data['cell_row'],
                    'cell_col': wp_data['cell_col']
                }

        print(f"[WAYPOINTS] Found {len(waypoints_by_cell)} manual waypoints with cell data")
        return waypoints_by_cell

    def get_manual_waypoint_by_cell(self, cell_row, cell_col):
        """
        Get the manual waypoint data for a specific cell.

        Args:
            cell_row: Row of the cell
            cell_col: Column of the cell

        Returns:
            dict or None: Waypoint data if found, otherwise None.
        """
        waypoints_by_cell = self.get_all_manual_waypoints_with_cells()
        return waypoints_by_cell.get((cell_row, cell_col))

    def find_shortest_cell_path_bfs(self, start_cell, end_cell, env_map):
        """
        Find shortest path between two cells using BFS on the grid.
        Only considers visited cells (value = 1) as valid path nodes.

        Args:
            start_cell: (row, col) tuple for start
            end_cell: (row, col) tuple for end
            env_map: EnvironmentMap instance with map data

        Returns:
            list or None: List of (row, col) cells in order from start to end,
                         or None if no path exists
        """
        from collections import deque

        print(f"\n[BFS] Finding shortest path from cell {start_cell} to {end_cell}")

        # Check if start and end are valid
        start_row, start_col = start_cell
        end_row, end_col = end_cell

        if env_map.map[start_row][start_col] != 1:
            print(f"[BFS] ERROR: Start cell {start_cell} is not visited (value={env_map.map[start_row][start_col]})")
            return None

        if env_map.map[end_row][end_col] != 1:
            print(f"[BFS] ERROR: End cell {end_cell} is not visited (value={env_map.map[end_row][end_col]})")
            return None

        # BFS initialization
        queue = deque([(start_cell, [start_cell])])
        visited = {start_cell}

        while queue:
            current_cell, path = queue.popleft()
            current_row, current_col = current_cell

            # Found the goal
            if current_cell == end_cell:
                print(f"[BFS] Path found with {len(path)} cells (distance: {len(path)-1} hops)")
                print(f"[BFS] Path: {' -> '.join([f'({r},{c})' for r,c in path])}")
                return path

            # Explore neighbors (4-connectivity: N, E, S, W)
            neighbors = [
                (current_row - 1, current_col),  # North
                (current_row, current_col + 1),  # East
                (current_row + 1, current_col),  # South
                (current_row, current_col - 1)   # West
            ]

            for neighbor_row, neighbor_col in neighbors:
                # Check bounds
                if (0 <= neighbor_row < env_map.rows and
                    0 <= neighbor_col < env_map.cols):

                    neighbor_cell = (neighbor_row, neighbor_col)

                    # Only visit cells that are visited (value = 1) and not yet explored
                    if (env_map.map[neighbor_row][neighbor_col] == 1 and
                        neighbor_cell not in visited):

                        visited.add(neighbor_cell)
                        queue.append((neighbor_cell, path + [neighbor_cell]))

        print(f"[BFS] No path found from {start_cell} to {end_cell}")
        return None

    def _path_exists_in_graph(self, from_waypoint_id, to_waypoint_id, graph, cell_from, cell_to, env_map):
        """
        Check if a path exists between two waypoints in the graph using BFS.
        Only considers waypoints that are INSIDE the two cells being checked.

        This allows MULTI-HOP paths through intermediate waypoints (automatic or manual)
        as long as all intermediate waypoints are inside cell_from or cell_to.

        Args:
            from_waypoint_id: ID of the source waypoint
            to_waypoint_id: ID of the destination waypoint
            graph: The downloaded graph object
            cell_from: (row, col) tuple of the source cell
            cell_to: (row, col) tuple of the destination cell
            env_map: EnvironmentMap instance to check if waypoints are in cells

        Returns:
            bool: True if a path exists through waypoints inside the two cells, False otherwise
        """

        from collections import deque

        if from_waypoint_id == to_waypoint_id:
            return True

        # Get positions of all waypoints and check which are inside the two cells
        waypoints_in_cells = set()
        waypoint_names = {}  # Map ID -> name for logging

        waypoints_in_cells.add(from_waypoint_id)  # Always include source
        waypoints_in_cells.add(to_waypoint_id)    # Always include destination

        for wp in graph.waypoints:
            wp_x = wp.waypoint_tform_ko.position.x
            wp_y = wp.waypoint_tform_ko.position.y
            wp_name = wp.annotations.name if wp.annotations.name else f"auto_{wp.id[:8]}"
            waypoint_names[wp.id] = wp_name

            # Check if this waypoint is inside cell_from or cell_to
            if (env_map.is_point_in_cell(wp_x, wp_y, cell_from[0], cell_from[1]) or
                env_map.is_point_in_cell(wp_x, wp_y, cell_to[0], cell_to[1])):
                waypoints_in_cells.add(wp.id)

        # Build adjacency list from edges (only for waypoints in the two cells)
        adjacency = {}
        edge_count = 0

        for edge in graph.edges:
            from_id = edge.id.from_waypoint
            to_id = edge.id.to_waypoint

            # Only include edges where BOTH endpoints are in the valid cells
            if from_id in waypoints_in_cells and to_id in waypoints_in_cells:
                if from_id not in adjacency:
                    adjacency[from_id] = []
                if to_id not in adjacency:
                    adjacency[to_id] = []

                # Edges are bidirectional
                adjacency[from_id].append(to_id)
                adjacency[to_id].append(from_id)
                edge_count += 1

        print(f"  Found {edge_count} edges connecting these waypoints")

        # BFS to find path (allows multi-hop)
        if from_waypoint_id not in adjacency:
            print(f"  ✗ Source waypoint has no edges inside cells")
            return False

        queue = deque([(from_waypoint_id, [from_waypoint_id])])  # (current, path)
        visited = {from_waypoint_id}

        while queue:
            current, path = queue.popleft()

            if current == to_waypoint_id:
                # Found path! Show it
                path_names = [waypoint_names.get(wp_id, wp_id[:8]) for wp_id in path]
                print(f"  ✓ Multi-hop path found ({len(path)} waypoints):")
                print(f"    {' → '.join(path_names)}")
                return True

            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        print(f"  ✗ No path found (searched {len(visited)} waypoints)")
        return False

    def verify_and_create_missing_edges(self, cell_path, waypoints_by_cell, env_map, check_loop=False):
        """
        Verify that edges exist between consecutive waypoints in the path.
        Returns list of missing edges that need to be created.

        IMPORTANT: This function also considers automatic waypoints, BUT only
        those that are INTERNAL to the two consecutive cells being considered.
        If a path already exists between two manual waypoints via automatic
        waypoints internal to the cells, a new edge is not created.

        Args:
            cell_path: List of (row, col) cells in order
            waypoints_by_cell: Dictionary mapping cells to waypoint data
            env_map: EnvironmentMap instance to check if waypoints are in cells
            check_loop: If True, also check edge from last waypoint back to first (for closed loops)

        Returns:
            list: List of tuples (from_waypoint_name, to_waypoint_name) for missing edges
        """
        print(f"\n[EDGE_VERIFY] Checking edges for path with {len(cell_path)} cells")
        print(f"[EDGE_VERIFY] NOTE: Considering waypoints INSIDE the two cells being checked")
        if check_loop:
            print(f"[EDGE_VERIFY] LOOP mode: Will also check edge from LAST → FIRST waypoint")

        graph = self._get_graph()

        # Build set of existing direct edge pairs for logging
        existing_direct_edges = set()
        for edge in graph.edges:
            from_id = edge.id.from_waypoint
            to_id = edge.id.to_waypoint
            existing_direct_edges.add((from_id, to_id))
            existing_direct_edges.add((to_id, from_id))

        print(f"[EDGE_VERIFY] Graph has {len(graph.waypoints)} total waypoints")
        print(f"[EDGE_VERIFY] Graph has {len(graph.edges)} edges ({len(existing_direct_edges)//2} bidirectional pairs)")

        # Check each consecutive pair in path
        missing_edges = []

        for i in range(len(cell_path) - 1):
            cell_from = cell_path[i]
            cell_to = cell_path[i + 1]

            # Get waypoint data for these cells
            if cell_from not in waypoints_by_cell:
                print(f"[EDGE_VERIFY] WARNING: No waypoint found for cell {cell_from}")
                continue

            if cell_to not in waypoints_by_cell:
                print(f"[EDGE_VERIFY] WARNING: No waypoint found for cell {cell_to}")
                continue

            wp_from = waypoints_by_cell[cell_from]
            wp_to = waypoints_by_cell[cell_to]

            # First check direct edge
            direct_edge_exists = ((wp_from['id'], wp_to['id']) in existing_direct_edges or
                                  (wp_to['id'], wp_from['id']) in existing_direct_edges)

            if direct_edge_exists:
                print(f"[EDGE_VERIFY] ✓ Direct edge exists: {wp_from['name']} <-> {wp_to['name']}")
            else:
                # Check if there's an indirect path through waypoints INSIDE the two cells
                path_exists = self._path_exists_in_graph(
                    wp_from['id'], wp_to['id'], graph, cell_from, cell_to, env_map
                )

                if path_exists:
                    print(f"[EDGE_VERIFY] ✓ Indirect path exists: {wp_from['name']} <-> {wp_to['name']} (via waypoints inside cells)")
                else:
                    print(f"[EDGE_VERIFY] ✗ No path found: {wp_from['name']} <-> {wp_to['name']}")
                    missing_edges.append((wp_from['name'], wp_to['name']))

        # Check loop edge (last → first) if requested
        if check_loop and len(cell_path) > 1:
            print(f"\n[EDGE_VERIFY] === LOOP CHECK: Last waypoint → First waypoint ===")

            cell_from = cell_path[-1]
            cell_to = cell_path[0]

            if cell_from in waypoints_by_cell and cell_to in waypoints_by_cell:
                wp_from = waypoints_by_cell[cell_from]
                wp_to = waypoints_by_cell[cell_to]

                # Check direct edge
                direct_edge_exists = ((wp_from['id'], wp_to['id']) in existing_direct_edges or
                                      (wp_to['id'], wp_from['id']) in existing_direct_edges)

                if direct_edge_exists:
                    print(f"[EDGE_VERIFY] ✓ Direct loop edge exists: {wp_from['name']} <-> {wp_to['name']}")
                else:
                    # Check if there's an indirect path through waypoints INSIDE the two cells
                    path_exists = self._path_exists_in_graph(
                        wp_from['id'], wp_to['id'], graph, cell_from, cell_to, env_map
                    )

                    if path_exists:
                        print(f"[EDGE_VERIFY] ✓ Indirect loop path exists: {wp_from['name']} <-> {wp_to['name']}")
                    else:
                        print(f"[EDGE_VERIFY] ✗ No loop path found: {wp_from['name']} <-> {wp_to['name']}")
                        missing_edges.append((wp_from['name'], wp_to['name']))

        print(f"\n[EDGE_VERIFY] Summary: {len(missing_edges)} missing edges")
        return missing_edges

    def create_edge_between_waypoints(self, from_waypoint_name, to_waypoint_name):
        """
        Create a new edge between two waypoints.

        Args:
            from_waypoint_name: Name of source waypoint (e.g., 'wp_5')
            to_waypoint_name: Name of destination waypoint (e.g., 'wp_7')

        Returns:
            bool: True if edge created successfully, False otherwise
        """
        print(f"\n[EDGE_CREATE] Creating edge: {from_waypoint_name} -> {to_waypoint_name}")

        # Get current graph (use cache for reading waypoint info)
        graph = self._get_graph()

        # Find waypoint objects
        from_wp = None
        to_wp = None

        for waypoint in graph.waypoints:
            if waypoint.annotations.name == from_waypoint_name:
                from_wp = waypoint
            if waypoint.annotations.name == to_waypoint_name:
                to_wp = waypoint

        if from_wp is None:
            print(f"[EDGE_CREATE] ERROR: Waypoint '{from_waypoint_name}' not found in graph")
            return False

        if to_wp is None:
            print(f"[EDGE_CREATE] ERROR: Waypoint '{to_waypoint_name}' not found in graph")
            return False

        # Calculate transform between waypoints
        edge_transform = self._get_transform(from_wp, to_wp)

        # Create new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_wp.id
        new_edge.id.to_waypoint = to_wp.id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f"[EDGE_CREATE] Transform calculated: {edge_transform}")

        # Send request to add edge
        try:
            self._recording_client.create_edge(edge=new_edge)
            self.invalidate_graph_cache()  # Invalidate cache since we added a new edge
            print(f"[EDGE_CREATE] ✓ Edge created successfully")
            return True
        except Exception as e:
            print(f"[EDGE_CREATE] ✗ Failed to create edge: {e}")
            return False

    def find_nearest_waypoint_cell_to_target(self, target_cell, waypoints_by_cell, env_map):
        """
        Find the visited cell with a waypoint that is closest to the target cell.
        Uses Manhattan distance and prioritizes cells adjacent to target.

        Args:
            target_cell: (row, col) the target cell without waypoint
            waypoints_by_cell: dict mapping cells to waypoint data
            env_map: EnvironmentMap instance

        Returns:
            tuple: (row, col) of nearest waypoint cell, or None if not found
        """
        target_row, target_col = target_cell

        # First, check adjacent cells (distance 1)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        adjacent_with_waypoint = []
        for dr, dc in directions:
            adj_row, adj_col = target_row + dr, target_col + dc
            adj_cell = (adj_row, adj_col)

            # Check if cell is valid, visited, and has a waypoint
            if (0 <= adj_row < env_map.rows and
                0 <= adj_col < env_map.cols and
                adj_cell in waypoints_by_cell and
                env_map.map[adj_row][adj_col] == 1):  # visited
                adjacent_with_waypoint.append(adj_cell)

        if adjacent_with_waypoint:
            # Return the first adjacent cell with waypoint
            print(f"[NEAREST_WP] Found adjacent waypoint at {adjacent_with_waypoint[0]}")
            return adjacent_with_waypoint[0]

        # If no adjacent waypoint, search all waypoints and find closest
        min_distance = float('inf')
        nearest_cell = None

        for cell in waypoints_by_cell.keys():
            cell_row, cell_col = cell

            # Skip unvisited cells
            if env_map.map[cell_row][cell_col] != 1:
                continue

            # Calculate Manhattan distance
            distance = abs(cell_row - target_row) + abs(cell_col - target_col)

            if distance < min_distance:
                min_distance = distance
                nearest_cell = cell

        if nearest_cell:
            print(f"[NEAREST_WP] Nearest waypoint at {nearest_cell} (distance: {min_distance})")

        return nearest_cell

    def find_and_optimize_path(self, start_cell, end_cell, env_map, create_missing_edges=True):
        """
        Complete workflow to find shortest path and ensure all edges exist.

        This method:
        1. Gets all manual waypoints with cell data
        2. Finds shortest path using BFS on the grid
        3. Verifies edges exist between consecutive waypoints
        4. Creates missing edges if requested
        5. Returns the optimized path

        IMPORTANT: If the target cell does NOT have a waypoint, finds the nearest
        (adjacent) waypoint to the target cell and returns the path to it.

        IMPORTANT: For each pair of consecutive cells in the path, checks if
        a direct edge exists between manual waypoints OR a path that goes through
        automatic waypoints that are INTERNAL to the two cells being considered.
        If no valid path exists, creates a new direct edge.

        Args:
            start_cell: (row, col) starting cell
            end_cell: (row, col) destination cell
            env_map: EnvironmentMap instance
            create_missing_edges: If True, create missing edges automatically

        Returns:
            dict: {
                'success': bool,
                'cell_path': list of cells or None,
                'waypoint_names': list of waypoint names or None,
                'missing_edges': list of missing edge pairs,
                'edges_created': list of created edge pairs,
                'target_cell_has_waypoint': bool - True se la cella target ha un waypoint
            }
        """
        print(f"\n{'='*70}")
        print(f"[PATH_OPTIMIZE] Starting path optimization")
        print(f"[PATH_OPTIMIZE] From: {start_cell} -> To: {end_cell}")
        print(f"{'='*70}")

        result = {
            'success': False,
            'cell_path': None,
            'waypoint_names': None,
            'missing_edges': [],
            'edges_created': [],
            'target_cell_has_waypoint': False
        }

        # Step 1: Get all waypoints with cell data
        waypoints_by_cell = self.get_all_manual_waypoints_with_cells()

        if start_cell not in waypoints_by_cell:
            print(f"[PATH_OPTIMIZE] ERROR: No waypoint at start cell {start_cell}")
            return result

        # Check if end_cell has a waypoint
        actual_end_cell = end_cell
        if end_cell not in waypoints_by_cell:
            print(f"[PATH_OPTIMIZE] Target cell {end_cell} has no waypoint - finding nearest waypoint")

            # Find the nearest visited cell with a waypoint (adjacent to target)
            nearest_wp_cell = self.find_nearest_waypoint_cell_to_target(end_cell, waypoints_by_cell, env_map)

            if nearest_wp_cell is None:
                print(f"[PATH_OPTIMIZE] ERROR: No reachable waypoint found near target cell {end_cell}")
                return result

            actual_end_cell = nearest_wp_cell
            result['target_cell_has_waypoint'] = False
            print(f"[PATH_OPTIMIZE] Will navigate to nearest waypoint at cell {actual_end_cell}")
        else:
            result['target_cell_has_waypoint'] = True

        # Step 2: Find shortest path using BFS
        cell_path = self.find_shortest_cell_path_bfs(start_cell, actual_end_cell, env_map)

        if cell_path is None:
            print(f"[PATH_OPTIMIZE] ERROR: No path found")
            return result

        result['cell_path'] = cell_path

        # Convert to waypoint names
        waypoint_names = []
        for cell in cell_path:
            if cell in waypoints_by_cell:
                waypoint_names.append(waypoints_by_cell[cell]['name'])
            else:
                print(f"[PATH_OPTIMIZE] WARNING: No waypoint for cell {cell} in path")

        result['waypoint_names'] = waypoint_names

        # Step 3: Verify edges (considering all waypoints for indirect paths)
        missing_edges = self.verify_and_create_missing_edges(cell_path, waypoints_by_cell, env_map)
        result['missing_edges'] = missing_edges

        # Step 4: Create missing edges if requested
        if create_missing_edges and len(missing_edges) > 0:
            print(f"\n[PATH_OPTIMIZE] Creating {len(missing_edges)} missing edges...")

            for from_name, to_name in missing_edges:
                success = self.create_edge_between_waypoints(from_name, to_name)
                if success:
                    result['edges_created'].append((from_name, to_name))

            print(f"[PATH_OPTIMIZE] Created {len(result['edges_created'])} / {len(missing_edges)} edges")

        # Success if path exists (edges are optional)
        result['success'] = True


        print(f"\n{'='*70}")
        print(f"[PATH_OPTIMIZE] Optimization complete")
        print(f"[PATH_OPTIMIZE] Path length: {len(cell_path)} cells ({len(cell_path)-1} hops)")
        print(f"[PATH_OPTIMIZE] Waypoints: {' -> '.join(waypoint_names)}")
        if create_missing_edges:
            print(f"[PATH_OPTIMIZE] Edges created: {len(result['edges_created'])}")
        print(f"{'='*70}\n")

        return result

