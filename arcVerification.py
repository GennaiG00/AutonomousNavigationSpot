import threading

from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.robot_state import RobotStateClient

import spotUtils
import spotGrid
import time
import logging

LOGGER = logging.getLogger(__name__)

def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.1)

class AsyncArcVerificationTracker(AsyncPeriodicQuery):

    def __init__(self, robot_state_client):
        super(AsyncArcVerificationTracker, self).__init__('arc_verification', robot_state_client, LOGGER,
                                                          period_sec=0.2)

    def _start_query(self):
        return self._client.get_arc_verification(['arc_verification'])

class ArcVerificationTracker:
    """Tracks which PRM arcs have been verified as safe to traverse."""

    def __init__(self, robot):
        self.robot = robot
        self.verified_arcs = set()
        self.blocked_arcs = set()
        self._robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self._local_grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)
        self.path = []
        self.path_lock = threading.Lock()
        self._running = False
        self._thread= None

    def update_path(self, new_path):
        """Aggiorna il path in modo thread-safe ricevendo tuple (node_id, x, y)."""
        with self.path_lock:
            # Salviamo: node_id, x, y, status
            self.path = [(node_id, x, y, None) for node_id, x, y in new_path]
            print(f"Update path: {len(self.path)} nodes")

    def set_waypoint_status(self, index, status):
        with self.path_lock:
            if 0 <= index < len(self.path):
                node_id, x, y, _ = self.path[index]
                self.path[index] = (node_id, x, y, status)
                print(f"Set waypoint status: {status}")

    def get_path_copy(self):
        with self.path_lock:
            return list(self.path)

    def mark_arc_verified(self, node_id1, node_id2):
        """Mark arc as verified. Order-independent."""
        arc = tuple(sorted([node_id1, node_id2]))
        self.verified_arcs.add(arc)

    def mark_arc_blocked(self, node_id1, node_id2):
        """Mark arc as blocked."""
        arc = tuple(sorted([node_id1, node_id2]))
        self.blocked_arcs.add(arc)

    def is_arc_verified(self, node_id1, node_id2):
        """Check if arc has been verified."""
        arc = tuple(sorted([node_id1, node_id2]))
        return arc in self.verified_arcs

    def is_arc_blocked(self, node_id1, node_id2):
        """Check if arc is known to be blocked."""
        arc = tuple(sorted([node_id1, node_id2]))
        return arc in self.blocked_arcs

    def start(self):
        """Starts the background verification thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("ArcVerificationTracker started successfully in background.")

    def stop(self):
            self._running = False
            print('ArcVerificationTracker stopped.')

    def _process_loop(self):
        while self._running:
            current_path = self.get_path_copy()

            if len(current_path) < 2:
                time.sleep(0.2)
                continue

            try:
                proto = self._local_grid_client.get_local_grids(['obstacle_distance'])
                pts, cells_obstacle_dist, _ = spotGrid.create_vtk_obstacle_grid(proto, self._robot_state_client)
            except Exception as e:
                LOGGER.error(f"Errore nel recupero dati sensori/stato: {e}")
                time.sleep(0.2)
                continue

            # Scorriamo i nodi del percorso (il primo elemento è già la posizione live del robot)
            for i in range(len(current_path) - 1):
                node1, x1, y1, _ = current_path[i]
                node2, x2, y2, _ = current_path[i + 1]

                # Se l'arco è già stato catalogato dal tracker, passiamo al successivo
                if self.is_arc_verified(node1, node2) or self.is_arc_blocked(node1, node2):
                    continue

                # Se l'arco geometrico ricade nel campo visivo delle telecamere, lo analizziamo
                if is_arc_in_fov(x1, y1, x2, y2, pts):
                    status = verify_arc_safety(x1, y1, x2, y2, pts, cells_obstacle_dist)

                    if status == 'clear':
                        self.mark_arc_verified(node1, node2)
                        self.set_waypoint_status(i, True)
                        print(f"[BACKGROUND-TRACKER] Arco {node1}-{node2} verificato: LIBERO")

                    elif status == 'blocked':
                        self.mark_arc_blocked(node1, node2)
                        self.set_waypoint_status(i, False)
                        print(f"[BACKGROUND-TRACKER] Arco {node1}-{node2} rilevato: BLOCCATO!")

            time.sleep(0.2)

def is_arc_in_fov(x1, y1, x2, y2, pts, fov_margin=0.0):
    """
    Determine if an arc (edge) is within the camera's field of view.
    Both endpoints and midpoint must be within local grid bounds.

    Returns: True if arc is visible, False otherwise
    """
    if len(pts) == 0:
        return False

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    # Expand bounds with margin
    x_min -= fov_margin
    x_max += fov_margin
    y_min -= fov_margin
    y_max += fov_margin

    # Check if both endpoints are in FOV
    p1_in_fov = (x_min <= x1 <= x_max) and (y_min <= y1 <= y_max)
    p2_in_fov = (x_min <= x2 <= x_max) and (y_min <= y2 <= y_max)

    # Check midpoint as well
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    mid_in_fov = (x_min <= mid_x <= x_max) and (y_min <= mid_y <= y_max)

    return p1_in_fov and p2_in_fov and mid_in_fov


def verify_arc_safety(x1, y1, x2, y2, pts, cells_obstacle_dist):
    """
    Verify if an arc is safe to traverse (no obstacles blocking it).
    Uses check_line_of_sight with obstacle grid data.

    Returns: 'clear' if safe, 'blocked' if obstacles found, 'unseen' if not fully visible
    """
    return spotUtils.check_line_of_sight(x1, y1, x2, y2, pts, cells_obstacle_dist)


def get_visible_arcs_from_path(path_coords, robot_x, robot_y, pts, prm_graph):
    """
    Get list of arcs (edges) from the path that are currently visible in FOV.
    Also returns the node IDs for tracking verification.

    Returns: List of tuples (node_id1, node_id2, status)
    """
    if len(path_coords) < 2:
        return []

    visible_arcs = []

    # Check arcs between robot position and first waypoint
    if len(path_coords) > 0:
        next_x, next_y = path_coords[0]
        if is_arc_in_fov(robot_x, robot_y, next_x, next_y, pts):
            # Find node IDs
            start_node = prm_graph.get_nearest_node(robot_x, robot_y)
            next_node = prm_graph.get_nearest_node(next_x, next_y)
            visible_arcs.append((start_node, next_node, 'visible'))

    # Check arcs between consecutive waypoints
    for i in range(len(path_coords) - 1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i + 1]

        if is_arc_in_fov(x1, y1, x2, y2, pts):
            node1 = prm_graph.get_nearest_node(x1, y1)
            node2 = prm_graph.get_nearest_node(x2, y2)
            visible_arcs.append((node1, node2, 'visible'))

    return visible_arcs