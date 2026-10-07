import spotUtils
import spotGrid

class ArcVerificationTracker:
    """Tracks which PRM arcs have been verified as safe to traverse."""

    def __init__(self):
        self.verified_arcs = set()  # Set of tuples (node_id1, node_id2)
        self.blocked_arcs = set()

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


def is_arc_in_fov(x1, y1, x2, y2, pts, fov_margin=0.3):
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

