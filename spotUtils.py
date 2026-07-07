from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME, ODOM_FRAME_NAME
import numpy as np

def getPosition(robot_state_client):
    robot_state = robot_state_client.get_robot_state()
    transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

    return vision_tform_body.position.x, vision_tform_body.position.y, vision_tform_body.position.z, vision_tform_body.rotation

def check_line_of_sight(x1, y1, x2, y2, pts, cells, obstacle_threshold=0.0, max_valid_dist=0.2):
    """
    Check if there's a clear line of sight between two points.
    Uses sampling along the line to check for obstacles.

    Uses the obstacle_distance grid where:
        dist < 0   -> strictly inside an obstacle (blocked)
        dist >= 0  -> border or free space (passable – zero padding)
    """
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_checks = max(10, int(distance * 10))  # 10 checks per meter

    for i in range(num_checks):
        t = i / max(1, num_checks - 1)
        check_x = x1 + t * (x2 - x1)
        check_y = y1 + t * (y2 - y1)

        # Find nearest grid point
        distances = np.sqrt((pts[:, 0] - check_x) ** 2 + (pts[:, 1] - check_y) ** 2)
        nearest_idx = np.argmin(distances)
        min_dist = distances[nearest_idx]

        if min_dist > max_valid_dist:
            return 'unseen'

        if cells[nearest_idx] < obstacle_threshold:
            return 'blocked'  # Path blocked

    return 'clear'  # Path clear