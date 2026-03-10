import numpy as np
from bosdyn.client.frame_helpers import *
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2

def create_vtk_no_step_grid(proto, robot_state_client):
    """Generate VTK polydata for the no step grid from the local grid response."""
    local_grid_proto = None
    cell_size = 0.0
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            cell_size = local_grid_found.local_grid.extent.cell_size

    # If no relevant local grid found, return empty arrays (caller can handle)
    if local_grid_proto is None:
        return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Unpack the data field for the local grid.
    cells_no_step = unpack_grid(local_grid_proto).astype(np.float32)
    # Populate the x,y values with a complete combination of all possible pairs for the dimensions in the grid extent.
    ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
                      0:local_grid_proto.local_grid.extent.num_cells_y]
    # Get the estimated height (z value) of the ground in the vision frame as if the robot was standing.
    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    z_ground_in_vision_frame = compute_ground_height_in_vision_frame(robot_state_client)
    # Numpy vstack makes it so that each column is (x,y,z) for a single no step grid point. The height values come
    # from the estimated height of the ground plane.
    cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
    cells_est_height = np.ones(cell_count) * z_ground_in_vision_frame
    pts = np.vstack(
        [np.ravel(xs).astype(np.float32),
         np.ravel(ys).astype(np.float32), cells_est_height]).T
    pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                       local_grid_proto.local_grid.extent.cell_size)
    # Determine the coloration based on whether or not the region is steppable. The regions that Spot considers it
    # cannot safely step are colored red, and the regions that are considered safe to step are colored blue.
    color = np.zeros([cell_count, 3], dtype=np.uint8)
    color[:, 0] = (cells_no_step <= 0.0)
    color[:, 2] = (cells_no_step > 0.0)
    color *= 255
    # Offset the grid points to be in the vision frame instead of the local grid frame.
    vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                            local_grid_proto.local_grid.frame_name_local_grid_data)
    pts = offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

    return pts, cells_no_step, color

def compute_ground_height_in_vision_frame(robot_state_client):
    """Get the z-height of the ground plane in vision frame from the current robot state."""
    robot_state = robot_state_client.get_robot_state()
    vision_tform_ground_plane = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                              VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    return vision_tform_ground_plane.position.z

def offset_grid_pixels(pts, vision_tform_local_grid, cell_size):
    """Offset the local grid's pixels to be in the world frame instead of the local grid frame."""
    x_base = vision_tform_local_grid.position.x + cell_size * 0.5
    y_base = vision_tform_local_grid.position.y + cell_size * 0.5
    pts[:, 0] += x_base
    pts[:, 1] += y_base
    return pts

def unpack_grid(local_grid_proto):
    """Unpack the local grid proto."""
    # Determine the data type for the bytes data.
    data_type = get_numpy_data_type(local_grid_proto.local_grid)
    if data_type is None:
        print('Cannot determine the dataformat for the local grid.')
        return None
    # Decode the local grid.
    if local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
        full_grid = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
    elif local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
        full_grid = expand_data_by_rle_count(local_grid_proto, data_type=data_type)
    else:
        # Return nothing if there is no encoding type set.
        return None
    # Apply the offset and scaling to the local grid.
    if local_grid_proto.local_grid.cell_value_scale == 0:
        return full_grid
    full_grid_float = full_grid.astype(np.float64)
    full_grid_float *= local_grid_proto.local_grid.cell_value_scale
    full_grid_float += local_grid_proto.local_grid.cell_value_offset
    return full_grid_float

def get_numpy_data_type(local_grid_proto):
    """Convert the cell format of the local grid proto to a numpy data type."""
    if local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16:
        return np.uint16
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT16:
        return np.int16
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8:
        return np.uint8
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT8:
        return np.int8
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64:
        return np.float64
    elif local_grid_proto.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32:
        return np.float32
    else:
        return None

def expand_data_by_rle_count(local_grid_proto, data_type=np.int16):
    """Expand local grid data to full bytes data using the RLE count."""
    cells_pz = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
    cells_pz_full = []
    # For each value of rle_counts, we expand the cell data at the matching index
    # to have that many repeated, consecutive values.
    for i in range(0, len(local_grid_proto.local_grid.rle_counts)):
        for j in range(0, local_grid_proto.local_grid.rle_counts[i]):
            cells_pz_full.append(cells_pz[i])
    return np.array(cells_pz_full)

# def analyze_navigation_zones(pts, cells_no_step, robot_x, robot_y, robot_yaw,
#                             front_distance=0.5, lateral_distance=1.5, lateral_width=1.0,
#                             rear_distance=1.0):
#     """    Analyze zones in front, to the right, to the left and behind the robot to determine whether the path is clear.
#
#     Args:
#         pts: array (N, 3) with coordinates [x, y, z] of the cells in the VISION frame
#         cells_no_step: array (N,) with no-step values (<=0 = non-steppable, >0 = steppable)
#         robot_x, robot_y: robot position in meters (VISION frame)
#         robot_yaw: robot orientation in radians
#         front_distance: how far to look ahead (meters)
#         lateral_distance: how far to look sideways (meters)
#         lateral_width: width of the lateral area to consider (meters)
#         rear_distance: how far to look behind (meters)
#
#     Returns:
#         dict with keys:
#             - 'front_blocked': bool, True if front is blocked
#             - 'left_free': bool, True if left side is free
#             - 'right_free': bool, True if right side is free
#             - 'rear_free': bool, True if rear side is free
#             - 'front_free_ratio': float 0-1, fraction of free cells in front
#             - 'left_free_ratio': float 0-1, fraction of free cells on the left
#             - 'right_free_ratio': float 0-1, fraction of free cells on the right
#             - 'rear_free_ratio': float 0-1, fraction of free cells on the rear
#             - 'recommendation': str, suggestion ('GO_STRAIGHT', 'TURN_LEFT', 'TURN_RIGHT', 'BLOCKED')
#     """
#
#     # Direction vectors for the robot (body X-axis in the VISION frame)
#     front_dir_x = np.cos(robot_yaw)
#     front_dir_y = np.sin(robot_yaw)
#
#     # Perpendicular direction vectors (right and left)
#     right_dir_x = np.cos(robot_yaw - np.pi/2)  # -90° = right
#     right_dir_y = np.sin(robot_yaw - np.pi/2)
#     left_dir_x = np.cos(robot_yaw + np.pi/2)   # +90° = left
#     left_dir_y = np.sin(robot_yaw + np.pi/2)
#
#     # Coordinates of grid cells relative to the robot
#     dx = pts[:, 0] - robot_x
#     dy = pts[:, 1] - robot_y
#
#     # Projection onto the forward direction (longitudinal distance)
#     proj_front = dx * front_dir_x + dy * front_dir_y
#     # Projection onto the lateral direction (transverse distance, + = right, - = left)
#     proj_lateral = dx * right_dir_x + dy * right_dir_y
#
#     # --- FRONT ZONE ---
#     # Cells in front of the robot: proj_front > 0 and < front_distance, |proj_lateral| < 0.5m (robot width ~0.8m)
#     mask_front = (proj_front > 0) & (proj_front <= front_distance) & (np.abs(proj_lateral) <= 0.5)
#     cells_front = cells_no_step[mask_front]
#
#     if len(cells_front) > 0:
#         front_free_count = np.sum(cells_front > 0.0)
#         front_free_ratio = front_free_count / len(cells_front)
#     else:
#         front_free_ratio = 1.0  # no cells = consider free
#
#     # Threshold: if less than 70% of cells are free, consider the path blocked
#     front_blocked = front_free_ratio < 0.7
#
#     # --- LEFT ZONE ---
#     # Cells to the left of the robot (full side, not just front-left)
#     mask_left = (proj_lateral < 0) & (proj_lateral >= -lateral_width) & \
#                 (np.abs(proj_front) <= lateral_distance)
#     cells_left = cells_no_step[mask_left]
#
#     if len(cells_left) > 0:
#         left_free_count = np.sum(cells_left > 0.0)
#         left_free_ratio = left_free_count / len(cells_left)
#     else:
#         left_free_ratio = 1.0
#
#     left_free = left_free_ratio > 0.7
#
#     # --- RIGHT ZONE ---
#     # Cells to the right of the robot (full side, not just front-right)
#     mask_right = (proj_lateral > 0) & (proj_lateral <= lateral_width) & \
#                  (np.abs(proj_front) <= lateral_distance)
#     cells_right = cells_no_step[mask_right]
#
#     if len(cells_right) > 0:
#         right_free_count = np.sum(cells_right > 0.0)
#         right_free_ratio = right_free_count / len(cells_right)
#     else:
#         right_free_ratio = 1.0
#
#     right_free = right_free_ratio > 0.7
#
#     # --- REAR ZONE ---
#     # Cells behind the robot: proj_front < 0 and > -rear_distance, |proj_lateral| < 0.5m (robot width)
#     mask_rear = (proj_front < 0) & (proj_front >= -rear_distance) & (np.abs(proj_lateral) <= 0.5)
#     cells_rear = cells_no_step[mask_rear]
#
#     if len(cells_rear) > 0:
#         rear_free_count = np.sum(cells_rear > 0.0)
#         rear_free_ratio = rear_free_count / len(cells_rear)
#     else:
#         rear_free_ratio = 1.0
#
#     rear_free = rear_free_ratio > 0.7
#
#     # --- RECOMMENDATION ---
#     if not front_blocked:
#         recommendation = 'GO_STRAIGHT'
#     elif left_free and right_free:
#         # Both sides are free: choose the one with more space
#         recommendation = 'TURN_LEFT' if left_free_ratio >= right_free_ratio else 'TURN_RIGHT'
#     elif left_free:
#         recommendation = 'TURN_LEFT'
#     elif right_free:
#         recommendation = 'TURN_RIGHT'
#     else:
#         recommendation = 'BLOCKED'
#
#     return {
#         'front_blocked': front_blocked,
#         'left_free': left_free,
#         'right_free': right_free,
#         'rear_free': rear_free,
#         'front_free_ratio': front_free_ratio,
#         'left_free_ratio': left_free_ratio,
#         'right_free_ratio': right_free_ratio,
#         'rear_free_ratio': rear_free_ratio,
#         'recommendation': recommendation,
#         'masks': {  # for debug/visualization
#             'front': mask_front,
#             'left': mask_left,
#             'right': mask_right,
#             'rear': mask_rear
#         }
#     }
