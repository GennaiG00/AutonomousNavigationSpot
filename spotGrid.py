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
            local_grid_found.local_grid.extent.cell_size = 0.5
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


def create_vtk_obstacle_grid(proto, robot_state_client):
    """Generate points, cell values and colors for the obstacle distance grid.

    The obstacle_distance grid encodes the signed distance (in metres) from each
    cell to the nearest detected obstacle:
        dist < 0   -> strictly inside an obstacle  → blocked  (red)
        dist >= 0  -> border or free space          → passable (blue)

    Zero-padding policy: no safety margin is added around obstacles.
    A cell is passable as soon as its distance is >= 0.
    Unlike the no_step grid, grass is NOT classified as an obstacle here,
    making this grid more suitable for outdoor environments.

    Returns:
        pts (np.ndarray, shape (N,3)): cell positions in the VISION frame
        cells_obstacle_dist (np.ndarray, shape (N,)): raw signed-distance values
        color (np.ndarray, shape (N,3) uint8): RGB colours per cell
    """
    local_grid_proto = None
    cell_size = 0.0
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'obstacle_distance':
            local_grid_proto = local_grid_found
            cell_size = local_grid_found.local_grid.extent.cell_size

    # If no relevant local grid found, return empty arrays (caller can handle)
    if local_grid_proto is None:
        return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Unpack the raw distance values.
    cells_obstacle_dist = unpack_grid(local_grid_proto).astype(np.float32)

    # Build (x, y) grid coordinates.
    ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
                      0:local_grid_proto.local_grid.extent.num_cells_y]

    # Use ground-plane height for the z coordinate.
    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
    z_ground_in_vision_frame = compute_ground_height_in_vision_frame(robot_state_client)
    cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
    z = np.ones(cell_count, dtype=np.float32) * z_ground_in_vision_frame

    pts = np.vstack([np.ravel(xs).astype(np.float32),
                     np.ravel(ys).astype(np.float32), z]).T
    pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                       local_grid_proto.local_grid.extent.cell_size)

    # Colour coding (zero-padding – no border zone):
    #   red  -> strictly inside obstacle  (dist < 0)
    #   blue -> passable: border or free  (dist >= 0)
    color = np.zeros([cell_count, 3], dtype=np.uint8)
    color[:, 0] = (cells_obstacle_dist < 0.0)   # red  = blocked
    color[:, 2] = (cells_obstacle_dist >= 0.0)  # blue = passable
    color *= 255

    # Offset to VISION frame.
    vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                            local_grid_proto.local_grid.frame_name_local_grid_data)
    pts = offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

    return pts, cells_obstacle_dist, color


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
