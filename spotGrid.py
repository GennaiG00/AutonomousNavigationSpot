import numpy as np
import bosdyn.client
from bosdyn.client.frame_helpers import *
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from bosdyn.client.local_grid import LocalGridClient

class LocalGrid:
    def __init__(self, robot):
        self.local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)

    def _create_vtk_no_step_grid(self, proto, robot_state_client, local_grid_found):
        """Generate VTK polydata for the no step grid from the local grid response."""
        local_grid_proto = None
        cell_size = 0.0
        local_grid_proto = local_grid_found
        local_grid_found.local_grid.extent.cell_size = 0.5
        cell_size = local_grid_found.local_grid.extent.cell_size

        if local_grid_proto is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        # CORRETTO: Aggiunto self. per chiamare il metodo della classe
        cells_no_step = self._unpack_grid(local_grid_proto).astype(np.float32)

        ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
        0:local_grid_proto.local_grid.extent.num_cells_y]

        transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
        vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

        z_ground_in_vision_frame = self._compute_ground_height_in_vision_frame(robot_state_client)

        cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
        cells_est_height = np.ones(cell_count) * z_ground_in_vision_frame
        pts = np.vstack(
            [np.ravel(xs).astype(np.float32),
             np.ravel(ys).astype(np.float32), cells_est_height]).T
        pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                           local_grid_proto.local_grid.extent.cell_size)

        color = np.zeros([cell_count, 3], dtype=np.uint8)
        color[:, 0] = (cells_no_step <= 0.0)
        color[:, 2] = (cells_no_step > 0.0)
        color *= 255

        vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                                local_grid_proto.local_grid.frame_name_local_grid_data)

        pts = self._offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

        return pts, cells_no_step, color

    def _create_vtk_obstacle_grid(self, proto, robot_state_client, local_grid_found):
        """Generate points, cell values and colors for the obstacle distance grid."""
        local_grid_proto = local_grid_found
        cell_size = local_grid_found.local_grid.extent.cell_size

        if local_grid_proto is None:
            return np.empty((0, 3), dtype=np.float32), np.array([], dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        cells_obstacle_dist = self._unpack_grid(local_grid_proto).astype(np.float32)

        ys, xs = np.mgrid[0:local_grid_proto.local_grid.extent.num_cells_x,
        0:local_grid_proto.local_grid.extent.num_cells_y]

        transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot

        z_ground_in_vision_frame = self._compute_ground_height_in_vision_frame(robot_state_client)
        cell_count = local_grid_proto.local_grid.extent.num_cells_x * local_grid_proto.local_grid.extent.num_cells_y
        z = np.ones(cell_count, dtype=np.float32) * z_ground_in_vision_frame

        pts = np.vstack([np.ravel(xs).astype(np.float32),
                         np.ravel(ys).astype(np.float32), z]).T
        pts[:, [0, 1]] *= (local_grid_proto.local_grid.extent.cell_size,
                           local_grid_proto.local_grid.extent.cell_size)

        color = np.zeros([cell_count, 3], dtype=np.uint8)
        color[:, 0] = (cells_obstacle_dist < 0.0)
        color[:, 2] = (cells_obstacle_dist >= 0.0)
        color *= 255

        vision_tform_local_grid = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME,
                                                local_grid_proto.local_grid.frame_name_local_grid_data)

        pts = self._offset_grid_pixels(pts, vision_tform_local_grid, cell_size)

        return pts, cells_obstacle_dist, color

    def _compute_ground_height_in_vision_frame(self, robot_state_client):
        """Get the z-height of the ground plane in vision frame from the current robot state."""
        robot_state = robot_state_client.get_robot_state()
        vision_tform_ground_plane = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                                  VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
        return vision_tform_ground_plane.position.z

    def _offset_grid_pixels(self, pts, vision_tform_local_grid, cell_size):
        """Offset the local grid's pixels to be in the world frame instead of the local grid frame."""
        x_base = vision_tform_local_grid.position.x + cell_size * 0.5
        y_base = vision_tform_local_grid.position.y + cell_size * 0.5
        pts[:, 0] += x_base
        pts[:, 1] += y_base
        return pts

    def _unpack_grid(self, local_grid_proto):
        """Unpack the local grid proto."""
        # CORRETTO: Aggiunto self.
        data_type = self._get_numpy_data_type(local_grid_proto.local_grid)
        if data_type is None:
            print('Cannot determine the dataformat for the local grid.')
            return None

        if local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
            full_grid = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        elif local_grid_proto.local_grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
            # CORRETTO: Aggiunto self.
            full_grid = self._expand_data_by_rle_count(local_grid_proto, data_type=data_type)
        else:
            return None

        if local_grid_proto.local_grid.cell_value_scale == 0:
            return full_grid
        full_grid_float = full_grid.astype(np.float64)
        full_grid_float *= local_grid_proto.local_grid.cell_value_scale
        full_grid_float += local_grid_proto.local_grid.cell_value_offset
        return full_grid_float

    def _get_numpy_data_type(self, local_grid_proto):
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

    def _expand_data_by_rle_count(self, local_grid_proto, data_type=np.int16):
        """Expand local grid data to full bytes data using the RLE count."""
        cells_pz = np.frombuffer(local_grid_proto.local_grid.data, dtype=data_type)
        cells_pz_full = []
        for i in range(0, len(local_grid_proto.local_grid.rle_counts)):
            for j in range(0, local_grid_proto.local_grid.rle_counts[i]):
                cells_pz_full.append(cells_pz[i])
        return np.array(cells_pz_full)

    def return_local_grid(self, type_of_grid, robot_state_client):
        """Get the local grid from the local grid proto."""
        p = self.local_grid_client.get_local_grids([type_of_grid])

        local_grid_proto = None
        pts = None
        cells_obstacle_dist = None
        color = None
        for local_grid_found in p:
            if local_grid_found.local_grid_type_name == type_of_grid:
                local_grid_proto = local_grid_found
                if type_of_grid == 'obstacle_distance':
                    pts, cells_obstacle_dist, color = self._create_vtk_obstacle_grid(p, robot_state_client,
                                                                                     local_grid_found)
                elif type_of_grid == 'no_step':
                    pts, cells_obstacle_dist, color = self._create_vtk_no_step_grid(p, robot_state_client,
                                                                                     local_grid_found)
                else:
                    pts, cells_obstacle_dist, color = None, None, None
                break

        if local_grid_proto is None:
            print(f"[ERROR] No '{type_of_grid}' grid found")
            return False

        return pts, cells_obstacle_dist, color, local_grid_proto, p