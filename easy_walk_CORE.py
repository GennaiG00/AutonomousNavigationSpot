import os
import sys
import time
from time import sleep
from datetime import datetime
import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client import payload
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from types import SimpleNamespace
import navGraphUtils
import movements
import spotGrid
import spotLogInUtils
import environmentMap
import spotUtils

import matplotlib
matplotlib.use('Agg') # Forza Matplotlib a lavorare senza schermo (solo salvataggio file)

# TODO: check if we can avoid to set a sleep after each movement command
# TODO: change the folder destination of the name download of graph

def check_line_of_sight(x1, y1, x2, y2, pts, cells, obstacle_threshold=0.0):
    """
    Check if there's a clear line of sight between two points.
    Uses sampling along the line to check for obstacles.

    Uses the obstacle_distance grid where:
        dist < 0   -> strictly inside an obstacle (blocked)
        dist >= 0  -> border or free space (passable – zero padding)

    Args:
        x1, y1: Start coordinates (robot position)
        x2, y2: End coordinates (target point)
        pts: Grid points array from obstacle_distance grid
        cells: obstacle_distance values per cell
        obstacle_threshold: Cells with distance strictly less than this value are
                            considered blocked.  Default 0.0 = zero padding (no
                            safety margin around obstacles).

    Returns:
        bool: True if path is clear, False if blocked
    """
    # Number of points to check along the line
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    num_checks = max(10, int(distance * 10))  # 10 checks per meter

    for i in range(num_checks):
        t = i / max(1, num_checks - 1)
        check_x = x1 + t * (x2 - x1)
        check_y = y1 + t * (y2 - y1)

        # Find nearest grid point
        distances = np.sqrt((pts[:, 0] - check_x)**2 + (pts[:, 1] - check_y)**2)
        nearest_idx = np.argmin(distances)

        # Blocked only when strictly inside an obstacle (dist < threshold).
        # With threshold=0.0 this gives zero padding: the obstacle border (dist=0)
        # is already considered passable.
        if cells[nearest_idx] < obstacle_threshold:
            return False  # Path blocked

    return True  # Path clear


def sample_cell_points(env, cell_row, cell_col, num_samples=200):
    """
    Sample random points within a cell.

    Args:
        env: EnvironmentMap instance
        cell_row: Row index of the cell
        cell_col: Column index of the cell
        num_samples: Number of random points to generate

    Returns:
        list of (x, y) tuples representing sampled points in world coordinates
    """
    # Get cell center in world coordinates
    world_pos = env.get_world_position_from_cell(cell_row, cell_col)
    if world_pos is None:
        return []

    cell_center_x, cell_center_y = world_pos
    half_size = env.cell_size / 2.0

    # Generate random offsets within the cell (in grid frame)
    samples = []
    for _ in range(num_samples):
        # Random offset from center in grid frame
        offset_x = np.random.uniform(-half_size * 0.8, half_size * 0.8)  # 80% to avoid edges
        offset_y = np.random.uniform(-half_size * 0.8, half_size * 0.8)

        # Rotate offset to world frame
        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

        # Final world position
        sample_x = cell_center_x + world_offset_x
        sample_y = cell_center_y + world_offset_y

        samples.append((sample_x, sample_y))

    return samples


def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, pts, cells_obstacle_dist):
    """
    Sample 20 random points in a cell and find the one with clear path that is closest to cell center.

    Args:
        robot_x, robot_y: Current robot position
        env: EnvironmentMap instance
        cell_row, cell_col: Target cell coordinates
        pts: Grid points array from local grid
        cells_obstacle_dist: Cell values from obstacle_distance grid
                             (<=0 = inside obstacle, 0..0.33 = border, >=0.33 = free)

    Returns:
        tuple: (best_x, best_y, valid_samples, rejected_samples) or (None, None, [], []) if no valid point found
    """
    # Sample random points in the cell
    sampled_points = sample_cell_points(env, cell_row, cell_col, num_samples=100)

    if not sampled_points:
        return None, None, [], []

    # Get cell center coordinates
    cell_center = env.get_world_position_from_cell(cell_row, cell_col)
    if cell_center is None:
        return None, None, [], []

    cell_center_x, cell_center_y = cell_center

    valid_samples = []
    rejected_samples = []

    # Check each sampled point
    for sample_x, sample_y in sampled_points:
        # Check if path is clear (obstacle_distance > 0 means outside obstacle)
        if check_line_of_sight(robot_x, robot_y, sample_x, sample_y, pts, cells_obstacle_dist, obstacle_threshold=0.15):
            valid_samples.append((sample_x, sample_y))
        else:
            rejected_samples.append((sample_x, sample_y))

    # If no valid samples, return None
    if not valid_samples:
        print(f"[WARNING] No clear path found to any sampled point in cell ({cell_row},{cell_col})")
        return None, None, valid_samples, rejected_samples

    # Choose the valid point that is CLOSEST to the cell center
    best_point = None
    min_distance = float('inf')

    for sample_x, sample_y in valid_samples:
        dist = np.sqrt((sample_x - cell_center_x)**2 + (sample_y - cell_center_y)**2)
        if dist < min_distance:
            min_distance = dist
            best_point = (sample_x, sample_y)

    #print(f"[OK] Found {len(valid_samples)} valid points in cell ({cell_row},{cell_col}), chose closest to center at {min_distance:.2f}m from center")

    return best_point[0], best_point[1], valid_samples, rejected_samples


def attempt_enter_cell_from_position(local_grid_client, robot_state_client, command_client,
                                      env, target_row, target_col, mission_folder=None, iteration=0, recordingInterface=None):
    """
    Attempt to enter a target cell from the current robot position.

    This method:
    1. Gets the local grid
    2. Samples n random points in the target cell
    3. Finds the best point (closest to center) with clear line of sight
    4. If found, moves the robot to that point
    5. Returns success/failure

    Args:
        local_grid_client: Client for local grid
        robot_state_client: Client for robot state
        command_client: Client for robot commands
        env: EnvironmentMap instance
        target_row: Row of target cell to enter
        target_col: Column of target cell to enter
        mission_folder: Path to save visualization files
        iteration: Current iteration number for file naming
        recordingInterface: RecordingInterface for getting edge data

    Returns:
        bool: True if successfully entered cell, False otherwise
    """
    print(f"\n[ATTEMPT] Trying to enter cell ({target_row},{target_col}) from current position...")

    # Get local grid to check obstacles (obstacle_distance is better for outdoor/grass environments
    # because the no_step grid incorrectly marks grass as an obstacle)
    proto = local_grid_client.get_local_grids(['obstacle_distance'])
    pts, cells_obstacle_dist, color = spotGrid.create_vtk_obstacle_grid(proto, robot_state_client)

    # Get grid proto
    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'obstacle_distance':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("[ERROR] No 'obstacle_distance' grid found")
        return False

    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot

    # Get robot position
    vision_tform_body = get_a_tform_b(
        transforms_snapshot,
        VISION_FRAME_NAME,
        BODY_FRAME_NAME
    )
    robot_x = vision_tform_body.position.x
    robot_y = vision_tform_body.position.y

    print(f"[INFO] Robot position: ({robot_x:.2f}, {robot_y:.2f})")

    # Sample random points in the target cell and find the best one with clear path
    print(f"[INFO] Sampling 20 random points in cell ({target_row},{target_col})...")
    target_x, target_y, valid_samples, rejected_samples = find_best_point_in_cell(
        robot_x, robot_y, env, target_row, target_col, pts, cells_obstacle_dist
    )

    if target_x is None or target_y is None:
        print(f"[FAIL] No clear path found to cell ({target_row},{target_col}) from current position")

        return False

    print(f"[OK] Target point in cell ({target_row},{target_col}): ({target_x:.2f}, {target_y:.2f})")

    # Calculate distance and direction
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    print(f"[INFO] Distance to target: {distance:.2f}m")

    # Visualize target with sampled points and save to mission folder
    save_path = None
    if mission_folder:
        save_path = os.path.join(mission_folder, f"iteration_{iteration}_cell_{target_row}_{target_col}.png")

    # Calculate yaw to face the target
    target_yaw = np.arctan2(dy, dx)

    # Get current yaw
    quat = vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y),
                             1.0 - 2.0 * (quat.y**2 + quat.z**2))

    # Calculate rotation needed
    dyaw = target_yaw - current_yaw
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))  # Normalize to [-π, π]

    print(f"[INFO] Required rotation: {np.rad2deg(dyaw):.1f}°")

    # First rotate to face target
    print("[INFO] Step 1: Rotating to face target...")
    success_rot = movements.relative_move(0, 0, dyaw, "vision",
                                         command_client, robot_state_client)


    #time.sleep(0.5)

    # Then move forward
    print(f"[INFO] Step 2: Moving forward {distance:.2f}m...")
    success_move = movements.relative_move(distance, 0, 0, "vision",
                                          command_client, robot_state_client)

    if success_move:
        # Wait for movement to complete
        #time.sleep(0.5)

        # VERIFY: Check if the robot is actually in the target cell
        x_final, y_final, z_final, _ = spotUtils.getPosition(robot_state_client)
        check_position_in_cell = env.is_point_in_cell(x_final, y_final, target_row, target_col)

        if check_position_in_cell:
            print(f"We are in the right cell")
            return True
        else:
            print(f"[FAIL] We are in the wrong cell")
            return False

    else:
        print(f"[FAIL] Movement command failed for cell ({target_row},{target_col})")
        return False

def find_new_borders(env, robot_row, robot_col, path, frontier):
    new_borders = env.get_adjacent_frontier_cells(robot_row, robot_col, path)
    new_borders_cells = []
    if len(new_borders) != 0:
        for new_border in new_borders:
            if new_border not in frontier and env.is_cell_visited(new_border[0], new_border[1]) != 1:
                new_borders_cells.append(new_border)
    return new_borders_cells

def easy_walk(options):
    robot, lease_client, robot_state_client, client_metadata = spotLogInUtils.setLogInfo(options)
    robot.authenticate_from_payload_credentials(*bosdyn.client.util.get_guid_and_secret(options))

    estop = spotLogInUtils.SimpleEstop(robot, options.name + "_estop")

    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()

    mission_log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot...')
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        blocking_stand(command_client)

        # Clear any existing map first
        recordingInterface.clear_map()

        # Start recording BEFORE trying to initialize with fiducial
        recordingInterface.start_recording()

        # Try to initialize with fiducial (optional - if it fails, we can still create waypoints manually)
        fiducial_success = recordingInterface.initialize_with_fiducial(robot_state_client, 549)
        if not fiducial_success:
            print("[WARNING] Fiducial initialization failed. Continuing without fiducial origin.")
            print("[INFO] The map origin will be set when creating the first waypoint.")

        # Grid start cell used consistently for origin, wp_0 and serpentine ranking.
        start_row, start_col = 0, 0

        # Create first waypoint in initial cell (wp_0)
        recordingInterface.create_default_waypoint(cell_row=start_row, cell_col=start_col)
        #TODO Controlla che il branch sia quello giusto
        env = environmentMap.EnvironmentMap(rows=7, cols=7, cell_size=2)
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)

        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y ** 2 + quat_boot.z ** 2))

        env.set_origin(x_boot, y_boot, yaw_boot, start_row=start_row, start_col=start_col)

        print(f'[INIT] Boot position: x={x_boot:.3f}, y={y_boot:.3f}, z={z_boot:.3f}')
        print(
            f'[INIT] Boot orientation: reale {np.rad2deg(yaw_boot):.1f}° -> allineata alla griglia: {np.rad2deg(yaw_boot):.1f}°')


        mission_timestamp = datetime.now().strftime("Mission_%d-%m-%Y_%H-%M-%S")
        base_graph_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph")
        os.makedirs(base_graph_folder, exist_ok=True)
        graph_folder = os.path.join(base_graph_folder, mission_timestamp)
        os.makedirs(graph_folder, exist_ok=True)

        # Also create MissionMap folder for visualizations (separate from graphs)
        mission_map_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionMap", mission_timestamp)
        os.makedirs(mission_map_folder, exist_ok=True)
        mission_folder = mission_map_folder  # For visualization files

        # Create mission log folder/file with the same timestamp convention.
        mission_log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionLogs", mission_timestamp)
        os.makedirs(mission_log_folder, exist_ok=True)
        mission_log_path = os.path.join(mission_log_folder, "mission_log.txt")

        mission_log_file = open(mission_log_path, "a", buffering=1)

        print(f"[INIT] Mission folder created: {mission_folder}")
        print(f"[INIT] Graph folder created: {graph_folder}")
        print(f"[INIT] Mission log file: {mission_log_path}")

        # Update recording interface to save graph in mission folder
        recordingInterface.set_download_filepath(graph_folder)

        # Generate serpentine path starting from the configured start cell.
        path = env.generate_serpentine_path(start_cell=env.start_cell)

        frontier = []
        current_path_index = 0
        visualization_counter = 0  # Counter for visualization files

        x, y, z, _ = spotUtils.getPosition(robot_state_client)
        robot_row, robot_col = env.get_cell_from_world(x, y)
        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

        while(True):
            print(frontier)
            print(f"\n{'#'*70}")
            print(f"### PATH STEP: {current_path_index + 1}/{len(path)} ###")
            print(f"{'#'*70}\n")

            x, y, z, _ = spotUtils.getPosition(robot_state_client)
            robot_row, robot_col = env.get_cell_from_world(x, y)

            borders = env.get_adjacent_frontier_cells(robot_row, robot_col, path)
            borders_in_frontier = []

            for border in borders:
                is_in_frontier = any(
                    (f[0] == border[0] and f[1] == border[1])
                    for f in frontier
                )
                if is_in_frontier:
                    borders_in_frontier.append(border)

            if len(borders_in_frontier) != 0:
                # Select the border with the lowest rank (index 2 of the tuple)
                selected_border = min(borders_in_frontier, key=lambda b: b[2])
                print(f"[BORDER] Selected border with lowest rank: ({selected_border[0]},{selected_border[1]}) rank={selected_border[2]}")

                check = attempt_enter_cell_from_position(local_grid_client, robot_state_client, command_client, env, selected_border[0], selected_border[1], mission_folder, visualization_counter, recordingInterface)
                visualization_counter += 1
                frontier.remove(selected_border)
                #recordingInterface.auto_close_loops(False, True)
                if check:
                    env.update_position(x, y)
                    env.print_map()
                    # Create waypoint saving the cell we entered
                    recordingInterface.create_default_waypoint(cell_row=selected_border[0], cell_col=selected_border[1])
                    env.add_waypoint(x, y)
                    env.mark_cell_visited(selected_border[0], selected_border[1])
                    x_new, y_new, _, _ = spotUtils.getPosition(robot_state_client)
                    robot_row, robot_col = env.get_cell_from_world(x_new, y_new)
                    frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
            else:
                lowest_rank_cell = env.get_lowest_rank_from_frontier_list(frontier, path)

                if lowest_rank_cell is not None:
                    target_row, target_col, rank = lowest_rank_cell

                    print(f"\n[TARGET] Cell with lowest rank: ({target_row},{target_col}) rank={rank}")

                    # Get current position
                    x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)
                    current_row, current_col = env.get_cell_from_world(x_current, y_current)

                    # Use grid-based path optimization to find shortest path
                    print(f"\n[PATH_OPTIMIZE] Finding optimized path from ({current_row},{current_col}) to ({target_row},{target_col})")

                    # Stop recording to analyze graph (no need to download snapshots, _get_graph() is used internally)
                    recordingInterface.stop_recording()

                    target_cell = (target_row, target_col)
                    waypoints_by_cell = recordingInterface.get_all_manual_waypoints_with_cells()
                    nearest_cell = recordingInterface.find_nearest_waypoint_cell_to_target(target_cell, waypoints_by_cell, env)
                    nearest_wp = recordingInterface.get_manual_waypoint_by_cell(nearest_cell[0], nearest_cell[1])

                    waypoints_by_cell = recordingInterface.get_all_manual_waypoints_with_cells()

                    for cell, wp_data in waypoints_by_cell.items():
                        if wp_data['name'] == nearest_wp['name']:
                                    waypoint_data = wp_data
                                    break

                    navigation_success = recordingInterface.navigate_to_waypoint(nearest_wp['id'], robot_state_client)

                    if navigation_success:
                        # Resume recording at target waypoint
                        recordingInterface.start_recording()

                        # Try to enter the target cell
                        check = attempt_enter_cell_from_position(
                            local_grid_client, robot_state_client, command_client,
                            env, target_row, target_col, mission_folder, visualization_counter, recordingInterface
                        )
                        visualization_counter += 1

                        if check:
                            # Success - create waypoint and update map
                            x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
                            recordingInterface.create_default_waypoint(cell_row=target_row, cell_col=target_col)
                            env.add_waypoint(x_final, y_final)

                            # Mark cell as visited (IMPORTANT!)
                            env.mark_cell_visited(target_row, target_col)

                            # Remove from frontier
                            frontier.remove((target_row, target_col, rank))

                            # Update robot position and find new borders
                            robot_row, robot_col = env.get_cell_from_world(x_final, y_final)
                            frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

                            print(f"[SUCCESS] Entered cell ({target_row},{target_col}) via optimized path")
                        else:
                            # Failure - remove from frontier anyway
                            frontier.remove((target_row, target_col, rank))
                            print(f"[ERROR] Could not enter cell ({target_row},{target_col}) after navigating optimized path")
                    else:
                        print(f"[ERROR] Navigation failed along optimized path")
                        # IMPORTANT: Resume recording even in case of failure
                        recordingInterface.start_recording()
                        # Remove from frontier
                        frontier.remove((target_row, target_col, rank))
                else:
                    print(f"[ERROR] No path found to cell ({target_row},{target_col})")
                    # IMPORTANT: Resume recording even in case of failure
                    recordingInterface.start_recording()
                    # Remove from frontier - unreachable
                    frontier.remove((target_row, target_col, rank))

            if len(frontier) == 0:
                break
        print(f"\n{'='*70}")
        print(f"EXPLORATION COMPLETE")
        print(f"{'='*70}")
        print(f"Cells explored: {sum(sum(row) for row in env.map)}")
        env.print_map()

        # Create final waypoint (current robot position)
        x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
        final_row, final_col = env.get_cell_from_world(x_final, y_final)
        recordingInterface.create_default_waypoint(cell_row=final_row, cell_col=final_col)
        recordingInterface.get_recording_status()
        # Note: Edges are created during path optimization, no need for create_new_edge

        # --- END OF SIMPLE MISSION ---

        robot.logger.info('Robot mission completed.')
        log_comment = 'Easy autowalk with obstacle avoidance.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        print(f"\n{'='*70}")
        print(f"[RETURN_OPTIMIZE] Optimizing return path to base (wp_0)")
        print(f"{'='*70}")

        # Get current position and find optimal path back to start
        x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)
        current_row, current_col = env.get_cell_from_world(x_current, y_current)
        start_row, start_col = env.start_cell

        print(f"[RETURN_OPTIMIZE] Current position: cell ({current_row},{current_col})")
        print(f"[RETURN_OPTIMIZE] Target: wp_0 at cell ({start_row},{start_col})")

        print(f"{'='*70}\n")
        recordingInterface.auto_close_loops(True, False)
        recordingInterface.stop_recording()
        recordingInterface.optimize_anchoring()
        #recordingInterface.find_nearest_waypoint_to_position(x_current, y_current)
        recordingInterface.navigate_to_first_waypoint(robot_state_client)

        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(3)
        robot.power_off(cut_immediately=False)

        # Save the final map to disk (includes return path optimization)
        recordingInterface.download_full_graph()
        estop.stop()

# FIXME Change hostname for Jetson/localhost
def main():
    # Instead of argparse, create an options object manually
    options = SimpleNamespace()
    options.name = "autonomousMission"
    options.hostname = "192.168.50.3"
    options.guid = "04d1e376-7819-45a8-9bbe-9430607ef4d3"
    options.secret = "5b73c34f-5e2c-441a-b3e9-936ba9a9e5fe"
    options.verbose = False
    options.recording_user_name = ""
    options.recording_session_name = ""
    options.download_filepath = os.getcwd()

    try:
        easy_walk(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error('Hello, Spot! threw an exception: %r', exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
