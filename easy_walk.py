import os
import sys
import time
from time import sleep
import numpy as np
from datetime import datetime

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client import robot
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.frame_helpers import get_a_tform_b
from bosdyn.api import local_grid_pb2
from types import SimpleNamespace
import navGraphUtils
import movements
import spotGrid
import spotLogInUtils
import environmentMap
import spotUtils

# TODO: check if we can avoid to set a sleep after each movement command

def find_nearest_waypoint_to_cell(env, target_cell, recording_interface):
    """
    Find the nearest waypoint to a target cell.

    Args:
        env: EnvironmentMap instance
        target_cell: (row, col) tuple
        recording_interface: RecordingInterface to access waypoints

    Returns:
        str: Waypoint ID of nearest waypoint, or None if no waypoints exist
    """
    # Get target world position
    target_world = env.get_world_position_from_cell(target_cell[0], target_cell[1])
    if target_world is None:
        return None

    target_x, target_y = target_world

    # Get all waypoints from the graph
    waypoints = recording_interface.get_waypoint_list()
    if not waypoints:
        print("[WARNING] No waypoints available yet")
        return None

    # Find closest waypoint
    min_distance = float('inf')
    nearest_waypoint_id = None

    for wp in waypoints:
        # Get waypoint position from graph
        wp_x = wp.waypoint_tform_ko.position.x
        wp_y = wp.waypoint_tform_ko.position.y

        # Calculate distance
        dist = np.sqrt((wp_x - target_x)**2 + (wp_y - target_y)**2)

        if dist < min_distance:
            min_distance = dist
            nearest_waypoint_id = wp.id

    print(f"[NAV] Nearest waypoint to cell {target_cell}: {nearest_waypoint_id} (distance: {min_distance:.2f}m)")
    return nearest_waypoint_id


def navigate_to_cell_via_waypoint(local_grid_client, robot_state_client, command_client,
                                   env, recording_interface, target_cell):
    """
    Navigate to a cell by first going to the nearest waypoint, then moving to the cell.

    Returns:
        tuple: (success: bool, target_cell: tuple)
    """
    print(f"\n[NAV] Navigating to cell {target_cell} via waypoint...")

    waypoint_id = find_nearest_waypoint_to_cell(env, target_cell, recording_interface)
    if waypoint_id is None:
        print("[ERROR] No waypoints available for navigation")
        return False, target_cell

    # Navigate to that waypoint using graph_nav
    print(f"[NAV] Step 1: Navigating to waypoint {waypoint_id}...")
    nav_success = recording_interface.navigate_to_waypoint(waypoint_id)

    if not nav_success:
        print(f"[ERROR] Failed to navigate to waypoint {waypoint_id}")
        return False, target_cell

    print(f"[OK] Reached waypoint {waypoint_id}")
    time.sleep(0.5)

    return True, target_cell

def check_line_of_sight(x1, y1, x2, y2, pts, cells, obstacle_threshold=0.0):
    """
    Check if there's a clear line of sight between two points.
    Uses sampling along the line to check for obstacles.

    Args:
        x1, y1: Start coordinates (robot position)
        x2, y2: End coordinates (target point)
        pts: Grid points array
        cells: Grid cell values (<=0=obstacle, >0=free)
        obstacle_threshold: Cell value below which it's considered blocked

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

        # Check if this point is an obstacle (cells_no_step <= 0 means obstacle)
        if cells[nearest_idx] <= obstacle_threshold:
            return False  # Path blocked

    return True  # Path clear


def sample_cell_points(env, cell_row, cell_col, num_samples=20):
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


def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, pts, cells_no_step):
    """
    Sample 20 random points in a cell and find the one with clear path that is closest to cell center.

    Args:
        robot_x, robot_y: Current robot position
        env: EnvironmentMap instance
        cell_row, cell_col: Target cell coordinates
        pts: Grid points array from local grid
        cells_no_step: Cell values from local grid

    Returns:
        tuple: (best_x, best_y, valid_samples, rejected_samples) or (None, None, [], []) if no valid point found
    """
    # Sample random points in the cell
    sampled_points = sample_cell_points(env, cell_row, cell_col, num_samples=20)

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
        # Check if path is clear
        if check_line_of_sight(robot_x, robot_y, sample_x, sample_y, pts, cells_no_step):
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


def draw_explored_sides(ax, cell_x, cell_y, half_size, sides_status, cos_yaw, sin_yaw):
    """
    Draw red lines on the edges of a cell to show which sides have been explored.

    Args:
        ax: Matplotlib axis object
        cell_x, cell_y: Center coordinates of the cell
        half_size: Half of the cell size
        sides_status: 4-bit value representing explored sides
        cos_yaw, sin_yaw: Rotation parameters for coordinate transformation
    """
    if sides_status == 0b0000:
        return  # Nothing to draw

    edge_inset = 0.05  # Small inset to make lines visible

    # North edge (top) - Bit 3: 0b1000
    if sides_status & 0b1000:
        north_start = (-half_size + edge_inset, half_size)
        north_end = (half_size - edge_inset, half_size)
        # Rotate to world frame
        ns_wx = cell_x + (north_start[0] * cos_yaw - north_start[1] * sin_yaw)
        ns_wy = cell_y + (north_start[0] * sin_yaw + north_start[1] * cos_yaw)
        ne_wx = cell_x + (north_end[0] * cos_yaw - north_end[1] * sin_yaw)
        ne_wy = cell_y + (north_end[0] * sin_yaw + north_end[1] * cos_yaw)
        ax.plot([ns_wx, ne_wx], [ns_wy, ne_wy], 'r-', linewidth=4, alpha=0.8, zorder=4)

    # East edge (right) - Bit 2: 0b0100
    if sides_status & 0b0100:
        east_start = (half_size, -half_size + edge_inset)
        east_end = (half_size, half_size - edge_inset)
        # Rotate to world frame
        es_wx = cell_x + (east_start[0] * cos_yaw - east_start[1] * sin_yaw)
        es_wy = cell_y + (east_start[0] * sin_yaw + east_start[1] * cos_yaw)
        ee_wx = cell_x + (east_end[0] * cos_yaw - east_end[1] * sin_yaw)
        ee_wy = cell_y + (east_end[0] * sin_yaw + east_end[1] * cos_yaw)
        ax.plot([es_wx, ee_wx], [es_wy, ee_wy], 'r-', linewidth=4, alpha=0.8, zorder=4)

    # South edge (bottom) - Bit 1: 0b0010
    if sides_status & 0b0010:
        south_start = (-half_size + edge_inset, -half_size)
        south_end = (half_size - edge_inset, -half_size)
        # Rotate to world frame
        ss_wx = cell_x + (south_start[0] * cos_yaw - south_start[1] * sin_yaw)
        ss_wy = cell_y + (south_start[0] * sin_yaw + south_start[1] * cos_yaw)
        se_wx = cell_x + (south_end[0] * cos_yaw - south_end[1] * sin_yaw)
        se_wy = cell_y + (south_end[0] * sin_yaw + south_end[1] * cos_yaw)
        ax.plot([ss_wx, se_wx], [ss_wy, se_wy], 'r-', linewidth=4, alpha=0.8, zorder=4)

    # West edge (left) - Bit 0: 0b0001
    if sides_status & 0b0001:
        west_start = (-half_size, -half_size + edge_inset)
        west_end = (-half_size, half_size - edge_inset)
        # Rotate to world frame
        ws_wx = cell_x + (west_start[0] * cos_yaw - west_start[1] * sin_yaw)
        ws_wy = cell_y + (west_start[0] * sin_yaw + west_start[1] * cos_yaw)
        we_wx = cell_x + (west_end[0] * cos_yaw - west_end[1] * sin_yaw)
        we_wy = cell_y + (west_end[0] * sin_yaw + west_end[1] * cos_yaw)
        ax.plot([ws_wx, we_wx], [ws_wy, we_wy], 'r-', linewidth=4, alpha=0.8, zorder=4)


def visualize_grid_with_candidates(pts, cells_no_step, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env=None, save_path=None, recordingInterface=None):
    """
    Visualize the no-step grid with sampled candidates and chosen point.
    Optionally overlay global grid map (only cells visible within local grid bounds).

    Args:
        save_path: If provided, save the figure to this path. The plot is always shown.
        recordingInterface: RecordingInterface instance to get edge information for waypoint connections.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot local grid points
    x = pts[:, 0]
    y = pts[:, 1]
    colors_norm = color.astype(np.float32) / 255.0
    ax.scatter(x, y, c=colors_norm, s=2, alpha=0.4, label='Local Grid (obstacles)')

    # Calculate local grid bounds
    local_x_min, local_x_max = x.min(), x.max()
    local_y_min, local_y_max = y.min(), y.max()

    # Overlay global grid if provided (only cells within local grid bounds)
    if env is not None:
        for row in range(env.rows):
            for col in range(env.cols):
                # Get world position of cell center
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue

                cell_x, cell_y = world_pos

                # Check if cell is within local grid bounds (with small margin)
                margin = env.cell_size
                if not (local_x_min - margin <= cell_x <= local_x_max + margin and
                       local_y_min - margin <= cell_y <= local_y_max + margin):
                    continue  # Skip cells outside local grid view

                half_size = env.cell_size / 2.0

                # Calculate corners in grid frame
                grid_corners = [
                    (-half_size, -half_size),
                    (half_size, -half_size),
                    (half_size, half_size),
                    (-half_size, half_size)
                ]

                # Rotate corners to world frame
                cos_yaw = np.cos(env.origin_yaw)
                sin_yaw = np.sin(env.origin_yaw)

                world_corners = []
                for gx, gy in grid_corners:
                    wx = cell_x + (gx * cos_yaw - gy * sin_yaw)
                    wy = cell_y + (gx * sin_yaw + gy * cos_yaw)
                    world_corners.append((wx, wy))

                # Draw cell
                cell_status, _ = env.get_cell_status(row, col)
                if cell_status == 1:
                    # Visited - green fill
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen',
                                          facecolor='lightgreen', alpha=0.3, zorder=2)
                elif cell_status == -1:
                    # Blocked - red fill
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkred',
                                          facecolor='lightcoral', alpha=0.4, zorder=2)
                else:
                    # Unvisited - gray outline only
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray',
                                          facecolor='none', alpha=0.6, linestyle='--', zorder=2)
                ax.add_patch(rect)

                # Add cell label
                ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=3,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                # Draw explored sides as RED LINES on cell borders
                cell_status, sides_status = env.get_cell_status(row, col)
                if cell_status != 1 and sides_status != 0b0000:  # Show for unvisited and blocked cells
                    draw_explored_sides(ax, cell_x, cell_y, half_size, sides_status, cos_yaw, sin_yaw)

    # Plot rejected candidates (red X)
    if 'rejected' in candidates:
        for point in candidates['rejected']:
            ax.plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    # Plot valid candidates (yellow circles)
    if 'valid' in candidates:
        for point in candidates['valid']:
            ax.plot(point[0], point[1], 'yo', markersize=10, markerfacecolor='yellow',
                    markeredgewidth=2, markeredgecolor='orange', zorder=5)

    # Plot chosen point (large green star)
    if chosen_point is not None:
        ax.plot(chosen_point[0], chosen_point[1], 'g*', markersize=25,
                markeredgewidth=2, label='Target', zorder=6)

        # Calculate and draw distance from robot to target with annotation
        target_dist = np.sqrt((chosen_point[0] - robot_x)**2 + (chosen_point[1] - robot_y)**2)
        ax.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]],
                'g--', linewidth=2.5, alpha=0.8, zorder=4)

        # Add distance text near the middle of the line
        mid_x = (robot_x + chosen_point[0]) / 2
        mid_y = (robot_y + chosen_point[1]) / 2
        ax.text(mid_x, mid_y, f'{target_dist:.2f}m', fontsize=9, color='darkgreen',
               weight='bold', zorder=6,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                        alpha=0.9, edgecolor='darkgreen'))

    # Draw waypoints and robot path
    if type(env.waypoints) != int:
        if env is not None and hasattr(env, 'waypoints') and isinstance(env.waypoints, list) and len(env.waypoints) > 0:
            # Collect visible waypoints
            visible_waypoints = []

            for i, waypoint in enumerate(env.waypoints):
                if not isinstance(waypoint, (tuple, list)):
                    continue
                if type(waypoint) != int:
                    if len(waypoint) >= 2:  # Ensure it's a valid tuple/list
                        wp_x, wp_y = waypoint[0], waypoint[1]
                        if (local_x_min - 0.5 <= wp_x <= local_x_max + 0.5 and
                            local_y_min - 0.5 <= wp_y <= local_y_max + 0.5):
                            visible_waypoints.append((wp_x, wp_y, i))

            # Draw lines connecting waypoints based on actual edges in the graph
            if type(visible_waypoints) != int:
                if isinstance(visible_waypoints, list) and len(visible_waypoints) > 1 and recordingInterface is not None:
                    # Get all edges from the graph
                    edges = recordingInterface.get_edge_list()

                    if edges:
                        # Get all waypoints from graph to map IDs to positions
                        graph_waypoints = recordingInterface.get_waypoint_list()

                        # Create mapping from waypoint ID to position (from env.waypoints)
                        waypoint_id_to_pos = {}
                        for i, wp in enumerate(graph_waypoints):
                            if i < len(env.waypoints) and isinstance(env.waypoints[i], (tuple, list)) and len(env.waypoints[i]) >= 2:
                                waypoint_id_to_pos[wp.id] = (env.waypoints[i][0], env.waypoints[i][1])

                        # Draw edges
                        edge_drawn = False
                        for edge in edges:
                            from_id = edge.id.from_waypoint
                            to_id = edge.id.to_waypoint

                            # Check if both waypoints of this edge are visible
                            if from_id in waypoint_id_to_pos and to_id in waypoint_id_to_pos:
                                from_pos = waypoint_id_to_pos[from_id]
                                to_pos = waypoint_id_to_pos[to_id]

                                # Check if both waypoints are within visible bounds
                                if ((local_x_min - 0.5 <= from_pos[0] <= local_x_max + 0.5 and
                                     local_y_min - 0.5 <= from_pos[1] <= local_y_max + 0.5) and
                                    (local_x_min - 0.5 <= to_pos[0] <= local_x_max + 0.5 and
                                     local_y_min - 0.5 <= to_pos[1] <= local_y_max + 0.5)):

                                    ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
                                           'm--', linewidth=2, alpha=0.5, zorder=3,
                                           label='Waypoint edges' if not edge_drawn else '')
                                    edge_drawn = True

            # Draw waypoints with numbers on top
                for wp_x, wp_y, idx in visible_waypoints:
                    ax.plot(wp_x, wp_y, 'mo', markersize=12, markerfacecolor='magenta',
                           markeredgewidth=2.5, markeredgecolor='purple', zorder=7,
                           label='Waypoints' if idx == 0 else '')
                    ax.text(wp_x + 0.12, wp_y + 0.12, f'W{idx+1}', fontsize=9, color='purple',
                           weight='bold', zorder=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='purple'))

    # Draw ROBOT PATH TRACES with different colors for exploration vs navigation
    if type(env.robot_path) != int:
        if env is not None and hasattr(env, 'robot_path') and isinstance(env.robot_path, list) and len(env.robot_path) > 0:
            # Collect all visible positions
            all_positions = []

            for entry in env.robot_path:
                if not isinstance(entry, (tuple, list)):
                    continue

                # Handle both old format (x, y) and new format (x, y, movement_type)
                if type(entry) != int:
                    if len(entry) >= 2:
                        pos_x, pos_y = entry[0], entry[1]
                        movement_type = entry[2] if len(entry) >= 3 else 'explore'

                        # Check if within visible bounds
                        if (local_x_min - 0.5 <= pos_x <= local_x_max + 0.5 and
                            local_y_min - 0.5 <= pos_y <= local_y_max + 0.5):
                            all_positions.append((pos_x, pos_y, movement_type))

            # Draw traces connecting ALL robot positions in sequence
            if type(all_positions) != int:
                if isinstance(all_positions, list) and len(all_positions) > 1:
                    for i in range(len(all_positions) - 1):
                        pos1 = all_positions[i]
                        pos2 = all_positions[i + 1]

                        # Color based on movement type
                        if pos1[2] == 'navigate' or pos2[2] == 'navigate':
                            # Navigation movement - red dashed line
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                   'r--', linewidth=2.5, alpha=0.7, zorder=4,
                                   label='Navigation' if i == 0 and pos1[2] == 'navigate' else '')
                        else:
                            # Exploration movement - green solid line
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                                   'g-', linewidth=2.5, alpha=0.7, zorder=4,
                                   label='Exploration' if i == 0 else '')

            # Draw position markers
            for i, (pos_x, pos_y, movement_type) in enumerate(all_positions):
                if movement_type == 'navigate':
                    ax.plot(pos_x, pos_y, 'o', color='orange', markersize=5, alpha=0.8, zorder=5)
                else:
                    ax.plot(pos_x, pos_y, 'o', color='lime', markersize=5, alpha=0.8, zorder=5)


    # Draw robot position
    ax.plot(robot_x, robot_y, 'bo', markersize=18, label='Robot', zorder=7)

    # Add distance circles
    for r in [1.0, 2.0]:
        circle = patches.Circle((robot_x, robot_y), r, fill=False,
                               linestyle=':', linewidth=1,
                               edgecolor='blue', alpha=0.3, zorder=1)
        ax.add_patch(circle)

    # Set axis limits to focus on local grid
    ax.set_xlim(local_x_min - 0.5, local_x_max + 0.5)
    ax.set_ylim(local_y_min - 0.5, local_y_max + 0.5)

    ax.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')

    title = f'Iteration {iteration}: Robot Path Visualization'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Save figure to file if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VISUALIZATION] Saved to: {save_path}")

    # Always show the plot for real-time monitoring
    plt.pause(0.5)  # Show plot and continue execution
    plt.close()


def attempt_enter_cell_from_position(local_grid_client, robot_state_client, command_client,
                                      env, target_row, target_col, mission_folder=None, iteration=0, recordingInterface=None):
    """
    Attempt to enter a target cell from the current robot position.

    This method:
    1. Gets the local grid
    2. Samples 20 random points in the target cell
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

    Returns:
        bool: True if successfully entered cell, False otherwise
    """
    print(f"\n[ATTEMPT] Trying to enter cell ({target_row},{target_col}) from current position...")

    # Get local grid to check obstacles
    proto = local_grid_client.get_local_grids(['no_step'])
    pts, cells_no_step, color = spotGrid.create_vtk_no_step_grid(proto, robot_state_client)

    # Get grid proto
    local_grid_proto = None
    for local_grid_found in proto:
        if local_grid_found.local_grid_type_name == 'no_step':
            local_grid_proto = local_grid_found
            break

    if local_grid_proto is None:
        print("[ERROR] No 'no_step' grid found")
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

    # Sample 20 random points in the target cell and find the best one with clear path
    print(f"[INFO] Sampling 20 random points in cell ({target_row},{target_col})...")
    target_x, target_y, valid_samples, rejected_samples = find_best_point_in_cell(
        robot_x, robot_y, env, target_row, target_col, pts, cells_no_step
    )

    if target_x is None or target_y is None:
        print(f"[FAIL] No clear path found to cell ({target_row},{target_col}) from current position")
        # Skip visualization for blocked paths
        return False

    print(f"[OK] Target point in cell ({target_row},{target_col}): ({target_x:.2f}, {target_y:.2f})")

    # Calculate distance and direction
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = np.sqrt(dx ** 2 + dy ** 2)

    print(f"[INFO] Distance to target: {distance:.2f}m")

    # Save visualization with sampled points
    save_path = None
    if mission_folder:
        save_path = os.path.join(mission_folder, f"iteration_{iteration}_sampling.png")

    visualize_grid_with_candidates(
        pts, cells_no_step, color, robot_x, robot_y,
        {'rejected': rejected_samples, 'valid': valid_samples},
        (target_x, target_y), iteration, env, save_path, recordingInterface
    )

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


    time.sleep(0.5)

    # Then move forward
    print(f"[INFO] Step 2: Moving forward {distance:.2f}m...")
    success_move = movements.relative_move(distance, 0, 0, "vision",
                                          command_client, robot_state_client)

    if success_move:
        # Wait for movement to complete
        time.sleep(0.5)

        # VERIFICA: Controlla se il robot è effettivamente nella cella target
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

def realign_robot_to_waypoint_orientation(self, waypoint_name):
    """
    Versione CORRETTA: Usa velocity_command per ruotare fisicamente il robot.
    La versione precedente con 'stand' non funzionava per grandi angoli.
    """
    if waypoint_name not in self.waypoint_poses:
        print(f"[REALIGN] ✗ Nessuna pose salvata per '{waypoint_name}'")
        return False

    target_yaw = self.waypoint_poses[waypoint_name]['yaw']

    # Clienti necessari
    command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
    robot_state_client = self.robot.ensure_client('robot-state')

    print(f"[REALIGN] Allineamento a {waypoint_name} (Target Yaw: {np.degrees(target_yaw):.1f}°)...")

    # Loop di controllo rotazione (max 10 secondi)
    for _ in range(20):
        # Calcola Yaw attuale
        x, y, z, quat = spotUtils.getPosition(robot_state_client)
        current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))

        # Calcola errore
        delta_yaw = target_yaw - current_yaw
        while delta_yaw > np.pi: delta_yaw -= 2 * np.pi
        while delta_yaw < -np.pi: delta_yaw += 2 * np.pi

        if abs(delta_yaw) < np.radians(2.0):  # Tolleranza 2 gradi
            print("[REALIGN] ✓ Allineato.")
            command_client.robot_command(RobotCommandBuilder.stop_command())
            return True

        # Ruota (Velocità proporzionale ma limitata)
        rot_speed = np.clip(delta_yaw, -0.6, 0.6)
        # Minima velocità per non stallare
        if abs(rot_speed) < 0.2: rot_speed = 0.2 * np.sign(rot_speed)

        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0, v_y=0, v_rot=rot_speed)
        command_client.robot_command(cmd)
        time.sleep(0.5)

    print("[REALIGN] ⚠️ Timeout rotazione.")
    command_client.robot_command(RobotCommandBuilder.stop_command())
    return False

#TODO: try to use EXTENT local grid

def easy_walk(options):
    robot, lease_client, robot_state_client, client_metadata = spotLogInUtils.setLogInfo(options)

    estop = spotLogInUtils.SimpleEstop(robot, options.name + "_estop")

    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()

    #TODO controlla che tutte le volte lui si va a scaricare gli ultimi waypoint

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        local_grid_client = robot.ensure_client(LocalGridClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot... This may take several seconds.')
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

        # Create first waypoint in initial cell (0, 0)
        recordingInterface.create_default_waypoint(cell_row=0, cell_col=0)

        env = environmentMap.EnvironmentMap(rows=4, cols=4, cell_size=1)
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)

        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y**2 + quat_boot.z**2))

        # Set origin with position AND orientation
        env.set_origin(x_boot, y_boot, yaw_boot, start_row=0, start_col=0)

        # Register initial waypoint
        env.add_waypoint(x_boot, y_boot)

        # Load any existing manual waypoints from the graph
        # This is useful if we're reconnecting to an existing map
        loaded_count = recordingInterface.load_manual_waypoints_from_graph()
        if loaded_count > 0:
            # Update cell information for loaded waypoints based on current environment map
            recordingInterface.update_waypoint_cells_from_map(env)

        print(f'[INIT] Boot position: x={x_boot:.3f}, y={y_boot:.3f}, z={z_boot:.3f}')
        print(f'[INIT] Boot orientation: yaw={np.rad2deg(yaw_boot):.1f}°')

        # Create mission folder for saving visualizations
        mission_timestamp = datetime.now().strftime("Mission-%Y%m%d_%H%M%S")
        mission_folder = os.path.join(os.path.dirname(__file__), "MissionMap", mission_timestamp)
        os.makedirs(mission_folder, exist_ok=True)
        print(f'[INIT] Mission folder created: {mission_folder}')

        # Generate serpentine path (lawnmower pattern)
        path = env.generate_serpentine_path()

        frontier = []
        cell_attempt_count = {}
        current_path_index = 0

        x, y, z, _ = spotUtils.getPosition(robot_state_client)
        robot_row, robot_col = env.get_cell_from_world(x, y)
        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

        while(True):
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
                # Seleziona il border con rank più basso (index 2 della tupla)
                selected_border = min(borders_in_frontier, key=lambda b: b[2])
                print(f"[BORDER] Selezionato border con rank minore: ({selected_border[0]},{selected_border[1]}) rank={selected_border[2]}")

                check = attempt_enter_cell_from_position(local_grid_client, robot_state_client, command_client, env, selected_border[0], selected_border[1], mission_folder, current_path_index, recordingInterface)
                frontier.remove(selected_border)
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

                    print(f"\n[TARGET] Cella con rank più basso: ({target_row},{target_col}) rank={rank}")

                    # Find nearest waypoint to target cell
                    target_waypoint = recordingInterface.find_nearest_waypoint_to_cell(target_row, target_col)

                    if target_waypoint is not None:
                        # Stop registrazione prima di navigare
                        recordingInterface.stop_recording()
                        # Download lightweight graph (structure only, no snapshots) - faster and lighter
                        recordingInterface.download_full_graph(include_snapshots=False)

                        # Get current position to find starting waypoint
                        x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)
                        current_waypoint = recordingInterface.find_nearest_waypoint_to_position(x_current, y_current)

                        if current_waypoint:
                            # Use shortest path navigation with automatic edge creation
                            print(f"[PATHFINDING] Computing shortest path from {current_waypoint['name']} to {target_waypoint['name']}")
                            success = recordingInterface.navigate_shortest_path(
                                current_waypoint['name'],
                                target_waypoint['name'],
                                env,
                                robot_state_client
                            )
                        else:
                            # Fallback to direct navigation if current waypoint not found
                            print(f"[PATHFINDING] Current waypoint not found, using direct navigation")
                            success = recordingInterface.navigate_to_waypoint(target_waypoint['id'], robot_state_client)
                            if success:
                                recordingInterface.realign_robot_to_waypoint_orientation(target_waypoint['name'])

                        if success:
                            recordingInterface.start_recording()

                            # Prova a entrare nella cella target
                            check = attempt_enter_cell_from_position(
                                local_grid_client, robot_state_client, command_client,
                                env, lowest_rank_cell[0], lowest_rank_cell[1], mission_folder, current_path_index, recordingInterface
                            )

                            if check:
                                # Success - create waypoint and update map
                                x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
                                # Create waypoint saving the cell we entered
                                recordingInterface.create_default_waypoint(cell_row=lowest_rank_cell[0], cell_col=lowest_rank_cell[1])
                                env.add_waypoint(x_final, y_final)

                                # Remove from frontier
                                frontier.remove((lowest_rank_cell[0], lowest_rank_cell[1], lowest_rank_cell[2]))

                                # Update robot position
                                robot_row, robot_col = env.get_cell_from_world(x_final, y_final)

                                # Find new borders
                                frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

                                print(f"[SUCCESS] Entered cell ({lowest_rank_cell[0]},{lowest_rank_cell[1]}) via waypoint navigation")
                            else:
                                # Failure - remove from frontier anyway
                                frontier.remove((lowest_rank_cell[0], lowest_rank_cell[1], lowest_rank_cell[2]))
                                print(f"[ERROR] Could not enter cell {lowest_rank_cell[0], lowest_rank_cell[1]} after navigating to waypoint {target_waypoint['name']}")
                        else:
                            print(f"[ERROR] Could not navigate to waypoint {target_waypoint['name']} near cell {lowest_rank_cell[0], lowest_rank_cell[1]}")
                    else:
                        print(f"[ERROR] No waypoint found near cell {lowest_rank_cell[0], lowest_rank_cell[1]}")

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
        recordingInterface.create_new_edge()

        # --- END OF SIMPLE MISSION ---

        robot.logger.info('Robot mission completed.')
        log_comment = 'Easy autowalk with obstacle avoidance.'
        robot.operator_comment(log_comment)
        robot.logger.info('Added comment "%s" to robot log.', log_comment)

        # Stop recording and download the graph
        recordingInterface.stop_recording()
        x, y, z, _ = spotUtils.getPosition(robot_state_client)
        recordingInterface.find_nearest_waypoint_to_position(x, y)
        recordingInterface.navigate_to_first_waypoint(robot_state_client)
        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(1)
        robot.power_off(cut_immediately=False)
        # Download the complete map with snapshots at the end
        recordingInterface.download_full_graph(include_snapshots=True)
        estop.stop()

# FIXME Change hostname for Jetson/localhost
def main():
    # Instead of argparse, create an options object manually
    options = SimpleNamespace()
    options.name = "easyWalk"
    options.hostname = "192.168.80.3"
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
