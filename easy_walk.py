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

# TODO: check if we can avoid to set a sleep after each movement command
# TODO: change the folder destination of the name download of graph

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


# def navigate_to_cell_via_waypoint(env, recording_interface, target_cell):
#     """
#     Navigate to a cell by first going to the nearest waypoint, then moving to the cell.
#
#     Returns:
#         tuple: (success: bool, target_cell: tuple)
#     """
#     print(f"\n[NAV] Navigating to cell {target_cell} via waypoint...")
#
#     waypoint_id = find_nearest_waypoint_to_cell(env, target_cell, recording_interface)
#     if waypoint_id is None:
#         print("[ERROR] No waypoints available for navigation")
#         return False, target_cell
#
#     # Navigate to that waypoint using graph_nav
#     print(f"[NAV] Step 1: Navigating to waypoint {waypoint_id}...")
#     nav_success = recording_interface.navigate_to_waypoint(waypoint_id)
#
#     if not nav_success:
#         print(f"[ERROR] Failed to navigate to waypoint {waypoint_id}")
#         return False, target_cell
#
#     print(f"[OK] Reached waypoint {waypoint_id}")
#     time.sleep(0.5)
#
#     return True, target_cell

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
    sampled_points = sample_cell_points(env, cell_row, cell_col, num_samples=50)

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
        if check_line_of_sight(robot_x, robot_y, sample_x, sample_y, pts, cells_obstacle_dist):
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


def visualize_grid_with_candidates(pts, cells_obstacle_dist, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env=None, save_path=None):
    """
    Visualize the obstacle-distance grid with sampled candidates and chosen point.
    Zero-padding colour scheme: red = inside obstacle (dist<0), blue = passable (dist>=0).
    Optionally overlay global grid map (only cells visible within local grid bounds).

    Args:
        save_path: If provided, save the figure to this path
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

                # Calculate corners in grid frame (non-rotated square)
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
                    # Apply rotation and translation
                    wx = cell_x + (gx * cos_yaw - gy * sin_yaw)
                    wy = cell_y + (gx * sin_yaw + gy * cos_yaw)
                    world_corners.append((wx, wy))

                # Draw cell
                cell_status = env.get_cell_status(row, col)
                if cell_status == 1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen',
                                          facecolor='lightgreen', alpha=0.3, zorder=2)
                elif cell_status == -1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkred',
                                          facecolor='lightcoral', alpha=0.4, zorder=2)
                else:
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray',
                                          facecolor='none', alpha=0.6, linestyle='--', zorder=2)
                ax.add_patch(rect)

                # Add cell label
                ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=3,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

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

            # Draw waypoints with numbers on top (edges removed for cleaner visualization)
            if isinstance(visible_waypoints, list) and len(visible_waypoints) > 0:
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

    # Save figure if save_path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VISUALIZATION] Saved to: {save_path}")

    plt.pause(0.5)
    plt.close()

    # ------------------------------------------------------------------ #
    # SECOND FIGURE: global map with ACCUMULATED local scans overlaid    #
    # ------------------------------------------------------------------ #
    if env is not None:
        # ---- Merge current scan into the persistent accumulated map -----
        # env._accumulated_pts: dict (ix, iy) -> [R, G, B]  (0-255 integers)
        # Points are quantised to ACCUM_RES metres so nearby pts merge.
        ACCUM_RES = 0.05  # metres per accumulated pixel

        if not hasattr(env, '_accumulated_pts'):
            env._accumulated_pts = {}

        for pt_idx in range(len(pts)):
            px_w, py_w = float(pts[pt_idx, 0]), float(pts[pt_idx, 1])
            r_new = int(colors_norm[pt_idx, 0] * 255)
            g_new = int(colors_norm[pt_idx, 1] * 255) if colors_norm.shape[1] > 1 else 0
            b_new = int(colors_norm[pt_idx, 2] * 255) if colors_norm.shape[1] > 2 else 0
            key = (round(px_w / ACCUM_RES), round(py_w / ACCUM_RES))

            if key not in env._accumulated_pts:
                env._accumulated_pts[key] = [r_new, g_new, b_new]
            else:
                # Blue channel > 0  →  free/steppable pixel: blue always wins
                if b_new > 0:
                    env._accumulated_pts[key] = [r_new, g_new, b_new]
                # Red (obstacle) only stays if the slot was empty (already handled above)

        # ---- Build numpy arrays from the accumulated dict ---------------
        if env._accumulated_pts:
            accum_keys   = np.array(list(env._accumulated_pts.keys()),   dtype=np.float32)
            accum_wx     = accum_keys[:, 0] * ACCUM_RES
            accum_wy     = accum_keys[:, 1] * ACCUM_RES
            accum_colors = np.array(list(env._accumulated_pts.values()), dtype=np.float32) / 255.0
        else:
            accum_wx     = np.array([robot_x], dtype=np.float32)
            accum_wy     = np.array([robot_y], dtype=np.float32)
            accum_colors = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        fig2, ax2 = plt.subplots(figsize=(18, 14))

        # --- plot ACCUMULATED local grid (all past scans merged) ---------
        ax2.scatter(accum_wx, accum_wy, c=accum_colors, s=2, alpha=0.6,
                    label='Accumulated Local Grid')

        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        # --- draw ALL global grid cells (no bounds clipping) -------------
        for row in range(env.rows):
            for col in range(env.cols):
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None:
                    continue
                cell_x, cell_y = world_pos
                half_size = env.cell_size / 2.0

                # cell_x/cell_y is already in the world frame —
                # do NOT rotate corner offsets again.
                world_corners = [
                    (cell_x - half_size, cell_y - half_size),
                    (cell_x + half_size, cell_y - half_size),
                    (cell_x + half_size, cell_y + half_size),
                    (cell_x - half_size, cell_y + half_size),
                ]

                cell_status, sides_status = env.get_cell_status(row, col)
                if cell_status == 1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen',
                                           facecolor='lightgreen', alpha=0.3, zorder=2)
                elif cell_status == -1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkred',
                                           facecolor='lightcoral', alpha=0.4, zorder=2)
                else:
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray',
                                           facecolor='none', alpha=0.6, linestyle='--', zorder=2)
                ax2.add_patch(rect)

                ax2.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center',
                         fontsize=7, color='black', weight='bold', zorder=3,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

                if cell_status != 1 and sides_status != 0b0000:
                    draw_explored_sides(ax2, cell_x, cell_y, half_size, sides_status, cos_yaw, sin_yaw)

        # --- rejected candidates ---
        if 'rejected' in candidates:
            for point in candidates['rejected']:
                ax2.plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

        # --- valid candidates ---
        if 'valid' in candidates:
            for point in candidates['valid']:
                ax2.plot(point[0], point[1], 'yo', markersize=10, markerfacecolor='yellow',
                         markeredgewidth=2, markeredgecolor='orange', zorder=5)

        # --- chosen point ---
        if chosen_point is not None:
            ax2.plot(chosen_point[0], chosen_point[1], 'g*', markersize=25,
                     markeredgewidth=2, label='Target', zorder=6)
            target_dist = np.sqrt((chosen_point[0] - robot_x)**2 + (chosen_point[1] - robot_y)**2)
            ax2.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]],
                     'g--', linewidth=2.5, alpha=0.8, zorder=4)
            mid_x = (robot_x + chosen_point[0]) / 2
            mid_y = (robot_y + chosen_point[1]) / 2
            ax2.text(mid_x, mid_y, f'{target_dist:.2f}m', fontsize=9, color='darkgreen',
                     weight='bold', zorder=6,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen',
                               alpha=0.9, edgecolor='darkgreen'))

        # --- waypoints (all, not clipped) ---
        if env is not None and hasattr(env, 'waypoints') and isinstance(env.waypoints, list) and len(env.waypoints) > 0:
            for i, waypoint in enumerate(env.waypoints):
                if not isinstance(waypoint, (tuple, list)) or len(waypoint) < 2:
                    continue
                wp_x, wp_y = waypoint[0], waypoint[1]
                ax2.plot(wp_x, wp_y, 'mo', markersize=12, markerfacecolor='magenta',
                         markeredgewidth=2.5, markeredgecolor='purple', zorder=7,
                         label='Waypoints' if i == 0 else '')
                ax2.text(wp_x + 0.12, wp_y + 0.12, f'W{i+1}', fontsize=9, color='purple',
                         weight='bold', zorder=8,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   alpha=0.9, edgecolor='purple'))

        # --- robot path traces (all, not clipped) ---
        if env is not None and hasattr(env, 'robot_path') and isinstance(env.robot_path, list) and len(env.robot_path) > 0:
            all_pos_global = []
            for entry in env.robot_path:
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                px, py = entry[0], entry[1]
                mt = entry[2] if len(entry) >= 3 else 'explore'
                all_pos_global.append((px, py, mt))

            for i in range(len(all_pos_global) - 1):
                p1, p2 = all_pos_global[i], all_pos_global[i + 1]
                if p1[2] == 'navigate' or p2[2] == 'navigate':
                    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                             'r--', linewidth=2.5, alpha=0.7, zorder=4,
                             label='Navigation' if i == 0 and p1[2] == 'navigate' else '')
                else:
                    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                             'g-', linewidth=2.5, alpha=0.7, zorder=4,
                             label='Exploration' if i == 0 else '')

            for px, py, mt in all_pos_global:
                color_marker = 'orange' if mt == 'navigate' else 'lime'
                ax2.plot(px, py, 'o', color=color_marker, markersize=5, alpha=0.8, zorder=5)

        # --- robot position ---
        ax2.plot(robot_x, robot_y, 'bo', markersize=18, label='Robot', zorder=7)
        for r in [1.0, 2.0]:
            circle2 = patches.Circle((robot_x, robot_y), r, fill=False,
                                     linestyle=':', linewidth=1,
                                     edgecolor='blue', alpha=0.3, zorder=1)
            ax2.add_patch(circle2)

        # Draw a rectangle showing the CURRENT local scan extent
        rect_local = patches.Rectangle(
            (local_x_min, local_y_min),
            local_x_max - local_x_min,
            local_y_max - local_y_min,
            linewidth=2, edgecolor='cyan', facecolor='none',
            linestyle='-', alpha=0.8, zorder=6, label='Current local scan'
        )
        ax2.add_patch(rect_local)

        ax2.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')
        ax2.set_title(
            f'Iteration {iteration}: Global Map View (accumulated local scans)',
            fontsize=13, fontweight='bold'
        )
        ax2.axis('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            global_save_path = f"{base}_global{ext}"
            fig2.savefig(global_save_path, dpi=150, bbox_inches='tight')
            print(f"[VISUALIZATION] Global map saved to: {global_save_path}")

        plt.pause(0.5)
        plt.close(fig2)


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

        # Visualize the blocked path
        visualize_grid_with_candidates(
            pts, cells_obstacle_dist, color, robot_x, robot_y,
            {'rejected': rejected_samples, 'valid': []},
            None, 0, env
        )

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

    visualize_grid_with_candidates(
        pts, cells_obstacle_dist, color, robot_x, robot_y,
        {'rejected': rejected_samples, 'valid': valid_samples},
        (target_x, target_y), iteration, env, save_path
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

    estop = spotLogInUtils.SimpleEstop(robot, options.name + "_estop")

    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()

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

        # Create first waypoint in initial cell (0, 0)
        recordingInterface.create_default_waypoint(cell_row=0, cell_col=0)

        env = environmentMap.EnvironmentMap(rows=4, cols=4, cell_size=1.5)
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)

        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y ** 2 + quat_boot.z ** 2))

        # Arrotondiamo lo yaw al multiplo di 90° (pi/2) più vicino.
        # Mantiene i quadrati dritti sul grafico, ma fa espandere la griglia in avanti rispetto al robot.
        snapped_yaw = round(yaw_boot / (np.pi / 2.0)) * (np.pi / 2.0)

        env.set_origin(x_boot, y_boot, snapped_yaw, start_row=0, start_col=0)

        print(f'[INIT] Boot position: x={x_boot:.3f}, y={y_boot:.3f}, z={z_boot:.3f}')
        print(
            f'[INIT] Boot orientation: reale {np.rad2deg(yaw_boot):.1f}° -> allineata alla griglia: {np.rad2deg(snapped_yaw):.1f}°')


        mission_timestamp = datetime.now().strftime("Mission_%d-%m-%Y_%H-%M-%S")
        base_graph_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph")
        os.makedirs(base_graph_folder, exist_ok=True)
        graph_folder = os.path.join(base_graph_folder, mission_timestamp)
        os.makedirs(graph_folder, exist_ok=True)

        # Also create MissionMap folder for visualizations (separate from graphs)
        mission_map_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionMap", mission_timestamp)
        os.makedirs(mission_map_folder, exist_ok=True)
        mission_folder = mission_map_folder  # For visualization files

        print(f'[INIT] Mission folder created: {mission_folder}')
        print(f'[INIT] Graph folder created: {graph_folder}')

        # Update recording interface to save graph in mission folder
        recordingInterface.set_download_filepath(graph_folder)

        # Generate serpentine path (lawnmower pattern)
        path = env.generate_serpentine_path()

        frontier = []
        current_path_index = 0
        visualization_counter = 0  # Counter for visualization files

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
                # Select the border with the lowest rank (index 2 of the tuple)
                selected_border = min(borders_in_frontier, key=lambda b: b[2])
                print(f"[BORDER] Selected border with lowest rank: ({selected_border[0]},{selected_border[1]}) rank={selected_border[2]}")

                check = attempt_enter_cell_from_position(local_grid_client, robot_state_client, command_client, env, selected_border[0], selected_border[1], mission_folder, visualization_counter, recordingInterface)
                visualization_counter += 1
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

                    print(f"\n[TARGET] Cell with lowest rank: ({target_row},{target_col}) rank={rank}")

                    # Get current position
                    x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)
                    current_row, current_col = env.get_cell_from_world(x_current, y_current)

                    # Use grid-based path optimization to find shortest path
                    print(f"\n[PATH_OPTIMIZE] Finding optimized path from ({current_row},{current_col}) to ({target_row},{target_col})")

                    # Stop recording to analyze graph (no need to download snapshots, _get_graph() is used internally)
                    recordingInterface.stop_recording()

                    # Find shortest path (WITHOUT creating edges yet)
                    path_result = recordingInterface.find_and_optimize_path(
                        start_cell=(current_row, current_col),
                        end_cell=(target_row, target_col),
                        env_map=env,
                        create_missing_edges=False  # Don't create edges yet
                    )

                    if path_result['success']:
                        print(f"\n[PATH_OPTIMIZE] Path found: {len(path_result['cell_path'])-1} hops")
                        print(f"[PATH_OPTIMIZE] Waypoints: {' -> '.join(path_result['waypoint_names'])}")

                        # Create missing edges if needed (requires recording to be active)
                        if path_result['missing_edges']:
                            print(f"[PATH_OPTIMIZE] Found {len(path_result['missing_edges'])} missing edges - creating them...")

                            # Restart recording to create edges
                            recordingInterface.start_recording()

                            for from_name, to_name in path_result['missing_edges']:
                                success = recordingInterface.create_edge_between_waypoints(from_name, to_name)
                                if success:
                                    path_result['edges_created'].append((from_name, to_name))

                            print(f"[PATH_OPTIMIZE] Created {len(path_result['edges_created'])} edges")

                            # Stop recording again for navigation
                            recordingInterface.stop_recording()
                            # Note: No need to download_full_graph - cache is invalidated by create_edge_between_waypoints

                        if path_result['edges_created']:
                            print(f"[PATH_OPTIMIZE] Created {len(path_result['edges_created'])} missing edges")
                            # Note: No need to download_full_graph - _get_graph() will fetch fresh data

                        # Get waypoints by cell for navigation
                        waypoints_by_cell = recordingInterface.get_all_manual_waypoints_with_cells()

                        # Check if the last cell in the path is the target cell
                        waypoints_to_navigate = path_result['waypoint_names'][1:]  # Skip first (already there)

                        # If the last waypoint is in the target cell, we don't need to navigate to it
                        # but stop at the previous waypoint
                        last_cell_in_path = path_result['cell_path'][-1] if path_result['cell_path'] else None

                        if last_cell_in_path == (target_row, target_col) and len(waypoints_to_navigate) > 1:
                            # Last waypoint is in target cell - stop at the previous one
                            waypoints_to_navigate = waypoints_to_navigate[:-1]
                            print(f"[NAV] Target cell has waypoint - stopping at previous waypoint")
                        elif last_cell_in_path == (target_row, target_col) and len(waypoints_to_navigate) == 1:
                            # There's only one waypoint and it's in the target cell - don't navigate, we're already close
                            waypoints_to_navigate = []
                            print(f"[NAV] Target cell is adjacent - no navigation needed, attempting direct entry")

                        # Navigate through waypoints (stop BEFORE target cell)
                        navigation_success = True
                        for i, waypoint_name in enumerate(waypoints_to_navigate, 1):
                            print(f"\n[NAV] Step {i}/{len(waypoints_to_navigate)}: Navigating to {waypoint_name}")

                            # Find waypoint data
                            waypoint_data = None
                            for cell, wp_data in waypoints_by_cell.items():
                                if wp_data['name'] == waypoint_name:
                                    waypoint_data = wp_data
                                    break

                            if waypoint_data:
                                success = recordingInterface.navigate_to_waypoint(
                                    waypoint_data['id'],
                                    robot_state_client
                                )

                                if success:
                                    recordingInterface.realign_robot_to_waypoint_orientation(waypoint_data['name'])
                                    print(f"[NAV] Reached {waypoint_name}")
                                else:
                                    print(f"[NAV] Failed to reach {waypoint_name}")
                                    navigation_success = False
                                    break
                            else:
                                print(f"[NAV] ERROR: Waypoint data not found for {waypoint_name}")
                                navigation_success = False
                                break

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
        start_row, start_col = 0, 0  # wp_0 is at cell (0,0)

        print(f"[RETURN_OPTIMIZE] Current position: cell ({current_row},{current_col})")
        print(f"[RETURN_OPTIMIZE] Target: wp_0 at cell ({start_row},{start_col})")

        # Find optimal path back to start
        return_path_result = recordingInterface.find_and_optimize_path(
            start_cell=(current_row, current_col),
            end_cell=(start_row, start_col),
            env_map=env,
            create_missing_edges=True  # Create edges if needed
        )

        if return_path_result['success']:
            print(f"\n[RETURN_OPTIMIZE] ✓ Return path found: {len(return_path_result['cell_path'])-1} hops")
            print(f"[RETURN_OPTIMIZE] Waypoints: {' -> '.join(return_path_result['waypoint_names'])}")

            if return_path_result['edges_created']:
                print(f"[RETURN_OPTIMIZE] Created {len(return_path_result['edges_created'])} edges for return path")
                for from_name, to_name in return_path_result['edges_created']:
                    print(f"  - {from_name} → {to_name}")
        else:
            print(f"[RETURN_OPTIMIZE] ✗ Could not optimize return path")

        print(f"{'='*70}\n")

        recordingInterface.stop_recording()
        recordingInterface.find_nearest_waypoint_to_position(x_current, y_current)
        recordingInterface.navigate_to_first_waypoint(robot_state_client)

        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(3)
        robot.power_off(cut_immediately=False)

        # Save the final map to disk (includes return path optimization)
        recordingInterface.download_full_graph()
        estop.stop()

# FIXME Change hostname for Jetson/localhost but anyway it's broken
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
