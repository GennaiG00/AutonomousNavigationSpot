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
from bosdyn.client.frame_helpers import get_a_tform_b
from types import SimpleNamespace
import navGraphUtils
import movements
import spotGrid
import spotLogInUtils
import environmentMap
import spotUtils
#import velodyneClient
import arcVerification

import global_sampler
import prm_graph

# TODO: check if we can avoid to set a sleep after each movement command
# TODO: change the folder destination of the name download of graph

def sample_cell_points(env, cell_row, cell_col, num_samples=200):
    """Sample random points within a cell."""
    world_pos = env.get_world_position_from_cell(cell_row, cell_col)
    if world_pos is None:
        return []

    cell_center_x, cell_center_y = world_pos
    half_size = env.cell_size / 2.0

    samples = []
    for _ in range(num_samples):
        offset_x = np.random.uniform(-half_size * 0.8, half_size * 0.8)
        offset_y = np.random.uniform(-half_size * 0.8, half_size * 0.8)

        cos_yaw = np.cos(env.origin_yaw)
        sin_yaw = np.sin(env.origin_yaw)

        world_offset_x = offset_x * cos_yaw - offset_y * sin_yaw
        world_offset_y = offset_x * sin_yaw + offset_y * cos_yaw

        sample_x = cell_center_x + world_offset_x
        sample_y = cell_center_y + world_offset_y

        samples.append((sample_x, sample_y))

    return samples

#TODO: test this part. Now this method take a point nearest to the center of the cell.
def find_best_point_in_cell(robot_x, robot_y, env, cell_row, cell_col, pts, cells_obstacle_dist, global_sampler):
    """Sample random points in a cell and explicitly return the cell center as the main target point."""
    target_cell_points = global_sampler.get_point_in_cell(cell_row, cell_col)

    cell_center = env.get_world_position_from_cell(cell_row, cell_col)
    if cell_center is None:
        return None, None, [], []

    cell_center_x, cell_center_y = cell_center
    rejected_samples = []

    return cell_center_x, cell_center_y, target_cell_points, rejected_samples


def visualize_grid_with_candidates(pts, cells_obstacle_dist, color, robot_x, robot_y,
                                   candidates, chosen_point, iteration, env=None, save_path=None, prm_graph=None, chosen_path=None):
    """
    Visualize the obstacle-distance grid with sampled candidates, chosen point, PRM Graph, and Chosen Path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(14, 12))

    x = pts[:, 0]
    y = pts[:, 1]
    PADDING_THRESHOLD = 0.15

    # --- Disegna PRM graph (nodi + archi) sulla mappa LOCALE se fornito ---
    if prm_graph is not None and hasattr(prm_graph, 'nodes'):
        # Disegna i nodi (puntini neri)
        for nid, (nx, ny) in prm_graph.nodes.items():
            ax.plot(nx, ny, 'k.', markersize=3, alpha=0.5, zorder=1)

        # Disegna gli archi (linee grigie sottili)
        if hasattr(prm_graph, 'edges'):
            for node_id, edges in prm_graph.edges.items():
                if node_id in prm_graph.nodes:
                    nx1, ny1 = prm_graph.nodes[node_id]
                    for neighbor_id in edges:
                        if neighbor_id in prm_graph.nodes:
                            nx2, ny2 = prm_graph.nodes[neighbor_id]
                            ax.plot([nx1, nx2], [ny1, ny2], color='gray', linewidth=0.5, alpha=0.3, zorder=1)

    # --- [NEW] Disegna il path scelto sulla mappa LOCALE ---
    if chosen_path is not None and len(chosen_path) > 1:
        path_x = [p[0] for p in chosen_path if p is not None]
        path_y = [p[1] for p in chosen_path if p is not None]
        ax.plot(path_x, path_y, color='magenta', linewidth=4.0, linestyle='-', zorder=6, label='Chosen PRM Path')
        ax.plot(path_x, path_y, 'mo', markersize=8, markeredgecolor='white', zorder=7)

    # Calculate local grid bounds
    local_x_min, local_x_max = x.min(), x.max()
    local_y_min, local_y_max = y.min(), y.max()

    if env is not None:
        for row in range(env.rows):
            for col in range(env.cols):
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None: continue
                cell_x, cell_y = world_pos

                margin = env.cell_size
                if not (local_x_min - margin <= cell_x <= local_x_max + margin and
                        local_y_min - margin <= cell_y <= local_y_max + margin):
                    continue

                half_size = env.cell_size / 2.0
                grid_corners = [(-half_size, -half_size), (half_size, -half_size), (half_size, half_size),
                                (-half_size, half_size)]
                cos_yaw, sin_yaw = np.cos(env.origin_yaw), np.sin(env.origin_yaw)
                world_corners = []
                for gx, gy in grid_corners:
                    wx = cell_x + (gx * cos_yaw - gy * sin_yaw)
                    wy = cell_y + (gx * sin_yaw + gy * cos_yaw)
                    world_corners.append((wx, wy))

                cell_status, _ = env.get_cell_status(row, col) if len(env.get_cell_status(row, col)) == 2 else (
                    env.get_cell_status(row, col), None)
                if cell_status == 1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen', facecolor='lightgreen',
                                           alpha=0.3, zorder=2)
                elif cell_status == -1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkred', facecolor='lightcoral',
                                           alpha=0.4, zorder=2)
                else:
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray', facecolor='none', alpha=0.6,
                                           linestyle='--', zorder=2)
                ax.add_patch(rect)
                ax.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center', fontsize=7, color='black',
                        weight='bold', zorder=3, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    if 'rejected' in candidates:
        for point in candidates['rejected']:
            ax.plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2.5, zorder=5)

    if 'valid' in candidates:
        for point in candidates['valid']:
            ax.plot(point[0], point[1], 'yo', markersize=10, markerfacecolor='yellow', markeredgewidth=2,
                    markeredgecolor='orange', zorder=5)

    if chosen_point is not None:
        ax.plot(chosen_point[0], chosen_point[1], 'g*', markersize=25, markeredgewidth=2, label='Target', zorder=6)
        target_dist = np.sqrt((chosen_point[0] - robot_x) ** 2 + (chosen_point[1] - robot_y) ** 2)
        ax.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]], 'g--', linewidth=2.5, alpha=0.8, zorder=4)
        mid_x, mid_y = (robot_x + chosen_point[0]) / 2, (robot_y + chosen_point[1]) / 2
        ax.text(mid_x, mid_y, f'{target_dist:.2f}m', fontsize=9, color='darkgreen', weight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen'))

    if type(env.waypoints) != int:
        if env is not None and hasattr(env, 'waypoints') and isinstance(env.waypoints, list) and len(env.waypoints) > 0:
            visible_waypoints = []
            for i, waypoint in enumerate(env.waypoints):
                if not isinstance(waypoint, (tuple, list)): continue
                if type(waypoint) != int and len(waypoint) >= 2:
                    wp_x, wp_y = waypoint[0], waypoint[1]
                    if (
                            local_x_min - 0.5 <= wp_x <= local_x_max + 0.5 and local_y_min - 0.5 <= wp_y <= local_y_max + 0.5):
                        visible_waypoints.append((wp_x, wp_y, i))
            if isinstance(visible_waypoints, list) and len(visible_waypoints) > 0:
                for wp_x, wp_y, idx in visible_waypoints:
                    ax.plot(wp_x, wp_y, 'mo', markersize=12, markerfacecolor='magenta', markeredgewidth=2.5,
                            markeredgecolor='purple', zorder=7, label='Waypoints' if idx == 0 else '')
                    ax.text(wp_x + 0.12, wp_y + 0.12, f'W{idx + 1}', fontsize=9, color='purple', weight='bold',
                            zorder=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='purple'))

    if type(env.robot_path) != int:
        if env is not None and hasattr(env, 'robot_path') and isinstance(env.robot_path, list) and len(
                env.robot_path) > 0:
            all_positions = []
            for entry in env.robot_path:
                if not isinstance(entry, (tuple, list)): continue
                if type(entry) != int and len(entry) >= 2:
                    pos_x, pos_y = entry[0], entry[1]
                    movement_type = entry[2] if len(entry) >= 3 else 'explore'
                    if (
                            local_x_min - 0.5 <= pos_x <= local_x_max + 0.5 and local_y_min - 0.5 <= pos_y <= local_y_max + 0.5):
                        all_positions.append((pos_x, pos_y, movement_type))

            if type(all_positions) != int and isinstance(all_positions, list) and len(all_positions) > 1:
                for i in range(len(all_positions) - 1):
                    pos1, pos2 = all_positions[i], all_positions[i + 1]
                    if pos1[2] == 'navigate' or pos2[2] == 'navigate':
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r--', linewidth=2.5, alpha=0.7, zorder=4,
                                label='Navigation' if i == 0 and pos1[2] == 'navigate' else '')
                    else:
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'g-', linewidth=2.5, alpha=0.7, zorder=4,
                                label='Exploration' if i == 0 else '')

            for i, (pos_x, pos_y, movement_type) in enumerate(all_positions):
                color_marker = 'orange' if movement_type == 'navigate' else 'lime'
                ax.plot(pos_x, pos_y, 'o', color=color_marker, markersize=5, alpha=0.8, zorder=5)

    ax.plot(robot_x, robot_y, 'bo', markersize=18, label='Robot', zorder=7)

    for r in [1.0, 2.0]:
        circle = patches.Circle((robot_x, robot_y), r, fill=False, linestyle=':', linewidth=1, edgecolor='blue',
                                alpha=0.3, zorder=1)
        ax.add_patch(circle)

    ax.set_xlim(local_x_min - 0.5, local_x_max + 0.5)
    ax.set_ylim(local_y_min - 0.5, local_y_max + 0.5)
    ax.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')
    ax.set_title(f'Iteration {iteration}: Robot Path Visualization', fontsize=13, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VISUALIZATION] Saved to: {save_path}")

    plt.pause(0.5)
    plt.close()

    # ------------------------------------------------------------------ #
    # SECOND FIGURE: global map
    # ------------------------------------------------------------------ #
    if env is not None:
        ACCUM_RES = 0.05
        if not hasattr(env, '_accumulated_pts'):
            env._accumulated_pts = {}

        if env._accumulated_pts:
            accum_keys = np.array(list(env._accumulated_pts.keys()), dtype=np.float32)
            accum_wx = accum_keys[:, 0] * ACCUM_RES
            accum_wy = accum_keys[:, 1] * ACCUM_RES
            accum_colors = np.array(list(env._accumulated_pts.values()), dtype=np.float32) / 255.0
        else:
            accum_wx = np.array([robot_x], dtype=np.float32)
            accum_wy = np.array([robot_y], dtype=np.float32)
            accum_colors = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        fig2, ax2 = plt.subplots(figsize=(18, 14))
        ax2.scatter(accum_wx, accum_wy, c=accum_colors, s=2, alpha=0.6,
                    label='Accumulated Local Grid (obstacle/padding/free)')

        # --- Disegna PRM graph (nodi + archi) sulla mappa GLOBALE ---
        if prm_graph is not None and hasattr(prm_graph, 'nodes'):
            # Nodi sulla mappa globale (punti neri leggermente più grandi)
            for nid, (nx, ny) in prm_graph.nodes.items():
                ax2.plot(nx, ny, 'k.', markersize=5, alpha=0.6, zorder=3)

            # Archi sulla mappa globale (linee grigie)
            if hasattr(prm_graph, 'edges'):
                for node_id, edges in prm_graph.edges.items():
                    if node_id in prm_graph.nodes:
                        nx1, ny1 = prm_graph.nodes[node_id]
                        for neighbor_id in edges:
                            if neighbor_id in prm_graph.nodes:
                                nx2, ny2 = prm_graph.nodes[neighbor_id]
                                ax2.plot([nx1, nx2], [ny1, ny2], color='gray', linewidth=0.6, alpha=0.4, zorder=2)

        # --- [NEW] Disegna il path scelto sulla mappa GLOBALE ---
        if chosen_path is not None and len(chosen_path) > 1:
            path_x = [p[0] for p in chosen_path if p is not None]
            path_y = [p[1] for p in chosen_path if p is not None]
            ax2.plot(path_x, path_y, color='magenta', linewidth=4.0, linestyle='-', zorder=6, label='Chosen PRM Path')
            ax2.plot(path_x, path_y, 'mo', markersize=8, markeredgecolor='white', zorder=7)

        cos_yaw, sin_yaw = np.cos(env.origin_yaw), np.sin(env.origin_yaw)

        for row in range(env.rows):
            for col in range(env.cols):
                world_pos = env.get_world_position_from_cell(row, col)
                if world_pos is None: continue
                cell_x, cell_y = world_pos
                half_size = env.cell_size / 2.0

                world_corners = [(cell_x - half_size, cell_y - half_size), (cell_x + half_size, cell_y - half_size),
                                 (cell_x + half_size, cell_y + half_size), (cell_x - half_size, cell_y + half_size)]

                cell_status, _ = env.get_cell_status(row, col) if len(env.get_cell_status(row, col)) == 2 else (
                    env.get_cell_status(row, col), None)
                if cell_status == 1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkgreen', facecolor='lightgreen',
                                           alpha=0.3, zorder=2)
                elif cell_status == -1:
                    rect = patches.Polygon(world_corners, linewidth=2, edgecolor='darkred', facecolor='lightcoral',
                                           alpha=0.4, zorder=2)
                else:
                    rect = patches.Polygon(world_corners, linewidth=1.5, edgecolor='gray', facecolor='none', alpha=0.6,
                                           linestyle='--', zorder=2)
                ax2.add_patch(rect)
                ax2.text(cell_x, cell_y, f'{row},{col}', ha='center', va='center', fontsize=7, color='black',
                         weight='bold', zorder=3, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        if 'rejected' in candidates:
            for point in candidates['rejected']: ax2.plot(point[0], point[1], 'rx', markersize=10, markeredgewidth=2.5,
                                                          zorder=5)

        if 'valid' in candidates:
            for point in candidates['valid']: ax2.plot(point[0], point[1], 'yo', markersize=10,
                                                       markerfacecolor='yellow', markeredgewidth=2,
                                                       markeredgecolor='orange', zorder=5)

        if chosen_point is not None:
            ax2.plot(chosen_point[0], chosen_point[1], 'g*', markersize=25, markeredgewidth=2, label='Target', zorder=6)
            ax2.plot([robot_x, chosen_point[0]], [robot_y, chosen_point[1]], 'g--', linewidth=2.5, alpha=0.8, zorder=4)

        rect_local = patches.Rectangle((local_x_min, local_y_min), local_x_max - local_x_min, local_y_max - local_y_min,
                                       linewidth=2, edgecolor='cyan', facecolor='none', linestyle='-', alpha=0.8,
                                       zorder=6, label='Current local scan')
        ax2.add_patch(rect_local)

        ax2.set_xlabel('X [m] (VISION)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y [m] (VISION)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Iteration {iteration}: Global Map View (accumulated local scans)', fontsize=13,
                      fontweight='bold')
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


def attempt_enter_cell_from_position(local_grid, robot_state_client, command_client,
                                     env, target_row, target_col, global_sampler, prm_graph, mission_folder=None,
                                     iteration=0,
                                     recordingInterface=None, verification_tracker=None):
    """Attempt to enter a target cell from the current robot position using background tracking."""
    print(f"\n[ATTEMPT] Trying to enter cell ({target_row},{target_col}) from current position...")


    pts, cells_obstacle_dist, color, local_grid_proto , _= local_grid.return_local_grid('obstacle_distance', robot_state_client)
    transforms_snapshot = local_grid_proto.local_grid.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    robot_x, robot_y = vision_tform_body.position.x, vision_tform_body.position.y

    target_x, target_y, valid_samples, rejected_samples = find_best_point_in_cell(
        robot_x, robot_y, env, target_row, target_col, pts, cells_obstacle_dist, global_sampler)

    if target_x is None or target_y is None:
        print(f"[FAIL] No clear path found to cell ({target_row},{target_col}) from current position")
        visualize_grid_with_candidates(pts, cells_obstacle_dist, color, robot_x, robot_y,
                                       {'rejected': rejected_samples, 'valid': []}, None, 0, env, prm_graph=prm_graph,
                                       chosen_path=None)
        return False

    # --- Calcolo del percorso sul PRM ---
    start_id = prm_graph.get_nearest_node(robot_x, robot_y)
    goal_id = prm_graph.get_nearest_node(target_x, target_y)
    path_ids = prm_graph.find_path_dijkstra(start_id, goal_id)

    full_path_coords = None
    path_waypoints = []

    if path_ids is not None:
        full_path_coords = [prm_graph.get_node_position(nid) for nid in path_ids]

        for nid in path_ids:
            nx, ny = prm_graph.get_node_position(nid)
            path_waypoints.append((nid, nx, ny))

        if len(path_waypoints) > 0:
            path_waypoints.pop(0)  # Rimuove il primo punto (posizione attuale)
    else:
        path_waypoints = None

    save_path = os.path.join(mission_folder,
                             f"iteration_{iteration}_cell_{target_row}_{target_col}.png") if mission_folder else None

    visualize_grid_with_candidates(pts, cells_obstacle_dist, color, robot_x, robot_y,
                                   {'rejected': rejected_samples, 'valid': valid_samples}, (target_x, target_y),
                                   iteration, env, save_path, prm_graph=prm_graph, chosen_path=full_path_coords)

    # --- Ciclo di navigazione waypoint per waypoint ---
    while path_waypoints and len(path_waypoints) > 0:
        robot_x, robot_y, _, _ = spotUtils.getPosition(robot_state_client)
        current_node_id = prm_graph.get_nearest_node(robot_x, robot_y)
        next_node_id, next_x, next_y = path_waypoints[0]

        # 1. AGGIORNA IL TRACKER
        tracker_payload = [(current_node_id, robot_x, robot_y)] + path_waypoints
        verification_tracker.update_path(tracker_payload)

        # 2. STAMPA LIVE DELLO STATO DEL THREAD SECONDARIO
        # Estraiamo una copia del path dal tracker per vedere lo stato assegnato ad ogni arco
        print("\n================ [LIVE TRACKER MONITOR] ================")
        current_tracker_path = verification_tracker.get_path_copy()
        for idx, (nid, nx, ny, status) in enumerate(current_tracker_path):
            if idx == 0:
                print(f" -> [POS ATTUALE ROBOT] Nodo ID: {nid} ({nx:.2f}, {ny:.2f})")
            else:
                status_str = "ATTESA VERIFICA ⏳" if status is None else ("LIBERO ✅" if status is True else "BLOCCATO ❌")
                print(f"    Segmento {idx}: Verso Nodo ID: {nid} ({nx:.2f}, {ny:.2f}) -> {status_str}")
        print("========================================================\n")

        # 3. VERIFICA E ATTESA SICURA SULL'ARCO CORRENTE
        print(f"[INFO] Controllo sicurezza arco corrente: {current_node_id} -> {next_node_id}")

        # Se l'arco è già noto come bloccato dal tracker, ci fermiamo subito
        if verification_tracker.is_arc_blocked(current_node_id, next_node_id):
            print(f"[FAIL] L'arco corrente {current_node_id}-{next_node_id} è BLOCCATO. Interruzione percorso!")
            return False

        # Se non è ancora verificato come libero, entriamo in un loop di attesa sicuro con timeout
        if not verification_tracker.is_arc_verified(current_node_id, next_node_id):
            print(
                f"[INFO] L'arco {current_node_id}-{next_node_id} non è ancora verificato. Attesa elaborazione background...")

            timeout = 4.0
            start_wait = time.time()
            abort_mission = False
            verified_clear = False

            while time.time() - start_wait < timeout:
                if verification_tracker.is_arc_blocked(current_node_id, next_node_id):
                    print(f"[FAIL] Il thread secondario ha rilevato l'arco come BLOCCATO durante l'attesa!")
                    abort_mission = True
                    break
                if verification_tracker.is_arc_verified(current_node_id, next_node_id):
                    verified_clear = True
                    break
                time.sleep(0.1)

            if abort_mission:
                return False

            # Se scatta il timeout (es. l'arco è fuori dal FOV locale e il tracker non lo analizza)
            if not verified_clear:
                print(f"[WARNING] Timeout di attesa superato. Eseguo un controllo istantaneo di fallback...")
                pts_fb, cells_fb, _, _, _=local_grid.return_local_grid('obstacle_distance', robot_state_client)

                if arcVerification.is_arc_in_fov(robot_x, robot_y, next_x, next_y, pts_fb):
                    safety_status = arcVerification.verify_arc_safety(robot_x, robot_y, next_x, next_y, pts_fb,
                                                                      cells_fb)
                    if safety_status == 'blocked':
                        print(f"[FAIL] Fallback manuale: Rilevato ostacolo sull'arco. Interruzione!")
                        return False
                    elif safety_status == 'clear':
                        print(f"[OK] Fallback manuale: L'arco è libero. Procedo.")
                else:
                    # Se non è nemmeno nel FOV (es. alle spalle del robot), ci fidiamo del PRM globale per questa frazione
                    print(
                        f"[WARNING] L'arco non è nel FOV locale delle telecamere. Procedo con cautela basandomi sul PRM.")

        print(f"[OK] Arco {current_node_id}-{next_node_id} pronto per essere percorso.")

        _,_,_,_,proto_updated = local_grid.return_local_grid('obstacle_distance', robot_state_client)
        if not proto_updated:
            print("[ERROR] Impossibile aggiornare la griglia locale")
            return False

        vision_tform_body_current = get_a_tform_b(proto_updated[0].local_grid.transforms_snapshot, VISION_FRAME_NAME,
                                                  BODY_FRAME_NAME)

        success_move = navigate_to(next_x, next_y, robot_x, robot_y, robot_state_client, command_client,
                                   vision_tform_body_current)

        if success_move:
            print(f"[INFO] Spostamento completato su ({next_x:.2f}, {next_y:.2f})")
            path_waypoints.pop(0)
        else:
            print(f"[FAIL] Comando di movimento fallito per ({next_x:.2f}, {next_y:.2f})")
            return False

    robot_x, robot_y, _, _ = spotUtils.getPosition(robot_state_client)
    return env.is_point_in_cell(robot_x, robot_y, target_row, target_col)

def navigate_to(target_x, target_y, robot_x, robot_y, robot_state_client, command_client, vision_tform_body):
    dx, dy = target_x - robot_x, target_y - robot_y
    distance = np.sqrt(dx ** 2 + dy ** 2)
    target_yaw = np.arctan2(dy, dx)

    quat = vision_tform_body.rotation
    current_yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))
    dyaw = np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw))

    print("[INFO] Step 1: Rotating to face target...")
    # FIXME: Try before with relative move and PRM. After that we can try with relative_move_velocity_command.
    movements.relative_move(0, 0, dyaw, "vision", command_client, robot_state_client)
    # movements.relative_move_velocity_command(0, 0, 0, command_client, robot_state_client, VISION_FRAME_NAME)

    print(f"[INFO] Step 2: Moving forward {distance:.2f}m...")
    success_move, _ = movements.relative_move(distance, 0, 0, "vision", command_client, robot_state_client)

    return success_move


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
    local_grid = spotGrid.LocalGrid(robot)
    recordingInterface = navGraphUtils.RecordingInterface(robot, options.download_filepath, client_metadata)
    recordingInterface.stop_recording()
    recordingInterface.clear_map()

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        robot.time_sync.wait_for_sync()
        robot.logger.info('Powering on robot...')
        robot.power_on()
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')
        blocking_stand(command_client)

        print("[INIT] Initializing Velodyne client...")
        recordingInterface.clear_map()
        recordingInterface.start_recording()

        recordingInterface.initialize_with_fiducial(robot_state_client, 549)

        start_row, start_col = 0, 0
        recordingInterface.create_default_waypoint(cell_row=start_row, cell_col=start_col)

        env = environmentMap.EnvironmentMap(rows=3, cols=5, cell_size=1.5)

        # --- [FIX CRUCIALE] Ricaviamo la posizione di boot PRIMA di configurare ed elaborare il grafo PRM ---
        x_boot, y_boot, z_boot, quat_boot = spotUtils.getPosition(robot_state_client)
        yaw_boot = np.arctan2(2.0 * (quat_boot.w * quat_boot.z + quat_boot.x * quat_boot.y),
                              1.0 - 2.0 * (quat_boot.y ** 2 + quat_boot.z ** 2))
        env.set_origin(x_boot, y_boot, yaw_boot, start_row=start_row, start_col=start_col)

        gb_sampler = global_sampler.GlobalSampler(env, 10)
        gb_sampler.sample_global_grid()

        prm = prm_graph.PRM(max_edge_length=1, connection_radius=4, min_edge_length=0.2)
        prm.add_nodes_from_sampler(gb_sampler)

        # [NEW] Inseriamo il punto iniziale (Boot Node) dentro la lista dei nodi permanenti prima di generare gli archi
        current_max_id = max(prm.nodes.keys(), default=-1)
        start_node_id = current_max_id + 1
        prm.add_node(start_node_id, x_boot, y_boot)

        # [NEW] Inseriamo forzatamente tutti i centri geometrici delle celle come nodi del PRM
        current_max_id = start_node_id
        for r in range(env.rows):
            for c in range(env.cols):
                world_pos = env.get_world_position_from_cell(r, c)
                if world_pos is not None:
                    current_max_id += 1
                    prm.add_node(current_max_id, world_pos[0], world_pos[1])

        prm.build_graph()
        verification_tracker = arcVerification.ArcVerificationTracker(robot)
        verification_tracker.start()

        mission_timestamp = datetime.now().strftime("Mission_%d-%m-%Y_%H-%M-%S")
        base_graph_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph")
        graph_folder = os.path.join(base_graph_folder, mission_timestamp)
        mission_map_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionMap", mission_timestamp)
        mission_log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MissionLogs", mission_timestamp)
        os.makedirs(graph_folder, exist_ok=True)
        os.makedirs(mission_map_folder, exist_ok=True)
        os.makedirs(mission_log_folder, exist_ok=True)
        mission_folder = mission_map_folder
        mission_log_path = os.path.join(mission_log_folder, "mission_log.txt")
        open(mission_log_path, "a", buffering=1)

        recordingInterface.set_download_filepath(graph_folder)
        path = env.generate_serpentine_path(start_cell=env.start_cell)

        frontier = []
        visualization_counter = 0

        x, y, z, _ = spotUtils.getPosition(robot_state_client)
        robot_row, robot_col = env.get_cell_from_world(x, y)
        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))

        while (True):
            print(frontier)
            x, y, z, _ = spotUtils.getPosition(robot_state_client)
            robot_row, robot_col = env.get_cell_from_world(x, y)

            borders = env.get_adjacent_frontier_cells(robot_row, robot_col, path)
            borders_in_frontier = [b for b in borders if any((f[0] == b[0] and f[1] == b[1]) for f in frontier)]

            if len(borders_in_frontier) != 0:
                selected_border = min(borders_in_frontier, key=lambda b: b[2])
                check = attempt_enter_cell_from_position(local_grid, robot_state_client, command_client, env,
                                                         selected_border[0], selected_border[1], gb_sampler, prm,
                                                         mission_folder,
                                                         visualization_counter, recordingInterface, verification_tracker)
                visualization_counter += 1
                frontier.remove(selected_border)
                x_new, y_new, _, _ = spotUtils.getPosition(robot_state_client)
                robot_row, robot_col = env.get_cell_from_world(x_new, y_new)
                if check:
                    env.update_position(x, y)
                    recordingInterface.create_default_waypoint(cell_row=selected_border[0], cell_col=selected_border[1])
                    env.add_waypoint(x, y)
                    env.mark_cell_visited(selected_border[0], selected_border[1])
                    frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                else:
                    #TODO qui se non ho trovato il percorso e devo verificare se sono dentro la cella esatta o meno
                    if selected_border == (robot_row, robot_col):
                        env.update_position(x, y)
                        recordingInterface.create_default_waypoint(cell_row=selected_border[0],
                                                                   cell_col=selected_border[1])
                        env.add_waypoint(x, y)
                        env.mark_cell_visited(selected_border[0], selected_border[1])
                        frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                        #TODO qui sono dentro la cella ma n on sono arrivato nel punto desiderato e quindi va bene lo stesso
                    else:
                        #TODO qui niente fallisco vuol dire nemmeno sono entrato nella cella e quindi procedo con il normale algoritmo
                        #devo però controllare che ci se ho già trovato un path allora provo a trovarne un altro che sta sotto un certo costo
                        #se non ho trovato nessun path libero all'inizio (o perchè il corsto è elevato o perchè è tutto bloccato) allora proedo normale
                        #se invece era partito e dopo non ha trovato gli elementi, deve andare a cercare un altro percorso sempre che abbia un costo minore di una certa soglia
            else:
                lowest_rank_cell = env.get_lowest_rank_from_frontier_list(frontier, path)
                if lowest_rank_cell is not None:
                    target_row, target_col, rank = lowest_rank_cell
                    x_current, y_current, _, _ = spotUtils.getPosition(robot_state_client)

                    recordingInterface.stop_recording()
                    target_cell = (target_row, target_col)
                    waypoints_by_cell = recordingInterface.get_all_manual_waypoints_with_cells()
                    nearest_cell = recordingInterface.find_nearest_waypoint_cell_to_target(target_cell,
                                                                                           waypoints_by_cell, env)
                    nearest_wp = recordingInterface.get_manual_waypoint_by_cell(nearest_cell[0], nearest_cell[1])
                    navigation_success = recordingInterface.navigate_to_waypoint(nearest_wp['id'], robot_state_client)

                    if navigation_success:
                        recordingInterface.start_recording()
                        check = attempt_enter_cell_from_position(local_grid, robot_state_client, command_client,
                                                                 env, target_row, target_col, gb_sampler, prm,
                                                                 mission_folder,
                                                                 visualization_counter, recordingInterface, verification_tracker)
                        visualization_counter += 1

                        if check:
                            x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
                            recordingInterface.create_default_waypoint(cell_row=target_row, cell_col=target_col)
                            env.add_waypoint(x_final, y_final)
                            env.mark_cell_visited(target_row, target_col)
                            frontier.remove((target_row, target_col, rank))
                            robot_row, robot_col = env.get_cell_from_world(x_final, y_final)
                            frontier.extend(find_new_borders(env, robot_row, robot_col, path, frontier))
                        else:
                            frontier.remove((target_row, target_col, rank))
                    else:
                        recordingInterface.start_recording()
                        frontier.remove((target_row, target_col, rank))
                else:
                    if len(frontier) > 0:
                        frontier.remove(frontier[0])

            if len(frontier) == 0:
                break

        env.print_map()
        x_final, y_final, _, _ = spotUtils.getPosition(robot_state_client)
        final_row, final_col = env.get_cell_from_world(x_final, y_final)
        recordingInterface.create_default_waypoint(cell_row=final_row, cell_col=final_col)

        recordingInterface.auto_close_loops(True, False)
        recordingInterface.stop_recording()
        recordingInterface.optimize_anchoring()
        recordingInterface.navigate_to_first_waypoint(robot_state_client)

        command_client.robot_command(RobotCommandBuilder.synchro_sit_command(), end_time_secs=time.time() + 20)
        sleep(3)
        robot.power_off(cut_immediately=False)
        recordingInterface.download_full_graph()
        estop.stop()


def main():
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