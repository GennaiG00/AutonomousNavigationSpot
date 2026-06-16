# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
This is a test application for communicating with the velodyne over the API.
It should only be used to test the API connection.
"""

import argparse
import collections
import logging
import sys
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import bosdyn
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.frame_helpers import get_odom_tform_body, get_a_tform_b
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_state import RobotStateClient

matplotlib.use('Qt5agg')
LOGGER = logging.getLogger(__name__)
TEXT_SIZE = 10
SPOT_YELLOW = '#FBD403'


def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.1)


class AsyncPointCloud(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncPointCloud, self).__init__('point_clouds', robot_state_client, LOGGER,
                                              period_sec=1)

    def _start_query(self):
        return self._client.get_point_cloud_from_sources_async(['velodyne-point-cloud'])


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=1)

    def _start_query(self):
        return self._client.get_robot_state_async()


def window_closed(ax):
    fig = ax.figure.canvas.manager
    active_managers = plt._pylab_helpers.Gcf.figs.values()
    return fig not in active_managers


def main():
    sdk = bosdyn.client.create_standard_sdk('VelodyneClient')
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()

    _point_cloud_client = robot.ensure_client('velodyne-point-cloud')
    _robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    _point_cloud_task = AsyncPointCloud(_point_cloud_client)
    _robot_state_task = AsyncRobotState(_robot_state_client)
    _task_list = [_point_cloud_task, _robot_state_task]
    _async_tasks = AsyncTasks(_task_list)
    print('Connected.')

    update_thread = threading.Thread(target=_update_thread, args=[_async_tasks])
    update_thread.daemon = True
    update_thread.start()

    # Wait for the first responses.
    while any(task.proto is None for task in _task_list):
        time.sleep(1)

    fig = plt.figure(figsize=(8, 8))  # Finestra quadrata

    body_tform_butt = SE3Pose(-0.5, 0, 0, Quat())
    body_tform_head = SE3Pose(0.5, 0, 0, Quat())

    # --- GRAFICO IN 2D (Senza projection='3d') ---
    ax = fig.add_subplot(111)

    while True:
        if _point_cloud_task.proto[0].point_cloud:
            data = np.frombuffer(_point_cloud_task.proto[0].point_cloud.data, dtype=np.float32)
            plot_data = data

            ax.clear()
            ax.set_title("Occupancy Grid (2D Map)")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')

            if _robot_state_task.proto:
                snapshot = _robot_state_task.proto.kinematic_state.transforms_snapshot
                odom_tform_body = get_odom_tform_body(snapshot).to_proto()
                helper_se3 = SE3Pose.from_proto(odom_tform_body)
                odom_tform_butt = helper_se3.mult(body_tform_butt)
                odom_tform_head = helper_se3.mult(body_tform_head)

                # Disegna Spot dall'alto in 2D (Solo X e Y)
                ax.plot([odom_tform_butt.x], [odom_tform_butt.y], 'o', color=SPOT_YELLOW, markersize=8)
                ax.plot([odom_tform_head.x], [odom_tform_head.y], 'o', color=SPOT_YELLOW, markersize=8)
                ax.plot([odom_tform_butt.x, odom_tform_head.x], [odom_tform_butt.y, odom_tform_head.y], linewidth=6,
                        color=SPOT_YELLOW)

                ax.text(odom_tform_butt.x, odom_tform_butt.y, '  Butt', size=TEXT_SIZE, zorder=5, color='k',
                        va='center')
                ax.text(odom_tform_head.x, odom_tform_head.y, '  Head', size=TEXT_SIZE, zorder=5, color='k',
                        va='center')

                body_tform_gpe = get_a_tform_b(snapshot, 'body', 'gpe')
                ground_z = helper_se3.z - abs(body_tform_gpe.z)

                x_coords = plot_data[0::3]
                y_coords = plot_data[1::3]
                z_coords = plot_data[2::3]

                # 1. DOWNSAMPLING A CUBETTI
                downsampling_cube = 0.05
                cube = np.vstack([(x_coords / downsampling_cube).astype(int),
                                  (y_coords / downsampling_cube).astype(int),
                                  (z_coords / downsampling_cube).astype(int)]).T

                _, cube_index = np.unique(cube, axis=0, return_index=True)
                x_coords = x_coords[cube_index]
                y_coords = y_coords[cube_index]
                z_coords = z_coords[cube_index]

                # 2. FILTRO RAGGIO 10 METRI (Rispetto al centro globale di Spot)
                distanze_2d = np.sqrt((x_coords - helper_se3.x) ** 2 + (y_coords - helper_se3.y) ** 2)
                radius_mask = distanze_2d <= 10.0

                x_coords = x_coords[radius_mask]
                y_coords = y_coords[radius_mask]
                z_coords = z_coords[radius_mask]

                if len(x_coords) == 0:
                    plt.pause(0.2)
                    continue

                z_coords_rel = z_coords - ground_z
                high_height_mask = z_coords_rel > 0.40

                # 3. ALGORITMO CELLE 2D
                size_of_cell = 0.1  # Griglia di 10x10 cm
                x_indices = (x_coords / size_of_cell).astype(int)
                y_indices = (y_coords / size_of_cell).astype(int)

                obstacle_points = np.vstack([x_indices[high_height_mask], y_indices[high_height_mask]]).T
                blocked_cells = set(map(tuple, obstacle_points))

                all_point = np.vstack((x_indices, y_indices)).T

                # Estraiamo SOLO le celle uniche 2D per disegnare ogni pixel una sola volta
                unique_cells = list(set(map(tuple, all_point)))
                unique_cells_array = np.array(unique_cells)

                if len(unique_cells_array) == 0:
                    plt.pause(0.2)
                    continue

                # Riconvertiamo gli indici interi in coordinate metriche per disegnarli
                cell_x_coords = unique_cells_array[:, 0] * size_of_cell
                cell_y_coords = unique_cells_array[:, 1] * size_of_cell

                # Controlliamo quali di queste celle uniche sono ostacoli
                vertical_obstacle_mask = np.array([tuple(p) in blocked_cells for p in unique_cells_array], dtype=bool)

                # Colori: grigio chiaro per terreno libero, rosso intenso per ostacoli
                colori = np.full(len(vertical_obstacle_mask), '#cccccc', dtype=object)
                colori[vertical_obstacle_mask] = '#ff0000'

            # 4. RENDERING 2D OTTIMIZZATO
            # Usiamo marker='s' (square) per disegnare dei pixel quadrati al posto dei puntini rotondi
            # zorder=0 assicura che la mappa stia sotto ai marker gialli del robot
            ax.scatter(cell_x_coords, cell_y_coords, c=colori, marker='s', s=15, alpha=0.9, zorder=0)

            # Legenda personalizzata
            ax.plot([], [], 's', color='#cccccc', label='Area Libera (< 40cm)')
            ax.plot([], [], 's', color='#ff0000', label='Ostacolo (≥ 40cm)')
            ax.legend(loc='upper right')

            # 5. TELECAMERA FISSA E PROPORZIONATA
            ax.set_aspect('equal', adjustable='box')  # Forza le proporzioni 1:1
            ax.set_xlim([helper_se3.x - 10, helper_se3.x + 10])
            ax.set_ylim([helper_se3.y - 10, helper_se3.y + 10])

            plt.draw()
            plt.pause(0.2)
            if window_closed(ax):
                sys.exit(0)


if __name__ == '__main__':
    if not main():
        sys.exit(1)