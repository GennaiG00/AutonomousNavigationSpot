# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""
This is a test application for communicating with the velodyne over the API.
It should only be used to test the API connection.
"""
import logging
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import bosdyn
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.frame_helpers import get_odom_tform_body, get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_state import RobotStateClient

LOGGER = logging.getLogger(__name__)


#TODO: put this outside this file
def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.1)
#-----------------------------------

class AsyncPointCloud(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncPointCloud, self).__init__('point_clouds', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_point_cloud_from_sources_async(['velodyne-point-cloud'])


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()

class VelodyneProcessor:
    def __init__(self, robot, size_of_cell=0.1, down_sampling_cube=0.05):
        self.robot = robot
        self._point_cloud_client = self.robot.ensure_client('velodyne-point-cloud')
        self._robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

        self._point_cloud_task = AsyncPointCloud(self._point_cloud_client)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)

        self._task_list = [self._point_cloud_task, self._robot_state_task]
        self._async_tasks = AsyncTasks(self._task_list)

        self._running = False
        self._obstacles_2d = np.array([])
        self._lock = threading.Lock()

        self.size_of_cell = size_of_cell
        self.down_sampling_cube = down_sampling_cube

    def start(self):
        self._running = True

        print('VelodyneProcessor started.')

        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        while any(task.proto is None for task in self._task_list):
            time.sleep(0.1)

        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        print('VelodyneProcessor running.')

    def stop(self):
            self._running = False
            print('VelodyneProcessor stopped.')

    def _update_loop(self):
            while self._running:
                self._async_tasks.update()
                time.sleep(0.1)

    def _process_loop(self):
        """Processa continuamente la Point Cloud e aggiorna l'array degli ostacoli."""
        size_of_cell = 0.1
        downsampling_cube = 0.05

        while self._running:
            if not self._point_cloud_task.proto or not self._point_cloud_task.proto[0].point_cloud:
                time.sleep(0.1)
                continue

            data = np.frombuffer(self._point_cloud_task.proto[0].point_cloud.data, dtype=np.float32)
            if self._robot_state_task.proto:
                snapshot = self._robot_state_task.proto.kinematic_state.transforms_snapshot

                # Otteniamo la trasformazione VISION rispetto a BODY (perché il main usa VISION)
                vision_tform_body = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
                body_tform_gpe = get_a_tform_b(snapshot, BODY_FRAME_NAME, 'gpe')

                # Calcolo altezza del terreno
                ground_z = vision_tform_body.position.z - abs(body_tform_gpe.position.z)

                x_coords = data[0::3]
                y_coords = data[1::3]
                z_coords = data[2::3]

                # Trasformiamo i punti locali del sensore/body nel frame VISION
                # Nota: qui per semplicità approssimiamo traslando, se serve la rotazione esatta va applicata la matrice
                # Usiamo la trasformazione semplice per l'esempio
                quat = vision_tform_body.rotation
                # Estrazione yaw (semplificata)
                yaw = np.arctan2(2.0 * (quat.w * quat.z + quat.x * quat.y), 1.0 - 2.0 * (quat.y ** 2 + quat.z ** 2))

                # Rotazione e traslazione
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                x_world = x_coords * cos_yaw - y_coords * sin_yaw + vision_tform_body.position.x
                y_world = x_coords * sin_yaw + y_coords * cos_yaw + vision_tform_body.position.y
                z_world = z_coords + vision_tform_body.position.z

                # Filtro altezza (Ostacoli > 40cm dal suolo)
                z_coords_rel = z_world - ground_z
                high_height_mask = z_coords_rel > 0.40

                x_obs = x_world[high_height_mask]
                y_obs = y_world[high_height_mask]

                if len(x_obs) > 0:
                    # Grigliamo gli ostacoli (Celle 2D da 10cm) per ridurre i punti
                    x_indices = (x_obs / size_of_cell).astype(int)
                    y_indices = (y_obs / size_of_cell).astype(int)

                    obstacle_points = np.vstack((x_indices, y_indices)).T
                    unique_cells = np.unique(obstacle_points, axis=0)

                    # Convertiamo di nuovo in metri
                    final_x = unique_cells[:, 0] * size_of_cell
                    final_y = unique_cells[:, 1] * size_of_cell

                    # Salvataggio thread-safe
                    with self._lock:
                        self._obstacles_2d = np.vstack((final_x, final_y)).T
                else:
                    with self._lock:
                        self._obstacles_2d = np.array([])

            time.sleep(0.2)

    def get_latest_obstacles(self):
        """Restituisce la lista di array (X,Y) degli ostacoli nel frame VISION."""
        with self._lock:
            return np.copy(self._obstacles_2d)