import time
import numpy as np
from bosdyn.client.frame_helpers import *
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

def relative_move(dx, dy, dyaw, frame_name, robot_command_client, robot_state_client, stairs=False):
    """Move the robot relative to its current pose.

    Returns:
        tuple: (success: bool, distance_traveled: float)
               - success: True if goal reached, False if failed
               - distance_traveled: meters traveled before stopping/failing
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Save initial position
    initial_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    initial_x = initial_tform_body.x
    initial_y = initial_tform_body.y

    # Build the transform for where we want the robot to be relative to where the body currently is.
    body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
    out_tform_body = get_se2_a_tform_b(transforms, frame_name, BODY_FRAME_NAME)
    out_tform_goal = out_tform_body * body_tform_goal

    # Command the robot to go to the goal point in the specified frame.

    obstacle_params = spot_command_pb2.ObstacleParams(disable_vision_foot_obstacle_avoidance=True)
    mobility_params = spot_command_pb2.MobilityParams(obstacle_params=obstacle_params)

    robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
        goal_x=out_tform_goal.x, goal_y=out_tform_goal.y, goal_heading=out_tform_goal.angle,
        frame_name=frame_name, params=mobility_params)
    end_time = 6000.0
    cmd_id = robot_command_client.robot_command(lease=None, command=robot_cmd,
                                                end_time_secs=time.time() + end_time)

    # Wait until the robot has reached the goal or fails
    while True:
        feedback = robot_command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback

        # Get current position
        current_state = robot_state_client.get_robot_state()
        current_transforms = current_state.kinematic_state.transforms_snapshot
        current_tform_body = get_se2_a_tform_b(current_transforms, frame_name, BODY_FRAME_NAME)

        # Calculate distance traveled
        distance_traveled = np.sqrt((current_tform_body.x - initial_x) ** 2 +
                                    (current_tform_body.y - initial_y) ** 2)

        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print(f'Failed to reach the goal (traveled {distance_traveled:.2f}m)')
            return False, distance_traveled

        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if traj_feedback.status == traj_feedback.STATUS_AT_GOAL:
            print(f'Arrived at the goal (traveled {distance_traveled:.2f}m)')
            return True, distance_traveled


def relative_move_velocity_command(v_x, v_y, v_rot, robot_command_client, robot_state_client, frame_name):
    obstacle_params = spot_command_pb2.ObstacleParams(disable_vision_foot_obstacle_avoidance=True)
    mobility_params = spot_command_pb2.MobilityParams(obstacle_params=obstacle_params)

    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot, params=mobility_params)

    # Il comando viene lanciato. Timeout di sicurezza a 1 secondo in caso di perdita del segnale.
    robot_command_client.robot_command(command=cmd, end_time_secs=time.time() + 3.0)