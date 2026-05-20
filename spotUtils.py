from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, BODY_FRAME_NAME, ODOM_FRAME_NAME

def getPosition(robot_state_client):
    robot_state = robot_state_client.get_robot_state()
    transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
    vision_tform_body = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)

    return vision_tform_body.position.x, vision_tform_body.position.y, vision_tform_body.position.z, vision_tform_body.rotation
