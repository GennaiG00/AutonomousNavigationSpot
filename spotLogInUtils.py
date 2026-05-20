import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive

def setLogInfo(options):
    bosdyn.client.util.setup_logging(options.verbose)
    sdk = bosdyn.client.create_standard_sdk(options.name)
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    session_name = options.recording_session_name
    if session_name == '':
        session_name = os.path.basename(
            '/Users/gianmariagennai/Documents/Unifi/Magistrale/spot/autowalk/TestAuto'
        )
    user_name = options.recording_user_name
    if user_name == '':
        user_name = robot._current_user
    client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name,
        client_username=user_name,
        client_id='RecordingClient',
        client_type='Python SDK',
    )
    return robot, lease_client, robot_state_client, client_metadata

class SimpleEstop:
    """
    Simple E-Stop manager without GUI.
    Call start() to activate, stop() to release.
    """

    def __init__(self, robot, name='simple_estop'):
        self.estop_client = robot.ensure_client(EstopClient.default_service_name)
        self.endpoint = EstopEndpoint(self.estop_client, name, estop_timeout=9.0)
        self.endpoint.force_simple_setup()
        self.keepalive = EstopKeepAlive(self.endpoint)
        # Start in allowed state
        self.keepalive.allow()
        print(f"[OK] E-Stop '{name}' activated and released")

    def stop(self):
        """Trigger E-Stop (robot stops immediately)."""
        self.keepalive.stop()
        print("[ESTOP] Robot stopped")

    def allow(self):
        """Release E-Stop (robot can move)."""
        self.keepalive.allow()
        print("[ESTOP] Robot allowed to move")
