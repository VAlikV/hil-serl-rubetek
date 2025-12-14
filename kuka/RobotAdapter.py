from dataclasses import dataclass
import numpy as np, time
from .RobotSocket import RobotSocket

@dataclass
class Obs:
    images: dict             # {"cam_front": np.ndarray(H,W,3), "cam_side": ...}
    tcp_pos: np.ndarray
    tcp_vel: np.ndarray
    # torque: np.ndarray
    # gripper_pos: float
    timestamp: float

class RobotAdapter:
    def __init__(self, own_ip, own_port, robot_ip, robot_port, cameras, image_keys=("cam_front","cam_side")):

        
        self.cams = cameras
        self.image_keys = image_keys

        self.robot_socket = RobotSocket(own_ip, own_port, robot_ip, robot_port)

        self.pos, self.orient = self.robot_socket.readState()

        self.robot_socket.sendCommand(self.pos.copy(), self.orient.copy())

        self.start_pos = self.pos.copy()
        self.start_orient = self.orient.copy()

    # ====================================================================================================

    def observe(self) -> Obs:

        pos, orient = self.robot_socket.readState()

        imgs  = {k: self.cams[k].get_image() for k in self.image_keys}

        vel = pos - self.prev_pos
        self.prev_pos = pos

        pos_obs = pos - self.start_pos

        return Obs(imgs, pos_obs, vel, time.time())

    # ====================================================================================================

    def apply_action(self, delta, a_gripper):

        self.pos[0:3] += delta[0:3]
        self.robot_socket.sendCommand(self.pos.copy(), self.orient.copy())

    # ====================================================================================================

    def emergency_stop(self, reason=""):
        pass

    # ====================================================================================================

    def reset(self):
        self.reset_pos = self.start_pos.copy()
        self.reset_pos[0:3] += np.random.uniform(-0.03, 0.03, size=3)

        self.robot_socket.sendCommand(self.reset_pos.copy(), self.orient.copy())

        self.pos = self.reset_pos.copy()

        while not self._check_reset():
            time.sleep(0.001)
    
    # ====================================================================================================

    def _check_reset(self):

        pos, orient = self.robot_socket.readState()

        if (np.abs(self.reset_pos - pos[0:3]) >= 0.002).any():
            print(np.abs(self.reset_pos - pos[0:3]))
            return False
        else:
            return True
