from dataclasses import dataclass
import numpy as np, time
from .RobotSocket import RobotSocket
from .Gripper import Gripper

@dataclass
class Obs:
    images: dict             # {"cam_front": np.ndarray(H,W,3), "cam_side": ...}
    tcp_pos: np.ndarray
    tcp_vel: np.ndarray
    # torque: np.ndarray
    # gripper_pos: float
    timestamp: float

class RobotAdapter:
    def __init__(self, own_ip, own_port, robot_ip, robot_port, cameras, image_keys=("cam_front","cam_side"), gripper="/dev/ttyUSB0"):

        
        self.cams = cameras
        self.image_keys = image_keys

        self.robot_socket = RobotSocket(own_ip, own_port, robot_ip, robot_port)

        self.gripper = Gripper(device=gripper, boudrate=115200, timeout=1)
        self.gripper.send(0)

        self.pos, self.orient = self.robot_socket.readState()
        self.prev_pos = self.pos.copy()

        self.robot_socket.sendCommand(self.pos.copy(), self.orient.copy())

        self.start_pos = np.array([0.6, 0.0, 0.4])
        # self.start_orient = np.array([[0.635366, 0.0545912, 0.770279],
        #                                 [0.0855156, -0.996337, 0.0],
        #                                 [0.767461, 0.0658235, -0.637707]])
        
        # self.start_orient = np.array([[0.5, 0.0, 0.866],
        #                             [0.0, -1.0, 0.0],
        #                             [0.866, 0.0, -0.5]])
        
        self.start_orient = np.array([[-1.0, 0.0, 0.0,],
                                    [0.0, 1.0, 0.0,],
                                    [0.0, 0.0, -1.0]])

    # ====================================================================================================

    def observe(self) -> Obs:

        pos, orient = self.robot_socket.readState()

        imgs  = {k: self.cams[k].get_image() for k in self.image_keys}

        vel = pos - self.prev_pos
        self.prev_pos = pos

        # pos_obs = pos - self.start_pos
        pos_obs = pos

        return Obs(imgs, pos_obs, vel, time.time())

    # ====================================================================================================

    def apply_action(self, delta, a_gripper):

        self.pos[0:3] += delta[0:3]
        self.robot_socket.sendCommand(self.pos.copy(), self.orient.copy())
        self.gripper.send(a_gripper)

    # ====================================================================================================

    def emergency_stop(self, reason=""):
        pass

    # ====================================================================================================

    def reset(self):
        self.reset_pos = self.start_pos.copy()
        self.reset_pos[0:3] += np.random.uniform(-0.03, 0.03, size=3)

        self.reset_orient = self.start_orient.copy()

        self.robot_socket.sendCommand(self.reset_pos.copy(), self.reset_orient.copy())

        # self.pos = self.reset_pos.copy()
        # self.orient = self.reset_orient.copy()

        while not self._check_reset():
            time.sleep(0.001)

        self.pos, self.orient = self.robot_socket.readState()

        time.sleep(2)
    
    # ====================================================================================================

    def _check_reset(self):

        pos, orient = self.robot_socket.readState()

        if (np.abs(self.reset_pos - pos[0:3]) >= 0.02).any():
            print(np.abs(self.reset_pos - pos[0:3]))
            return False
        else:
            return True
