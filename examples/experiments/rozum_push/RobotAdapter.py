from dataclasses import dataclass
import numpy as np, time

@dataclass
class Obs:
    images: dict             # {"cam_front": np.ndarray(H,W,3), "cam_side": ...}
    q: np.ndarray
    dq: np.ndarray
    tcp_pos: np.ndarray
    tcp_vel: np.ndarray
    # gripper_pos: float
    timestamp: float

class RobotAdapter:
    def __init__(self, robot, cameras, image_keys=("cam_front","cam_side")):
        self.ctrl = robot
        self.cams = cameras
        self.image_keys = image_keys

        q, dq, tcp, tcp_vel, torque = self.ctrl.get_current_info()

        self.pos = tcp[0:3]
        self.orient = np.array([[1.0, 0.0, 0.0], 
                                [0.0, -1.0, 0.0], 
                                [0.0, 0.0, -1.0]])
        
        self.ctrl.set_new_point(self.pos.copy(), self.orient.copy())

        self.start_pos = tcp[0:3].copy()
        self.start_orient = self.orient.copy()

    # ====================================================================================================

    def observe(self) -> Obs:
        q, dq, tcp, tcp_vel, torque = self.ctrl.get_robot_attr()
        imgs  = {k: self.cams[k].get_image() for k in self.image_keys}

        return Obs(imgs, q, dq, tcp, tcp_vel, torque, time.time())

    # ====================================================================================================

    def apply_action(self, delta, a_gripper):

        # if delta[0:3].any() >= 0.0001:
        self.pos += delta[0:3]
        self.ctrl.set_target(self.pos.copy(), self.orient.copy())    # Δx, Δr (roll/pitch/yaw) в рамке ЭЭ
        self.ctrl.step()
        # self.ctrl.set_gripper_cmd(a_gripper)

    # ====================================================================================================

    def emergency_stop(self, reason=""):  # по желанию
        self.ctrl.stop()

    # ====================================================================================================

    def reset(self):
        self.ctrl.set_target(self.start_pos, self.start_orient)

        while self._check_reset():
            time.sleep(0.001)
    
    # ====================================================================================================

    def _check_reset(self):

        tcp = self.ctrl.get_current_tcp()

        if np.abs(self.start_pos - tcp[0:3]).any() >= 0.001:
            return False
        else:
            return True