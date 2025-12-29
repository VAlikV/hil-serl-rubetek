import gymnasium as gym
import numpy as np
import socket
import time
from pynput import keyboard
import copy

class RealRobotEnv(gym.Env):
    metadata={"render_modes":[]}

    def __init__(
        self,
        robot_adapter,
        image_keys=("cam_front","cam_side"),
        teleop_set=False,
        teleop_ip="127.0.0.1",
        teleop_port=8081,
        reward_model=None,
        classifier_keys=[],
        enable_keyboard_listener=True,
        fake_env=False
    ):
        
        self.hz = 10
        
        self.robot = robot_adapter
        self.image_keys = image_keys

        self.reward_model = reward_model

        if enable_keyboard_listener:
            listener = keyboard.Listener(on_press=self._on_press,)
            listener.start()

        self.done = False
        self.gripper_state = 1

        self.action_scale = 100
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*4, dtype=np.float32),
                                           high=np.array([+1.0]*4, dtype=np.float32),
                                           dtype=np.float32)

        # H, W = 360, 480
        H, W = 128, 128
        self.observation_space = gym.spaces.Dict(
            {
            "state": gym.spaces.Dict(
                {
                    # "joints_pos": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    # "joints_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                    "tcp_pos": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    "gripper_state":gym.spaces.Box(-1, 1, shape=(1,))
                }
            ),
            "images": gym.spaces.Dict({key: gym.spaces.Box(0, 255, shape=(H, W, 3), dtype=np.uint8) for key in self.image_keys})
            }
        )
        
        self.last_obs = None

        self._t = 0
        self._max_ep_steps = 300

        if fake_env:
            return

        self.teleop_set = teleop_set
        if teleop_set:
            self.haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.haptic_sock.bind((teleop_ip, teleop_port))
            self.haptic_sock.settimeout(0.001)
            self.haptic_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)

        self.first = True
        self.last_pos = np.array([0.0, 0.0, 0.0,
                                  0.0, -1.0, 0.0,
                                  -1.0, 0.0, 0.0,
                                  0.0, 0.0, -1.0])
        
    # ========================================================================================

    def _obs_from_robot(self, o):
        
        proprio = {"tcp_pos": o.tcp_pos,
                   "tcp_vel": o.tcp_vel,
                   "gripper_state":self.gripper_state}

        imgs = {k: o.images[k] for k in self.image_keys}

        return copy.deepcopy({'state': proprio, "images": imgs})
    
    # ========================================================================================

    def reset(self, *, seed=None, options=None):

        self.robot.reset()

        o = self.robot.observe()
        self.last_obs = self._obs_from_robot(o)
        self._t = 0
        info = {}
        
        return self.last_obs, info
    
    # ========================================================================================

    def step(self, action):

        start_time = time.time()

        act = action.copy()

        info = {}

        if self.teleop_set:
            success, message = self._read_teleop()

            if success:
                act = message[0:4]*self.action_scale
                info["intervene_action"] = act

        a = np.asarray(act, dtype=np.float32)/self.action_scale
        self.gripper_state = a[3]

        self.robot.apply_action(a[0:3], self.gripper_state)

        o = self.robot.observe()
        obs = self._obs_from_robot(o)

        if self.reward_model is not None:
            img_dict = {k: o.images[k] for k in self.image_keys}
            score = self.reward_model(img_dict)
            reward = round(score)
            terminated = bool(reward > 0.95)
            # terminated = False

            if terminated: info["succeed"] = True

        else:
            reward = 0.0
            terminated = False

        if self.done:
            terminated = True
            info["succeed"] = True
            self.done = False

        if not terminated:
            if ((obs['state']["tcp_pos"][0] <= -0.02) and ((obs['state']["tcp_pos"][0] >= -0.1))) \
            and ((obs['state']["tcp_pos"][1] <= 0.65) and (obs['state']["tcp_pos"][1] >= 0.55)):
                terminated = True
                reward = -3

        truncated = (self._t >= self._max_ep_steps)
        if truncated: reward = -1
        self.last_obs = obs
        self._t += 1

        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))
        
        return obs.copy(), reward, terminated, truncated, info
    
    # ========================================================================================
    
    def _read_teleop(self):

        delta_pos = np.array([])

        success = False

        try:
            data, addr = self.haptic_sock.recvfrom(1024)
            message = np.array(list(map(float, data.decode()[1:-1].split(","))))
            message = np.append(message, self.gripper_state)
            if len(message):
                if self.first:
                    self.last_pos = message
                    self.first = False
                else:
                    delta_pos = message - self.last_pos
                    self.last_pos = message
                    success = True

        except socket.timeout:
            data, addr = None, None  # или просто continue

        return success, delta_pos
    
    # ========================================================================================

    def _on_press(self, key):
        if key == keyboard.Key.shift:
            self.done = True

        if key == "o":
            self.gripper_state = 1
        
        if key == "c":
            self.gripper_state = -1
            # print("Shift is currently pressed")
        
    # ========================================================================================

    def stop(self):
        self.robot.emergency_stop()
