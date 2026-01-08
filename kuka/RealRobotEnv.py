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
        self.teleop_gripper_state = 1
        self.teleop_gripper_flag = False

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

        # -----------------------------

        act, info = self._pre_step(act, info)

        self.gripper_state = act[3]
        self.robot.apply_action(act[0:3], self.gripper_state)

        # -----------------------------

        o = self.robot.observe()
        obs = self._obs_from_robot(o)

        # -----------------------------

        reward = self._get_reward(obs)

        reward, terminated, truncated, info = self._get_dones(reward, obs, info)

        # -----------------------------

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
            # message = np.append(message, self.teleop_gripper_state)

            print("Message: ", message)
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

        if hasattr(key, "char"):
            if key.char == 'o':
                self.teleop_gripper_state = 1
                self.teleop_gripper_flag = True
            
            if key.char == "c":
                self.teleop_gripper_state = -1
                self.teleop_gripper_flag = True
        else:
            if key == keyboard.Key.shift:
                self.done = True
        
    # ========================================================================================

    def stop(self):
        self.robot.emergency_stop()

    # ========================================================================================
    
    def _check_position(self, obs, xlim=[0.4, 0.8], ylim=[-0.2, 0.2], zlim=[0.32, 0.45]):
        pos = obs['state']["tcp_pos"]

        return pos[0] >= xlim[0] and pos[0] <= xlim[1] and \
                pos[1] >= ylim[0] and pos[1] <= ylim[1] and \
                pos[2] >= zlim[0] and pos[2] <= zlim[1]
    
    # ========================================================================================

    def _pre_step(self, action, info):

        if self.teleop_set:
            success, message = self._read_teleop()

            if success:
                if self.teleop_gripper_flag:
                    message[3] = self.teleop_gripper_state

                action = message[0:4]*self.action_scale
                info["intervene_action"] = action
            
            self.teleop_gripper_flag = False

        action = np.asarray(action, dtype=np.float32)/self.action_scale
        action[3] = action[3]*self.action_scale

        return action, info
    
    # ========================================================================================

    def _get_reward(self, obs):
        if self.reward_model is not None:
            img_dict = {k: obs["images"][k] for k in self.image_keys}
            score = self.reward_model(img_dict)
            reward = round(score)
        else:
            reward = 0.0

        return reward

    # ========================================================================================

    def _get_dones(self, reward, obs, info):

        terminated = bool(reward > 0.95)
        if terminated: info["succeed"] = True

        if self.done:
            terminated = True
            info["succeed"] = True
            self.done = False
        else:
            if not self._check_position(obs=obs):
                terminated = True
                reward = -3

        truncated = (self._t >= self._max_ep_steps)
        if truncated: reward = -1
        self.last_obs = obs
        self._t += 1

        return reward, terminated, truncated, info