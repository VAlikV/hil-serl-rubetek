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
            listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            listener.start()

        self.pressed = set()

        self.done = False
        self.gripper_state = 1
        self.teleop_gripper_state = 1
        self.teleop_gripper_flag = False

        self.action_scale = 100
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*7, dtype=np.float32),
                                           high=np.array([+1.0]*7, dtype=np.float32),
                                           dtype=np.float32)
        
        self.obs_scale = 100.0
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
        
        proprio = {"tcp_pos": o.tcp_pos*self.obs_scale,
                   "tcp_vel": o.tcp_vel*self.obs_scale,
                   "gripper_state":self.gripper_state}

        imgs = {k: o.images[k] for k in self.image_keys}

        return copy.deepcopy({'state': proprio, "images": imgs})
    
    # ========================================================================================

    def reset(self, *, seed=None, options=None):

        zero = np.array([0,0,0])

        self.robot.apply_action(zero, 1)

        time.sleep(0.5)

        self.robot.reset()

        o = self.robot.observe()
        self.last_obs = self._obs_from_robot(o)
        self._t = 0
        info = {}
        
        return self.last_obs, info
    
    # ========================================================================================

    def step(self, action):

        self._check_keyboard()

        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        act = action.copy()

        info = {}

        # -----------------------------

        act, info = self._pre_step(act, info)

        self.gripper_state = act[6]
        self.robot.apply_action(act[0:3], self.gripper_state)

        # -----------------------------

        o = self.robot.observe()
        obs = self._obs_from_robot(o)

        # -----------------------------

        reward = self._get_reward(obs)
        reward, terminated, truncated, info = self._get_dones(reward, obs, info)

        # -----------------------------

        # print("Action: ", act[0:4])
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

            print("Teleop")
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
            self.pressed.add(key.char)
        else:
            self.pressed.add(key)
        
    # ========================================================================================
    
    def _check_keyboard(self):

        if keyboard.Key.shift in self.pressed:
            self.done = True

    # ========================================================================================

    def _on_release(self, key):

        if hasattr(key, "char"):
            self.pressed.discard(key.char)
        else:
            self.pressed.discard(key)

    # ========================================================================================

    def stop(self):
        self.robot.emergency_stop()

    # ========================================================================================
    
    def _check_position(self, obs, xlim=[-0.1, 0.2], ylim=[-0.2, 0.2], zlim=[-0.08, 0.05]):
        pos = obs['state']["tcp_pos"]/self.obs_scale

        return pos[0] >= xlim[0] and pos[0] <= xlim[1] and \
                pos[1] >= ylim[0] and pos[1] <= ylim[1] and \
                pos[2] >= zlim[0] and pos[2] <= zlim[1]
    
    # ========================================================================================

    def _pre_step(self, action, info):

        if self.teleop_set:
            success, message = self._read_teleop()

            if success:
                message[3] = self.gripper_state
                if self.teleop_gripper_flag:
                    message[3] = self.teleop_gripper_state

                # message[3:6] = 0.0
                action = message[0:2]*self.action_scale
                action = np.concatenate((message[0:2]*self.action_scale, np.zeros(3), message[3]))
                info["intervene_action"] = action
            
            self.teleop_gripper_flag = False

        # action = np.asarray(action, dtype=np.float32)
        action[0:6] = action[0:6]/self.action_scale

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
                # reward = -3

        truncated = (self._t >= self._max_ep_steps)
        # if truncated: reward = -1
        self.last_obs = obs
        self._t += 1

        return reward, terminated, truncated, info
    

# ========================================================================================
# ========================================================================================
# ========================================================================================


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        print(obs["state"])
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < 0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info