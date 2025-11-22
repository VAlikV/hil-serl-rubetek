import gymnasium as gym
import numpy as np
import socket
import time
from pynput import keyboard

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
    ):
        self.robot = robot_adapter
        self.image_keys = image_keys

        self.reward_model = reward_model

        if enable_keyboard_listener:
            listener = keyboard.Listener(on_press=self._on_press,)
            listener.start()

        self.done = False

        # ---- определите пространства наблюдений/действий:
        H, W = 360, 480  # пример; подгоните под свой препроцессинг
        obs_space = {
          "state": gym.spaces.Box(
                        -np.inf, np.inf, 
                        shape=(6 + 6 + 6 + 6, ), 
                        dtype=np.float32),
                        
                    **{k: gym.spaces.Box(
                        0, 255,
                        shape=(H, W, 3), 
                        dtype=np.uint8) for k in self.image_keys},
        }
        self.observation_space = gym.spaces.Dict(obs_space)

        # Действия: 6D ΔЭЭ + дискретный хват через «hybrid»-режим (см. ниже)
        self.action_space = gym.spaces.Box(low=np.array([-2.0]*2, dtype=np.float32),
                                           high=np.array([+2.0]*2, dtype=np.float32),
                                           dtype=np.float32)

        self.last_obs = None
        self._t = 0
        self._max_ep_steps = 500

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

    def _obs_from_robot(self, o):

        proprio = np.concatenate([
            o.q, o.dq, o.tcp_pos, o.tcp_vel,
        ]).astype(np.float32)
        imgs = {k: o.images[k] for k in self.image_keys}
        return {'state': proprio, **imgs}

    def reset(self, *, seed=None, options=None):
        # сделайте reset сцены/объектов/позиции, при необходимости
        self.robot.reset()

        o = self.robot.observe()
        self.last_obs = self._obs_from_robot(o)
        self._t = 0
        info = {}
        
        return self.last_obs, info

    def step(self, action):
        # action: np.array(6,), а хват — через info["intervene_action"]/внешний канал (см. ниже режимы)
        # Действие - смещение в декартовой системе (первые 3 учитываются, остальные игнорируются)

        act = action.copy()

        info = {}

        if self.teleop_set:
            success, message = self._read_teleop()

            if success:
                # print("SOSI")
                act = message[0:2]
                info["intervene_action"] = act

        a = np.asarray(act, dtype=np.float32)
        a_gripper = 0  # 0=open, 1=close, 2=stay (если fixed-gripper — просто игнорим)

        # t = time.time()
        self.robot.apply_action(a, a_gripper)
        # print((time.time()-t)*1000)

        o = self.robot.observe()
        obs = self._obs_from_robot(o)

        if self.reward_model is not None:
            img_dict = {k: o.images[k] for k in self.image_keys}
            score = self.reward_model(img_dict)    # [0..1]
            reward = float(score)                  # или 2*score-1, или (score>0.9)*1.0
            terminated = bool(score > 0.95)        # например, эпизод успешен
        else:
            reward = 0.0
            terminated = False

        if self.done:
            terminated = True
            info["succeed"] = True
            self.done = False

        truncated = (self._t >= self._max_ep_steps)
        self.last_obs = obs
        self._t += 1

        # вознаграждение: часто даётся внешним визуал-классификатором; тут — заглушка
        
        return obs, reward, terminated, truncated, info
    
    def _read_teleop(self):

        delta_pos = np.array([])

        success = False

        try:
            data, addr = self.haptic_sock.recvfrom(1024)
            message = np.array(list(map(float, data.decode()[1:-1].split(","))))
            if len(message):
                if self.first:
                    self.last_pos = message
                    self.first = False
                else:
                    delta_pos = message - self.last_pos
                    self.last_pos = message
                    success = True
                    delta_pos[2] = 0.0

        except socket.timeout:
            data, addr = None, None  # или просто continue

        return success, delta_pos

    def _on_press(self, key):
        if key == keyboard.Key.shift:
            self.done = True
            # print("Shift is currently pressed")

    def stop(self):
        self.robot.emergency_stop()
