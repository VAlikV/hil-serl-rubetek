import gymnasium as gym
import numpy as np

class RealRobotEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self, robot_adapter, image_keys=("cam_front","cam_side")):
        self.robot = robot_adapter
        self.image_keys = image_keys

        # ---- определите пространства наблюдений/действий:
        H, W = 480, 640  # пример; подгоните под свой препроцессинг
        obs_space = {
          "proprio": gym.spaces.Box(
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
        self.action_space = gym.spaces.Box(low=np.array([-2.0]*6, dtype=np.float32),
                                           high=np.array([+2.0]*6, dtype=np.float32),
                                           dtype=np.float32)

        self.last_obs = None
        self._t = 0
        self._max_ep_steps = 500

    def _obs_from_robot(self, o):

        proprio = np.concatenate([
            o.q, o.dq, o.tcp_pos, o.tcp_vel,
        ]).astype(np.float32)
        imgs = {k: o.images[k] for k in self.image_keys}
        return {"proprio": proprio, **imgs}

    def reset(self, *, seed=None, options=None):
        # сделайте reset сцены/объектов/позиции, при необходимости
        o = self.robot.observe()
        self.last_obs = self._obs_from_robot(o)
        self._t = 0
        info = {}
        return self.last_obs, info

    def step(self, action):
        # action: np.array(6,), а хват — через info["intervene_action"]/внешний канал (см. ниже режимы)
        a = np.asarray(action, dtype=np.float32)
        a_gripper = 0  # 0=open, 1=close, 2=stay (если fixed-gripper — просто игнорим)
        self.robot.apply_action(a, a_gripper)

        o = self.robot.observe()
        obs = self._obs_from_robot(o)

        # вознаграждение: часто даётся внешним визуал-классификатором; тут — заглушка
        reward = 0.0
        terminated = False
        truncated = (self._t >= self._max_ep_steps)
        info = {}
        self.last_obs = obs
        self._t += 1
        return obs, reward, terminated, truncated, info
