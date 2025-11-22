import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from examples.experiments.rozum_push.RealRobotEnv import RealRobotEnv


@dataclass
class FakeObs:
    images: Dict[str, np.ndarray]
    q: np.ndarray
    dq: np.ndarray
    tcp_pos: np.ndarray
    tcp_vel: np.ndarray
    timestamp: float


class FakeRobotAdapter:
    def __init__(
        self,
        cameras: Dict[str, object] | None = None,
        image_keys: Iterable[str] = ("cam_front", "cam_side"),
        noise_std: float = 5e-4,
        seed: int | None = None,
        image_shape: Tuple[int, int, int] = (360, 480, 3),
    ):
        self.cams = cameras or {}
        self.image_keys = tuple(image_keys)
        self.noise_std = noise_std
        self.image_shape = image_shape
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.q = np.zeros(6, dtype=np.float32)
        self.dq = np.zeros(6, dtype=np.float32)
        self.tcp_pos = np.zeros(6, dtype=np.float32)
        self.tcp_vel = np.zeros(6, dtype=np.float32)
        self._last_t = time.time()

    def observe(self) -> FakeObs:
        now = time.time()
        dt = max(now - self._last_t, 1e-3)
        self._last_t = now

        # simple fake dynamics: current vel decays a bit over time
        self.tcp_vel *= np.exp(-dt)
        self.tcp_pos += self.tcp_vel * dt

        imgs = {}
        for k in self.image_keys:
            frame = None
            cam = self.cams.get(k)
            if cam is not None:
                frame = cam.get_image()
            if frame is None:
                frame = self._synth_image()
            imgs[k] = frame

        return FakeObs(
            images=imgs,
            q=self.q.copy(),
            dq=self.dq.copy(),
            tcp_pos=self.tcp_pos.copy(),
            tcp_vel=self.tcp_vel.copy(),
            timestamp=now,
        )

    def apply_action(self, delta, a_gripper):
        delta = np.asarray(delta, dtype=np.float32)
        padded = np.zeros(6, dtype=np.float32)
        padded[: min(len(delta), 2)] = delta[: min(len(delta), 2)]

        self.tcp_vel[:2] = padded[:2]
        self.tcp_pos[:2] = np.clip(self.tcp_pos[:2] + padded[:2], -1.0, 1.0)

        noise = self.rng.normal(scale=self.noise_std, size=self.q.shape).astype(np.float32)
        self.q += noise
        self.dq = self.tcp_vel.copy()

    def emergency_stop(self, reason: str = ""):
        return

    def _synth_image(self) -> np.ndarray:
        return self.rng.integers(0, 255, size=self.image_shape, dtype=np.uint8)


class FakeRobotEnv(RealRobotEnv):
    def __init__(
        self,
        cameras: Dict[str, object] | None = None,
        image_keys: Iterable[str] = ("cam_front", "cam_side"),
        teleop_set: bool = False,
        reward_model=None,
        classifier_keys=None,
        seed: int | None = None,
    ):
        adapter = FakeRobotAdapter(cameras=cameras, image_keys=image_keys, seed=seed)
        super().__init__(
            robot_adapter=adapter,
            image_keys=image_keys,
            teleop_set=teleop_set,
            reward_model=reward_model,
            classifier_keys=classifier_keys or [],
        )
