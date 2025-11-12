# low_level_ctrl.py
from __future__ import annotations
import time, threading, math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable
import numpy as np

def minjerk_segment(p0, v0, pf, vf, T, t):
    """Мини-рывковый профиль pos(t) при граничных p,v на [0,T]. 1D версия, векторизуем."""
    if T <= 1e-6:  # избегаем деления
        return pf, vf, 0.0
    tau = np.clip(t / T, 0.0, 1.0)
    # классическая траектория 5-й степени с v-краями
    # обозначения для удобства
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float)
    pf = np.asarray(pf, float); vf = np.asarray(vf, float)
    # нормируем по T
    dp = pf - p0
    A = 6*dp - (4*vf + 2*v0)*T
    B = -15*dp + (8*vf + 7*v0)*T
    C = 10*dp - (6*vf + 4*v0)*T
    pos = p0 + (v0*T)*tau + A*(tau**3) + B*(tau**4) + C*(tau**5)
    vel = (v0 +
           (3*A*(tau**2) + 4*B*(tau**3) + 5*C*(tau**4)) * (1.0/T))
    acc = ((6*A*tau + 12*B*(tau**2) + 20*C*(tau**3)) * (1.0/(T**2)))
    return pos, vel, acc

# ------------------ основной контроллер -----------------------------

class LowLevelCtrl:
    def __init__(self, sdk):
        self.sdk = sdk

        # состояние
        self._q = np.zeros(6, float)
        self._dq = np.zeros(6, float)

        self._p = np.zeros(3, float)
        self._v = np.zeros(3, float)

        # цели текущего сегмента
        # self._seg_t0 = 0.0
        # self._seg_T = self.cfg.segment_T
        # self._p0 = np.zeros(3, float)
        # self._v0 = np.zeros(3, float)
        # self._p_goal = np.zeros(3, float)
        # self._v_goal = np.zeros(3, float)
        # self._q0 = np.array([0,0,0,1], float)
        # self._q_goal = np.array([0,0,0,1], float)

        self.max_lin = 0.1

        # управление потоком
        self._lock = threading.Lock()
        self._run_flag = False
        self._thread: Optional[threading.Thread] = None
        self._last_cmd_time = time.time()

    # ---------- публичный API (используется RobotAdapter) ------------

    def start(self):
        if self._run_flag: return
        self._run_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run_flag = False
        if self._thread: self._thread.join(timeout=2.0)

    def soft_stop(self):
        # плавное затухание скоростей/торков
        try:
            self.sdk.stop()  # TODO: integrate robot SDK
        except Exception:
            pass

    def update_target(self, a_ee6: np.ndarray, absolute: bool=False):
        """
        a_ee6: [dx, dy, dz, droll, dpitch, dyaw] в рамке ЭЭ или абсолютная цель (absolute=True)
        Ставит новый 100мс сегмент min-jerk (позиция) + SLERP (ориентация).
        """
        with self._lock:
            self._last_cmd_time = time.time()
            # текущие
            p_cur = self._p.copy()
            v_cur = self._v.copy()

            dp = a_ee6[:3]
            # drot_rpy = a_ee6[3:]
            # rate limit (ленточка) по скоростям
            # max_lin = self.cfg.limits.v_max_lin * self._seg_T
            # max_ang = self.cfg.limits.v_max_ang * self._seg_T
            dp = np.clip(dp, -self.max_lin, +self.max_lin)
            # drot_rpy = np.clip(drot_rpy, -max_ang, +max_ang)

            self._p = p_cur + dp

    # def set_gripper_cmd(self, cmd: int):
    #     """
    #     cmd: 0=open, 1=close, 2=stay
    #     """
    #     try:
    #         if cmd == 0: self.sdk.gripper_open()
    #         elif cmd == 1: self.sdk.gripper_close()
    #         else: pass
    #     except Exception: pass

    def get_joint_state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._q.copy(), self._dq.copy()

    def get_tcp_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._p.copy()

    def get_tcp_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._v.copy()

    # -------------------- внутренний цикл ----------------------------

    def _loop(self):
        dt = 1.0 / 10
        next_t = time.time()
        while self._run_flag:
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue
            next_t += dt

            # 1) прочитать текущее состояние из SDK
            self._read_state_from_sdk()

            # # 2) watchdog / аварии
            # if (time.time() - self._last_cmd_time) > self.cfg.watchdog_s:
            #     self._apply_soft_hold()
            #     continue
            # if self._ft_guard_tripped():
            #     self.soft_stop()
            #     continue

            # 3) сгенерировать микрошаг цели
            with self._lock:
                t_seg = time.time() - self._seg_t0
                xd_pos, xd_lin, _ = minjerk_segment(self._p0, self._v0,
                                                    self._p_goal, self._v_goal,
                                                    self._seg_T, t_seg)
                # ориентация — через SLERP: скорости оценим численно
                alpha = np.clip(t_seg / self._seg_T, 0.0, 1.0)
                xd_quat = quat_slerp(self._q0, self._q_goal, alpha)

            # 6) отправить команду
            self._send_position_to_sdk(tau)

    # ------------------------- SDK I/O --------------------------------

    def _read_state_from_sdk(self):
        """
        Читать q,dq, TCP, скорости, FT, грейпер из SDK.
        Здесь только шаблон — подставьте свои вызовы.
        """
        try:
            # TODO: integrate robot SDK reads
            # self._q, self._dq = self.sdk.get_joint_state()           # (n,), (n,)
            # self._p, self._q4 = self.kin.fk(self._q)                  # TCP pose
            # self._v, self._w  = self.sdk.get_tcp_velocity()          # (3,), (3,)
            # self._ft          = self.sdk.read_ft()                    # (6,)
            # self._gripper     = self.sdk.get_gripper_pos()
            pass
        except Exception:
            pass

    def _send_position_to_sdk(self, tau: np.ndarray):
        try:
            # TODO: integrate robot SDK writes
            # self.sdk.send_torque(tau)
            # или, если торки недоступны:
            # dq_ref = np.linalg.pinv(self.kin.J(self._q)) @ (self._v_ref_from_F() )
            # self.sdk.send_joint_velocity(qdot_clipped)
            pass
        except Exception:
            pass

    # ------------------------ вспомогательное --------------------------

    # @staticmethod
    # def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    #     roll, pitch, yaw = rpy
    #     cr = math.cos(roll/2); sr = math.sin(roll/2)
    #     cp = math.cos(pitch/2); sp = math.sin(pitch/2)
    #     cy = math.cos(yaw/2); sy = math.sin(yaw/2)
    #     # xyzw
    #     x = sr*cp*cy - cr*sp*cy
    #     y = cr*sp*sy + sr*cp*sy
    #     z = cr*cp*sy - sr*sp*cy
    #     w = cr*cp*cy + sr*sp*sy
    #     return quat_normalize(np.array([x,y,z,w], float))
