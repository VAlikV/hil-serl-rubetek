import os, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import logging
from scipy.signal import savgol_filter
import threading
import ast

from API.rc_api import RobotApi

class GovnoBot():
    def __init__(self, ip):
        
        # Constants
        self.X_MIN = -0.4
        self.X_MAX = 0.4
        self.Y_MIN = 0.4
        self.Y_MAX = 1.0
        self.Z_MIN = 0.0
        self.Z_MAX = 0.7

        self.WINDOW_SIZE = 10
        self.V_GAIN = 2.0
        self.W_GAIN = 0.4

        self.DEADZONE_VEL = 0.001     
        self.DEADZONE_ANG = 3
        self.UNITS = 'deg'
        self.ANG_SPEED = 20
        self.ANG_ACCEL = 30
        self.BLEND = 0.0005

        self.IP = ip

        self.robot = RobotApi(
            self.IP,
            enable_logger=True,
            log_std_level=logging.DEBUG,
            enable_logfile=False
        )
        self.robot.payload.set(mass=0, tcp_mass_center=(0, 0, 0))
        self.robot.motion.scale_setup.set(velocity=1, acceleration=1)
        self.robot.controller_state.set('run', await_sec=120)

        # ----------------------------------------------------------

        self.RZ = np.array([[0,-1,0],[1,0,0],[0,0,1]])

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.omg = np.zeros(3)
        self.R_transform = np.zeros((3,3))

        self.time_stamps = []
        self.pos_window = []
        self.rpy_window = []

        X_start, Y_start, Z_start, Roll_start, Pitch_start, Yaw_start = self.robot.motion.linear.get_actual_position(orientation_units=self.UNITS)
        self.last_point = np.array([X_start, Y_start, Z_start, Roll_start, Pitch_start, Yaw_start])

        # print('Запуск устройства')
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.sock.bind(("127.0.0.1", 8081))

        self.start = self.last_tick = self.last_send = time.time()

        Vx, Vy, Vz = 0, 0, 0

    # =======================================================================================
    
    def _normalize_angle_deg(self, angle):
        return (angle + 180) % 360 - 180
    
    # =======================================================================================

    def _point_update(self, last_point, Vx, Vy, Vz, wx, wy, wz, dt):
        # Обновление позиции
        pos = last_point[:3] + np.array([Vx, Vy, Vz]) * dt
        pos[0] = np.clip(pos[0], self.X_MIN, self.X_MAX)
        pos[1] = np.clip(pos[1], self.Y_MIN, self.Y_MAX)
        pos[2] = np.clip(pos[2], self.Z_MIN, self.Z_MAX)

        # Обновление ориентации с нормализацией углов
        rpy = last_point[3:] + np.array([wx, wy, wz]) * dt
        rpy = np.array([self._normalize_angle_deg(a) for a in rpy])

        return np.concatenate([pos, rpy])
    
    # =======================================================================================
    
    def _is_safety(self, point):
        X, Y, Z, _, _, _ = point
        return (self.X_MIN <= X <= self.X_MAX) and (self.Y_MIN <= Y <= self.Y_MAX) and (self.Z_MIN <= Z <= self.Z_MAX)
    
    # =======================================================================================
    
    def _get_euler_from_matrix(self, R):
        assert isinstance(R, np.ndarray) and R.shape == (3, 3)
        
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0

        return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)

    # =======================================================================================

    def _get_safety_rpy(self, R_transform, rpy_window):
        r, p, y = R.from_matrix(R_transform).as_euler('xyz', degrees=True)
        if len(rpy_window) == 0:
            return r,p,y
        rpy_window.append([r,p,y])
        rpy_ar = np.array(rpy_window)
        r_ar = rpy_ar[:,0]
        p_ar = rpy_ar[:,1]
        y_ar = rpy_ar[:,2]

        for ar in [r_ar, p_ar, y_ar]:
            delta = np.diff(ar)
            idx_plus = np.where(delta < -200)
            idx_minus = np.where(delta > 200)

            if idx_plus[0].shape[0] >0:
                for i in idx_plus[0]:
                    ar[i+1:] += 360 
            if idx_minus[0].shape[0] >0:
                for j in idx_minus[0]:
                    ar[j+1:] -= 360

        return r_ar[-1], p_ar[-1], y_ar[-1]

    # =======================================================================================

    def _calculate_velocity(self, pos, R_transform, t, pos_window, rpy_window, time_stamps):

        r, p, y = self._get_safety_rpy(R_transform, rpy_window.copy())

        vx, vy, vz = 0.0, 0.0, 0.0
        wx, wy, wz = 0.0, 0.0, 0.0

        if len(time_stamps) < self.WINDOW_SIZE:
            pos_window.append(pos.copy())
            rpy_window.append([r, p, y])
            time_stamps.append(t)

        if len(time_stamps) == self.WINDOW_SIZE:
            pos_ar = np.array(pos_window)
            rpy_ar = np.array(rpy_window)
            time_ar = np.array(time_stamps)

            pos_window = pos_window[1:]
            rpy_window = rpy_window[1:]
            time_stamps = time_stamps[1:]

            dt = time_ar - time_ar[0]

            vx = np.polyfit(dt, pos_ar[:, 0], 1)[0]
            vy = np.polyfit(dt, pos_ar[:, 1], 1)[0]
            vz = np.polyfit(dt, pos_ar[:, 2], 1)[0]

            wx = np.polyfit(dt, rpy_ar[:, 0], 1)[0]
            wy = np.polyfit(dt, rpy_ar[:, 1], 1)[0]
            wz = np.polyfit(dt, rpy_ar[:, 2], 1)[0]

        if len(time_stamps) > self.WINDOW_SIZE:
            pos_window = []
            rpy_window = []
            time_stamps = []

        return vx, vy, vz, wx, wy, wz, pos_window.copy(), rpy_window.copy(), time_stamps.copy()
    
    # =======================================================================================
    
    def _move_robot(self, new_point_joints):
        if new_point_joints is not None:
            self.robot.motion.joint.add_new_waypoint(angle_pose=new_point_joints,
                                            speed=self.ANG_SPEED,
                                            accel=self.ANG_ACCEL,
                                            blend=self.BLEND,
                                            units='deg')
            self.robot.motion.mode.set('move')

    # =======================================================================================

    def step(self):

        self.pos = self.RZ @ self.pos

        t = time.time() - self.start
        dt = t - self.last_tick
        self.last_tick = t

        self.vel[0], self.vel[1], self.vel[2], self.omg[0], self.omg[1], self.omg[2], self.pos_window, self.rpy_window, self.time_stamps = self._calculate_velocity(self.pos.copy(), 
                                                                                                                                                                    self.R_transform.copy(), 
                                                                                                                                                                    t, 
                                                                                                                                                                    self.pos_window.copy(), 
                                                                                                                                                                    self.rpy_window.copy(), 
                                                                                                                                                                    self.time_stamps.copy())
                                                                                        
        roll, pitch, yaw = R.from_matrix(self.R_transform).as_euler('xyz', degrees=True)
        print(f'Roll: {roll:.2f},\tPitch: {pitch:.2f},\tYaw: {yaw:.2f}')

        self.vel[np.abs(self.vel) < self.DEADZONE_VEL] = 0.0
        self.omg[np.abs(self.omg) < self.DEADZONE_ANG] = 0.0
        
        Vx, Vy, Vz = self.vel[0]*self.V_GAIN, self.vel[1]*self.V_GAIN, self.vel[2]*self.V_GAIN
        wx, wy, wz = self.omg[0]*self.W_GAIN, self.omg[1]*self.W_GAIN, self.omg[2]*self.W_GAIN
        new_point = self._point_update(self.last_point, Vx, Vy, Vz, wx, wy, wz, dt=dt)
    
        new_point_joints = self.robot.motion.kinematics.get_inverse(tcp_pose=(new_point[0], 
                                                                              new_point[1], 
                                                                              new_point[2], 
                                                                              new_point[3], 
                                                                              new_point[4], 
                                                                              new_point[5]), orientation_units='deg', get_all=False)

        if (wx or wy or wz or Vx or Vy or Vz) and self._is_safety(new_point) and (time.time() - self.last_send > 0.11):
            # await asyncio.to_thread(self._move_robot, new_point_joints)
            # print('delta',time.time() - last_send)
            # print("AAAAAAAAAAAAAAAAAAAAAA")
            thread = threading.Thread(target=self._move_robot, args=(new_point_joints,))
            thread.start()
            self.last_send = time.time()
            self.last_point = new_point

    # =======================================================================================

    def set_new_point(self, position, orientation):
        # data, addr = self.sock.recvfrom(1024)
        # line = data.decode()
        # numbers = ast.literal_eval(line)
        self.pos[0], self.pos[1], self.pos[2] = position[0], position[1], position[2]
        self.R_transform[0,0], self.R_transform[0,1], self.R_transform[0,2] = orientation[0], orientation[1], orientation[2]
        self.R_transform[1,0], self.R_transform[1,1], self.R_transform[1,2] = orientation[3], orientation[4], orientation[5]
        self.R_transform[2,0], self.R_transform[2,1], self.R_transform[2,2] = orientation[6], orientation[7], orientation[8]

    # =======================================================================================

    def get_robot_attr(self):
        rtd = self.robot._rtd_receiver.rt_data
        q = np.expand_dims(np.array(rtd.act_q), axis=0)
        dq = np.expand_dims(np.array(rtd.act_qd), axis=0)
        tool_tcp = np.expand_dims(np.array(rtd.act_tcp_x), axis=0)
        tool_tcp_vel = np.expand_dims(np.array(rtd.act_tcp_xd), axis=0)
        return q, dq, tool_tcp, tool_tcp_vel
    
    # =======================================================================================

    def stop(self):
        self.robot.motion.mode.set('hold')
    
# ===================================================================================================================================================
# ===================================================================================================================================================
# ===================================================================================================================================================