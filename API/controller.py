import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from API.rc_api import RobotApi

import time

def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

class JointSpaceJogController:
    def __init__(self, ip="10.10.10.10", rate_hz=100, velocity=1, acceleration=1, treshold=0.5):
        """
        :param ip: IP робота
        :param rate_hz: частота управления (Гц)
        :param velocity: скорость движения к точке [0.0 - 1.0]
        :param acceleration: ускорение при движении к точке [0.0 - 1.0]
        :param treshold: точность прибложения к целевой точке в градусах
        """

        self.robot = RobotApi(
            ip,
            enable_logger=True,
            # log_std_level=logging.DEBUG,
            enable_logfile=False,
            show_std_traceback=True
        )

        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz

        self.robot.payload.set(mass=0, tcp_mass_center=(0, 0, 0))
        self.robot.motion.scale_setup.set(velocity=velocity, acceleration=acceleration)
        self.robot.controller_state.set('run', await_sec=120)

        self.target_q = None  # целевая конфигурация в joint_space
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.treshold = treshold * np.pi/180      

    # ====================================================================================================

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.start()

    # ====================================================================================================

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    # ====================================================================================================

    def set_target_joint(self, target_q):
        """
        :param target_q: целевые углы в джоинтах [q1, q2, q3, ..., q6] в радианах
        """
        with self.lock:
            self.target_q = np.array(target_q)

    # ====================================================================================================

    def set_target(self, target_pos, target_rot):
        """
        :param target_pos: координаты целевой точки [x, y, z] в метрах
        :param target_rot: матрица поворота для целевой ориентации [[...], [...], [...]]
        """

        roll, pitch, yaw = R.from_matrix(target_rot).as_euler('xyz', degrees=False)

        target_q = self.robot.motion.kinematics.get_inverse(tcp_pose=(target_pos[0], 
                                                                    target_pos[1], 
                                                                    target_pos[2], 
                                                                    roll, 
                                                                    pitch, 
                                                                    yaw), orientation_units='rad', get_all=False)
        with self.lock:
            self.target_q = np.array(target_q)

    # ====================================================================================================

    def get_current_joint(self):
        '''
        Текущие значения углов в джоинтах [q1 ... q6]
        '''
        rtd = self.robot._rtd_receiver.rt_data
        q = np.array(rtd.act_q)

        return q
    
    # ====================================================================================================
    
    def get_current_joint_vel(self):
        '''
        Текущие значения скоростей в джоинтах [dq1 ... dq6]
        '''
        rtd = self.robot._rtd_receiver.rt_data
        dq = np.array(rtd.act_qd)

        return dq
    
    # ====================================================================================================
    
    def get_current_tcp(self):
        '''
        Текущие координаты эндефектора [x, y, z, r, p, y]. Углы в радианах
        '''
        rtd = self.robot._rtd_receiver.rt_data
        tcp = np.array(rtd.act_tcp_x)

        return tcp
    
    # ====================================================================================================
    
    def get_current_tcp_vel(self):
        '''
        Текущие скорости эндефектора [dx, dy, dz, dr, dp, dy]. Угловые скорости в рад/с (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data
        tcp_vel = np.array(rtd.act_tcp_xd)

        return tcp_vel
    
    # ====================================================================================================
    
    def get_current_torque(self):
        '''
        Текущие значения с тензодатчиков [t1, t2, ... t3]. Н*м (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data 
        act_trq = np.array(rtd.act_trq)

        return act_trq

    # ====================================================================================================

    def get_current_info(self):
        '''
        Текущие значения углов в джоинтах [q1 ... q6]
        значения скоростей в джоинтах [dq1 ... dq6]
        координаты эндефектора [x, y, z, r, p, y]. Углы в радианах
        скорости эндефектора [dx, dy, dz, dr, dp, dy]. Угловые скорости в рад/с (?)
        значения с тензодатчиков [t1, t2, ... t3]. Н*м (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data
        q = np.array(rtd.act_q)
        dq = np.array(rtd.act_qd)
        tcp = np.array(rtd.act_tcp_x)
        tcp_vel = np.array(rtd.act_tcp_xd)
        act_trq = np.array(rtd.act_trq)

        return q, dq, tcp, tcp_vel, act_trq

    # ====================================================================================================
        
    def _run_loop(self):
        while self.running:
            start_time = time.time()

            with self.lock:
                target_q = self.target_q

            if target_q is not None:
                current_q = self.get_current_joint()
                error = target_q - current_q
                dir = self._compute_jog_step(error)
                self.robot.motion.joint.jog_once_all_joints(dir)

            elapsed = time.time() - start_time
            time_to_sleep = self.dt - elapsed
            if time_to_sleep > 0:
                precise_sleep(dt=time_to_sleep)

    # ====================================================================================================

    def _compute_jog_step(self, error):

        error = np.sign(np.where((np.abs(error)<self.treshold), 0, error))
        dir = ['+' if i > 0 else '-' if i < 0 else '0' for i in error]
        
        return dir
    
# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

class TaskSpaceJogController:
    def __init__(self, ip="10.10.10.10", rate_hz=100, velocity=1, acceleration=1, treshold_position=0.001, treshold_angel=1):
        """
        :param ip: IP робота
        :param rate_hz: частота управления (Гц)
        :param velocity: скорость движения к точке [0.0 - 1.0]
        :param acceleration: ускорение при движении к точке [0.0 - 1.0]
        :param treshold_position: точность приближения к целевой точке в метрах
        :param treshold_angel: точность прибложения к целевой ориентации в градусах
        """

        self.robot = RobotApi(
            ip,
            enable_logger=False,
            # log_std_level=logging.DEBUG,
            enable_logfile=False,
            show_std_traceback=True
        )

        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz

        self.robot.payload.set(mass=0, tcp_mass_center=(0, 0, 0))
        self.robot.motion.scale_setup.set(velocity=velocity, acceleration=acceleration)
        self.robot.controller_state.set('run', await_sec=120)

        self.target_pose = self.get_current_tcp()  # целевая конфигурация в joint_space
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.treshold_pos = treshold_position
        self.treshold_angel = treshold_angel * np.pi/180

    # ====================================================================================================

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.start()

    # ====================================================================================================

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    # ====================================================================================================

    def set_target(self, target_pos, target_rot):
        """
        :param target_pos: координаты целевой точки [x, y, z] в метрах
        :param target_rot: матрица поворота для целевой ориентации [[...], [...], [...]]
        """

        roll, pitch, yaw = R.from_matrix(target_rot).as_euler('xyz', degrees=False)

        target_pose = np.array([target_pos[0], target_pos[1], target_pos[2], roll, pitch, yaw])

        with self.lock:
            self.target_pose = target_pose
        
    # ====================================================================================================

    def get_current_joint(self):
        '''
        Текущие значения углов в джоинтах [q1 ... q6]
        '''
        rtd = self.robot._rtd_receiver.rt_data
        q = np.array(rtd.act_q)

        return q
    
    # ====================================================================================================
    
    def get_current_joint_vel(self):
        '''
        Текущие значения скоростей в джоинтах [dq1 ... dq6]
        '''
        rtd = self.robot._rtd_receiver.rt_data
        dq = np.array(rtd.act_qd)

        return dq
    
    # ====================================================================================================
    
    def get_current_tcp(self):
        '''
        Текущие координаты эндефектора [x, y, z, r, p, y]. Углы в радианах
        '''
        rtd = self.robot._rtd_receiver.rt_data
        tcp = np.array(rtd.act_tcp_x)

        return tcp
    
    # ====================================================================================================
    
    def get_current_tcp_vel(self):
        '''
        Текущие скорости эндефектора [dx, dy, dz, dr, dp, dy]. Угловые скорости в рад/с (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data
        tcp_vel = np.array(rtd.act_tcp_xd)

        return tcp_vel
    
    # ====================================================================================================
    
    def get_current_torque(self):
        '''
        Текущие значения с тензодатчиков [t1, t2, ... t3]. Н*м (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data 
        act_trq = np.array(rtd.act_trq)

        return act_trq

    # ====================================================================================================

    def get_current_info(self):
        '''
        Текущие значения углов в джоинтах [q1 ... q6]
        значения скоростей в джоинтах [dq1 ... dq6]
        координаты эндефектора [x, y, z, r, p, y]. Углы в радианах
        скорости эндефектора [dx, dy, dz, dr, dp, dy]. Угловые скорости в рад/с (?)
        значения с тензодатчиков [t1, t2, ... t3]. Н*м (?)
        '''
        rtd = self.robot._rtd_receiver.rt_data
        q = np.array(rtd.act_q)
        dq = np.array(rtd.act_qd)
        tcp = np.array(rtd.act_tcp_x)
        tcp_vel = np.array(rtd.act_tcp_xd)
        act_trq = np.array(rtd.act_trq)

        return q, dq, tcp, tcp_vel, act_trq
    
    # ====================================================================================================
        
    def _run_loop(self):
        while self.running:
            start_time = time.time()

            with self.lock:
                target_pose = self.target_pose

            # print(target_q)

            if target_pose is not None:
                current_pose = self.get_current_tcp()
                error = target_pose - current_pose
                dir = self._compute_jog_step(error)
                self.robot.motion.linear.jog_once_all_axis(dir)

            elapsed = time.time() - start_time
            time_to_sleep = self.dt - elapsed
            if time_to_sleep > 0:
                precise_sleep(dt=time_to_sleep)

    # ====================================================================================================

    def _compute_jog_step(self, error):

        error_pos = np.sign(np.where((np.abs(error[0:3])<self.treshold_pos), 0, error[0:3]))
        error_angle = np.sign(np.where((np.abs(error[3:6])<self.treshold_angel), 0, error[3:6]))

        dir = ['+' if i > 0 else '-' if i < 0 else '0' for i in error_pos] + ['0']*3 # ['+' if i > 0 else '-' if i < 0 else '0' for i in error_angle]
        
        return dir