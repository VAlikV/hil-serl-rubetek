from types import SimpleNamespace
from .RealRobotEnv import RealRobotEnv
from .RobotAdapter import RobotAdapter
from .LowLevelCtrl import LowLevelCtrl   # твой модуль
from .Camera import make_cameras     # словарь камер

class MyRobotConfig:
    def __init__(self):
        # режимы: 'single-arm-fixed-gripper' | 'single-arm-learned-gripper' | ...
        self.setup_mode = 'single-arm-learned-gripper'
        self.image_keys = ['cam_front','cam_side']
        self.encoder_type = 'resnet10'     # см. фабрики make_sac_pixel_agent*
        self.discount = 0.99

        # буфера/ритмы
        self.replay_buffer_capacity = 200_000
        self.training_starts = 5_000
        self.random_steps = 2_000
        self.max_steps = 300_000
        self.batch_size = 256
        self.cta_ratio = 4
        self.steps_per_update = 1000
        self.log_period = 500
        self.checkpoint_period = 10_000
        self.buffer_period = 5_000

    def get_environment(self, save_video=False, classifier=True):
        # fake_env=True при запуске learner, чтобы он не трогал реальное железо
        # if fake_env:
        #     # можно вернуть «обёртку-заглушку», которая только валидирует shape’ы
        #     return DummyRealRobotLikeEnv(self.image_keys)
        # реальный робот
        llc = LowLevelCtrl(...)                     # твой НУ-контроллер
        cams = make_cameras(self.image_keys, ...)   # {"cam_front": Cam(), ...}
        robot = RobotAdapter(llc, cams, self.image_keys)
        return RealRobotEnv(robot, self.image_keys)
