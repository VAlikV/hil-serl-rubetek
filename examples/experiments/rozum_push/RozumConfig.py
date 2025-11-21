from .RobotAdapter import RobotAdapter
from .Camera import Camera
from .RealRobotEnv import RealRobotEnv, FakeEnv
# from .Robot import GovnoBot
from API.controller import TaskSpaceJogController
from .VisualReward import VisualReward
import numpy as np

class RobotConfig:
    def __init__(self):
        # режимы: 'single-arm-fixed-gripper' | 'single-arm-learned-gripper' | ...
        self.setup_mode = 'single-arm-learned-gripper'
        self.image_keys = ['cam_front','cam_side']
        self.classifier_keys = ['cam_front','cam_side']
        self.encoder_type = 'resnet-pretrained'     # см. фабрики make_sac_pixel_agent*
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

    def get_environment(self, fake_env=False, save_video=False, classifier=True):

        # реальный робот

        if fake_env:
            env = FakeEnv(image_keys=self.image_keys)
        else:
            cameras = {"cam_front": Camera(2),"cam_side": Camera(4)}
            robot = TaskSpaceJogController(ip="10.10.10.10",
                                            rate_hz=100,
                                            velocity=1,
                                            acceleration=1,
                                            treshold_position=0.001,
                                            treshold_angel=1)
            adapter = RobotAdapter(robot=robot, cameras=cameras, image_keys=["cam_front","cam_side"])
            reward_model = None
            if classifier:
                # берём один sample_obs из env, чтобы создать classifier (или формируем руками)
                sample_obs = {
                    k: np.zeros((1, 360, 480, 3), np.uint8) for k in self.image_keys
                }
                reward_model = VisualReward(
                    ckpt_dir="classifier_ckpt",
                    sample_observations=sample_obs,
                    classifier_keys=self.image_keys,
                )
            
            env = RealRobotEnv(robot_adapter=adapter, image_keys=self.image_keys, teleop_set=True, teleop_ip="127.0.0.1", teleop_port=8081, reward_model=None, classifier_keys=self.image_keys)

        return env
