from rozum.RobotAdapter import RobotAdapter
from rozum.Camera import Camera
from rozum.RealRobotEnv import RealRobotEnv
from rozum.FakeRobotEnv import FakeRobotEnv
from API.controller import TaskSpaceJogController
from rozum.VisualReward import VisualReward
import numpy as np

from experiments.config import DefaultTrainingConfig

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

class RobotConfig(DefaultTrainingConfig):
    def __init__(self):
        self.setup_mode = 'single-arm-fixed-gripper'

        self.proprio_keys = ["joints_pos", "joints_vel", "tcp_pos", "tcp_vel"]
        self.image_keys = ['cam_front','cam_side']
        self.classifier_keys = ['cam_front','cam_side']

    def get_environment(self, fake_env=False, save_video=False, classifier=True):

        if not fake_env:
            cameras = {"cam_front": Camera(4),"cam_side": Camera(2)}
            robot = TaskSpaceJogController(ip="10.10.10.10",
                                            rate_hz=100,
                                            velocity=1,
                                            acceleration=1,
                                            treshold_position=0.001,
                                            treshold_angel=1)
            adapter = RobotAdapter(robot=robot, cameras=cameras, image_keys=["cam_front","cam_side"])

            reward_model = None

            if classifier:

                sample_obs = {
                    k: np.zeros((1, 360, 480, 3), np.uint8) for k in self.image_keys
                }

                reward_model = VisualReward(
                    ckpt_dir="/home/valikv/Desktop/Robotics/hil-serl-rubetek/classifier_ckpt",
                    sample_observations=sample_obs,
                    classifier_keys=self.image_keys,
                )
            
            env = RealRobotEnv(robot_adapter=adapter, 
                                image_keys=self.image_keys, 
                                fake_env=fake_env,
                                teleop_set=True, 
                                teleop_ip="127.0.0.1", 
                                teleop_port=8081, 
                                reward_model=reward_model, 
                                classifier_keys=self.image_keys)
            
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        else:
            env = RealRobotEnv(robot_adapter=None, 
                                image_keys=self.image_keys, 
                                fake_env=fake_env)
            
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        return env
