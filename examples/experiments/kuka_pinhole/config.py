from kuka.RobotAdapter import RobotAdapter
from kuka.Camera import Camera
from kuka.RealRobotEnv import RealRobotEnv
from kuka.RealRobotEnv import GripperPenaltyWrapper
from kuka.VisualReward import VisualReward
import numpy as np
import time

from experiments.config import DefaultTrainingConfig

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

class KukaPinHoleConfig(DefaultTrainingConfig):
    def __init__(self):
        self.setup_mode = 'single-arm-learned-gripper'

        self.proprio_keys = ["tcp_pos", "tcp_vel", "gripper_state"]
        self.image_keys = ['cam_front','cam_side']
        self.classifier_keys = ['cam_front','cam_side']

    def get_environment(self, fake_env=False, save_video=False, classifier=True):

        if not fake_env:

            reward_model = None

            if classifier:

                sample_obs = {
                    k: np.zeros((1, 128, 128, 3), np.uint8) for k in self.image_keys
                }

                reward_model = VisualReward(
                    ckpt_dir="/home/valikv/Desktop/Robotics/hil-serl-rubetek/classifier_ckpt",
                    sample_observations=sample_obs,
                    classifier_keys=self.classifier_keys,
                )

            cameras = {"cam_front": Camera(2),"cam_side": Camera(4)}
            
            time.sleep(3)
            
            adapter = RobotAdapter(own_ip="127.0.0.1", own_port=8082, 
                                   robot_ip="127.0.0.1", robot_port=8081, 
                                   cameras=cameras, image_keys=["cam_front", "cam_side"])
            
            env = RealRobotEnv(robot_adapter=adapter, 
                               image_keys=["cam_front", "cam_side"], 
                               teleop_set=True, 
                               teleop_ip="127.0.0.1", teleop_port=8083,
                               reward_model=reward_model,
                               classifier_keys=self.image_keys)

            
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
            env = GripperPenaltyWrapper(env)

        else:
            env = RealRobotEnv(robot_adapter=None, 
                                image_keys=self.image_keys, 
                                fake_env=fake_env)
            
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
            env = GripperPenaltyWrapper(env)

        return env
