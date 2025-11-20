from examples.experiments.rozum_push.RobotAdapter import RobotAdapter
from examples.experiments.rozum_push.Camera import Camera
from examples.experiments.rozum_push.RealRobotEnv import RealRobotEnv
from examples.experiments.rozum_push.Robot import GovnoBot
from API.controller import TaskSpaceJogController

import numpy as np
import socket
import time
import cv2

cameras = {"cam_front": Camera(2),"cam_side": Camera(4)}
robot = TaskSpaceJogController(ip="10.10.10.10",
                                        rate_hz=100,
                                        velocity=1,
                                        acceleration=1,
                                        treshold_position=0.001,
                                        treshold_angel=1)

adapter = RobotAdapter(robot=robot, cameras=cameras, image_keys=["cam_front","cam_side"])
env = RealRobotEnv(robot_adapter=adapter, image_keys=["cam_front","cam_side"], teleop_set=True)

delta_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

last_obs, info = env.reset()

i = 0
T = 1000

while True:

    # delta_pos[0] = np.sign(np.sin(2*np.pi*i/T))/1000
    # t = time.time()
    obs, reward, terminated, truncated, info = env.step(delta_pos)
    # print((time.time()-t)*1000)
    # delta_pos = np.array([0.0, 0.0, 0.0])

    # print(obs)

    image_1 = obs["cam_front"]
    image_2 = obs["cam_side"]

    cv2.imshow("cam_front", image_1)
    cv2.imshow("cam_side", image_2)

    if terminated:
        env.reset()

    i += 1

    # time.sleep(0.01)
    cv2.waitKey(1) 

last_obs, info = env.reset()

print(last_obs)

while True:
    time.sleep(0.01)




