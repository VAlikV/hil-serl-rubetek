from kuka.RobotAdapter import RobotAdapter
from kuka.Camera import Camera
from kuka.RealRobotEnv import RealRobotEnv

import numpy as np
import socket
import time
import cv2

cameras = {"cam_front": Camera(2),"cam_side": Camera(4)}
adapter = RobotAdapter(own_ip="127.0.0.1", own_port=8084, robot_ip="127.0.0.1", robot_port=8080, cameras=cameras, image_keys=["cam_front","cam_side"])

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

    image_1 = obs["images"]["cam_front"]
    image_2 = obs["images"]["cam_side"]

    # image_1 = cv2.resize(image_1, (118, 118))
    # image_2 = cv2.resize(image_2, (118, 118))

    cv2.imshow("cam_front", image_1)
    cv2.imshow("cam_side", image_2)

    if terminated:
        env.reset()

    print(reward)
    print(obs["state"]["tcp_pos"])

    i += 1

    # time.sleep(0.01)
    cv2.waitKey(1) 

last_obs, info = env.reset()

print(last_obs)

while True:
    time.sleep(0.01)




