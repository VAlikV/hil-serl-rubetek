from RobotAdapter import RobotAdapter
from Camera import Camera
from RealRobotEnv import RealRobotEnv
from Robot import GovnoBot

import numpy as np
import socket
import time

cameras = {"cam_front": Camera(0),"cam_side": Camera(2)}
robot = GovnoBot("10.10.10.10")
adapter = RobotAdapter(robot=robot, cameras=cameras, image_keys=["cam_front","cam_side"])
env = RealRobotEnv(robot_adapter=adapter, image_keys=["cam_front","cam_side"], teleop_set=True)

delta_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

last_obs, info = env.reset()

i = 0
T = 1000

while True:

    delta_pos[0] = np.sign(np.sin(2*np.pi*i/T))/1000
    # t = time.time()
    obs, reward, terminated, truncated, info = env.step(delta_pos)
    # print((time.time()-t)*1000)
    # delta_pos = np.array([0.0, 0.0, 0.0])

    # print(obs)

    i += 1

    time.sleep(0.01)

# data, addr = self.sock.recvfrom(1024)
# line = data.decode()
# numbers = ast.literal_eval(line)


