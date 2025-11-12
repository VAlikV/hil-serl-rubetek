from .RobotAdapter import RobotAdapter
from .Camera import Camera
from .RealRobotEnv import RealRobotEnv
from .Robot import GovnoBot

import numpy as np
import socket
import time

haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
haptic_sock.bind(("127.0.0.1", 8081))
haptic_sock.settimeout(0.001)

robot = GovnoBot("10.10.10.10")
cameras = {"cam_front": Camera(0),"cam_side": Camera(1)}
adapter = RobotAdapter(robot=robot, cameras=cameras, image_keys=["cam_front","cam_side"])
env = RealRobotEnv(robot_adapter=adapter, image_keys=["cam_front","cam_side"])

delta_pos = np.array([0.0, 0.0, 0.0])

last_obs, info = env.reset()

first = True
last_pos = np.array([0.0, 0.0, 0.0])


while True:
    try:
        data, addr = haptic_sock.recvfrom(1024)
        message = np.array(list(map(float, data.decode()[1:-1].split(","))))

    except socket.timeout:
        data, addr = None, None  # или просто continue

    if len(message):
        # print(message)
        # print(message[0:3], message[3:12])
        # env.
        if first:
            last_pos = message[0:3]
            first = False
        else:
            delta_pos = message[0:3] - last_pos
            last_pos = message[0:3]
        # robot.set_new_point(message[0:3], message[3:12])

    obs, reward, terminated, truncated, info = env.step(delta_pos)

    delta_pos = np.array([0.0, 0.0, 0.0])

    print(obs)

    time.sleep(0.01)

# data, addr = self.sock.recvfrom(1024)
# line = data.decode()
# numbers = ast.literal_eval(line)


