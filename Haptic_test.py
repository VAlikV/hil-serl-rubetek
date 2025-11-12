import numpy as np
import socket
from Robot import GovnoBot
import time
import asyncio

robot = GovnoBot("10.10.10.10")

haptic_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
haptic_sock.bind(("127.0.0.1", 8081))
haptic_sock.settimeout(0.001)

message = np.array([])

while True:
    try:
        data, addr = haptic_sock.recvfrom(1024)
        message = np.array(list(map(float, data.decode()[1:-1].split(","))))

    except socket.timeout:
        data, addr = None, None  # или просто continue

    if len(message):
        # print(message)
        # print(message[0:3], message[3:12])
        robot.set_new_point(message[0:3], message[3:12])

    robot.step()

    q, dq, tool_tcp, tool_tcp_vel = robot.get_robot_attr()

    time.sleep(0.01)

# data, addr = self.sock.recvfrom(1024)
# line = data.decode()
# numbers = ast.literal_eval(line)
