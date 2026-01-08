import numpy as np
import socket

class RobotSocket:
    def __init__(self, own_ip, own_port, robot_ip, robot_port):

        self.robot_ip = robot_ip
        self.robot_port = robot_port

        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_socket.bind((own_ip, own_port))
        self.robot_socket.settimeout(0.001)
        self.robot_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)


    def readState(self):

        while(True):
            try:
                data, addr = self.robot_socket.recvfrom(1024)
                message = np.array(list(map(float, data.decode()[1:-1].split(","))))
                if len(message):
                    pos = message[0:3]
                    orient = np.array([message[3:6], 
                                        message[6:9], 
                                        message[9:12]])
                    break

            except socket.timeout:
                data, addr = None, None

        return pos, orient
    
    
    def sendCommand(self, pos, orient):

        command = np.concatenate([pos, orient[0], orient[1], orient[2]])

        # command = str(command).replace("", ",")
        # print(command)

        command = self.toJSONstr(command)

        self.robot_socket.sendto(command.encode("utf-8"), (self.robot_ip, self.robot_port))


    def toJSONstr(self, command):

        msg = "[" + str(command[0])

        for i in range(1, len(command)):
            msg += ","
            msg += str(command[i])

        msg += "]"

        return msg