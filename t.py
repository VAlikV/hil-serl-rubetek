from kuka.Gripper import Gripper
import time
grip = Gripper()
while 1:
    grip.send(0)
    time.sleep(2)
    grip.send(1)
    time.sleep(2)