from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


# with open('demo_data/rozum_push_1_demos_2025-11-20_17-11-38.pkl', 'rb') as f:
with open('classifier_data/success_images_2025-11-20_18-32-25.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[0].keys())

actions = np.array([[0.0, 0.0]])
dones = []
teleop = []
t_actions = np.array([[0.0, 0.0]])

for i in range(len(data)):

    dones.append(data[i]["dones"])
    actions = np.concatenate((actions, [data[i]["actions"]]), axis=0)

    # if "intervene_action" in data[i]["infos"].keys():
    #     teleop.append(1)
    #     t_actions = np.concatenate((t_actions, [data[i]["infos"]["intervene_action"]]), axis=0)
    # else:
    #     teleop.append(0)
    #     t_actions = np.concatenate((t_actions, [[0,0]]), axis=0)

    cam_1 = data[i]["observations"]["cam_front"]
    cam_2 = data[i]["observations"]["cam_side"]

    cv2.imshow("cam_front", cam_1)
    cv2.imshow("cam_side", cam_2)

    cv2.waitKey(0)

# print(actions.shape)

# plt.figure(0)
# plt.plot(actions)
# # plt.plot(t_actions)
# # plt.plot(np.array(teleop)/1000)

# plt.figure(1)
# plt.plot(teleop)

# plt.figure(2)
# plt.plot(dones)

# plt.show()