import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import cv2

# with open('demo_data/for_classifier.pkl', 'rb') as f:
with open('classifier_data/success_images_2025-11-20_18-32-25.pkl', 'rb') as f:
    data = pkl.load(f)   

successes = []
failures = []

for i in range(len(data)):

    obs = data[i]["observations"]
    actions = data[i]["actions"]
    next_obs = data[i]["next_observations"]
    rew = data[i]["rewards"]
    done = data[i]["dones"]

    transition = copy.deepcopy(
        dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=rew,
            masks=1.0 - done,
            dones=done,
        )
    )
    
    image_1 = data[i]["observations"]["cam_front"]
    image_2 = data[i]["observations"]["cam_side"]

    cv2.imshow("cam_front", image_1)
    cv2.imshow("cam_side", image_2)

    key = cv2.waitKey(0)

    if key == ord("s"):
        successes.append(transition)
    elif key == ord("f"):
        failures.append(transition)

    print("Успехов: ", len(successes), " Неуспехов: ", len(failures))
    print("Всего: ", i, " / ", len(data))

if not os.path.exists("./classifier_data"):
    os.makedirs("./classifier_data")
uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_name = f"./classifier_data/success_images_{uuid}.pkl"
with open(file_name, "wb") as f:
    pkl.dump(successes, f)
    print(f"saved {len(successes)} successful transitions to {file_name}")

file_name = f"./classifier_data/failure_images_{uuid}.pkl"
with open(file_name, "wb") as f:
    pkl.dump(failures, f)
    print(f"saved {len(failures)} failure transitions to {file_name}")

