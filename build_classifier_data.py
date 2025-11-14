import glob, os, pickle as pkl
import numpy as np

SRC_DIRS = [
    "logs/my_robot_run/demo_buffer",   # где лежат transitions_*.pkl для демо/интервенций
    "logs/my_robot_run/buffer"         # онлайн буфер (если там есть метки)
]
DST_DIR = "classifier_data"
IMAGE_KEYS = ["cam_front", "cam_side"]  # == config.classifier_keys

os.makedirs(DST_DIR, exist_ok=True)
success_samples, failure_samples = [], []

def obs_ok(obs):
    # изображения должны лежать напрямую по IMAGE_KEYS
    return all(k in obs and obs[k].dtype==np.uint8 for k in IMAGE_KEYS)

for d in SRC_DIRS:
    for f in glob.glob(os.path.join(d, "transitions_*.pkl")):
        transitions = pkl.load(open(f,"rb"))
        # простой хук: считаем "успех" если последний шаг эпизода имел info["episode"]["r"]>0
        # (поменяй под свою логику: например, флаг success в info, или ручная фильтрация)
        # Здесь возьмём последние кадры каждого эпизода по done=True
        episode = []
        for tr in transitions:
            episode.append(tr)
            if tr.get("dones", False) or (tr.get("masks", 1.0)==0.0):
                final = episode[-1]
                if "observations" in final and obs_ok(final["observations"]):
                    label_is_success = bool(final.get("rewards", 0.0) > 0.5)  # адаптируй под себя!
                    sample = {
                        "observations": {k: final["observations"][k] for k in IMAGE_KEYS},
                        "actions": np.zeros(1, np.float32),  # заглушка, скрипт всё равно заменит
                    }
                    if label_is_success:
                        success_samples.append(sample)
                    else:
                        failure_samples.append(sample)
                episode = []

# Сохраним порционно
with open(os.path.join(DST_DIR, "my_success_0.pkl"), "wb") as f:
    pkl.dump(success_samples, f)
with open(os.path.join(DST_DIR, "my_failure_0.pkl"), "wb") as f:
    pkl.dump(failure_samples, f)

print("success:", len(success_samples), "failure:", len(failure_samples))