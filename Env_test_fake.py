import cv2
import numpy as np

from examples.experiments.rozum_push.Camera import Camera
from examples.experiments.rozum_push.FakeRobotEnv import FakeRobotEnv


def build_cameras():
    camera_indices = {"cam_front": 0}
    cams = {}
    for name, idx in camera_indices.items():
        try:
            cams[name] = Camera(idx)
        except RuntimeError:
            print(f"Camera {name} (index {idx}) unavailable, using synthetic frames.")
    return cams


def main():
    image_keys = ("cam_front",)
    cameras = build_cameras()
    env = FakeRobotEnv(cameras=cameras, image_keys=image_keys, teleop_set=False)

    delta_pos = np.zeros(2, dtype=np.float32)
    obs, info = env.reset()

    try:
        while True:
            obs, reward, terminated, truncated, info = env.step(delta_pos)

            cv2.imshow("cam_front", obs["cam_front"])

            if terminated or truncated:
                obs, info = env.reset()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        env.stop()
        for cam in cameras.values():
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
