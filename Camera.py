import numpy as np
import cv2

class Camera():
    def __init__(self, number, img_size=(640,480)):
        self.cam = cv2.VideoCapture(number)
        if not self.cam.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
        self.img_size = img_size

    def get_image(self):
        ok, frame = self.cam.read()
        if not ok:
            print("Кадр не получен.")

        frame = cv2.resize(frame, self.img_size)

        return frame