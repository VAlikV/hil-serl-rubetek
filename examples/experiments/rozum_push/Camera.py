import cv2
import threading

class Camera:
    def __init__(self, number, img_size=(480, 360)):
        self.cam = cv2.VideoCapture(number)
        if not self.cam.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        self.img_size = img_size
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        # Запускаем отдельный поток
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cam.read()
            if ok:
                frame = cv2.resize(frame, self.img_size)
                with self.lock:
                    self.frame = frame

    def get_image(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join()
        self.cam.release()