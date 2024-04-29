import cv2
import threading
import time

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        if not self.cap.isOpened():
            raise ValueError(f"Camera {name} could not be opened.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.lock = threading.Lock()
        self.stopped = False
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self.stopped:
            with self.lock:
                ret = self.cap.grab()
                if not ret:
                    self.stopped = True
                    break
            time.sleep(1/35)

    def read(self):
        with self.lock:
            if self.stopped:
                return None
            ret, frame = self.cap.retrieve()
            return frame if ret else None

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def __del__(self):
        self.release()