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
        self._frame = None
        self._jpeg_buffer = None
        self._frame_time = None
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ret = self.cap.grab()
            if not ret:
                self.stopped = True
                break
            ret, frame = self.cap.retrieve()
            if ret:
                _, jpeg_buf = cv2.imencode('.jpg', frame)
                ts = time.time()
                with self.lock:
                    self._frame = frame
                    self._jpeg_buffer = jpeg_buf
                    self._frame_time = ts
            time.sleep(1/35)

    def read(self):
        with self.lock:
            if self.stopped or self._frame is None:
                return None
            return self._frame

    def read_jpeg(self):
        with self.lock:
            if self.stopped or self._jpeg_buffer is None:
                return None, None
            return self._jpeg_buffer.tobytes(), self._frame_time

    def release(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

    def __del__(self):
        self.release()