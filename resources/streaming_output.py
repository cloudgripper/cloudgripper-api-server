import io
from threading import Condition

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.frame_seq = 0
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.frame_seq += 1
            self.condition.notify_all()