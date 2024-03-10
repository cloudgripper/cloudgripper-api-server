import time
import cv2
import numpy as np
from resources.coordinate_normalizer import CoordinateNormalizer
from resources.limit_workspace import LimitWorkspace

class Robot():
    def __init__(self, teensy, camera_base, camera_top, limits):
        # Limit definitions
        self.xmin, self.xmax, self.ymin, self.ymax = limits[0]
        self.user_xmin, self.user_xmax, self.user_ymin, self.user_ymax = limits[1]

        # Coordinate normalization and workspace limitation
        self.xy_normalizer = CoordinateNormalizer(self.user_xmin, self.user_xmax, self.user_ymin, self.user_ymax)
        self.boundaryUser = LimitWorkspace((0.0, 0.0), (1.0, 1.0))

        # setup camera and teensy
        self.camera_base = camera_base
        self.camera_top = camera_top
        self.teensy = teensy
        self.teensy.flush()
        self.wake_teensy()

        # Robot State
        self.x_position = 0
        self.y_position = 0
        self.nudge = 0.05
        self.alpha = 0

        self.calibrate()
        time.sleep(5)
        
        print("Calibrating...")
        self.calibrate()
        self.rotate(180)
        self.move_to(0,0)

    def wake_teensy(self):
        self.teensy.flush()
        self.teensy.write("\r\n\r\n".encode('utf-8'))
        time.sleep(2)
        self.teensy.flushInput()
        for _ in range(5):
            self.teensy.readline()
        print("Teensy is awake.")

    def calibrate(self):
        self.teensy.write( ('T1' + '\n').encode('utf-8'))
        
    def grip_open_close(self, val):
        command_val = 90 - int(float(val)*90)
        self.teensy.write(('O'+str(val) + '\n').encode('utf-8'))
    
    def grip_up_down(self, val):
        command_val = 180 - int(float(val)*180)
        self.teensy.write(('P'+str(val) + '\n').encode('utf-8'))

    def rotate(self, angle):
        command_val = 180 - angle
        self.teensy.write(('R'+str(command_val) + '\n').encode('utf-8'))
        
    def step_right(self):
        self.x_position += self.nudge
        self.move_to( self.x_position + self.nudge, self.y_position )
        # self.teensy.write(('DD' + '\n').encode('utf-8'))

    def step_left(self):
        self.x_position -= self.nudge
        self.move_to( self.x_position - self.nudge, self.y_position )
        # self.teensy.write(('LL' + '\n').encode('utf-8'))

    def step_forward(self):
        self.y_position += self.nudge
        self.move_to( self.x_position, self.y_position + self.nudge )
        # self.teensy.write(('FF' + '\n').encode('utf-8'))

    def step_backward(self):
        self.y_position -= self.nudge
        self.move_to( self.x_position, self.y_position - self.nudge )
        # self.teensy.write(('BB' + '\n').encode('utf-8'))

    def get_image_from_base(self):
        ret, frame = self.camera_base.read()
        if not ret:
            self.camera_base.release()
            self.camera_base = cv2.VideoCapture("/dev/camdown0")
            if not self.camera_base.isOpened():
                print("Cannot open camera on base")
            else:
                self.camera_base.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return False, None, None 
        frame_time = time.time()
        return ret, frame, frame_time

    def get_image_from_top(self):
        try:
            with self.camera_top.condition:
                self.camera_top.condition.wait()
                frame = self.camera_top.frame
            frame_time = time.time()
            ret = True
            return ret, frame, frame_time
        except Exception as e:
            print(f"Failed to get image from top camera: {e}")
            return False, None, None  