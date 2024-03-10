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
        # wake the teensy 
        self.teensy.write("\r\n\r\n".encode('utf-8'))
        # wait for to wake up
        time.sleep(2)
        # flush startup text
        self.teensy.flushInput() 
        # Clear the serial
        for i in range(5) : self.teensy.readline()

        # Initiailizing
        print("Initiailizing")
        print("- Caliberating")

        self.grip_open_close(40)
        time.sleep(1)
        self.grip_up_down(10)
        time.sleep(1)
        self.rotate(0)
        time.sleep(1)

        self.calibrate()
        time.sleep(5)
        
        print("Initialization Completed")
        self.x_position = 0
        self.y_position = 0
        self.gripper_angle = 40
        self.gripper_rotation = 0
        self.gripper_z = 0
        
    def get_state(self):

        if self.gripper_angle>65:
            # close
            gripperval = 1
        else:
            # open
            gripperval = 0
        
        z_pos = self.gripper_z
        rot = self.gripper_rotation
        x_state = round(self.x_position, 2)
        y_state = round(self.y_position, 2)
        return x_state, y_state, z_pos, rot, gripperval, time.time()

    def calibrate(self):
        self.teensy.write( ('T1' + '\n').encode('utf-8'))
        
    def grip_open_close(self, val):
        self.gripper_angle = val
        self.teensy.write(('O'+str(val)+ '\n').encode('utf-8'))
    
    def grip_up_down(self, val):
        self.gripper_z = val
        self.teensy.write(('P'+str(val)+ '\n').encode('utf-8'))
        
    def rotate(self,angle):
        self.gripper_rotation = angle
        self.teensy.write(('R'+str(angle) + '\n').encode('utf-8'))
        
    def move_to(self,x,y):
        self.x_position = float(x)
        self.y_position = float(y)
        self.teensy.write( ('G00 X'+str(x)+' Y'+str(y) + '\n').encode('utf-8'))
    
    def step_right(self, nudge_mm = 5):
        self.x_position -= abs(nudge_mm)
        self.move_to( self.x_position, self.y_position )

    def step_left(self, nudge_mm = 5):
        self.x_position += abs(nudge_mm)
        self.move_to( self.x_position, self.y_position )

    def step_forward(self, nudge_mm = 5):
        self.y_position += abs(nudge_mm)
        self.move_to( self.x_position, self.y_position )

    def step_backward(self, nudge_mm = 5):
        self.y_position -= abs(nudge_mm)
        self.move_to( self.x_position, self.y_position )

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