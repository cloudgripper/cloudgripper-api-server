import time
import numpy as np
from resources.coordinate_normalizer import CoordinateNormalizer
from resources.limit_workspace import LimitWorkspace
from resources.video_capture import VideoCapture

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
        try:
            self.teensy = teensy
            self.teensy.flush()
            self.wake_teensy()
        except Exception as e:
            print(f"Failed to send command to Teensy: {e}")

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
        self.write_to_teensy("\r\n\r\n")
        time.sleep(2)
        try:
            self.teensy.flushInput()
            for _ in range(5):
                self.teensy.readline()
            print("Teensy is awake.")
        except Exception as e:
            print(f"Failed to send command to Teensy: {e}")

    def calibrate(self):
        self.write_to_teensy('T1' + '\n')

    def move_to(self, x, y):
        if self.boundaryUser.check_limits(x,y):
            x_mm, y_mm = self.xy_normalizer.xy_to_mm(x, y)
            x_mm = max(min(float(x_mm), self.xmax), self.xmin)
            y_mm = max(min(float(y_mm), self.ymax), self.ymin)
            self.x_position, self.y_position = x, y
            self.write_to_teensy('G00 X'+str(x_mm)+' Y'+str(y_mm) + '\n')
        else:
            print("Position out of bounds.")

    def get_state(self):
        #  x, y, z_angle, rotation_angle, claw_angle, z_current, rotation_current, claw_current, rotation_current
        self.write_to_teensy('S' + '\n')
        try:
            response = self.teensy.readline().decode('utf-8').strip()
            parts = response.split()
            if parts[0] == "STATE" and len(parts) == 9:
                x_raw, y_raw, z_angle, rotation_angle, claw_angle = map(float, parts[1:6])
                x_norm = self.xy_normalizer.x_to_norm(x_raw)
                y_norm = self.xy_normalizer.y_to_norm(y_raw)
                z_norm = 1 - z_angle / 180.0
                rotation = 180 - int(rotation_angle)
                claw_norm = 1 - claw_angle / 90.0

                z_current, rotation_current, claw_current = parts[6:9]
                return {
                    'x_norm': x_norm, 'y_norm': y_norm, 'z_norm': z_norm,
                    'rotation': rotation, 'claw_norm': claw_norm,
                    'z_current': z_current, 'rotation_current': rotation_current,
                    'claw_current': claw_current
                }, time.time()
        except Exception as e:
            print(f"Failed to read state: {e}")
        return None
        
    def grip_open_close(self, val):
        command_val = 90 - int(float(val)*90)
        self.write_to_teensy('O'+str(command_val) + '\n')
    
    def grip_up_down(self, val):
        command_val = 180 - int(float(val)*180)
        self.write_to_teensy('P'+str(command_val) + '\n')

    def rotate(self, angle):
        command_val = 180 - angle
        self.write_to_teensy('R'+str(command_val) + '\n')
        
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

    def write_to_teensy(self, command):
        try:
            self.teensy.write(command.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send command to Teensy: {e}.")

    def get_image_from_base(self):
        frame = self.camera_base.read()
        if frame is None:
            self.camera_base.release()
            try:
                self.camera_base = VideoCapture("/dev/camdown0")
            except Exception as e:
                print(f"Cannot open camera on base")
            return False, None, None 
        frame_time = time.time()
        return True, frame, frame_time

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