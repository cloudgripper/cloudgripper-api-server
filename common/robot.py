import time
import cv2

class Robot():
    def __init__(self, teensy, camera_base, camera_top):
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
        self.teensy.write(('O'+str(val)+ '\n').encode('utf-8'))
    
    def grip_up_down(self, val):
        self.teensy.write(('P'+str(val)+ '\n').encode('utf-8'))
        
    def rotate(self,angle):
        self.teensy.write(('R'+str(angle) + '\n').encode('utf-8'))
        
    def move_to(self,x,y):
        self.teensy.write( ('G00 X'+str(x)+' Y'+str(y) + '\n').encode('utf-8'))
    
    def step_right(self):
        self.teensy.write(('DD' + '\n').encode('utf-8'))
    def step_left(self):
        self.teensy.write(('LL' + '\n').encode('utf-8'))
    def step_forward(self):
        self.teensy.write(('FF' + '\n').encode('utf-8'))
    def step_backward(self):
        self.teensy.write(('BB' + '\n').encode('utf-8'))
    def get_image_from_base(self):
        ret, frame  = self.camera_base.read()
        if not ret:
            self.camera_base.release()
            self.camera_base = cv2.VideoCapture("/dev/camdown0")
            if not self.camera_base.isOpened():
                print("Cannot open camera on base")
            else:
                self.camera_base.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return ret, frame
    def get_image_from_top(self):
        with self.camera_top.condition:
            self.camera_top.condition.wait()
            frame = self.camera_top.frame
            frameTime = time.time()
        ret = True
        return ret, frame