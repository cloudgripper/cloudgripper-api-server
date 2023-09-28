import serial
import cv2
import os
from flask import Flask
from flask_restful import Api
from pymongo import MongoClient
import urllib.parse
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from common.robot import Robot
from resources.move_forward import MoveForward
from resources.move_backward import MoveBackward
from resources.move_left import MoveLeft
from resources.move_right import MoveRight
from resources.gripper import Gripper
from resources.rotate import Rotate
from resources.up_down import UpDown
from resources.gcode import Gcode
from resources.get_image_base import GetImageBase
from resources.get_image_top import GetImageTop
from resources.calibrate import Calibrate
from resources.register import Register
from resources.login import Login

# Making a Connection with MongoClient
mongoClientUsername = os.environ['MONGO_CLIENT_USERNAME']
mongoClientPassword = os.environ['MONGO_CLIENT_PASSWORD']
client = MongoClient("mongodb+srv://"+mongoClientUsername+":"+urllib.parse.quote(mongoClientPassword)+"@usercluster.kgmuck4.mongodb.net/?retryWrites=true&w=majority")
# database
db = client["api_database"]
# collection
users = db["Users"]
# Connect to teensy
teensy = serial.Serial('/dev/teensy', 9600, timeout=1)

# Connect to base camera
camera_base = cv2.VideoCapture("/dev/camdown0") # Original size: 640x480
if not camera_base.isOpened():
    raise Exception("Cannot open camera on base")
camera_base.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera_base.set(cv2.CAP_PROP_FRAME_WIDTH,320)
camera_base.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

# Connect to top camera
camera_top = cv2.VideoCapture("/dev/camtop0")
if not camera_top.isOpened():
    raise Exception("Cannot open camera on top")
camera_top.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera_top.set(cv2.CAP_PROP_FRAME_WIDTH,320)
camera_top.set(cv2.CAP_PROP_FRAME_HEIGHT,240)


# Initiate Robot
robot = Robot(teensy, camera_base, camera_top)
# Flask object
app = Flask(__name__)
api = Api(app)
#  JWT manager
jwt = JWTManager(app)
# JWT Config
app.config["JWT_SECRET_KEY"] = os.environ['JWT_SECRET_KEY']
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False
app.config['PROPAGATE_EXCEPTIONS'] = True


api.add_resource(Login, '/api/v1.1/robot/login', resource_class_kwargs={'users': users, 'app':app})
api.add_resource(Register, '/api/v1.1/robot/register', resource_class_kwargs={'users': users})
api.add_resource(MoveForward, '/api/v1.1/robot/moveUp', resource_class_kwargs={'robot': robot})
api.add_resource(MoveBackward, '/api/v1.1/robot/moveDown', resource_class_kwargs={'robot': robot})
api.add_resource(MoveRight, '/api/v1.1/robot/moveRight', resource_class_kwargs={'robot': robot})
api.add_resource(MoveLeft, '/api/v1.1/robot/moveLeft', resource_class_kwargs={'robot': robot})
api.add_resource(Gripper, '/api/v1.1/robot/grip/<string:grip_angle>', resource_class_kwargs={'robot': robot})
api.add_resource(Rotate, '/api/v1.1/robot/rotate/<string:rotate_angle>', resource_class_kwargs={'robot': robot})
api.add_resource(UpDown, '/api/v1.1/robot/up_down/<string:z_angle>', resource_class_kwargs={'robot': robot})
api.add_resource(Gcode, '/api/v1.1/robot/gcode/<string:x>/<string:y>', resource_class_kwargs={'robot': robot})
api.add_resource(Calibrate, '/api/v1.1/robot/calibrate', resource_class_kwargs={'robot': robot})
api.add_resource(GetImageBase, '/api/v1.1/robot/getImage', resource_class_kwargs={'robot': robot})
api.add_resource(GetImageTop, '/api/v1.1/robot/getImageTop', resource_class_kwargs={'robot': robot})

if __name__=="__main__":
    app.run(host='0.0.0.0')

