from flask_restful import Resource
from common.auth import restrict_in_tasks
import time

class Gripper(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    @restrict_in_tasks(['planar_pushing'])
    def get(self, grip_angle):
        self.robot.grip_open_close(grip_angle)
        return {"Pinch to ": grip_angle, "time": time.time()}, 200