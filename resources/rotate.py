from flask_restful import Resource
import time

class Rotate(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, rotate_angle):
        self.robot.rotate(int(rotate_angle))
        return {"Rotate to ": rotate_angle, "time": time.time()}, 200