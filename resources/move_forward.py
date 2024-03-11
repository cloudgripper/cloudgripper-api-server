from flask_restful import Resource
import time

class MoveForward(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        self.robot.step_forward()
        return {"move": "forward", "time": time.time()}, 200