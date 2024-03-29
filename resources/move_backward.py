from flask_restful import Resource
import time

class MoveBackward(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        self.robot.step_backward()
        return {"move": "backward", "time": time.time()}, 200