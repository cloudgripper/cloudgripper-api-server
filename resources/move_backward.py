from flask_restful import Resource

class MoveBackward(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        self.robot.step_backward()
        return {"move": "backward"}, 200