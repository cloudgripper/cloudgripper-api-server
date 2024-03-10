from flask_restful import Resource

class MoveLeft(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        self.robot.step_left()
        return {"move": "left"}, 200