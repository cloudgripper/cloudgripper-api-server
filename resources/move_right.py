from flask_restful import Resource

class MoveRight(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        self.robot.step_right()
        return {"move": "right", "time": time.time()}, 200