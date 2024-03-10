from flask_restful import Resource

class GetState(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        x, y, z, rotation_angle, grip_angle, current_time = self.robot.get_state()
        return {"state": [x, y, z, rotation_angle, grip_angle], 'time': current_time}, 200