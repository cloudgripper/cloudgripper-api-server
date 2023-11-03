from flask_restful import Resource
from flask_jwt_extended import jwt_required

class GetState(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    @jwt_required()
    def get(self):
        x, y, z, rotation_angle, grip_angle, current_time = self.robot.get_state()
        return {"state": [x, y, z, rotation_angle, grip_angle], 'time': current_time}, 200