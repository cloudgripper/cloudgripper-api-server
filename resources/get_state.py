from flask_restful import Resource

class GetState(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        # x, y, z, rotation_angle, grip_angle, current_time = self.robot.get_state()
        state, timestamp = self.robot.get_state()
        if state:
            return {"state": state, 'timestamp': timestamp}, 200
        else:
            return {"error": "Failed to retrieve the robot's state"}, 500