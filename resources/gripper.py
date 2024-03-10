from flask_restful import Resource

class Gripper(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, grip_angle):
        self.robot.grip_open_close(int(grip_angle))
        return {"Pinch to ": grip_angle}, 200