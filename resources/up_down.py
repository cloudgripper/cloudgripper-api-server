from flask_restful import Resource

class UpDown(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, z_angle):
        self.robot.grip_up_down(z_angle)
        return {"Up/Down to ": z_angle, "time": time.time()}, 200