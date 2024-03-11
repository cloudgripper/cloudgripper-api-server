from flask_restful import Resource

class Gcode(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, x, y):
        self.robot.move_to(float(x),float(y))
        return {"gcode": "G00 X"+str(x)+" Y"+str(y)}, 200