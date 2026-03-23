from flask_restful import Resource
from flask_jwt_extended import jwt_required, verify_jwt_in_request
import time


class Gcode(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, x, y):
        # Check if a valid JWT token is provided
        try:
            verify_jwt_in_request()
            # Valid JWT - privileged user, bypass boundary limits
            x_mm, y_mm = self.robot.move_to_admin(float(x), float(y))
            privileged = True
            return {"gcode": "G00 X"+str(x_mm)+" Y"+str(y_mm), "time": time.time()}, 200
        except Exception:
            # No JWT or invalid - normal user, enforce boundary limits
            result = self.robot.move_to(float(x), float(y))
            if result is None:
                return {"error": "Position out of bounds"}, 403
            x_mm, y_mm = result
            return {"gcode": "G00 X"+str(x_mm)+" Y"+str(y_mm), "time": time.time()}, 200