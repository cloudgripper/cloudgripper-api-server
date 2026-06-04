from flask_restful import Resource
import time

class StepAction(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self, x, y, z, rotation, grip):
        try:
            x, y, z, rotation, grip = [float(value) for value in [x, y, z, rotation, grip]]
        except (TypeError, ValueError):
            return {"error": "Action values must be numeric"}, 400

        self.robot.step_action(x, y, z, int(rotation), grip)
        return {
            "action": {
                "x": x,
                "y": y,
                "z": z,
                "rotation": int(rotation),
                "grip": grip
            },
            "time": time.time()
        }, 200
