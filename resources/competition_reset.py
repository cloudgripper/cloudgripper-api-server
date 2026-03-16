import os
import time
from flask_restful import Resource
from common.reset_policies import execute_pushing_reset, execute_rope_reset

VALID_OBJECT_TYPES = {'square_base', 'circle_base', 't_base'}


class EnvironmentReset(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        task_env = os.environ.get('RGMC_TASK_ENV')
        object_type = os.environ.get('RGMC_OBJECT_TYPE')

        try:
            if task_env == 'planar_pushing':
                if object_type not in VALID_OBJECT_TYPES:
                    return {"error": f"Invalid object type: {object_type}"}, 400
                execute_pushing_reset(self.robot, object_type)

            elif task_env == 'deformable_linear':
                execute_rope_reset(self.robot)

            else:
                return {"error": "Robot misconfigured. Task unknown."}, 500

            return {"status": "success", "message": "Environment reset successfully."}, 200

        except Exception as e:
            return {"error": f"Hardware reset failed: {str(e)}"}, 503