from flask_restful import Resource


class EvalTarget(Resource):
    def __init__(self, **kwargs):
        self.eval_manager = kwargs['eval_manager']

    def get(self):
        result, error = self.eval_manager.get_target()
        if error:
            return {"error": error}, 404
        return result, 200
