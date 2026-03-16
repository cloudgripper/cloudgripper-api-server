from flask_restful import Resource


class EvalStart(Resource):
    def __init__(self, **kwargs):
        self.eval_manager = kwargs['eval_manager']

    def get(self):
        result, error = self.eval_manager.start()
        if error:
            return {"error": error}, 400
        return result, 200
