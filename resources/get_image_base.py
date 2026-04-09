from flask_restful import Resource
import base64


class GetImageBase(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        ret, jpeg_bytes, frame_time = self.robot.get_jpeg_from_base()
        if not ret:
            return {'error': 'Cannot read frame from webcam.'}, 404

        image_str = base64.b64encode(jpeg_bytes).decode('latin1')
        return {'data': image_str, 'time': frame_time}, 200
