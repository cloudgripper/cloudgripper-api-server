from flask_restful import Resource
import cv2
import base64
from common.reset_policies import _build_iou_evaluator


class GetImageBaseUndistorted(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']
        self._iou_evaluator = _build_iou_evaluator()

    def get(self):
        ret, frame, frame_time = self.robot.get_image_from_base()
        if not ret or frame is None:
            return {'error': 'Cannot read frame from base camera.'}, 404

        undistorted = self._iou_evaluator.undistort_image(frame)
        encoded, buffer = cv2.imencode('.jpg', undistorted)
        image_str = base64.b64encode(buffer).decode('latin1')
        return {'data': image_str, 'time': frame_time}, 200
