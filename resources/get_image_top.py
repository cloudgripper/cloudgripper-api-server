from flask_restful import Resource
import cv2
import os
import base64

class GetImageTop(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        ret, frame, frame_time = self.robot.get_image_from_top()
        if not ret:
            return {'error': 'Cannot read frame from webcam.'},404
            
        jpg_as_text = base64.b64encode(frame)
        image_str = jpg_as_text.decode('latin1')
        return {'data': image_str, 'time': frame_time}, 200