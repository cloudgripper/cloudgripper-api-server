from flask_restful import Resource
from flask_jwt_extended import jwt_required
import cv2
import os
import base64

class GetImageTop(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    @jwt_required()
    def get(self):
        ret, frame = self.robot.get_image_from_top()
        if not ret:
            return {'error': 'Cannot read frame from webcam.'},404
            
        jpg_as_text = base64.b64encode(frame)
        image_str = jpg_as_text.decode('latin1')
        return {'data': image_str}, 200