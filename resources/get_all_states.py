from flask_restful import Resource
import cv2
import os
import base64

class GetAllStates(Resource):
    def __init__(self, **kwargs):
        self.robot = kwargs['robot']

    def get(self):
        ret_top, frame_top, frame_time_top = self.robot.get_image_from_top()
        ret_base, frame_base, frame_time_base = self.robot.get_image_from_base()
        state, timestamp_state = self.robot.get_state()

        if not ret_top:
            return {'error': 'Cannot read frame from top camera.'},404
        
        if not ret_base:
            return {'error': 'Cannot read frame from base camera.'},404
            
        if not state:
            return {"error": "Failed to retrieve the robot's state"}, 404

        encoded_frame_base, buffer_frame_base = cv2.imencode('.jpg', frame_base)

        jpg_as_text_top = base64.b64encode(frame_top)
        jpg_as_text_base = base64.b64encode(buffer_frame_base)

        image_str_top = jpg_as_text_top.decode('latin1')
        image_str_base = jpg_as_text_base.decode('latin1')

        return {'data_top_camera': image_str_top, 'time_top_camera': frame_time_top, 'data_base_camera': image_str_base, 'time_base_camera': frame_time_base, 'state': state, 'time_state': timestamp_state}, 200