import json
import os
import struct
import time
import traceback

import cv2
from flask import request


_iou_evaluator = None


def restrict_in_tasks(restricted_tasks):
    """Block a WS action when the current RGMC_TASK_ENV is in restricted_tasks."""
    def decorator(fn):
        def wrapper(robot, ws, msg, is_admin):
            current_env = os.environ.get('RGMC_TASK_ENV')
            if current_env in restricted_tasks and not is_admin:
                action = msg.get('action', '')
                _send_json(ws, {
                    'action': action,
                    'error': f"Action forbidden. Endpoint is locked during '{current_env}' task.",
                })
                return
            return fn(robot, ws, msg, is_admin)
        return wrapper
    return decorator


def jwt_required_ws(fn):
    """Block a WS action unless the connection was established with a valid JWT."""
    def wrapper(robot, ws, msg, is_admin):
        if not is_admin:
            action = msg.get('action', '')
            _send_json(ws, {'action': action, 'error': 'This action requires admin (JWT) authentication.'})
            return
        return fn(robot, ws, msg, is_admin)
    return wrapper


def _send_binary(ws, meta_dict, *binary_parts):
    meta = json.dumps(meta_dict).encode()
    ws.send(struct.pack('>I', len(meta)) + meta + b''.join(binary_parts))


def _send_json(ws, data):
    ws.send(json.dumps(data))


# ---------------------------------------------------------------------------
# Image handlers (binary response)
# ---------------------------------------------------------------------------

def handle_get_image_base(robot, ws, msg, is_admin):
    ret, jpeg_bytes, frame_time = robot.get_jpeg_from_base()
    if not ret:
        return _send_json(ws, {'action': 'getImageBase', 'error': 'Cannot read frame from webcam.'})
    _send_binary(ws, {'action': 'getImageBase', 'time': frame_time}, jpeg_bytes)


@restrict_in_tasks(['planar_pushing'])
def handle_get_image_top(robot, ws, msg, is_admin):
    ret, frame, frame_time = robot.get_image_from_top()
    if not ret:
        return _send_json(ws, {'action': 'getImageTop', 'error': 'Cannot read frame from webcam.'})
    _send_binary(ws, {'action': 'getImageTop', 'time': frame_time}, frame)


def handle_get_image_base_undistorted(robot, ws, msg, is_admin):
    ret, frame, frame_time = robot.get_image_from_base()
    if not ret or frame is None:
        return _send_json(ws, {'action': 'getImageBaseUndistorted', 'error': 'Cannot read frame from base camera.'})
    undistorted = _iou_evaluator.undistort_image(frame)
    _, buffer = cv2.imencode('.jpg', undistorted)
    _send_binary(ws, {'action': 'getImageBaseUndistorted', 'time': frame_time}, buffer.tobytes())


@restrict_in_tasks(['planar_pushing'])
def handle_get_all_states(robot, ws, msg, is_admin):
    ret_top, frame_top, time_top = robot.get_image_from_top()
    ret_base, jpeg_base, time_base = robot.get_jpeg_from_base()
    state, time_state = robot.get_state()

    if not ret_top:
        return _send_json(ws, {'action': 'getAllStates', 'error': 'Cannot read frame from top camera.'})
    if not ret_base:
        return _send_json(ws, {'action': 'getAllStates', 'error': 'Cannot read frame from base camera.'})
    if not state:
        return _send_json(ws, {'action': 'getAllStates', 'error': "Failed to retrieve the robot's state"})

    _send_binary(ws, {
        'action': 'getAllStates',
        'state': state, 'time_state': time_state,
        'time_top_camera': time_top, 'time_base_camera': time_base,
        'top_image_size': len(frame_top),
    }, frame_top, jpeg_base)


# ---------------------------------------------------------------------------
# State handler (JSON response)
# ---------------------------------------------------------------------------

def handle_get_state(robot, ws, msg, is_admin):
    state, timestamp = robot.get_state()
    if state:
        _send_json(ws, {'action': 'getState', 'state': state, 'timestamp': timestamp})
    else:
        _send_json(ws, {'action': 'getState', 'error': "Failed to retrieve the robot's state"})


# ---------------------------------------------------------------------------
# Movement handlers (JSON response)
# ---------------------------------------------------------------------------

def handle_move_up(robot, ws, msg, is_admin):
    robot.step_forward()
    _send_json(ws, {'action': 'moveUp', 'move': 'forward', 'time': time.time()})


def handle_move_down(robot, ws, msg, is_admin):
    robot.step_backward()
    _send_json(ws, {'action': 'moveDown', 'move': 'backward', 'time': time.time()})


def handle_move_left(robot, ws, msg, is_admin):
    robot.step_left()
    _send_json(ws, {'action': 'moveLeft', 'move': 'left', 'time': time.time()})


def handle_move_right(robot, ws, msg, is_admin):
    robot.step_right()
    _send_json(ws, {'action': 'moveRight', 'move': 'right', 'time': time.time()})


def handle_gcode(robot, ws, msg, is_admin):
    x, y = msg.get('x'), msg.get('y')
    if x is None or y is None:
        return _send_json(ws, {'action': 'gcode', 'error': 'Missing "x" or "y"'})
    if is_admin:
        result = robot.move_to_admin(float(x), float(y))
    else:
        result = robot.move_to(float(x), float(y))
    if result is None:
        return _send_json(ws, {'action': 'gcode', 'error': 'Position out of bounds'})
    x_mm, y_mm = result
    _send_json(ws, {'action': 'gcode', 'gcode': f'G00 X{x_mm} Y{y_mm}', 'time': time.time()})


# ---------------------------------------------------------------------------
# Gripper / rotation handlers (JSON response)
# ---------------------------------------------------------------------------

@restrict_in_tasks(['planar_pushing'])
def handle_grip(robot, ws, msg, is_admin):
    grip_angle = msg.get('grip_angle')
    if grip_angle is None:
        return _send_json(ws, {'action': 'grip', 'error': 'Missing "grip_angle"'})
    robot.grip_open_close(grip_angle)
    _send_json(ws, {'action': 'grip', 'grip_angle': grip_angle, 'time': time.time()})


@restrict_in_tasks(['planar_pushing'])
def handle_rotate(robot, ws, msg, is_admin):
    rotate_angle = msg.get('rotate_angle')
    if rotate_angle is None:
        return _send_json(ws, {'action': 'rotate', 'error': 'Missing "rotate_angle"'})
    robot.rotate(int(rotate_angle))
    _send_json(ws, {'action': 'rotate', 'rotate_angle': rotate_angle, 'time': time.time()})


@restrict_in_tasks(['planar_pushing'])
def handle_up_down(robot, ws, msg, is_admin):
    z_angle = msg.get('z_angle')
    if z_angle is None:
        return _send_json(ws, {'action': 'upDown', 'error': 'Missing "z_angle"'})
    robot.grip_up_down(z_angle)
    _send_json(ws, {'action': 'upDown', 'z_angle': z_angle, 'time': time.time()})


# ---------------------------------------------------------------------------
# Calibrate & environment reset (JSON response)
# ---------------------------------------------------------------------------

@jwt_required_ws
def handle_calibrate(robot, ws, msg, is_admin):
    robot.calibrate()
    _send_json(ws, {'action': 'calibrate', 'status': 'Calibrate', 'time': time.time()})


def handle_env_reset(robot, ws, msg, is_admin):
    from common.reset_policies import execute_pushing_reset, execute_rope_reset
    task_env = os.environ.get('RGMC_TASK_ENV')
    object_type = os.environ.get('RGMC_OBJECT_TYPE')
    try:
        if task_env == 'planar_pushing':
            if object_type not in {'square_base', 'circle_base', 't_base'}:
                return _send_json(ws, {'action': 'envReset', 'error': f'Invalid object type: {object_type}'})
            execute_pushing_reset(robot, object_type)
        elif task_env == 'deformable_linear':
            execute_rope_reset(robot)
        else:
            return _send_json(ws, {'action': 'envReset', 'error': 'Robot misconfigured. Task unknown.'})
        _send_json(ws, {'action': 'envReset', 'status': 'success', 'message': 'Environment reset successfully.'})
    except Exception as e:
        _send_json(ws, {'action': 'envReset', 'error': f'Hardware reset failed: {str(e)}'})


# ---------------------------------------------------------------------------
# Action routing table
# ---------------------------------------------------------------------------

ACTION_HANDLERS = {
    'getImageBase': handle_get_image_base,
    'getImageTop': handle_get_image_top,
    'getImageBaseUndistorted': handle_get_image_base_undistorted,
    'getAllStates': handle_get_all_states,
    'getState': handle_get_state,
    'moveUp': handle_move_up,
    'moveDown': handle_move_down,
    'moveLeft': handle_move_left,
    'moveRight': handle_move_right,
    'gcode': handle_gcode,
    'grip': handle_grip,
    'rotate': handle_rotate,
    'upDown': handle_up_down,
    'calibrate': handle_calibrate,
    'envReset': handle_env_reset,
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _verify_jwt(app):
    """Check if the handshake carried a valid JWT token. Returns True if admin."""
    token = request.args.get('token') or request.headers.get('Authorization', '').removeprefix('Bearer ')
    if not token:
        return False
    try:
        from flask_jwt_extended import decode_token
        decode_token(token)
        return True
    except Exception:
        return False


def register(sock, robot, app):
    global _iou_evaluator
    from common.reset_policies import _build_iou_evaluator
    _iou_evaluator = _build_iou_evaluator()

    @sock.route('/api/v1.1/robot/ws')
    def websocket_handler(ws):
        is_admin = _verify_jwt(app)

        while True:
            try:
                raw = ws.receive()
            except Exception:
                break
            if raw is None:
                break

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                _send_json(ws, {'error': 'Invalid JSON'})
                continue

            action = msg.get('action')
            if action is None:
                _send_json(ws, {'error': 'Missing "action" field'})
                continue

            handler = ACTION_HANDLERS.get(action)
            if handler is None:
                _send_json(ws, {'error': f'Unknown action: {action}'})
                continue

            try:
                handler(robot, ws, msg, is_admin)
            except Exception as e:
                traceback.print_exc()
                try:
                    _send_json(ws, {'action': action, 'error': str(e)})
                except Exception:
                    break
