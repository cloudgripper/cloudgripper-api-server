import json
import struct
import time
import traceback


def _send_binary(ws, meta_dict, *binary_parts):
    meta = json.dumps(meta_dict).encode()
    ws.send(struct.pack('>I', len(meta)) + meta + b''.join(binary_parts))


def _send_json(ws, data):
    ws.send(json.dumps(data))


def handle_get_image_base(robot, ws, msg, is_admin):
    ret, jpeg_bytes, frame_time = robot.get_jpeg_from_base()
    if not ret:
        return _send_json(ws, {'action': 'getImageBase', 'error': 'Cannot read frame from webcam.'})
    _send_binary(ws, {'action': 'getImageBase', 'time': frame_time}, jpeg_bytes)


def handle_get_image_top(robot, ws, msg, is_admin):
    ret, frame, frame_time = robot.get_image_from_top()
    if not ret:
        return _send_json(ws, {'action': 'getImageTop', 'error': 'Cannot read frame from webcam.'})
    _send_binary(ws, {'action': 'getImageTop', 'time': frame_time}, frame)


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
        'state': state,
        'time_state': time_state,
        'time_top_camera': time_top,
        'time_base_camera': time_base,
        'top_image_size': len(frame_top),
    }, frame_top, jpeg_base)


def handle_get_state(robot, ws, msg, is_admin):
    state, timestamp = robot.get_state()
    if state:
        _send_json(ws, {'action': 'getState', 'state': state, 'timestamp': timestamp})
    else:
        _send_json(ws, {'action': 'getState', 'error': "Failed to retrieve the robot's state"})


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
    robot.move_to(float(x), float(y))
    _send_json(ws, {'action': 'gcode', 'gcode': f'G00 X{x} Y{y}', 'time': time.time()})


def handle_grip(robot, ws, msg, is_admin):
    grip_angle = msg.get('grip_angle')
    if grip_angle is None:
        return _send_json(ws, {'action': 'grip', 'error': 'Missing "grip_angle"'})
    robot.grip_open_close(grip_angle)
    _send_json(ws, {'action': 'grip', 'grip_angle': grip_angle, 'time': time.time()})


def handle_rotate(robot, ws, msg, is_admin):
    rotate_angle = msg.get('rotate_angle')
    if rotate_angle is None:
        return _send_json(ws, {'action': 'rotate', 'error': 'Missing "rotate_angle"'})
    robot.rotate(int(rotate_angle))
    _send_json(ws, {'action': 'rotate', 'rotate_angle': rotate_angle, 'time': time.time()})


def handle_up_down(robot, ws, msg, is_admin):
    z_angle = msg.get('z_angle')
    if z_angle is None:
        return _send_json(ws, {'action': 'upDown', 'error': 'Missing "z_angle"'})
    robot.grip_up_down(z_angle)
    _send_json(ws, {'action': 'upDown', 'z_angle': z_angle, 'time': time.time()})


def handle_step(robot, ws, msg, is_admin):
    action = msg.get('action_vector')
    if not isinstance(action, list) or len(action) != 5:
        return _send_json(ws, {'action': 'step', 'error': 'Missing action_vector: [x, y, z, rotation, grip]'})
    x, y, z, rotation, grip = [float(value) for value in action]
    robot.step_action(x, y, z, int(rotation), grip)
    _send_json(ws, {
        'action': 'step',
        'action_vector': [x, y, z, int(rotation), grip],
        'time': time.time(),
    })


def handle_calibrate(robot, ws, msg, is_admin):
    robot.calibrate()
    _send_json(ws, {'action': 'calibrate', 'status': 'Calibrate', 'time': time.time()})


ACTION_HANDLERS = {
    'getImageBase': handle_get_image_base,
    'getImageTop': handle_get_image_top,
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
    'step': handle_step,
    'calibrate': handle_calibrate,
}


def register(sock, robot, app):
    @sock.route('/api/v1.1/robot/ws')
    def websocket_handler(ws):
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
                handler(robot, ws, msg, False)
            except Exception as e:
                traceback.print_exc()
                try:
                    _send_json(ws, {'action': action, 'error': str(e)})
                except Exception:
                    break
