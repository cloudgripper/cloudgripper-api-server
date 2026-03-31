import os
import time
import random
import socket
import cv2
import numpy as np
from resources.hand_eye_converter import HandEyeConverter
from rgmc_cloud_robotics_2026 import PushingTaskIOU_Calculator
from common.rope_segmentation_util import RopeSegmentationUtil
from rgmc_cloud_robotics_2026.shapes import (
    SQUARE_CORNERS, CIRCLE_CORNERS, T_CORNERS,
    T_TOTAL_WIDTH_MM, T_THICKNESS_MM,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_DIR = os.path.join(BASE_DIR, 'calibration')

OBJECT_CORNERS = {
    'square_base': SQUARE_CORNERS,
    'circle_base': CIRCLE_CORNERS,
    't_base': T_CORNERS,
}


def _get_robot_number():
    hostname = socket.gethostname()
    return int(hostname.replace('cr', ''))


def _get_robot_config():
    """
    Robot-specific reset policies configuration.
    """
    return {
        'push_reset_z': float(os.environ.get('PUSH_RESET_Z', '0.3')),
        'push_reset_grip_hold': float(os.environ.get('PUSH_RESET_GRIP_HOLD', '0')),
        # T-shape specific configs
        't_reset_z': float(os.environ.get('T_RESET_Z', '0')),
        't_reset_grip_hold': float(os.environ.get('T_RESET_GRIP_HOLD', '0')),
    }


def _build_iou_evaluator():
    robot_num = _get_robot_number()
    camera_params_path = os.path.join(CALIBRATION_DIR, f'camera-params-cr{robot_num:02d}.yaml')
    return PushingTaskIOU_Calculator(camera_params_path)


def _build_hand_eye_converter():
    robot_num = _get_robot_number()
    path = os.path.join(CALIBRATION_DIR, f'camera-to-robot-cr{robot_num}.yaml')
    return HandEyeConverter(path)


def _centroid_of_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def _t_long_end_points_undistorted(alignment_warp, iou_evaluator):
    """
    Return the centres of the two stem boxes in undistorted image space.
    Coordinates stay undistorted so they can feed straight into
    converter.px_py_to_x_y.
    """
    stem_junction_x = T_TOTAL_WIDTH_MM / 2 - T_THICKNESS_MM
    stem_tip_x = -T_TOTAL_WIDTH_MM / 2
    stem_length = stem_junction_x - stem_tip_x

    stem_pts_3d = np.array([
        [stem_junction_x - 0.25 * stem_length, 0, 0],
        [stem_junction_x - 0.75 * stem_length, 0, 0],
    ], dtype=np.float32)

    stem_pts_undist = iou_evaluator.project_object_to_image(stem_pts_3d, [0, 0], 0)
    stem_pts_warped = cv2.transform(stem_pts_undist.astype(np.float32), alignment_warp)

    pt1 = stem_pts_warped[0, 0]
    pt2 = stem_pts_warped[1, 0]
    return (float(pt1[0]), float(pt1[1])), (float(pt2[0]), float(pt2[1]))


def _random_position_away_from(x, y, gap=0.2):
    """Sample a random position that maintains at least `gap` distance from (x, y)."""
    usable = 1.0 - gap
    rx = random.uniform(0, usable)
    ry = random.uniform(0, usable)
    if rx > (x - gap / 2):
        rx += gap
    if ry > (y - gap / 2):
        ry += gap
    return float(np.clip(rx, 0, 1)), float(np.clip(ry, 0, 1))


def _execute_generic_pushing_reset(robot, object_type):
    iou_evaluator = _build_iou_evaluator()
    converter = _build_hand_eye_converter()
    obj_corners = OBJECT_CORNERS[object_type]
    config = _get_robot_config()

    robot.grip_open_close(1)
    time.sleep(0.5)
    robot.grip_up_down(config['push_reset_z'])
    time.sleep(1)

    ret, frame, _ = robot.get_image_from_base()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture base camera image.")

    target_contour_undistorted = iou_evaluator.generate_target_contour(
        obj_corners, offset_from_center_mm=[50, -50]
    )

    (
        frame_score,
        vis_img,
        obj_contour_distorted,
        target_contour,
        intersection_contour_img_undistorted,
        obj_contour_undistorted,
        target_contour_undistorted,
        alignment_angle_deg,
        alignment_warp,
    ) = iou_evaluator.calculate_iou_from_distorted_base(
        frame, target_contour_undistorted, obj_corners, debug=False
    )

    centroid = _centroid_of_contour(obj_contour_undistorted)
    if centroid is None:
        raise RuntimeError("Could not detect object centroid in base image.")

    robot_x, robot_y = converter.px_py_to_x_y(float(centroid[0]), float(centroid[1]))
    print(f"Centroid: {centroid}")
    print(f"Robot coordinates: {robot_x}, {robot_y}")
    robot_x = float(np.clip(robot_x, -0.08, 1.08))
    robot_y = float(np.clip(robot_y, -0.08, 1.5))

    robot.move_to_admin(robot_x, robot_y)
    time.sleep(3)
    robot.grip_open_close(config['push_reset_grip_hold'])
    time.sleep(0.5)

    drop_x = random.uniform(0.2, 0.8)
    drop_y = random.uniform(0.2, 0.8)
    robot.move_to(drop_x, drop_y)
    time.sleep(3)
    robot.rotate(int(random.uniform(0, 180)))
    time.sleep(1)
    robot.grip_open_close(1)
    time.sleep(0.5)

    away_x, away_y = _random_position_away_from(drop_x, drop_y)
    robot.move_to(away_x, away_y)
    time.sleep(3)

    robot.grip_open_close(0)
    time.sleep(0.5)
    robot.grip_up_down(0.1)
    robot.rotate(0)
    time.sleep(1)


def _execute_t_pushing_reset(robot):
    iou_evaluator = _build_iou_evaluator()
    converter = _build_hand_eye_converter()
    config = _get_robot_config()

    robot.grip_open_close(1)
    time.sleep(0.5)
    robot.grip_up_down(0.40)
    time.sleep(1)

    ret, frame, _ = robot.get_image_from_base()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture base camera image.")

    target_contour_undistorted = iou_evaluator.generate_target_contour(
        T_CORNERS, offset_from_center_mm=[50, -50]
    )

    (
        frame_score,
        vis_img,
        obj_contour_distorted,
        target_contour,
        intersection_contour_img_undistorted,
        obj_contour_undistorted,
        target_contour_undistorted,
        alignment_angle_deg,
        alignment_warp,
    ) = iou_evaluator.calculate_iou_from_distorted_base(
        frame, target_contour_undistorted, T_CORNERS, debug=False
    )

    pt1, pt2 = _t_long_end_points_undistorted(alignment_warp, iou_evaluator)
    print(f"frame_score = {frame_score}")
    print(f"pickup point (undistorted) = {pt2}")

    robot_x, robot_y = converter.px_py_to_x_y(pt2[0], pt2[1])
    print(f"T position in robot frame = {robot_x}, {robot_y}")
    robot_x = float(np.clip(robot_x, 0, 1))
    robot_y = float(np.clip(robot_y, 0, 1))

    robot.rotate(int(alignment_angle_deg))
    robot.move_to(robot_x, robot_y)
    time.sleep(3)

    robot.grip_up_down(config['t_reset_z'])
    time.sleep(1)
    robot.grip_open_close(config['t_reset_grip_hold'])
    time.sleep(0.5)
    robot.grip_up_down(0.40)
    time.sleep(0.5)

    drop_x = random.uniform(0.1, 0.9)
    drop_y = random.uniform(0.1, 0.9)
    robot.move_to(drop_x, drop_y)
    time.sleep(3)
    robot.rotate(int(random.uniform(0, 180)))
    time.sleep(1)
    robot.grip_open_close(1)
    time.sleep(0.5)

    away_x, away_y = _random_position_away_from(drop_x, drop_y)
    robot.move_to(away_x, away_y)
    time.sleep(3)

    robot.grip_open_close(0)
    time.sleep(0.5)
    robot.grip_up_down(0.1)
    robot.rotate(0)
    time.sleep(1)


def execute_pushing_reset(robot, object_type):
    """
    Locate the object via the base camera, pick it up, drop it at a random
    position with a random rotation, then park the gripper out of the way.
    """
    if object_type == 't_base':
        _execute_t_pushing_reset(robot)
    else:
        _execute_generic_pushing_reset(robot, object_type)


iou_evaluator = _build_iou_evaluator()
converter = _build_hand_eye_converter()
robot_id = socket.gethostname().replace("cr", "")
rope_segmenter = RopeSegmentationUtil(robot_id=robot_id)

def execute_rope_reset(robot):

    def _get_rope_points_with_retry(max_attempts=3):
        """Get rope points with retry logic - capture new image and retry if detection fails."""
        for attempt in range(max_attempts):
            ret, frame, _ = robot.get_image_from_base()
            if not ret or frame is None:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                    continue
                else:
                    raise RuntimeError("Failed to capture base camera image.")

            undistorted_frame = iou_evaluator.undistort_image(frame)
            points = rope_segmenter.get_rope_points(undistorted_frame, limit_rope_length=False)

            if points:
                return points, undistorted_frame
            else:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)

        raise RuntimeError("Failed to detect rope points after 3 attempts.")

    try:
        rope_points, _ = _get_rope_points_with_retry(max_attempts=3)
        xmin_rope, xmax_rope = detect_rope_base(converter, rope_points)
    except:
        xmin_rope, xmax_rope = 0.45, 0.55

    if xmin_rope < 0.3 or xmax_rope > 0.7:
        reset_rope(robot, 0.45, 0.55)

    reset_rope(robot, xmax_rope, xmin_rope)

    # Move robot to random position after resetting
    robot.move_to(0.5, 0.8, block=True)
    random_x = random.uniform(0.0, 1.0)
    random_y = random.uniform(0.85, 1.0)
    robot.move_to(random_x, random_y)
    robot.grip_open_close(0)


def reset_rope(robot, xmax_rope, xmin_rope):
    robot.grip_up_down(0.4)
    robot.move_to(0.625, 0, block=True)
    robot.grip_up_down(0)
    robot.grip_open_close(0)
    robot.move_to(xmax_rope, 0, block=True)
    robot.move_to(xmax_rope, 0.71, block=True)
    robot.grip_up_down(0.4)
    robot.move_to(0.275, 0, block=True)
    robot.grip_up_down(0)
    robot.move_to(xmin_rope, 0, block=True)
    robot.move_to(xmin_rope, 0.71, block=True)


def check_if_rope_is_straight(rope_points, threshold=0.6):
    if not rope_points or len(rope_points) < 2:
        return False

    converter = _build_hand_eye_converter()

    tip1_px = (rope_points[0][0], rope_points[0][1])
    tip2_px = (rope_points[-1][0], rope_points[-1][1])

    tip1 = converter.px_py_to_x_y(tip1_px[0], tip1_px[1])
    tip2 = converter.px_py_to_x_y(tip2_px[0], tip2_px[1])

    is_straight = (
        (tip1[1] >= threshold or tip2[1] >= threshold)
        and 0.4 <= tip1[0] <= 0.6
        and 0.4 <= tip2[0] <= 0.6
    )
    return is_straight

def detect_rope_base(converter, rope_points):
    if not rope_points:
        return None, None

    rope_base_x, rope_base_y = converter.px_py_to_x_y(rope_points[-1][0], rope_points[-1][1])

    xmin_rope = rope_base_x - 0.08
    xmax_rope = rope_base_x + 0.08

    return xmin_rope, xmax_rope

# def reset_helper(robot, converter, rope_points):
#     if not rope_points or len(rope_points) < 2:
#         print("Warning: Not enough rope points for reset helper")
#         return

#     robot.grip_open_close(0.9)
#     robot.grip_up_down(1)
#     time.sleep(0.25)

#     # Grab the free tip of the rope (rope_points[0] = lowest-x end = free end)
#     tip_px = rope_points[0]
#     tip_robot = converter.px_py_to_x_y(tip_px[0], tip_px[1])

#     tip_robot = np.clip(np.array(tip_robot), [0.0, 0.0], [1.0, 1.0])
#     robot.move_to(tip_robot[0], tip_robot[1])
#     time.sleep(1.7)

#     robot.grip_up_down(0.02)
#     time.sleep(0.25)
#     robot.grip_open_close(0.05)
#     time.sleep(0.25)

#     robot.move_to(0.5, 0.65)
#     time.sleep(1.7)
#     robot.grip_open_close(1)