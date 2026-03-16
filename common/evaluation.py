import os
import json
import time
import random
import socket
import threading
import numpy as np
from common.reset_policies import (
    _build_iou_evaluator,
    _build_hand_eye_converter,
    _centroid_of_contour,
    OBJECT_CORNERS,
    execute_pushing_reset,
)

MAX_DURATION = 180
SAMPLE_INTERVAL = 1.0
MAX_INITIAL_IOU = 15.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_LOGS_DIR = os.path.join(BASE_DIR, "eval_logs")
os.makedirs(EVAL_LOGS_DIR, exist_ok=True)


def _contour_to_points(contour):
    if contour is None or len(contour) == 0:
        return []
    pts = np.array(contour).reshape(-1, 2)
    return [{"x": round(float(p[0]), 1), "y": round(float(p[1]), 1)} for p in pts]


class EvaluationManager:
    def __init__(self, robot):
        self.robot = robot
        self._lock = threading.Lock()

        self.is_evaluating = False
        self.is_resetting = False
        self.start_time = None
        self.current_iou = 0.0
        self.history = []       # list of {"t": float, "iou": float}
        self.final_score = 0.0

        self._task_env = None
        self._object_type = None
        self._obj_corners = None
        self._iou_evaluator = None
        self._target_contour = None
        self._target_points = None

        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        with self._lock:
            if self.is_evaluating:
                return None, "Evaluation already in progress"
            if self.is_resetting:
                return None, "Environment is resetting, please wait"

            self._task_env = os.environ.get("RGMC_TASK_ENV")
            self._object_type = os.environ.get("RGMC_OBJECT_TYPE")

            if self._task_env != "planar_pushing":
                return None, f"Unsupported task environment: {self._task_env}"
            if self._object_type not in OBJECT_CORNERS:
                return None, f"Invalid object type: {self._object_type}"

            self._obj_corners = OBJECT_CORNERS[self._object_type]
            self._iou_evaluator = _build_iou_evaluator()

        target_contour, target_points = self._generate_target()

        with self._lock:
            if target_contour is None:
                return None, "Failed to generate target: could not detect object"

            self._target_contour = target_contour
            self._target_points = target_points
            self.start_time = time.time()
            self.current_iou = 0.0
            self.history = []
            self.final_score = 0.0
            self.is_evaluating = True

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._iou_loop, daemon=True)
            self._thread.start()

            return {
                "status": "Evaluation initialized",
                "max_duration_seconds": MAX_DURATION,
                "system_time_start": self.start_time,
            }, None

    def get_target(self):
        with self._lock:
            if self._target_points is None:
                return None, "No target available — start an evaluation first"

            obj_name = (self._object_type or "unknown").replace("_base", "")

            return {
                "task": self._task_env,
                "target_object": obj_name,
                "coordinate_space": "undistorted_pixel_2d",
                "geometry": {
                    "type": "polygon",
                    "points": self._target_points,
                },
            }, None

    def get_status(self):
        with self._lock:
            if not self.is_evaluating:
                if self.is_resetting:
                    return {
                        "status": "resetting",
                        "message": "Environment is resetting.",
                    }, None
                if self.start_time is not None:
                    return {
                        "status": "completed",
                        "final_score": round(self.final_score, 2),
                        "iou_history": self.history,
                        "message": "Evaluation complete. Call /eval/start for a new run.",
                    }, None
                return None, "No evaluation has been started"

            elapsed = time.time() - self.start_time
            if elapsed >= MAX_DURATION:
                self._finish()
                return {
                    "status": "completed",
                    "final_score": round(self.final_score, 2),
                    "iou_history": self.history,
                    "message": "Environment resetting.",
                }, None

            return {
                "status": "running",
                "time_elapsed": round(elapsed, 1),
                "time_remaining": round(MAX_DURATION - elapsed, 1),
                "current_iou": round(self.current_iou, 1),
            }, None

    def _generate_target(self):
        """Pick a random target placement that:
          - has a random orientation
          - has its centroid within [0.2, 0.8] in normalized robot space
          - has IoU with the current object below MAX_INITIAL_IOU
        Retries up to 30 times before falling back to a fixed offset.
        """
        ret, frame, _ = self.robot.get_image_from_base()
        if not ret or frame is None:
            return None, None

        converter = _build_hand_eye_converter()

        for _ in range(30):
            offset_x = random.uniform(-80, 80)
            offset_y = random.uniform(-80, 80)
            rotation_rad = random.uniform(0, 2 * np.pi)

            candidate = self._iou_evaluator.generate_target_contour(
                self._obj_corners,
                offset_from_center_mm=[offset_x, offset_y],
                rotation_angle_rad=rotation_rad,
            )

            # Reject if centroid falls outside the safe robot workspace
            centroid = _centroid_of_contour(candidate)
            if centroid is None:
                continue
            rx, ry = converter.px_py_to_x_y(float(centroid[0]), float(centroid[1]))
            if not (0.2 <= rx <= 0.8 and 0.2 <= ry <= 0.8):
                continue

            try:
                result = self._iou_evaluator.calculate_iou_from_distorted_base(
                    frame, candidate, self._obj_corners, debug=False
                )
            except Exception:
                continue

            score = result[0]
            target_contour_undistorted = result[6]
            if score < MAX_INITIAL_IOU:
                return candidate, _contour_to_points(target_contour_undistorted)

        # Fallback: fixed offset, no rotation
        candidate = self._iou_evaluator.generate_target_contour(
            self._obj_corners, offset_from_center_mm=[70, -70], rotation_angle_rad=0
        )
        try:
            result = self._iou_evaluator.calculate_iou_from_distorted_base(
                frame, candidate, self._obj_corners, debug=False
            )
            points = _contour_to_points(result[6])
        except Exception:
            points = _contour_to_points(candidate)
        return candidate, points


    def _iou_loop(self):
        while not self._stop_event.is_set():
            if time.time() - self.start_time >= MAX_DURATION:
                with self._lock:
                    if self.is_evaluating:
                        self._finish()
                return

            try:
                ret, frame, _ = self.robot.get_image_from_base()
                if not ret or frame is None:
                    self._stop_event.wait(SAMPLE_INTERVAL)
                    continue

                result = self._iou_evaluator.calculate_iou_from_distorted_base(
                    frame, self._target_contour, self._obj_corners, debug=False
                )
                iou = float(result[0])
                t = round(time.time() - self.start_time, 2)
                with self._lock:
                    self.current_iou = iou
                    self.history.append({"t": t, "iou": round(iou, 2)})
            except Exception as e:
                print(f"IoU background error: {e}")

            self._stop_event.wait(SAMPLE_INTERVAL)

    def _finish(self):
        """Compute integral score, flip state, and kick off environment reset.
        Caller must hold self._lock.  Idempotent — safe to call more than once."""
        if not self.is_evaluating:
            return
        self.is_evaluating = False
        self._stop_event.set()

        if len(self.history) >= 2:
            ts = np.array([p["t"] for p in self.history])
            ious = np.array([p["iou"] for p in self.history])
            self.final_score = float(np.trapz(ious, ts))
        else:
            self.final_score = 0.0

        self._save_run()

        self.is_resetting = True
        threading.Thread(target=self._reset, daemon=True).start()

    def _save_run(self):
        """Persist the completed run's score and IoU history to a JSON file.
        Caller must hold self._lock."""
        try:
            robot_id = socket.gethostname()
            start_str = time.strftime("%Y%m%dT%H%M%S", time.localtime(self.start_time))
            filename = f"{robot_id}_{self._object_type}_{start_str}.json"
            filepath = os.path.join(EVAL_LOGS_DIR, filename)

            payload = {
                "robot_id": robot_id,
                "task_env": self._task_env,
                "object_type": self._object_type,
                "start_time": self.start_time,
                "duration_seconds": round(self.history[-1]["t"], 2) if self.history else 0,
                "final_score": round(self.final_score, 2),
                "iou_history": self.history,
            }

            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2)

            print(f"Evaluation log saved: {filepath}")
        except Exception as e:
            print(f"Failed to save evaluation log: {e}")

    def _reset(self):
        try:
            execute_pushing_reset(self.robot, self._object_type)
        except Exception as e:
            print(f"Post-evaluation reset failed: {e}")
        finally:
            with self._lock:
                self.is_resetting = False
