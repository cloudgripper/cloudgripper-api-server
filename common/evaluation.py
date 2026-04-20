import os
import json
import time
import random
import socket
import threading
from pathlib import Path
import numpy as np
import gevent
from gevent.threadpool import ThreadPool as _GeventThreadPool
from common.reset_policies import (
    _build_iou_evaluator,
    _build_hand_eye_converter,
    _centroid_of_contour,
    OBJECT_CORNERS,
    execute_pushing_reset,
    execute_rope_reset,
)
from common.rope_segmentation_util import RopeSegmentationUtil
from common.rope_pca_sampler import RopePCASampler

MAX_DURATION = 180
SAMPLE_INTERVAL = 1.0
MAX_INITIAL_IOU = 15.0

_SCORE_POOL = _GeventThreadPool(maxsize=2)

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
        self.current_iou = 0.0    # Used for planar_pushing
        self.current_score = 0.0  # Used for deformable_linear
        self.history = []         # list of {"t": float, "iou"/"score": float}
        self.final_score = 0.0

        self._task_env = None
        self._object_type = None
        self._obj_corners = None
        self._rope_points = None
        self._iou_evaluator = None
        self._rope_evaluator = None
        self._target_contour = None
        self._target_points = None
        self._target_rope_points = None

        self._thread = None
        self._score_async = None
        self._stop_event = threading.Event()

        self._rope_segmenter = RopeSegmentationUtil(robot_id=socket.gethostname().replace("cr", ""))
        self._iou_evaluator = _build_iou_evaluator()

        DATASET_DIR = Path(BASE_DIR) / "datasets"
        try:
            self._rope_pca_sampler = RopePCASampler(robot_name=socket.gethostname().replace("cr", "robot"), dataset_dir=DATASET_DIR)
        except (FileNotFoundError, ValueError) as e:
            print(f"RopePCASampler unavailable (deformable_linear task will be disabled): {e}")
            self._rope_pca_sampler = None
        
    def start(self):
        with self._lock:
            if self.is_evaluating:
                return None, "Evaluation already in progress"
            if self.is_resetting:
                return None, "Environment is resetting, please wait"

            self._task_env = os.environ.get("RGMC_TASK_ENV")
            self._object_type = os.environ.get("RGMC_OBJECT_TYPE")

            if self._task_env == "planar_pushing":
                if self._object_type not in OBJECT_CORNERS:
                    return None, f"Invalid object type: {self._object_type}"
                self._obj_corners = OBJECT_CORNERS[self._object_type]

                target_contour, target_points = self._generate_target()


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

                self._score_async = _SCORE_POOL.spawn(self._score_loop)
                self._thread = None

                gevent.spawn_later(MAX_DURATION + 0.5, self._finish_if_expired)

                return {
                    "status": "Evaluation initialized",
                    "max_duration_seconds": MAX_DURATION,
                    "system_time_start": self.start_time,
                }, None


            elif self._task_env == "deformable_linear":
                self._rope_points = None
                self._rope_evaluator = None

                target_rope_points = self._generate_target()

                if target_rope_points is None:
                    return None, "Failed to generate target: could not detect rope"

                self._target_rope_points = target_rope_points
                self.start_time = time.time()
                self.current_score = 0.0
                self.history = []
                self.final_score = 0.0
                self.is_evaluating = True

                self._stop_event.clear()
                self._score_async = _SCORE_POOL.spawn(self._score_loop)
                self._thread = None
                gevent.spawn_later(MAX_DURATION + 0.5, self._finish_if_expired)

                return {
                    "status": "Evaluation initialized",
                    "max_duration_seconds": MAX_DURATION,
                    "system_time_start": self.start_time,
                }, None
            else:
                return None, f"Unsupported task environment: {self._task_env}"

    def get_target(self):
        with self._lock:
            if self._task_env == "deformable_linear":
                if self._target_rope_points is None:
                    return None, "No target available — start an evaluation first"
                return {
                    "task": self._task_env,
                    "target_object": "rope",
                    "coordinate_space": "undistorted_pixel_2d",
                    "geometry": {
                        "type": "segmented_points",
                        "points": self._target_rope_points.tolist() if isinstance(self._target_rope_points, np.ndarray) else self._target_rope_points,
                    },
                }, None
            else:
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

    def get_object(self):
        task_env = os.environ.get("RGMC_TASK_ENV")
        object_type = os.environ.get("RGMC_OBJECT_TYPE")

        ret, frame, _ = self.robot.get_image_from_base()
        undistorted_frame = self._iou_evaluator.undistort_image(frame)

        if not ret or frame is None:
            return None, "Failed to capture base camera image"

        if task_env == "deformable_linear":
            try:
                rope_points = self._rope_segmenter.get_rope_points(undistorted_frame)
                if not rope_points:
                    return None, "Rope detection failed: no rope points found"
            except Exception as e:
                return None, f"Rope detection failed: {e}"

            return {
                "object": "rope",
                "coordinate_space": "undistorted_pixel_2d",
                "geometry": {
                    "type": "segmented_points",
                    "points": rope_points,
                },
            }, None

        if object_type not in OBJECT_CORNERS:
            return None, f"Invalid or missing object type: {object_type}"

        obj_corners = OBJECT_CORNERS[object_type]

        try:
            dummy_target = self._iou_evaluator.generate_target_contour(
                obj_corners, offset_from_center_mm=[0, 0], rotation_angle_rad=0
            )
            (
                _,
                _,
                _,
                _,
                _,
                obj_contour_undistorted,
                *_,
            ) = self._iou_evaluator.calculate_iou_from_distorted_base(
                frame, dummy_target, obj_corners, debug=False
            )
        except Exception as e:
            return None, f"Object detection failed: {e}"

        obj_name = object_type.replace("_base", "")
        return {
            "object": obj_name,
            "coordinate_space": "undistorted_pixel_2d",
            "geometry": {
                "type": "polygon",
                "points": _contour_to_points(obj_contour_undistorted),
            },
        }, None

    def get_status(self):
        with self._lock:
            score_key = "current_iou" if self._task_env == "planar_pushing" else "current_score"
            history_key = "iou_history" if self._task_env == "planar_pushing" else "score_history"

            if not self.is_evaluating:
                if self.is_resetting:
                    return {
                        "status": "resetting",
                        "message": "Environment is resetting.",
                    }, None
                if self.start_time is not None:
                    result = {
                        "status": "completed",
                        "final_score": round(self.final_score, 2),
                        "message": "Evaluation complete. Call /eval/start for a new run.",
                    }
                    result[history_key] = self.history
                    return result, None
                return None, "No evaluation has been started"

            elapsed = time.time() - self.start_time
            if elapsed >= MAX_DURATION:
                self._finish()
                result = {
                    "status": "completed",
                    "final_score": round(self.final_score, 2),
                    "message": "Environment resetting.",
                }
                result[history_key] = self.history
                return result, None

            current_value = self.current_iou if self._task_env == "planar_pushing" else self.current_score
            result = {
                "status": "running",
                "time_elapsed": round(elapsed, 1),
                "time_remaining": round(MAX_DURATION - elapsed, 1),
            }
            result[score_key] = round(current_value, 1)
            return result, None

    def _generate_target(self):
        
        ret, frame, _ = self.robot.get_image_from_base()
        undistorted_frame = self._iou_evaluator.undistort_image(frame)

        if not ret or frame is None:
            return None, None

        if self._task_env == "planar_pushing":
            return self._generate_target_planar_pushing(frame)
        elif self._task_env == "deformable_linear":
            return self._generate_target_deformable_linear(undistorted_frame)

    def _generate_target_planar_pushing(self, frame):
        """Generate target for planar_pushing task environment.

            Pick a random target placement that:
            - has a random orientation
            - has its centroid within [0.2, 0.8] in normalized robot space
            - has IoU with the current object below MAX_INITIAL_IOU
            Retries up to 30 times before falling back to a fixed offset.
        """
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

    def _generate_target_deformable_linear(self, frame):
        if self._rope_pca_sampler is None:
            return None
        rope_points, _ = self._rope_pca_sampler.sample_y_gt_zero()
        if rope_points is None or len(rope_points) == 0:
            return None
        return rope_points

    def _score_loop(self):
        """Background loop for calculating and tracking score during evaluation."""
        while not self._stop_event.is_set():
            if time.time() - self.start_time >= MAX_DURATION:
                return

            try:
                ret, frame, _ = self.robot.get_image_from_base()
                undistorted_frame = self._iou_evaluator.undistort_image(frame)

                if not ret or frame is None:
                    self._stop_event.wait(SAMPLE_INTERVAL)
                    continue

                if self._task_env == "planar_pushing":
                    result = self._iou_evaluator.calculate_iou_from_distorted_base(
                        frame, self._target_contour, self._obj_corners, debug=False
                    )
                    score = float(result[0])
                    with self._lock:
                        self.current_iou = score
                        self.history.append({"t": round(time.time() - self.start_time, 2), "iou": round(score, 2)})
                elif self._task_env == "deformable_linear":
                    self._rope_points = self._rope_segmenter.get_rope_points(undistorted_frame)
                    if self._rope_points is not None and len(self._rope_points) > 0:
                        score = self._calculate_rope_score(self._rope_points, self._target_rope_points)
                    else:
                        score = 0.0
                    with self._lock:
                        self.current_score = score
                        self.history.append({"t": round(time.time() - self.start_time, 2), "score": round(score, 2)})
                else:
                    score = 0.0
                    with self._lock:
                        self.current_iou = score
                        self.history.append({"t": round(time.time() - self.start_time, 2), "iou": round(score, 2)})

            except Exception as e:
                print(f"Score background error: {e}")

            self._stop_event.wait(SAMPLE_INTERVAL)

    def _calculate_rope_score(self, current_points, target_points):
        """Calculate rope alignment score using RMSE.

        Score = max(0, 1 - E / 220), where E is the RMSE of the aligned rope points.
        """
        if current_points is None or target_points is None:
            return 0.0
        if len(current_points) == 0 or len(target_points) == 0:
            return 0.0

        current = np.array(current_points)
        target = np.array(target_points)

        n_current = len(current)
        n_target = len(target)

        if n_current == 0 or n_target == 0:
            return 0.0

        n_points = min(n_current, n_target)
        current = current[:n_points]
        target = target[:n_points]

        mse = np.mean(np.sum((current - target) ** 2, axis=1))
        rmse = np.sqrt(mse)

        score = max(0.0, 1.0 - rmse / 220.0)
        return float(score)

    def _finish_if_expired(self):
        """Hub-side fallback finaliser. Called by a gevent.spawn_later timer
        scheduled in start(), so it always runs on the main hub. Idempotent:
        does nothing if the eval has already been finalised by get_status()."""
        with self._lock:
            if self.is_evaluating and self.start_time is not None and \
                    (time.time() - self.start_time) >= MAX_DURATION:
                self._finish()

    def _finish(self):
        """Compute final score, flip state, and kick off environment reset.
        Caller must hold self._lock.  Idempotent — safe to call more than once."""
        if not self.is_evaluating:
            return
        self.is_evaluating = False
        self._stop_event.set()

        if len(self.history) >= 1:
            if self._task_env == "planar_pushing":
                # Integral of IoU over time for pushing task
                ts = np.array([p["t"] for p in self.history])
                scores = np.array([p.get("iou", 0) for p in self.history])
                self.final_score = float(np.trapz(scores, ts))
            else:
                # Latest score for rope task (deformable_linear)
                last_entry = self.history[-1]
                self.final_score = float(last_entry["score"])
        else:
            self.final_score = 0.0

        self._save_run()

        # self.is_resetting = True

        # threading.Thread(target=self._reset, daemon=True).start()

    def _save_run(self):
        """Persist the completed run's score and history to a JSON file.
        Caller must hold self._lock."""
        try:
            robot_id = socket.gethostname()
            start_str = time.strftime("%Y%m%dT%H%M%S", time.localtime(self.start_time))
            filename = f"{robot_id}_{self._object_type}_{start_str}.json"
            filepath = os.path.join(EVAL_LOGS_DIR, filename)

            history_key = "iou_history" if self._task_env == "planar_pushing" else "score_history"
            payload = {
                "robot_id": robot_id,
                "task_env": self._task_env,
                "object_type": self._object_type,
                "start_time": self.start_time,
                "duration_seconds": round(self.history[-1]["t"], 2) if self.history else 0,
                "final_score": round(self.final_score, 2),
                history_key: self.history,
            }

            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2)

            print(f"Evaluation log saved: {filepath}")
        except Exception as e:
            print(f"Failed to save evaluation log: {e}")

    def _reset(self):
        try:
            if self._task_env == "planar_pushing":
                execute_pushing_reset(self.robot, self._object_type)
            elif self._task_env == "deformable_linear":
                execute_rope_reset(self.robot)
        except Exception as e:
            print(f"Post-evaluation reset failed: {e}")
        finally:
            with self._lock:
                self.is_resetting = False
