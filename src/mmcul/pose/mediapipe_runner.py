"""MediaPipe Pose Landmarker (Tasks API) — the baseline of most clinical MMC papers.

3D output are MediaPipe's "world landmarks": metric, origin at the hip centre,
so global translation is lost (frame = "root").
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ..video import iter_frames
from .cache import PoseSeq, empty
from .person import closest_to_box

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")

NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)

# the trunk point is the shoulder midpoint for every source, see mmcul.kinematics
CANONICAL = {
    "pelvis": ("left_hip", "right_hip"),
    "thorax": ("left_shoulder", "right_shoulder"),
    "neck": ("left_shoulder", "right_shoulder"),
    "head": "nose",
    "r_shoulder": "right_shoulder", "r_elbow": "right_elbow", "r_wrist": "right_wrist",
    "r_hand": ("right_index", "right_pinky"),
    "l_shoulder": "left_shoulder", "l_elbow": "left_elbow", "l_wrist": "left_wrist",
    "l_hand": ("left_index", "left_pinky"),
}


def model_path(cache_dir: str | Path, variant: str = "heavy") -> Path:
    p = Path(cache_dir) / f"pose_landmarker_{variant}.task"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL.format(variant=variant), p)
    return p


class MediaPipeRunner:
    """Runs one video at a time in VIDEO mode (temporal tracking between frames).

    Tracking state lives in the landmarker, so a chunk always starts a fresh
    landmarker; chunks should therefore be long (the default 600 frames = 20 s).
    """

    def __init__(self, model_file: str | Path, num_poses: int = 2):
        self.model_file = str(model_file)
        self.num_poses = num_poses
        self.name = f"mediapipe-{Path(model_file).stem.removeprefix('pose_landmarker_')}"

    def _landmarker(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_file),
            running_mode=vision.RunningMode.VIDEO, num_poses=self.num_poses)
        return mp, vision.PoseLandmarker.create_from_options(opts)

    def run(self, video: str | Path, start: int, stop: int, boxes: np.ndarray, fps: float) -> PoseSeq:
        """boxes: (stop - start, 4) xyxy subject boxes used to pick the right person."""
        seq = empty(stop - start, NAMES, "root", self.name, fps)
        mp, lm = self._landmarker()
        with lm:
            for i, rgb in iter_frames(video, start, stop):
                h, w = rgb.shape[:2]
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(round(1000 * i / fps)))
                if not res.pose_landmarks:
                    continue
                cands = [np.array([[p.x * w, p.y * h] for p in lms]) for lms in res.pose_landmarks]
                k = closest_to_box(cands, boxes[i - start])
                if k is None:
                    continue
                t = i - start
                seq.kp2d[t] = cands[k]
                seq.kp3d[t] = [[p.x, p.y, p.z] for p in res.pose_world_landmarks[k]]
                seq.conf[t] = [p.visibility for p in res.pose_landmarks[k]]
                seq.valid[t] = True
        return seq
