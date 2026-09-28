"""Pose model runners and their on-disk cache format."""

from __future__ import annotations

from pathlib import Path

from ..kinematics import Pose
from .cache import PoseSeq, to_pose

MODELS = ("mediapipe-heavy", "mediapipe-full", "sam3db-dinov3", "sam3db-vith")


def canonical_pose(seq: PoseSeq) -> Pose:
    """Canonical joints from a cached sequence of any supported model."""
    if seq.model.startswith("mediapipe"):
        from .mediapipe_runner import CANONICAL
    elif seq.model.startswith("sam3db"):
        from .sam3db_runner import CANONICAL
    else:
        raise ValueError(f"unknown model {seq.model}")
    return to_pose(seq, CANONICAL)


def make_runner(name: str, model_dir: str | Path = "models"):
    """Runner with a .run(video, start, stop, boxes, fps) -> PoseSeq method."""
    if name.startswith("mediapipe-"):
        from .mediapipe_runner import MediaPipeRunner, model_path
        return MediaPipeRunner(model_path(model_dir, name.removeprefix("mediapipe-")))
    if name.startswith("sam3db-"):
        from .sam3db_runner import SAM3DBodyRunner
        return SAM3DBodyRunner(f"facebook/sam-3d-body-{name.removeprefix('sam3db-')}")
    raise ValueError(f"unknown model {name}, choose from {MODELS}")
