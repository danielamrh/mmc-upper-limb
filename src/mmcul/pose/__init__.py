"""Pose model runners and their on-disk cache format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..kinematics import Pose
from .cache import PoseSeq, to_pose

MODELS = ("mediapipe-heavy", "mediapipe-full",
          "sam3db-dinov3", "sam3db-vith", "sam3db-dinov3-fp16", "sam3db-vith-fp16")


def _mapping(seq: PoseSeq) -> dict:
    if seq.model.endswith("+fit"):  # biomechanical fit output is stored in canonical joints
        return {n: n for n in seq.names}
    if seq.model.startswith("mediapipe"):
        from .mediapipe_runner import CANONICAL
    elif seq.model.startswith("sam3db"):
        from .sam3db_runner import CANONICAL
    else:
        raise ValueError(f"unknown model {seq.model}")
    return CANONICAL


def canonical_pose(seq: PoseSeq) -> Pose:
    """Canonical joints from a cached sequence of any supported model."""
    return to_pose(seq, _mapping(seq))


def canonical_conf(seq: PoseSeq) -> dict:
    """Per-joint confidence (T,) for the canonical joints (mean over source keypoints)."""
    out = {}
    for canon, src in _mapping(seq).items():
        names = (src,) if isinstance(src, str) else tuple(src)
        out[canon] = np.mean([seq.conf[:, seq.names.index(n)] for n in names], axis=0)
    return out


def make_runner(name: str, model_dir: str | Path = "models"):
    """Runner with a .run(video, start, stop, boxes, fps) -> PoseSeq method."""
    if name.startswith("mediapipe-"):
        from .mediapipe_runner import MediaPipeRunner, model_path
        return MediaPipeRunner(model_path(model_dir, name.removeprefix("mediapipe-")))
    if name.startswith("sam3db-"):
        from .sam3db_runner import SAM3DBodyRunner
        backbone, _, precision = name.removeprefix("sam3db-").partition("-")
        return SAM3DBodyRunner(f"facebook/sam-3d-body-{backbone}", fp16=precision == "fp16")
    raise ValueError(f"unknown model {name}, choose from {MODELS}")
