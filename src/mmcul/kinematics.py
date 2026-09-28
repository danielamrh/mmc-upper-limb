"""Joint angles of the upper body from 3D joint positions.

A pose is a dict mapping canonical joint names to arrays of shape (T, 3) in
metres. Every data source (mocap ground truth, pose models, the biomechanical
fit) is converted to these names so that all measures are computed by the same
code:

    pelvis, thorax, neck, head,
    r_shoulder, r_elbow, r_wrist, r_hand,
    l_shoulder, l_elbow, l_wrist, l_hand

`*_shoulder` is the glenohumeral joint centre, `*_hand` the end effector.
"""

from __future__ import annotations

import numpy as np

Pose = dict[str, np.ndarray]

JOINTS = (
    "pelvis", "thorax", "neck", "head",
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",
)

_EPS = 1e-9


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), _EPS)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between vectors a and b (broadcast over leading axes)."""
    cos = np.sum(_unit(a) * _unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def trunk_frame(pose: Pose) -> np.ndarray:
    """Thorax frame as rotation matrices of shape (T, 3, 3).

    Columns follow the ISB convention: x anterior, y superior (pelvis -> shoulder
    midpoint), z to the subject's right (left -> right shoulder, made orthogonal
    to y).
    """
    mid = 0.5 * (pose["r_shoulder"] + pose["l_shoulder"])
    y = _unit(mid - pose["pelvis"])
    right = pose["r_shoulder"] - pose["l_shoulder"]
    z = _unit(right - np.sum(right * y, axis=-1, keepdims=True) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def elbow_flexion(pose: Pose, side: str = "r") -> np.ndarray:
    """Elbow flexion in degrees: 0 = fully extended, 90 = right angle."""
    upper = pose[f"{side}_shoulder"] - pose[f"{side}_elbow"]
    fore = pose[f"{side}_wrist"] - pose[f"{side}_elbow"]
    return 180.0 - angle_between(upper, fore)


def shoulder_elevation(pose: Pose, side: str = "r") -> np.ndarray:
    """Humerothoracic elevation in degrees: 0 = arm hanging along the trunk,
    90 = upper arm horizontal (relative to the trunk, not the world)."""
    humerus = pose[f"{side}_elbow"] - pose[f"{side}_shoulder"]
    down = -trunk_frame(pose)[..., 1]
    return angle_between(humerus, down)


def trunk_tilt(pose: Pose, world_up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Angle in degrees between the trunk axis and the world vertical."""
    return angle_between(trunk_frame(pose)[..., 1], np.broadcast_to(world_up, pose["pelvis"].shape))
