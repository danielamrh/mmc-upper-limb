"""Aligning estimated poses to the ground-truth coordinate frame.

Monocular models predict in their own frame (camera or hip-centred), so
positions and speeds are only comparable after alignment. One similarity
transform per *video* is fitted, never per frame: per-frame alignment would
remove exactly the trunk and hand motion the measures are about.
Note that fitting the scale uses the ground truth (an oracle scale); joint
angles need no alignment at all.
"""

from __future__ import annotations

import numpy as np

from .kinematics import Pose


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with dst ≈ s * R @ src + t for (N, 3) point sets
    (Umeyama 1991); R is a proper rotation."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(0).sum()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def align_sequence(pred: Pose, gt: Pose, joints: tuple[str, ...] | None = None,
                   with_scale: bool = True) -> Pose:
    """Apply one similarity transform, fitted on all frames and joints where both
    poses are finite, to every joint of `pred`."""
    joints = joints or tuple(k for k in pred if k in gt)
    P = np.concatenate([pred[j] for j in joints])
    G = np.concatenate([gt[j] for j in joints])
    ok = np.isfinite(P).all(1) & np.isfinite(G).all(1)
    s, R, t = umeyama(P[ok], G[ok], with_scale)
    return {k: s * v @ R.T + t for k, v in pred.items()}
