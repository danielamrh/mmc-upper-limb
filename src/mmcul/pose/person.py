"""Choosing the subject when several people are visible.

Pose accuracy is evaluated separately from person detection: the subject's
bounding box comes from the dataset's 2D ground truth (mocap joints projected
into the camera), a common "oracle box" protocol. Both models see the same box.
"""

from __future__ import annotations

import numpy as np


def bbox_from_2d(j2d: np.ndarray, width: int, height: int, margin: float = 0.15) -> np.ndarray:
    """xyxy box around (K, 2) points, enlarged by `margin` of its size per side
    and clipped to the image."""
    lo, hi = np.nanmin(j2d, axis=0), np.nanmax(j2d, axis=0)
    pad = margin * (hi - lo)
    x0, y0 = lo - pad
    x1, y1 = hi + pad
    return np.array([max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1)], np.float32)


def closest_to_box(candidates_2d: list[np.ndarray], box: np.ndarray) -> int | None:
    """Index of the candidate (K, 2) whose keypoint centre is nearest the box
    centre and inside the box; None if no candidate qualifies."""
    c = 0.5 * (box[:2] + box[2:])
    best, best_d = None, np.inf
    for i, kp in enumerate(candidates_2d):
        m = np.nanmean(kp, axis=0)
        inside = box[0] <= m[0] <= box[2] and box[1] <= m[1] <= box[3]
        d = np.linalg.norm(m - c)
        if inside and d < best_d:
            best, best_d = i, d
    return best
