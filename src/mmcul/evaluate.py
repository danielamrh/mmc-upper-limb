"""Compare an estimated pose sequence with ground truth, repetition by repetition."""

from __future__ import annotations

import numpy as np

from .align import align_sequence
from .kinematics import Pose, elbow_flexion, shoulder_elevation
from .measures import compute_measures, lowpass

ARM_JOINTS = ("pelvis", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist")
MEASURES = ("movement_time", "n_movement_units", "peak_speed", "time_to_peak_speed",
            "trunk_displacement", "min_elbow_flexion", "max_shoulder_elevation")


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over non-finite frames of a (T, 3) trajectory
    (edges are held constant)."""
    ok = np.isfinite(x).all(1)
    if ok.all() or not ok.any():
        return x
    t = np.arange(len(x))
    return np.stack([np.interp(t, t[ok], x[ok, d]) for d in range(x.shape[1])], axis=1)


def _sl(pose: Pose, a: int, b: int) -> Pose:
    return {k: v[a:b] for k, v in pose.items()}


def evaluate_sequence(pred: Pose, gt: Pose, valid: np.ndarray, segments: list[tuple[int, int]],
                      fs: float, side: str = "r", align_scale: bool = True) -> list[dict]:
    """One row per segment (first, last frame inclusive) with ground-truth and
    estimated measures, joint-angle RMSE and aligned MPJPE.

    `pred` is aligned to `gt` with one similarity transform for the whole
    sequence (see mmcul.align); frames without a detection are interpolated
    and their share is reported as `valid_frac`.
    """
    joints = tuple(j for j in ARM_JOINTS if j in pred and j in gt)
    pred = {k: np.where(valid[:, None], v, np.nan) for k, v in pred.items() if k in gt}
    aligned = align_sequence(pred, gt, joints, with_scale=align_scale)
    aligned = {k: fill_gaps(v) for k, v in aligned.items()}

    rows = []
    for first, last in segments:
        a, b = first, last + 1
        row = {"first_frame": first, "last_frame": last, "valid_frac": float(valid[a:b].mean())}
        if not valid[a:b].any():
            rows.append(row)
            continue
        g, p = _sl(gt, a, b), _sl(aligned, a, b)
        mg = compute_measures(g, fs, side, trunk_point="shoulder_mid").as_dict()
        mp = compute_measures(p, fs, side, trunk_point="shoulder_mid").as_dict()
        row.update({f"gt_{k}": v for k, v in mg.items()})
        row.update({f"pred_{k}": v for k, v in mp.items()})

        gf = {k: lowpass(v, fs) for k, v in g.items()}
        pf = {k: lowpass(v, fs) for k, v in p.items()}
        for name, fn in (("elbow_flexion", elbow_flexion), ("shoulder_elevation", shoulder_elevation)):
            d = fn(pf, side) - fn(gf, side)
            row[f"rmse_{name}"] = float(np.sqrt(np.mean(d ** 2)))
            row[f"r_{name}"] = float(np.corrcoef(fn(pf, side), fn(gf, side))[0, 1])
        err = np.stack([np.linalg.norm(p[j] - g[j], axis=1) for j in joints])
        row["mpjpe_mm"] = float(1000 * err.mean())
        rows.append(row)
    return rows
