"""Choose FitConfig hyperparameters on held-out tuning subjects only.

    python tools/tune_fit.py --root data/rehab24 --cache cache/rehab24 --model mediapipe-heavy

Fits every grid configuration on the tuning persons' Ex1/Ex2 videos (both
cameras) and prints per-configuration agreement with mocap. The persons used
here must be excluded from the reported benchmark (mmcul.benchmark.TUNING_PERSONS).
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmcul.benchmark import TUNING_PERSONS  # noqa: E402
from mmcul.datasets import rehab24  # noqa: E402
from mmcul.evaluate import evaluate_sequence  # noqa: E402
from mmcul.fit import FitConfig, fit_sequence  # noqa: E402
from mmcul.pose import canonical_conf, canonical_pose  # noqa: E402
from mmcul.pose.cache import PoseSeq  # noqa: E402

GRID = {"smooth": [1e-5, 1e-4, 1e-3], "depth_weight": [0.3, 1.0]}


def job(args):
    path, root, ex, vid, cam, segs, meta, cfg = args
    import torch
    torch.set_num_threads(2)
    seq = PoseSeq.load(path)
    gt = rehab24.to_pose(rehab24.load_joints(rehab24.joints_path(root, ex, vid)))
    n = min(len(seq), len(gt["pelvis"]))
    gt = {k: v[:n] for k, v in gt.items()}
    obs = {k: v[:n] for k, v in canonical_pose(seq).items()}
    conf = {k: v[:n] for k, v in canonical_conf(seq).items()}
    rows = []
    if cfg is None:
        res_pose, valid = obs, seq.valid[:n]
    else:
        res_pose, valid = fit_sequence(obs, seq.fps, conf, seq.valid[:n], cfg).pose, np.ones(n, bool)
    for r in evaluate_sequence(res_pose, gt, valid, segs, seq.fps):
        rows.append({**meta, "camera": cam, **r})
    return cfg, rows


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    reps = [r for r in rehab24.read_segmentation(a.root / "Segmentation.csv")
            if r.exercise in (1, 2) and r.person in TUNING_PERSONS and not r.mocap_erroneous]
    videos = {}
    for r in reps:
        videos.setdefault((r.exercise, r.video_id), []).append(r)
    configs = [None] + [replace(FitConfig(), **dict(zip(GRID, v))) for v in itertools.product(*GRID.values())]
    jobs = []
    for (ex, vid), vreps in videos.items():
        for cam in (17, 18):
            segs = [(r.first_frame, r.last_frame) for r in vreps]
            meta = {"exercise": ex, "orientation": vreps[0].orientation(cam)}
            for cfg in configs:
                jobs.append((a.cache / a.model / f"Ex{ex}" / f"{vid}-c{cam}.npz", a.root, ex, vid, cam,
                             segs, meta, cfg))
    print(f"{len(jobs)} fits on persons {TUNING_PERSONS}", flush=True)
    rows = []
    with mp.Pool(a.workers) as pool:
        for cfg, rs in pool.imap_unordered(job, jobs):
            label = "raw" if cfg is None else f"smooth={cfg.smooth:g} dw={cfg.depth_weight:g}"
            rows += [{**r, "config": label} for r in rs]
    df = pd.DataFrame(rows)
    for m in ("n_movement_units", "peak_speed", "max_shoulder_elevation", "min_elbow_flexion", "trunk_displacement"):
        df[f"ae_{m}"] = (df[f"pred_{m}"] - df[f"gt_{m}"]).abs()
    cols = ["mpjpe_mm", "rmse_elbow_flexion", "rmse_shoulder_elevation", "ae_n_movement_units", "ae_peak_speed",
            "ae_max_shoulder_elevation", "ae_min_elbow_flexion", "ae_trunk_displacement"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(df.groupby("config")[cols].median().round(3))
    print(df[df.orientation != "profile"].groupby("config")[cols].median().round(3))
    df.to_csv(a.cache / f"tune_fit_{a.model}.csv", index=False)


if __name__ == "__main__":
    main()
