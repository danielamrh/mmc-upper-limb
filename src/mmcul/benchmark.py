"""Phase 1 benchmark: cached pose-model output vs. REHAB24-6 ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .datasets import rehab24
from .evaluate import MEASURES, evaluate_sequence
from .pose import canonical_pose
from .pose.cache import PoseSeq

FPS = 30.0
# subjects used to choose the fit hyperparameters (tools/tune_fit.py); excluded from reported results
TUNING_PERSONS = (4, 5)


def evaluate_model(root: str | Path, cache: str | Path, model: str, exercises=(1, 2),
                   cameras=(17, 18), align_scale: bool = True, exclude_persons=()) -> list[dict]:
    """One row per (repetition, camera) with metadata, ground-truth and
    estimated measures and pose errors. Videos without a cache file are skipped
    (and listed in the `missing` attribute of the returned list). Pass
    exclude_persons=TUNING_PERSONS for reported results."""
    root, cache = Path(root), Path(cache)
    reps = [r for r in rehab24.read_segmentation(root / "Segmentation.csv")
            if r.exercise in exercises and not r.mocap_erroneous and r.person not in exclude_persons]
    by_video: dict[tuple[int, str], list] = {}
    for r in reps:
        by_video.setdefault((r.exercise, r.video_id), []).append(r)

    rows, missing = [], []
    for (ex, vid), vreps in sorted(by_video.items()):
        gt = None
        for cam in cameras:
            path = cache / model / f"Ex{ex}" / f"{vid}-c{cam}.npz"
            if not path.exists():
                missing.append(path)
                continue
            if gt is None:
                gt = rehab24.to_pose(rehab24.load_joints(rehab24.joints_path(root, ex, vid)))
            seq = PoseSeq.load(path)
            n = min(len(seq), len(gt["pelvis"]))
            pred = {k: v[:n] for k, v in canonical_pose(seq).items()}
            g = {k: v[:n] for k, v in gt.items()}
            segs = [(r.first_frame, min(r.last_frame, n - 1)) for r in vreps]
            for r, res in zip(vreps, evaluate_sequence(pred, g, seq.valid[:n], segs, FPS,
                                                       align_scale=align_scale)):
                rows.append({"model": model, "exercise": ex, "video_id": vid, "repetition": r.repetition,
                             "person": r.person, "camera": cam, "orientation": r.orientation(cam),
                             "correct": r.correct, "lights_on": r.lights_on,
                             "extra_person": r.extra_person_cam17 if cam == 17 else r.extra_person_cam18,
                             **res})
    rows = _Rows(rows)
    rows.missing = missing
    return rows


class _Rows(list):
    missing: list


def agreement_table(df, group=("model", "orientation")):
    """Bland-Altman / ICC per measure and group, from an evaluate_model DataFrame."""
    import pandas as pd
    from .agreement import summary
    out = []
    for key, d in df.groupby(list(group)):
        for m in MEASURES:
            if f"pred_{m}" not in d:
                continue
            s = summary(d[f"pred_{m}"].to_numpy(), d[f"gt_{m}"].to_numpy())
            out.append({**dict(zip(group, np.atleast_1d(key))), "measure": m, **s})
    return pd.DataFrame(out)
