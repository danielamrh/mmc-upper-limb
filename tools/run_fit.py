"""Biomechanical fit (mmcul.fit) on cached pose-model output (resumable, parallel).

    python tools/run_fit.py --cache cache/rehab24 --model mediapipe-heavy --workers 4

Reads <cache>/<model>/Ex*/<video>.npz and writes the fitted canonical joints to
<cache>/<model>+fit/Ex*/<video>.npz, so mmcul.benchmark evaluates "<model>+fit"
like any other model. The fitted segment lengths go to <video>.lengths.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmcul.fit import FitConfig, fit_sequence  # noqa: E402
from mmcul.kinematics import JOINTS  # noqa: E402
from mmcul.pose import canonical_conf, canonical_pose  # noqa: E402
from mmcul.pose.cache import PoseSeq  # noqa: E402

FIT_JOINTS = tuple(j for j in JOINTS if not j.endswith("_hand") and j not in ("neck", "head"))


def fit_file(args) -> str:
    src, dst, cfg, threads = args
    if dst.exists():
        return f"{dst.parent.name}/{dst.name}: cached"
    import torch
    torch.set_num_threads(threads)
    t0 = time.time()
    seq = PoseSeq.load(src)
    res = fit_sequence(canonical_pose(seq), seq.fps, canonical_conf(seq), seq.valid, cfg)
    names = tuple(j for j in FIT_JOINTS if j in res.pose)
    out = PoseSeq(kp3d=np.stack([res.pose[j] for j in names], 1).astype(np.float32),
                  kp2d=np.full((len(seq), len(names), 2), np.nan, np.float32),
                  conf=np.ones((len(seq), len(names)), np.float32), valid=np.ones(len(seq), bool),
                  names=names, frame=seq.frame, model=seq.model + "+fit", fps=seq.fps)
    out.save(dst)
    dst.with_suffix(".lengths.json").write_text(json.dumps(
        {"lengths": res.lengths, "loss": res.loss, "config": asdict(cfg)}, indent=1))
    return f"{dst.parent.name}/{dst.name}: {len(seq)} frames in {time.time() - t0:.0f} s, " \
           f"forearm {res.lengths['r_fore']:.3f} m, data loss {res.loss['data']:.2e}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exercises", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--suffix", default="+fit", help="output model suffix, e.g. +fit_dw1 for ablations")
    for f, v in asdict(FitConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = FitConfig(**{f: getattr(a, f) for f in asdict(FitConfig())})

    jobs = []
    for ex in a.exercises:
        for src in sorted((a.cache / a.model / f"Ex{ex}").glob("*.npz")):
            dst = a.cache / (a.model + a.suffix) / f"Ex{ex}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((src, dst, cfg, max(1, 8 // a.workers)))
    print(f"{len(jobs)} sequences · {a.model}{a.suffix} · {cfg}", flush=True)
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(fit_file, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(fit_file(j), flush=True)


if __name__ == "__main__":
    main()
