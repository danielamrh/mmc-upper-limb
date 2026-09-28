"""Cached pose-model output for one video.

Pose models are run once per video and stored in their native keypoint format;
conversion to canonical joints happens at evaluation time, so a mapping fix
never needs a rerun. Inference is written in chunks so a disconnected Colab
session resumes where it stopped.

npz fields:
    kp3d   (T, K, 3) float32  3D keypoints, metres ("camera" or "root" frame)
    kp2d   (T, K, 2) float32  2D keypoints in pixels
    conf   (T, K)    float32  per-keypoint confidence (1 if the model has none)
    valid  (T,)      bool     a pose was found in this frame
    names  (K,)      str      native keypoint names
    frame, model     str
    fps              float
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ..kinematics import Pose


@dataclass
class PoseSeq:
    kp3d: np.ndarray
    kp2d: np.ndarray
    conf: np.ndarray
    valid: np.ndarray
    names: tuple[str, ...]
    frame: str
    model: str
    fps: float

    def __len__(self) -> int:
        return len(self.valid)

    def joint(self, name: str) -> np.ndarray:
        return self.kp3d[:, self.names.index(name)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, kp3d=self.kp3d, kp2d=self.kp2d, conf=self.conf, valid=self.valid,
                            names=np.array(self.names), frame=self.frame, model=self.model, fps=self.fps)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "PoseSeq":
        d = np.load(path)
        return cls(kp3d=d["kp3d"], kp2d=d["kp2d"], conf=d["conf"], valid=d["valid"],
                   names=tuple(str(n) for n in d["names"]), frame=str(d["frame"]),
                   model=str(d["model"]), fps=float(d["fps"]))

    @classmethod
    def concat(cls, parts: list["PoseSeq"]) -> "PoseSeq":
        p0 = parts[0]
        return cls(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("kp3d", "kp2d", "conf", "valid")),
                   names=p0.names, frame=p0.frame, model=p0.model, fps=p0.fps)


def empty(n: int, names: tuple[str, ...], frame: str, model: str, fps: float) -> PoseSeq:
    K = len(names)
    return PoseSeq(kp3d=np.full((n, K, 3), np.nan, np.float32), kp2d=np.full((n, K, 2), np.nan, np.float32),
                   conf=np.zeros((n, K), np.float32), valid=np.zeros(n, bool),
                   names=names, frame=frame, model=model, fps=fps)


def run_chunked(out_path: str | Path, n_frames: int, run_chunk: Callable[[int, int], PoseSeq],
                chunk: int = 600, log: Callable[[str], None] = print) -> PoseSeq:
    """Run `run_chunk(start, stop)` over [0, n_frames) in chunks, keeping every
    finished chunk on disk next to `out_path`; merge them when all are done."""
    out_path = Path(out_path)
    if out_path.exists():
        return PoseSeq.load(out_path)
    part_dir = out_path.with_name(out_path.stem + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        p = part_dir / f"{start:07d}.npz"
        if not p.exists():
            t0 = time.time()
            seq = run_chunk(start, stop)
            assert len(seq) == stop - start, (len(seq), start, stop)
            seq.save(p)
            log(f"{out_path.name}: frames {start}-{stop} of {n_frames} "
                f"({(stop - start) / max(time.time() - t0, 1e-6):.1f} fps, {seq.valid.mean():.0%} valid)")
        parts.append(PoseSeq.load(p))
    seq = PoseSeq.concat(parts)
    seq.save(out_path)
    for p in part_dir.iterdir():
        p.unlink()
    part_dir.rmdir()
    return seq


def to_pose(seq: PoseSeq, mapping: dict[str, str | Iterable[str]]) -> Pose:
    """Canonical pose from native keypoints. A mapping value that is a tuple of
    names is averaged (e.g. pelvis = mean of both hips)."""
    pose = {}
    for canon, src in mapping.items():
        names = (src,) if isinstance(src, str) else tuple(src)
        pose[canon] = np.mean([seq.joint(n) for n in names], axis=0).astype(np.float64)
    return pose
