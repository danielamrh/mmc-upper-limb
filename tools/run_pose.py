"""Run a pose model on REHAB24-6 videos and cache the results (resumable).

    python tools/run_pose.py --root data/rehab24 --cache cache/rehab24 --fetch
    python tools/run_pose.py --root data/rehab24 --cache cache/rehab24 --model mediapipe-heavy --workers 4
    python tools/run_pose.py ... --model sam3db-dinov3          # needs a GPU and the sam-3d-body repo

Output: <cache>/<model>/Ex<e>/<video_id>-c<camera>.npz (see mmcul.pose.cache).
"""

from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mmcul.datasets import rehab24  # noqa: E402
from mmcul.pose import MODELS, make_runner  # noqa: E402
from mmcul.pose.cache import run_chunked  # noqa: E402
from mmcul.pose.person import bbox_from_2d  # noqa: E402
from mmcul.video import frame_count  # noqa: E402

FPS = 30.0
_runner = None


def fetch(root: Path, exercises: tuple[int, ...]) -> None:
    rehab24.download(["Segmentation.csv"], root)
    for zip_name, select in (("2d_joints.zip", lambda n: n.endswith("-30fps.npy")),
                             ("3d_joints.zip", lambda n: n.endswith("-30fps.npy")),
                             ("videos.zip", lambda n: True)):
        t0 = time.time()
        paths = rehab24.fetch_members(zip_name, root, exercises, select)
        print(f"{zip_name}: {len(paths)} files for Ex{exercises} ({time.time() - t0:.0f} s)", flush=True)


def jobs(root: Path, exercises, cameras) -> list[tuple[int, str, int]]:
    reps = rehab24.read_segmentation(root / "Segmentation.csv")
    videos = sorted({(r.exercise, r.video_id) for r in reps if r.exercise in exercises})
    return [(ex, vid, cam) for ex, vid in videos for cam in cameras]


def run_job(args) -> str:
    global _runner
    (ex, vid, cam), root, cache, model, model_dir, chunk = args
    out = cache / model / f"Ex{ex}" / f"{vid}-c{cam}.npz"
    if out.exists():
        return f"{out.name}: cached"
    if _runner is None:
        _runner = make_runner(model, model_dir)
    video = rehab24.video_path(root, ex, vid, cam)
    if not video.exists():
        raise FileNotFoundError(f"{video} missing - run with --fetch first")
    j2d = np.load(rehab24.joints2d_path(root, ex, vid, cam))
    n = min(frame_count(video), len(j2d))
    if n == 0:
        raise RuntimeError(f"{video}: no frames could be read")
    w, h = rehab24.image_size(cam)
    boxes = np.stack([bbox_from_2d(j, w, h) for j in j2d[:n]])
    t0 = time.time()
    seq = run_chunked(out, n, lambda a, b: _runner.run(video, a, b, boxes[a:b], FPS), chunk=chunk,
                      log=lambda s: print(s, flush=True))
    return f"{out.name}: {n} frames in {time.time() - t0:.0f} s, {seq.valid.mean():.1%} valid"


def acquire_lock(path: Path) -> None:
    """One run per cache/model: a second process on the same machine would
    compute the same chunks and halve the GPU throughput of both."""
    host = socket.gethostname()
    if path.exists():
        other_host, _, other_pid = path.read_text().partition(":")
        alive = other_host == host and other_pid.isdigit() and _pid_alive(int(other_pid))
        if alive:
            sys.exit(f"another run_pose.py (pid {other_pid}) is already working on {path.parent} - "
                     f"stop it first (pkill -f run_pose.py) or wait for it")
        print(f"taking over stale lock from {other_host}:{other_pid}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{host}:{os.getpid()}")
    atexit.register(lambda: path.unlink(missing_ok=True))


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="REHAB24-6 data directory")
    ap.add_argument("--cache", type=Path, required=True, help="output directory for pose caches")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--exercises", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--cameras", type=int, nargs="+", default=[17, 18])
    ap.add_argument("--fetch", action="store_true", help="download the needed files first")
    ap.add_argument("--workers", type=int, default=1, help="parallel videos (MediaPipe only)")
    ap.add_argument("--chunk", type=int, default=600, help="frames per resumable chunk")
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    a = ap.parse_args()

    exercises = tuple(a.exercises)
    if a.fetch:
        fetch(a.root, exercises)
    if a.model is None:
        return
    if a.workers > 1 and not a.model.startswith("mediapipe"):
        ap.error("--workers > 1 only for MediaPipe (one GPU model per process would not fit)")
    acquire_lock(a.cache / a.model / ".run.lock")
    todo = [(j, a.root, a.cache, a.model, a.model_dir, a.chunk) for j in jobs(a.root, exercises, a.cameras)]
    print(f"{len(todo)} videos · {a.model}", flush=True)
    if a.model.startswith("mediapipe"):
        make_runner(a.model, a.model_dir)  # download the .task file once, before workers start
    if a.workers > 1:
        with mp.Pool(a.workers) as pool:
            for msg in pool.imap_unordered(run_job, todo):
                print(msg, flush=True)
    else:
        for t in todo:
            print(run_job(t), flush=True)


if __name__ == "__main__":
    main()
