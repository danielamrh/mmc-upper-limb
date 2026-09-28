"""Frame access for video files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


def frame_count(path: str | Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, RGB frame) for frames start..stop-1, decoding sequentially."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while stop is None or i < stop:
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            i += 1
    finally:
        cap.release()
