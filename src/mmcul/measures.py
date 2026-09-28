"""Upper-limb movement quality measures after Alt Murphy et al.

Definitions follow the drinking-task kinematics of Alt Murphy, Willén &
Sunnerhagen (Neurorehabil Neural Repair 2011, 2012, 2013), as also used by
Huber et al. (2024) and Unger et al. (ICORR 2025):

* positions low-pass filtered with a 2nd-order zero-phase Butterworth, 6 Hz
* movement onset / offset: hand speed crosses 2 % of its peak
* movement time (MT): offset - onset
* movement units (NMU): a local minimum of hand speed followed by a maximum
  that is more than 20 mm/s higher, with at least 150 ms between counted peaks
* trunk displacement (TD): maximal distance of the thorax from its position at
  movement onset

All positions are in metres and all speeds in m/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from .kinematics import Pose, elbow_flexion, shoulder_elevation


def lowpass(x: np.ndarray, fs: float, cutoff: float = 6.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0."""
    b, a = butter(order, cutoff / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def speed(pos: np.ndarray, fs: float) -> np.ndarray:
    """Tangential speed of a (T, 3) trajectory."""
    return np.linalg.norm(np.gradient(pos, 1.0 / fs, axis=0), axis=-1)


def movement_bounds(v: np.ndarray, frac: float = 0.02) -> tuple[int, int]:
    """Onset and offset index (inclusive) around the speed peak: the last sample
    before the peak and the first after it where speed is below `frac` * peak."""
    peak = int(np.argmax(v))
    thr = frac * v[peak]
    below_before = np.flatnonzero(v[:peak] < thr)
    below_after = np.flatnonzero(v[peak:] < thr)
    start = int(below_before[-1]) if below_before.size else 0
    end = peak + int(below_after[0]) if below_after.size else len(v) - 1
    return start, end


def n_movement_units(v: np.ndarray, fs: float, min_rise: float = 0.020,
                     min_interval: float = 0.150) -> int:
    """Number of speed peaks rising more than `min_rise` (m/s) above the lowest
    speed since the previous counted peak, at least `min_interval` s apart."""
    peaks, _ = find_peaks(v)
    count, last = 0, None
    for p in peaks:
        lo = 0 if last is None else last
        if v[p] - v[lo:p + 1].min() <= min_rise:
            continue
        if last is not None and (p - last) / fs < min_interval:
            continue
        count, last = count + 1, p
    return count


@dataclass
class Measures:
    movement_time: float        # s
    n_movement_units: int
    peak_speed: float           # m/s
    time_to_peak_speed: float   # % of movement time
    trunk_displacement: float   # m
    min_elbow_flexion: float    # deg, smallest flexion = maximal extension
    max_shoulder_elevation: float  # deg

    def as_dict(self) -> dict:
        return asdict(self)


def compute_measures(pose: Pose, fs: float, side: str = "r", end_effector: str = "wrist",
                     onset_frac: float = 0.02, cutoff: float = 6.0) -> Measures:
    """All measures for one movement segment (e.g. one repetition).

    The segment should contain the movement plus some rest before and after;
    onset and offset are detected inside it from the end-effector speed.
    """
    pose = {k: lowpass(v, fs, cutoff) for k, v in pose.items()}
    v = speed(pose[f"{side}_{end_effector}"], fs)
    start, end = movement_bounds(v, onset_frac)
    seg = slice(start, end + 1)
    vs = v[seg]
    thorax = pose["thorax"][seg]
    return Measures(
        movement_time=(end - start) / fs,
        n_movement_units=n_movement_units(vs, fs),
        peak_speed=float(vs.max()),
        time_to_peak_speed=100.0 * int(np.argmax(vs)) / max(end - start, 1),
        trunk_displacement=float(np.linalg.norm(thorax - thorax[0], axis=-1).max()),
        min_elbow_flexion=float(elbow_flexion(pose, side)[seg].min()),
        max_shoulder_elevation=float(shoulder_elevation(pose, side)[seg].max()),
    )
