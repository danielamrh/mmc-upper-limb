import numpy as np
import pytest

from mmcul.measures import compute_measures, lowpass, movement_bounds, n_movement_units, speed

FS = 120.0


def min_jerk(t, t0, dur, dist):
    """Minimum-jerk displacement along one axis."""
    tau = np.clip((t - t0) / dur, 0.0, 1.0)
    return dist * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)


def reach(submovements, total=3.0):
    """Hand trajectory along x made of (start, duration, distance) submovements."""
    t = np.arange(0, total, 1 / FS)
    x = sum(min_jerk(t, *s) for s in submovements)
    return t, np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)


def pose_from_hand(hand, thorax=None):
    T = len(hand)
    const = lambda p: np.tile(np.asarray(p, float), (T, 1))
    return {
        "pelvis": const([0, 1.0, 0]),
        "thorax": const([0, 1.3, 0]) if thorax is None else thorax,
        "r_shoulder": const([0.2, 1.5, 0]),
        "l_shoulder": const([-0.2, 1.5, 0]),
        "r_elbow": const([0.2, 1.2, 0]),
        "r_wrist": hand + [0.2, 0.95, 0],
    }


def test_single_reach():
    _, hand = reach([(1.0, 1.0, 0.3)])
    m = compute_measures(pose_from_hand(hand), FS)
    assert m.n_movement_units == 1
    assert m.movement_time == pytest.approx(1.0, abs=0.12)  # 2 % threshold trims the tails
    assert m.peak_speed == pytest.approx(1.875 * 0.3, rel=0.02)  # min-jerk peak = 1.875 d/T
    assert m.time_to_peak_speed == pytest.approx(50.0, abs=3.0)
    assert m.trunk_displacement == pytest.approx(0.0, abs=1e-6)


def test_two_submovements():
    _, hand = reach([(0.8, 0.8, 0.25), (1.5, 0.8, 0.15)])
    assert compute_measures(pose_from_hand(hand), FS).n_movement_units == 2


def test_noise_does_not_add_units_after_filtering():
    rng = np.random.default_rng(0)
    _, hand = reach([(1.0, 1.0, 0.3)])
    noisy = hand + rng.normal(0, 0.001, hand.shape)
    assert compute_measures(pose_from_hand(noisy), FS).n_movement_units == 1


def test_small_bump_below_threshold_is_ignored():
    v = np.concatenate([np.linspace(0, 0.5, 60), np.linspace(0.5, 0.49, 5),
                        np.linspace(0.49, 0.505, 5), np.linspace(0.505, 0, 60)])
    assert n_movement_units(v, FS) == 1


def test_units_closer_than_150ms_count_once():
    t = np.arange(0, 1, 1 / FS)
    v = 0.3 + 0.1 * np.cos(2 * np.pi * 10 * t)  # peaks 100 ms apart, 200 mm/s deep
    assert n_movement_units(v, FS) == 5  # every second peak: 100, 300, 500, 700, 900 ms


def test_trunk_displacement():
    t, hand = reach([(1.0, 1.0, 0.3)])
    thorax = np.tile([0, 1.3, 0.0], (len(t), 1))
    thorax[:, 2] += min_jerk(t, 1.1, 0.4, 0.08) - min_jerk(t, 1.5, 0.4, 0.08)  # lean forward and back
    m = compute_measures(pose_from_hand(hand, thorax), FS)
    assert m.trunk_displacement == pytest.approx(0.08, abs=0.005)


def test_movement_bounds_and_lowpass_shapes():
    _, hand = reach([(1.0, 1.0, 0.3)])
    f = lowpass(hand, FS)
    assert f.shape == hand.shape
    s, e = movement_bounds(speed(f, FS))
    assert 0 < s < e < len(hand) - 1
