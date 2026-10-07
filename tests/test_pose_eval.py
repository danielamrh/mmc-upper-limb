import numpy as np
import pytest

from mmcul.evaluate import evaluate_sequence, fill_gaps
from mmcul.pose.cache import PoseSeq, empty, run_chunked, to_pose
from mmcul.pose.person import bbox_from_2d, closest_to_box

FS = 30.0


def moving_pose(T=240, seed=0):
    """Arm abduction-like movement: wrist circles up and down."""
    t = np.arange(T) / FS
    ang = np.radians(80) * (1 - np.cos(2 * np.pi * t / 4)) / 2
    c = lambda p: np.tile(np.asarray(p, float), (T, 1))
    elbow = np.stack([0.2 + 0.28 * np.sin(ang), 1.5 - 0.28 * np.cos(ang), np.zeros(T)], 1)
    wrist = np.stack([0.2 + 0.56 * np.sin(ang), 1.5 - 0.56 * np.cos(ang), np.zeros(T)], 1)
    return {"pelvis": c([0, 1.0, 0]), "r_shoulder": c([0.2, 1.5, 0]), "l_shoulder": c([-0.2, 1.5, 0]),
            "r_elbow": elbow, "r_wrist": wrist, "l_elbow": c([-0.2, 1.2, 0]), "l_wrist": c([-0.2, 0.95, 0])}


def test_fill_gaps():
    x = np.arange(12, dtype=float).reshape(4, 3)
    y = x.copy()
    y[1:3] = np.nan
    np.testing.assert_allclose(fill_gaps(y), x)


def test_evaluate_sequence_perfect_estimate_in_other_frame():
    gt = moving_pose()
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pred = {k: 0.8 * v @ R.T + [1, 2, 3] for k, v in gt.items()}  # other frame and scale
    valid = np.ones(len(gt["pelvis"]), bool)
    valid[50:53] = False  # a short detection gap
    row = evaluate_sequence(pred, gt, valid, [(0, 119), (120, 239)], FS)[0]
    assert row["valid_frac"] == pytest.approx(117 / 120)
    assert row["mpjpe_mm"] < 5
    assert row["rmse_elbow_flexion"] < 1.0
    for m in ("peak_speed", "max_shoulder_elevation", "movement_time"):
        assert row[f"pred_{m}"] == pytest.approx(row[f"gt_{m}"], rel=0.05, abs=0.5)


def test_run_chunked_resumes(tmp_path):
    names = ("a", "b")
    calls = []

    def run_chunk(a, b):
        calls.append((a, b))
        seq = empty(b - a, names, "camera", "m", FS)
        seq.kp3d[:] = np.arange(a, b)[:, None, None]
        seq.valid[:] = True
        return seq

    out = tmp_path / "v.npz"
    part_dir = tmp_path / "v.parts"
    part_dir.mkdir()
    run_chunk(0, 4).save(part_dir / "0000000.npz")  # pretend the first chunk survived a crash
    calls.clear()
    seq = run_chunked(out, 10, run_chunk, chunk=4, log=lambda s: None)
    assert calls == [(4, 8), (8, 10)]
    assert len(seq) == 10 and seq.kp3d[9, 0, 0] == 9
    assert not part_dir.exists()
    assert len(PoseSeq.load(out)) == 10
    pose = to_pose(seq, {"x": "a", "mid": ("a", "b")})
    np.testing.assert_allclose(pose["mid"][:, 0], np.arange(10))


def test_run_chunked_recomputes_corrupt_chunk(tmp_path):
    names = ("a",)
    calls = []

    def run_chunk(a, b):
        calls.append((a, b))
        seq = empty(b - a, names, "camera", "m", FS)
        seq.valid[:] = True
        return seq

    part_dir = tmp_path / "v.parts"
    part_dir.mkdir()
    run_chunk(0, 4).save(part_dir / "0000000.npz")
    (part_dir / "0000004.npz").write_bytes(b"truncated")  # write cut off by a disconnect
    calls.clear()
    seq = run_chunked(tmp_path / "v.npz", 8, run_chunk, chunk=4, log=lambda s: None)
    assert calls == [(4, 8)] and len(seq) == 8


def test_person_selection():
    box = bbox_from_2d(np.array([[100, 100], [200, 400]], float), 1920, 1080)
    np.testing.assert_allclose(box, [85, 55, 215, 445])
    far = np.array([[1500, 500], [1600, 800]], float)
    near = np.array([[120, 150], [190, 380]], float)
    assert closest_to_box([far, near], box) == 1
    assert closest_to_box([far], box) is None
