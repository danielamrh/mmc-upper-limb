import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mmcul.fit import FitConfig, fit_sequence, matrix_to_rot6d, rot6d_to_matrix  # noqa: E402
from mmcul.kinematics import elbow_flexion, shoulder_elevation  # noqa: E402

FS = 30.0


def synthetic(T=240, seed=0):
    """Subject facing the camera (camera frame: x right in image, y down, z depth).
    Right arm abducts to ~100° while the elbow flexes to ~60°; left arm hangs."""
    t = np.arange(T) / FS
    phase = (1 - np.cos(2 * np.pi * t / 4)) / 2
    abd = np.radians(100) * phase
    flex = np.radians(60) * phase
    # subject's right = image left (-x), up = -y, anterior = towards camera (-z)
    right, up, ant = np.array([-1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])
    pelvis = np.tile([0, 0, 3.0], (T, 1))
    rs = pelvis + 0.5 * up + 0.18 * right
    ls = pelvis + 0.5 * up - 0.18 * right
    hum = np.sin(abd)[:, None] * right - np.cos(abd)[:, None] * up
    re = rs + 0.30 * hum
    fore = np.cos(flex)[:, None] * hum + np.sin(flex)[:, None] * ant
    rw = re + 0.26 * fore
    le = ls - 0.30 * up
    lw = le - 0.26 * up
    pose = {"pelvis": pelvis, "r_shoulder": rs, "l_shoulder": ls, "r_elbow": re, "l_elbow": le,
            "r_wrist": rw, "l_wrist": lw}
    return pose, np.degrees(flex)


def corrupt(pose, seed=0):
    rng = np.random.default_rng(seed)
    T = len(pose["pelvis"])
    noisy = {}
    for k, v in pose.items():
        n = rng.normal(0, 0.01, v.shape)
        n[:, 2] = rng.normal(0, 0.06, T)          # MediaPipe-like depth jitter
        noisy[k] = v + n
    spikes = rng.choice(T, 6, replace=False)
    noisy["r_wrist"][spikes] += [0.0, 0.0, 0.4]  # depth outliers
    valid = np.ones(T, bool)
    valid[100:104] = False                        # a short detection gap
    return noisy, valid


def test_rot6d_roundtrip():
    R = rot6d_to_matrix(torch.tensor(np.random.default_rng(0).normal(size=(5, 6))))
    np.testing.assert_allclose(R.numpy().transpose(0, 2, 1) @ R.numpy(), np.tile(np.eye(3), (5, 1, 1)), atol=1e-12)
    back = rot6d_to_matrix(torch.tensor(matrix_to_rot6d(R.numpy()))).numpy()
    np.testing.assert_allclose(back, R.numpy(), atol=1e-12)


def test_clean_observations_are_reproduced():
    # checks the kinematic model, not the regulariser: with the tuned smoothing
    # (1e-4) clean peaks are pulled in by up to ~7 mm
    gt, flex = synthetic()
    res = fit_sequence(gt, FS, cfg=FitConfig(smooth=1e-6))
    for j in ("r_elbow", "r_wrist", "l_wrist"):
        assert np.abs(res.pose[j] - gt[j]).max() < 0.005
    assert res.lengths["r_upper"] == pytest.approx(0.30, abs=0.003)
    assert res.lengths["r_fore"] == pytest.approx(0.26, abs=0.003)
    np.testing.assert_allclose(res.elbow_flexion["r"], flex, atol=1.5)


def test_fit_beats_raw_noisy_keypoints():
    gt, flex = synthetic()
    noisy, valid = corrupt(gt)
    res = fit_sequence(noisy, FS, valid=valid)

    def err(p, j):
        return np.linalg.norm(p[j] - gt[j], axis=1).mean()

    for j in ("r_elbow", "r_wrist"):
        assert err(res.pose, j) < 0.6 * err(noisy, j)
    raw_elbow = np.sqrt(np.mean((elbow_flexion(noisy) - flex) ** 2))
    fit_elbow = np.sqrt(np.mean((res.elbow_flexion["r"] - flex) ** 2))
    assert fit_elbow < 0.5 * raw_elbow
    raw_elev = np.sqrt(np.mean((shoulder_elevation(noisy) - shoulder_elevation(gt)) ** 2))
    fit_elev = np.sqrt(np.mean((shoulder_elevation(res.pose) - shoulder_elevation(gt)) ** 2))
    assert fit_elev < raw_elev
    # elbow stays inside its range, segment lengths are constant by construction
    assert res.elbow_flexion["r"].min() > -1.0
    L = np.linalg.norm(res.pose["r_elbow"] - res.pose["r_shoulder"], axis=1)
    assert np.ptp(L) < 1e-9


def test_depth_weight_matters():
    gt, _ = synthetic()
    noisy, valid = corrupt(gt, seed=1)
    e = {}
    for dw in (1.0, 0.3):
        res = fit_sequence(noisy, FS, valid=valid, cfg=FitConfig(depth_weight=dw))
        e[dw] = np.linalg.norm(res.pose["r_wrist"] - gt["r_wrist"], axis=1).mean()
    assert e[0.3] < e[1.0]
