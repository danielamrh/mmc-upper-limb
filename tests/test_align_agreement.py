import numpy as np
import pytest

from mmcul.agreement import bland_altman, icc_a1, summary
from mmcul.align import align_sequence, umeyama


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R, s, t = random_rotation(rng), 1.7, np.array([0.3, -2.0, 5.0])
    dst = s * src @ R.T + t
    s_, R_, t_ = umeyama(src, dst)
    assert s_ == pytest.approx(s)
    np.testing.assert_allclose(R_, R, atol=1e-10)
    np.testing.assert_allclose(t_, t, atol=1e-10)


def test_align_sequence_ignores_nan_frames():
    rng = np.random.default_rng(1)
    gt = {j: rng.normal(size=(20, 3)) for j in ("a", "b", "c")}
    R = random_rotation(rng)
    pred = {j: 0.5 * v @ R.T + 1.0 for j, v in gt.items()}
    pred["a"][3] = np.nan
    out = align_sequence(pred, gt)
    np.testing.assert_allclose(out["b"], gt["b"], atol=1e-10)
    assert np.isnan(out["a"][3]).all()


def test_icc_perfect_and_biased():
    ref = np.arange(10.0)
    assert icc_a1(ref, ref) == pytest.approx(1.0)
    # constant bias lowers absolute agreement but not correlation
    assert icc_a1(ref + 3, ref) < 0.9
    assert summary(ref + 3, ref)["r"] == pytest.approx(1.0)


def test_icc_matches_residual_anova_formula():
    x = np.array([9.0, 6, 8, 7, 10, 6])
    y = np.array([2.0, 1, 4, 1, 5, 2])
    Y = np.stack([x, y], 1)
    n, k = Y.shape
    g = Y.mean()
    msr = k * ((Y.mean(1) - g) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - g) ** 2).sum() / (k - 1)
    mse = (((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + g) ** 2).sum()) / ((n - 1) * (k - 1))
    expected = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    assert icc_a1(x, y) == pytest.approx(expected)


def test_bland_altman():
    ba = bland_altman([2.0, 4.0, 6.0], [1.0, 3.0, 5.0])
    assert ba["bias"] == pytest.approx(1.0)
    assert ba["sd"] == pytest.approx(0.0)
