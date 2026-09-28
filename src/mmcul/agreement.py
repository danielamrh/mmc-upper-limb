"""Agreement statistics between an estimate and a reference method."""

from __future__ import annotations

import numpy as np


def bland_altman(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Bias (mean of pred - ref) and 95 % limits of agreement."""
    d = np.asarray(pred, float) - np.asarray(ref, float)
    bias, sd = d.mean(), d.std(ddof=1)
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def icc_a1(pred: np.ndarray, ref: np.ndarray) -> float:
    """ICC(A,1): two-way model, absolute agreement, single measurement
    (McGraw & Wong 1996; ICC(2,1) in Shrout & Fleiss notation)."""
    Y = np.stack([pred, ref], axis=1).astype(float)
    n, k = Y.shape
    grand = Y.mean()
    ss_rows = k * ((Y.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((Y.mean(0) - grand) ** 2).sum()
    ss_err = ((Y - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def summary(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    return {"n": int(ok.sum()), **bland_altman(pred, ref),
            "mae": float(np.abs(pred - ref).mean()),
            "r": float(np.corrcoef(pred, ref)[0, 1]), "icc_a1": icc_a1(pred, ref)}
