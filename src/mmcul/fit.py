"""Biomechanical fit: an upper-body kinematic chain fitted to noisy 3D keypoints.

Model (per sequence constant segment lengths, per frame joint angles):

    pelvis      p(t)                                   3 DOF
    trunk       R_t(t)                                 3 DOF (6D rotation)
    shoulders   p + R_t (0, L_trunk, ±w) + R_t d_s(t)   small clavicle offsets d_s, penalised
    upper arm   R_t R_s(t) (0, -L_upper, 0)             3 DOF shoulder (6D rotation)
    forearm     R_t R_s(t) R_z(θ) (0, -L_fore, 0)       1 DOF elbow flexion θ ∈ [0°, 160°]

Frames follow the ISB convention of mmcul.kinematics (x anterior, y superior,
z right); θ > 0 flexes the forearm anteriorly. Rotations use the continuous 6D
representation (Zhou et al. 2019), so arms raised overhead (Ex2) hit no
axis-angle singularity.

Loss, minimised with L-BFGS over the whole sequence at once:

    data     Huber distance to the observed keypoints, weighted by keypoint
             confidence and down-weighted along the observation's depth axis
             (monocular models are least reliable in depth)
    smooth   squared joint accelerations (m/s²) of all model joints
    limits   elbow outside [0°, 160°]
    clavicle size of the clavicle offsets

Observations must be in a camera-aligned frame (z = depth), which holds for
MediaPipe world landmarks and SAM 3D Body camera coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from scipy.ndimage import median_filter

from .evaluate import fill_gaps
from .kinematics import Pose, trunk_frame
from .measures import lowpass

OBS_JOINTS = ("pelvis", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow", "r_wrist", "l_wrist")
SIDES = (("r", 1.0), ("l", -1.0))


@dataclass
class FitConfig:
    depth_weight: float = 0.3      # weight of the depth axis relative to x / y
    huber_delta: float = 0.05      # m
    smooth: float = 1e-4           # weight per (m/s²)² of joint acceleration (tools/tune_fit.py)
    clavicle: float = 10.0         # weight per m² of clavicle offset
    length_prior: float = 1e-3     # weight per (log length ratio)² to the initial median lengths
    elbow_max_deg: float = 160.0
    limit_weight: float = 100.0    # per rad² outside the elbow range
    max_iter: int = 300            # L-BFGS iterations per stage (poses only, then all)


@dataclass
class FitResult:
    pose: Pose                      # fitted canonical joints, observation frame
    lengths: dict[str, float]       # metres
    elbow_flexion: dict[str, np.ndarray]  # degrees, model angle (not from positions)
    loss: dict[str, float] = field(default_factory=dict)
    params: dict[str, np.ndarray] = field(default_factory=dict)


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3), columns from Gram-Schmidt on two 3-vectors."""
    a, b = x[..., :3], x[..., 3:]
    c1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b = b - (c1 * b).sum(-1, keepdim=True) * c1
    c2 = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return torch.stack([c1, c2, torch.cross(c1, c2, dim=-1)], dim=-1)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _init_arm(Rt: np.ndarray, sh: np.ndarray, el: np.ndarray, wr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder rotation (trunk frame) and elbow angle from observed points."""
    u = _unit(np.einsum("tji,tj->ti", Rt, el - sh))   # humerus direction, trunk frame
    f = _unit(np.einsum("tji,tj->ti", Rt, wr - el))   # forearm direction
    theta = np.arccos(np.clip((u * f).sum(-1), -1, 1))
    x = f - (f * u).sum(-1, keepdims=True) * u        # forearm part normal to the humerus
    small = np.linalg.norm(x, axis=-1) < 1e-3          # straight arm: flexion plane undefined
    ant = np.array([1.0, 0.0, 0.0]) - u * u[:, :1]     # fall back to trunk-anterior direction
    x[small] = ant[small]
    x = _unit(x)
    y = -u
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1), theta


def fit_sequence(obs: Pose, fs: float, conf: dict[str, np.ndarray] | None = None,
                 valid: np.ndarray | None = None, cfg: FitConfig | None = None) -> FitResult:
    """Fit the kinematic model to observed joints (canonical names, (T, 3), metres)."""
    cfg = cfg or FitConfig()
    T = len(obs["pelvis"])
    valid = np.ones(T, bool) if valid is None else valid.astype(bool)
    o = {j: fill_gaps(np.where(valid[:, None], obs[j], np.nan)) for j in OBS_JOINTS}
    w = np.stack([valid.astype(float) * (1.0 if conf is None else np.nan_to_num(conf[j]))
                  for j in OBS_JOINTS], axis=1)                     # (T, J)

    # ---- initialisation from median-filtered, low-pass filtered observations ----
    # Starting from the raw keypoints, jitter makes the acceleration term dominate
    # and L-BFGS first shrinks the segments (a forearm collapsed to 6 mm in tests).
    o_init = {j: lowpass(median_filter(v, size=(5, 1), mode="nearest"), fs) for j, v in o.items()}
    Rt = trunk_frame(o_init)
    mid = 0.5 * (o_init["r_shoulder"] + o_init["l_shoulder"])
    med = lambda a, b: float(np.median(np.linalg.norm(o_init[a] - o_init[b], axis=1)))
    L = {"trunk": float(np.median(np.linalg.norm(mid - o_init["pelvis"], axis=1))),
         "half_width": 0.5 * med("r_shoulder", "l_shoulder")}
    init = {"pelvis": o_init["pelvis"], "trunk": matrix_to_rot6d(Rt)}
    for s, _ in SIDES:
        L[f"{s}_upper"] = med(f"{s}_shoulder", f"{s}_elbow")
        L[f"{s}_fore"] = med(f"{s}_elbow", f"{s}_wrist")
        Rs, th = _init_arm(Rt, o_init[f"{s}_shoulder"], o_init[f"{s}_elbow"], o_init[f"{s}_wrist"])
        init[f"{s}_shoulder"] = matrix_to_rot6d(Rs)
        init[f"{s}_elbow"] = th
        init[f"{s}_clavicle"] = np.zeros((T, 3))

    dt = torch.float64
    P = {k: torch.tensor(np.ascontiguousarray(v), dtype=dt, requires_grad=True) for k, v in init.items()}
    logL = {k: torch.tensor(np.log(v), dtype=dt, requires_grad=True) for k, v in L.items()}
    logL0 = {k: v.detach().clone() for k, v in logL.items()}
    O = torch.tensor(np.stack([o[j] for j in OBS_JOINTS], 1), dtype=dt)
    W = torch.tensor(w, dtype=dt)
    axis_w = torch.tensor([1.0, 1.0, cfg.depth_weight], dtype=dt)
    emax = np.radians(cfg.elbow_max_deg)

    def forward():
        Ln = {k: v.exp() for k, v in logL.items()}
        R_t = rot6d_to_matrix(P["trunk"])
        out = {"pelvis": P["pelvis"]}
        for s, sign in SIDES:
            off = torch.stack([torch.zeros((), dtype=dt), Ln["trunk"], sign * Ln["half_width"]])
            sh = P["pelvis"] + torch.einsum("tij,tj->ti", R_t, off + P[f"{s}_clavicle"])
            R_a = R_t @ rot6d_to_matrix(P[f"{s}_shoulder"])
            th = P[f"{s}_elbow"]
            up = torch.stack([torch.zeros((), dtype=dt), -Ln[f"{s}_upper"], torch.zeros((), dtype=dt)])
            el = sh + torch.einsum("tij,j->ti", R_a, up)
            fore = Ln[f"{s}_fore"] * torch.stack([th.sin(), -th.cos(), torch.zeros_like(th)], -1)
            out[f"{s}_shoulder"], out[f"{s}_elbow"] = sh, el
            out[f"{s}_wrist"] = el + torch.einsum("tij,tj->ti", R_a, fore)
        return out

    def losses():
        out = forward()
        X = torch.stack([out[j] for j in OBS_JOINTS], 1)
        r = ((X - O) * axis_w).norm(dim=-1)
        d = cfg.huber_delta
        hub = torch.where(r < d, 0.5 * r ** 2, d * (r - 0.5 * d))
        data = (W * hub).sum() / W.sum().clamp_min(1.0)
        acc = (X[2:] - 2 * X[1:-1] + X[:-2]) * fs ** 2
        smooth = cfg.smooth * (acc ** 2).sum(-1).mean()
        lim = sum(torch.relu(-P[f"{s}_elbow"]) ** 2 + torch.relu(P[f"{s}_elbow"] - emax) ** 2 for s, _ in SIDES)
        limits = cfg.limit_weight * lim.mean()
        clav = cfg.clavicle * sum((P[f"{s}_clavicle"] ** 2).sum(-1) for s, _ in SIDES).mean()
        lengths = cfg.length_prior * sum((logL[k] - logL0[k]) ** 2 for k in logL)
        return {"data": data, "smooth": smooth, "limits": limits, "clavicle": clav, "lengths": lengths}, out

    # stage 1: poses with the median lengths fixed, stage 2: everything
    for params in (list(P.values()), [*P.values(), *logL.values()]):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.max_iter, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            total = sum(losses()[0].values())
            total.backward()
            return total

        opt.step(closure)
    with torch.no_grad():
        parts, out = losses()
    pose = {k: v.detach().numpy().copy() for k, v in out.items()}
    pose["thorax"] = 0.5 * (pose["r_shoulder"] + pose["l_shoulder"])
    return FitResult(
        pose=pose,
        lengths={k: float(v.detach().exp()) for k, v in logL.items()},
        elbow_flexion={s: np.degrees(P[f"{s}_elbow"].detach().numpy()) for s, _ in SIDES},
        loss={k: float(v) for k, v in parts.items()},
        params={k: v.detach().numpy().copy() for k, v in P.items()},
    )
