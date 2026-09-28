"""SAM 3D Body (Meta, 2025) — single-image human mesh recovery on the MHR rig.

Needs a clone of github.com/facebookresearch/sam-3d-body on sys.path and access
to the gated Hugging Face checkpoints. Detector, segmentor and FOV estimator are
not used: the subject box is given (see mmcul.pose.person) and the default FOV
is kept, because REHAB24-6 publishes no camera intrinsics.

3D output = pred_keypoints_3d + pred_cam_t, i.e. camera coordinates in metres
(frame = "camera").
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from ..video import iter_frames
from .cache import PoseSeq, empty

# first 70 MHR keypoints (sam_3d_body/metadata/mhr70.py)
NAMES = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
    "left-big-toe-tip", "left-small-toe-tip", "left-heel", "right-big-toe-tip", "right-small-toe-tip", "right-heel",
    *[f"right-{f}-{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
      for j in ("tip", "first-joint", "second-joint", "third-joint")],
    "right-wrist",
    *[f"left-{f}-{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
      for j in ("tip", "first-joint", "second-joint", "third-joint")],
    "left-wrist",
    "left-olecranon", "right-olecranon", "left-cubital-fossa", "right-cubital-fossa",
    "left-acromion", "right-acromion", "neck",
)
assert len(NAMES) == 70

CANONICAL = {
    "pelvis": ("left-hip", "right-hip"),
    "thorax": ("left-shoulder", "right-shoulder"),
    "neck": "neck",
    "head": "nose",
    "r_shoulder": "right-shoulder", "r_elbow": "right-elbow", "r_wrist": "right-wrist",
    "l_shoulder": "left-shoulder", "l_elbow": "left-elbow", "l_wrist": "left-wrist",
}


class SAM3DBodyRunner:
    def __init__(self, hf_repo_id: str = "facebook/sam-3d-body-dinov3", device: str = "cuda",
                 inference_type: str = "body"):
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
        model, cfg = load_sam_3d_body_hf(hf_repo_id, device=device)
        self.estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
        self.inference_type = inference_type
        self.name = "sam3db-" + hf_repo_id.rsplit("-", 1)[-1]

    def run(self, video: str | Path, start: int, stop: int, boxes: np.ndarray, fps: float) -> PoseSeq:
        seq = empty(stop - start, NAMES, "camera", self.name, fps)
        seq.conf[:] = 1.0
        for i, rgb in iter_frames(video, start, stop):
            t = i - start
            with contextlib.redirect_stdout(io.StringIO()):  # the estimator prints on every call
                out = self.estimator.process_one_image(rgb, bboxes=boxes[t][None],
                                                       inference_type=self.inference_type)
            if not out:
                continue
            o = out[0]
            seq.kp3d[t] = o["pred_keypoints_3d"][:70] + o["pred_cam_t"][None]
            seq.kp2d[t] = o["pred_keypoints_2d"][:70]
            seq.valid[t] = True
        return seq
