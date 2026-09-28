import numpy as np
import pytest

from mmcul.benchmark import agreement_table, evaluate_model
from mmcul.datasets import rehab24
from mmcul.pose import mediapipe_runner as mpr
from mmcul.pose.cache import empty

pd = pytest.importorskip("pandas")

CSV = """video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;extra_person_in_cam17;extra_person_in_cam18;correctness
PM_000;1;1;1;0;89;front;0;right arm;1;0;0;1
PM_000;2;1;1;90;179;front;0;right arm;1;0;0;0
"""


def fake_dataset(tmp_path):
    """Ground truth: arm abduction; 'MediaPipe' output: the same joints, rotated and hip-centred."""
    T = 180
    t = np.arange(T) / 30.0
    ang = np.radians(80) * (1 - np.cos(2 * np.pi * t / 3)) / 2
    J = np.zeros((T, 26, 3))
    idx = rehab24.JOINT_NAMES.index
    J[:, idx("Hips")] = [0, 1.0, 0]
    J[:, idx("Spine1")] = [0, 1.35, 0]
    J[:, idx("LeftArm")] = [-0.2, 1.5, 0]
    J[:, idx("RightArm")] = [0.2, 1.5, 0]
    J[:, idx("LeftForeArm")] = [-0.2, 1.22, 0]
    J[:, idx("LeftHand")] = [-0.2, 0.95, 0]
    J[:, idx("RightForeArm")] = np.stack([0.2 + 0.28 * np.sin(ang), 1.5 - 0.28 * np.cos(ang), 0 * t], 1)
    J[:, idx("RightHand")] = np.stack([0.2 + 0.56 * np.sin(ang), 1.5 - 0.56 * np.cos(ang), 0 * t], 1)
    root = tmp_path / "data"
    (root / "3d_joints" / "Ex1").mkdir(parents=True)
    (root / "Segmentation.csv").write_text(CSV)
    np.save(rehab24.joints_path(root, 1, "PM_000"), np.concatenate([J, np.ones((T, 26, 1))], -1))

    seq = empty(T, mpr.NAMES, "root", "mediapipe-heavy", 30.0)
    hip = J[:, idx("Hips")]
    for mp_name, src in (("left_hip", "Hips"), ("right_hip", "Hips"), ("left_shoulder", "LeftArm"),
                         ("right_shoulder", "RightArm"), ("left_elbow", "LeftForeArm"),
                         ("right_elbow", "RightForeArm"), ("left_wrist", "LeftHand"), ("right_wrist", "RightHand")):
        p = J[:, idx(src)] - hip
        seq.kp3d[:, mpr.NAMES.index(mp_name)] = p * [1, -1, -1]  # 180° about x, like an image frame
    seq.valid[:] = True
    seq.save(tmp_path / "cache" / "mediapipe-heavy" / "Ex1" / "PM_000-c17.npz")
    return root, tmp_path / "cache"


def test_evaluate_model_end_to_end(tmp_path):
    root, cache = fake_dataset(tmp_path)
    rows = evaluate_model(root, cache, "mediapipe-heavy", exercises=(1,))
    assert len(rows) == 2 and len(rows.missing) == 1  # camera 18 not cached
    df = pd.DataFrame(rows)
    assert (df.orientation == "front").all()
    assert df.mpjpe_mm.max() < 1.0
    np.testing.assert_allclose(df.pred_max_shoulder_elevation, df.gt_max_shoulder_elevation, atol=0.1)
    tab = agreement_table(df)
    assert set(tab.measure) >= {"peak_speed", "max_shoulder_elevation"}
