import io

import numpy as np

from mmcul.datasets import rehab24
from mmcul.kinematics import JOINTS

CSV = """video_id;repetition_number;exercise_id;person_id;first_frame;last_frame;cam17_orientation;mocap_erroneous;exercise_subtype;lights_on;extra_person_in_cam17;extra_person_in_cam18;correctness
PM_000;1;1;1;180;377;front;0;right arm;0;3;0;1
PM_101;2;3;4;10;90;profile;1;;1;0;2;0
"""


def test_read_segmentation():
    reps = rehab24.read_segmentation(io.StringIO(CSV))
    assert len(reps) == 2
    a, b = reps
    assert (a.video_id, a.exercise, a.first_frame, a.last_frame) == ("PM_000", 1, 180, 377)
    assert a.correct and not a.mocap_erroneous and not a.lights_on
    assert a.orientation(17) == "front" and a.orientation(18) == "profile"
    assert b.orientation(18) == "front" and b.mocap_erroneous and not b.correct
    assert b.extra_person_cam18 == 2


def test_paths():
    p = rehab24.joints_path("root", 1, "PM_000", 120)
    assert p.as_posix() == "root/3d_joints/Ex1/PM_000-120fps.npy"
    assert rehab24.video_path("root", 2, "PM_005", 18).name == "PM_005-Camera18-30fps-transposed.mp4"


def test_load_and_canonical_pose(tmp_path):
    j = np.random.default_rng(0).normal(size=(7, 26, 3))
    h = np.concatenate([2 * j, np.full((7, 26, 1), 2.0)], axis=-1)  # homogeneous, w = 2
    np.save(tmp_path / "x.npy", h)
    loaded = rehab24.load_joints(tmp_path / "x.npy")
    np.testing.assert_allclose(loaded, j)
    pose = rehab24.to_pose(loaded)
    assert set(pose) == set(JOINTS)
    np.testing.assert_array_equal(pose["r_elbow"], j[:, rehab24.JOINT_NAMES.index("RightForeArm")])


def test_file_table_matches_names():
    assert len(rehab24.JOINT_NAMES) == 26
    assert set(rehab24.CANONICAL.values()) <= set(rehab24.JOINT_NAMES)
