import numpy as np
import pytest

from mmcul.kinematics import elbow_flexion, shoulder_elevation, trunk_frame, trunk_tilt


def upright_pose(elbow, wrist, T=1):
    """Subject standing upright, y up, right shoulder at +x."""
    p = {
        "pelvis": [0.0, 1.0, 0.0],
        "r_shoulder": [0.2, 1.5, 0.0],
        "l_shoulder": [-0.2, 1.5, 0.0],
        "r_elbow": elbow,
        "r_wrist": wrist,
    }
    return {k: np.tile(np.asarray(v, float), (T, 1)) for k, v in p.items()}


def test_trunk_frame_is_rotation():
    R = trunk_frame(upright_pose([0.2, 1.2, 0.0], [0.2, 0.95, 0.0]))[0]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[:, 1], [0, 1, 0])  # superior
    np.testing.assert_allclose(R[:, 2], [1, 0, 0])  # right


def test_arm_hanging_straight():
    pose = upright_pose([0.2, 1.2, 0.0], [0.2, 0.95, 0.0])
    assert elbow_flexion(pose)[0] == pytest.approx(0.0, abs=1e-6)
    assert shoulder_elevation(pose)[0] == pytest.approx(0.0, abs=1e-6)


def test_arm_abducted_horizontal_elbow_right_angle():
    pose = upright_pose([0.5, 1.5, 0.0], [0.5, 1.8, 0.0])
    assert shoulder_elevation(pose)[0] == pytest.approx(90.0)
    assert elbow_flexion(pose)[0] == pytest.approx(90.0)


def test_elevation_is_relative_to_trunk():
    # whole body leaned 30 deg sideways, arm hanging along the trunk
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pose = {k: v @ rot.T for k, v in upright_pose([0.2, 1.2, 0.0], [0.2, 0.95, 0.0]).items()}
    assert shoulder_elevation(pose)[0] == pytest.approx(0.0, abs=1e-6)
    assert trunk_tilt(pose)[0] == pytest.approx(30.0)
