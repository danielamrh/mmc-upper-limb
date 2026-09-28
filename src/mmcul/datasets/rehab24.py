"""REHAB24-6 (Černek et al. 2024): two RGB views + OptiTrack ground truth.

Zenodo record 13305826, CC BY-NC 4.0. Layout after `download` + `extract`:

    root/Segmentation.csv
    root/3d_joints/Ex{1..6}/{video_id}-{30,120}fps.npy      (T, 26, 4) homogeneous, metres, y up
    root/videos/Ex{1..6}/{video_id}-Camera17-30fps.mp4      horizontal camera
    root/videos/Ex{1..6}/{video_id}-Camera18-30fps-transposed.mp4

Segmentation frame indices refer to the 30 fps data only.
"""

from __future__ import annotations

import csv
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..kinematics import Pose

RECORD = "13305826"
FILES = {
    "Segmentation.csv": 53_290,
    "3d_joints.zip": 551_298_977,
    "2d_joints.zip": 693_401_748,
    "3d_markers.zip": 668_016_975,
    "2d_markers.zip": 1_106_241_715,
    "videos.zip": 2_651_613_914,
}

JOINT_NAMES = (
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end",
)

# canonical name -> OptiTrack skeleton joint ("*Arm" is the glenohumeral joint,
# "*Shoulder" the sternoclavicular one)
CANONICAL = {
    "pelvis": "Hips", "thorax": "Spine1", "neck": "Neck", "head": "Head",
    "r_shoulder": "RightArm", "r_elbow": "RightForeArm", "r_wrist": "RightHand", "r_hand": "RightHand_end",
    "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm", "l_wrist": "LeftHand", "l_hand": "LeftHand_end",
}

EXERCISES = {1: "Arm abduction", 2: "Arm VW", 3: "Push-ups", 4: "Leg abduction", 5: "Leg lunge", 6: "Squats"}

# the cameras are orthogonal, so the subject's orientation towards camera 18 follows from camera 17
_CAM18_ORIENTATION = {"front": "profile", "half-profile": "half-profile", "profile": "front"}


def url(name: str) -> str:
    return f"https://zenodo.org/records/{RECORD}/files/{name}?download=1"


def download(names: list[str], root: str | Path, chunk: int = 1 << 22) -> None:
    """Download files from Zenodo into root, skipping complete ones."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        dst = root / name
        if dst.exists() and dst.stat().st_size == FILES[name]:
            print(f"{name}: already there")
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"{name}: downloading {FILES[name] / 1e9:.2f} GB ...", flush=True)
        with urllib.request.urlopen(url(name)) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, chunk)
        if tmp.stat().st_size != FILES[name]:
            raise IOError(f"{name}: got {tmp.stat().st_size} bytes, expected {FILES[name]}")
        tmp.replace(dst)


def extract(zip_path: str | Path, root: str | Path) -> Path:
    """Unzip e.g. 3d_joints.zip into root/3d_joints/ (skipped if it exists)."""
    zip_path = Path(zip_path)
    out = Path(root) / zip_path.stem
    if not out.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out)
    return out


@dataclass(frozen=True)
class Repetition:
    video_id: str
    repetition: int
    exercise: int
    person: int
    first_frame: int
    last_frame: int  # inclusive, 30 fps
    cam17_orientation: str
    mocap_erroneous: bool
    subtype: str
    lights_on: bool
    extra_person_cam17: int
    extra_person_cam18: int
    correct: bool

    @property
    def cam18_orientation(self) -> str:
        return _CAM18_ORIENTATION[self.cam17_orientation]

    def orientation(self, camera: int) -> str:
        return self.cam17_orientation if camera == 17 else self.cam18_orientation


def read_segmentation(src: str | Path | io.TextIOBase) -> list[Repetition]:
    """Parse Segmentation.csv (semicolon separated)."""
    f = open(src, newline="", encoding="utf-8") if isinstance(src, (str, Path)) else src
    try:
        return [
            Repetition(
                video_id=r["video_id"],
                repetition=int(r["repetition_number"]),
                exercise=int(r["exercise_id"]),
                person=int(r["person_id"]),
                first_frame=int(r["first_frame"]),
                last_frame=int(r["last_frame"]),
                cam17_orientation=r["cam17_orientation"],
                mocap_erroneous=bool(int(r["mocap_erroneous"])),
                subtype=r["exercise_subtype"],
                lights_on=bool(int(r["lights_on"])),
                extra_person_cam17=int(r["extra_person_in_cam17"]),
                extra_person_cam18=int(r["extra_person_in_cam18"]),
                correct=bool(int(r["correctness"])),
            )
            for r in csv.DictReader(f, delimiter=";")
        ]
    finally:
        if f is not src:
            f.close()


def joints_path(root: str | Path, exercise: int, video_id: str, fps: int = 30) -> Path:
    return Path(root) / "3d_joints" / f"Ex{exercise}" / f"{video_id}-{fps}fps.npy"


def video_path(root: str | Path, exercise: int, video_id: str, camera: int) -> Path:
    suffix = "30fps" if camera == 17 else "30fps-transposed"
    return Path(root) / "videos" / f"Ex{exercise}" / f"{video_id}-Camera{camera}-{suffix}.mp4"


def load_joints(path: str | Path) -> np.ndarray:
    """Ground-truth joints as (T, 26, 3) in metres (homogeneous coordinate dropped)."""
    j = np.load(path)
    return j[..., :3] / j[..., 3:4]


def to_pose(joints: np.ndarray) -> Pose:
    """(T, 26, 3) OptiTrack joints -> canonical pose dict."""
    return {name: joints[:, JOINT_NAMES.index(src)] for name, src in CANONICAL.items()}


def repetition_pose(root: str | Path, rep: Repetition, pad: int = 0) -> Pose:
    """Ground-truth pose of one repetition at 30 fps, optionally with `pad` frames
    of context on each side. Repetitions follow each other without a gap, so any
    padding reaches into the neighbouring repetition."""
    j = load_joints(joints_path(root, rep.exercise, rep.video_id))
    lo, hi = max(rep.first_frame - pad, 0), min(rep.last_frame + 1 + pad, len(j))
    return to_pose(j[lo:hi])
