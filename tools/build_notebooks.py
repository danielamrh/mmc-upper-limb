"""Generate the Colab notebooks in notebooks/ (source of truth for their cells).

    python tools/build_notebooks.py
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
GH = "danielamrh/mmc-upper-limb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def badge(name):
    return (f'<a href="https://colab.research.google.com/github/{GH}/blob/main/notebooks/{name}.ipynb">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>')


SETUP = [
    md("## 0 · Drive + repo + install\nCode comes from GitHub (always fresh via `git pull`), data and results live on Google Drive."),
    code("""
import os, sys
from google.colab import drive
drive.mount("/content/drive")

REPO = "mmc-upper-limb"
REPO_DIR = f"/content/{REPO}"
if not os.path.exists(REPO_DIR):
    !git clone -q https://github.com/danielamrh/{REPO}.git {REPO_DIR}
else:
    !git -C {REPO_DIR} pull -q
%cd {REPO_DIR}
COMMIT = !git rev-parse --short HEAD
COMMIT = COMMIT[0]
print("commit:", COMMIT)

!pip install -q -e .
sys.path.insert(0, f"{REPO_DIR}/src")  # editable install is only picked up after a restart
# re-running this cell must pick up freshly pulled code
for name in [m for m in sys.modules if m == "mmcul" or m.startswith("mmcul.")]:
    del sys.modules[name]

DRIVE_ROOT = "/content/drive/MyDrive/mmc_upper_limb"
os.makedirs(DRIVE_ROOT, exist_ok=True)
"""),
]


def nb01():
    cells = [
        md(f"""
# 01 · REHAB24-6: data and ground-truth measures
{badge("01_rehab24_data")}

Downloads the REHAB24-6 ground truth (OptiTrack joints + repetition segmentation, ~0.55 GB) to Drive,
checks units and skeleton, and computes the movement quality measures on the **mocap ground truth**.
Every later notebook compares pose-model estimates against exactly these numbers.

Dataset: Černek, Sedmidubsky & Budikova, *REHAB24-6*, SISAP 2024 — Zenodo 13305826, CC BY-NC 4.0.
"""),
        *SETUP,
        md("## 1 · Download + extract\nThe 3D joints are small enough to extract on Drive; videos come later (2.7 GB)."),
        code("""
from pathlib import Path
from mmcul.datasets import rehab24

DATA = Path(DRIVE_ROOT) / "data" / "rehab24"
rehab24.download(["Segmentation.csv", "3d_joints.zip"], DATA)
rehab24.extract(DATA / "3d_joints.zip", DATA)
reps = rehab24.read_segmentation(DATA / "Segmentation.csv")
print(len(reps), "repetitions")
"""),
        md("## 2 · Overview\nRepetitions per exercise and viewpoint. The viewpoint label is what makes this dataset useful for the robustness study (phase 3)."),
        code("""
import pandas as pd
df = pd.DataFrame([r.__dict__ for r in reps])
df["exercise_name"] = df.exercise.map(rehab24.EXERCISES)
display(pd.crosstab(df.exercise_name, df.cam17_orientation, margins=True))
display(pd.crosstab(df.exercise_name, df.correct))
print("repetitions with mocap errors:", df.mocap_erroneous.sum())
"""),
        md("## 3 · Units and skeleton sanity check\nSegment lengths should be constant over time and plausible in metres."),
        code("""
import numpy as np
from mmcul.kinematics import JOINTS

rows = []
for vid, ex in df[["video_id", "exercise"]].drop_duplicates().itertuples(index=False):
    p = rehab24.to_pose(rehab24.load_joints(rehab24.joints_path(DATA, ex, vid)))
    for name, (a, b) in {"upper arm": ("r_shoulder", "r_elbow"), "forearm": ("r_elbow", "r_wrist"),
                         "shoulder width": ("l_shoulder", "r_shoulder")}.items():
        L = np.linalg.norm(p[a] - p[b], axis=1)
        rows.append(dict(video=vid, segment=name, mean_m=L.mean(), std_mm=1000 * L.std()))
seg = pd.DataFrame(rows)
display(seg.groupby("segment")[["mean_m", "std_mm"]].describe().round(3))
"""),
        md("## 4 · One repetition up close"),
        code("""
import matplotlib.pyplot as plt
from mmcul.kinematics import elbow_flexion, shoulder_elevation
from mmcul.measures import lowpass, speed, movement_bounds

FS = 30.0
rep = next(r for r in reps if r.exercise == 1 and not r.mocap_erroneous)
pose = {k: lowpass(v, FS) for k, v in rehab24.repetition_pose(DATA, rep).items()}
t = np.arange(len(pose["pelvis"])) / FS
v = speed(pose["r_wrist"], FS)
s, e = movement_bounds(v)

fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax[0].plot(t, v); ax[0].axvspan(t[s], t[e], alpha=.15); ax[0].set_ylabel("wrist speed [m/s]")
ax[1].plot(t, shoulder_elevation(pose)); ax[1].set_ylabel("shoulder elevation [°]")
ax[2].plot(t, elbow_flexion(pose)); ax[2].set_ylabel("elbow flexion [°]"); ax[2].set_xlabel("t [s]")
fig.suptitle(f"{rep.video_id} rep {rep.repetition} · {rehab24.EXERCISES[rep.exercise]} · correct={rep.correct}")
plt.tight_layout()
"""),
        md("""
## 5 · Ground-truth measures for the arm exercises
Ex1 (arm abduction) and Ex2 (arm VW). These are cyclic exercises, not the discrete drinking task, so
onset/offset detection can cut a repetition to its main phase; what matters later is that estimates and
ground truth go through the same code. Results are saved to Drive.
"""),
        code("""
from mmcul.measures import compute_measures

rows = []
for r in reps:
    if r.exercise not in (1, 2) or r.mocap_erroneous:
        continue
    m = compute_measures(rehab24.repetition_pose(DATA, r), FS)
    rows.append({**r.__dict__, **m.as_dict()})
gt = pd.DataFrame(rows)
out = Path(DRIVE_ROOT) / "results" / "rehab24_gt_measures.csv"
out.parent.mkdir(parents=True, exist_ok=True)
gt.to_csv(out, index=False)
print("saved", out, len(gt), "repetitions · commit", COMMIT)
display(gt.groupby(["exercise", "correct"])[["movement_time", "n_movement_units", "peak_speed",
        "trunk_displacement", "min_elbow_flexion", "max_shoulder_elevation"]].median().round(3))
"""),
        md("Do the measures separate correct from incorrect repetitions? (Only a sanity check: the mistakes vary per subject.)"),
        code("""
cols = ["movement_time", "n_movement_units", "trunk_displacement", "min_elbow_flexion", "max_shoulder_elevation"]
fig, axes = plt.subplots(2, len(cols), figsize=(3 * len(cols), 6))
for i, ex in enumerate((1, 2)):
    d = gt[gt.exercise == ex]
    for ax, c in zip(axes[i], cols):
        ax.boxplot([d[d.correct][c], d[~d.correct][c]], tick_labels=["correct", "incorrect"])
        ax.set_title(f"Ex{ex} · {c}", fontsize=9)
plt.tight_layout()
"""),
    ]
    return cells


def write(name, cells):
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = NB_DIR / f"{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    write("01_rehab24_data", nb01())
