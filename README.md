# How Trustworthy Are Webcam-Based Upper-Limb Movement Quality Measures?

Self-study computer vision project on **markerless motion capture (MMC) for upper-limb stroke rehabilitation**.

Kinematic measures of the drinking task — movement time, smoothness (number of movement units),
trunk displacement, elbow extension, shoulder elevation — separate impairment levels after stroke and
track recovery ([Alt Murphy et al. 2011](https://doi.org/10.1177/1545968310370748), 2012, 2013). They
normally need a marker-based lab. Multi-camera MMC with biomechanical fitting already comes close to
it ([Unger et al., ICORR 2025](https://arxiv.org/abs/2411.14992)). For **home rehabilitation with a single
webcam**, three questions are still open:

1. How accurate are current monocular pose models on these *clinical measures*, not just on joint positions?
2. How strongly do viewpoint, occlusion and image quality affect them?
3. Can the system say **when a measure is not trustworthy** (calibrated uncertainty from one camera)?

## Plan

| Phase | Goal | Status |
|---|---|---|
| 0 · Setup | Repo, measures (Alt Murphy definitions) with tests, REHAB24-6 loader | ✅ |
| 1 · Benchmark | MediaPipe, RTMPose/RTMW + lifting, SAM 3D Body vs. mocap: joint angles and measures (Bland-Altman, ICC) | ⏳ |
| 2 · Biomechanical fit | Arm + trunk kinematic model (constant bone lengths, joint limits, smoothness) vs. raw keypoints | ⏳ |
| 3 · Robustness | Viewpoint (front / half-profile / profile), occlusion, resolution, frame rate, blur | ⏳ |
| 4 · Method | Calibrated per-measure uncertainty from a single camera (ensembles, TTA, Laplace around the fit) | ⏳ |
| 5 · Application | Compensation on real stroke data (Toronto Rehab), own webcam drinking-task demo | ⏳ |

## Data

| Dataset | Content | Role |
|---|---|---|
| [REHAB24-6](https://zenodo.org/records/13305826) | 10 subjects, 2 RGB views + OptiTrack, 6 rehab exercises, viewpoint labels | main ground truth (CC BY-NC 4.0) |
| Human3.6M | 4 views + Vicon; *Eating / Phoning / Smoking* as hand-to-mouth proxy | drinking-task proxy |
| [MoVi](https://www.biomotionlab.ca/movi/) | 90 subjects, video + mocap + IMU | cross-subject validation |
| [TotalCapture](https://cvssp.org/data/totalcapture/) | 8 views, 13 IMUs, Vicon | viewpoint study |
| [Toronto Rehab Stroke Pose](https://dl.acm.org/doi/10.1145/3154862.3154925) | Kinect skeletons, 9 stroke survivors, expert compensation labels | real patient data |

No public dataset contains video + optical mocap of stroke survivors performing the drinking task; ground
truth therefore comes from healthy subjects (see limitations).

## Repository layout

```
src/mmcul/kinematics.py        canonical joints, trunk frame, elbow flexion, shoulder elevation
src/mmcul/measures.py          Alt Murphy measures: MT, NMU, peak speed, trunk displacement, ...
src/mmcul/datasets/rehab24.py  REHAB24-6 download, segmentation, ground-truth loader
tools/build_notebooks.py       source of truth for notebooks/ (edit this, not the .ipynb)
notebooks/                     Colab notebooks (T4), data and results on Google Drive
tests/                         pytest suite
```

Measure definitions: 2nd-order zero-phase Butterworth low-pass at 6 Hz; onset / offset where end-effector
speed crosses 2 % of its peak; a movement unit is a speed minimum followed by a maximum more than 20 mm/s
higher, with ≥ 150 ms between counted peaks; trunk displacement is the maximal thorax distance from its
position at onset.

## Running

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows; bin/ on Linux
.venv/Scripts/python -m pytest
python tools/build_notebooks.py
```

Notebooks run in Colab: open them via the badge, they clone this repo and write to
`MyDrive/mmc_upper_limb/`.

## Limitations

Ground truth comes from healthy subjects; transfer to stroke survivors is only checked indirectly
(Toronto Rehab, skeleton data) and by simulated compensation in own recordings. Human3.6M / MoVi actions
are proxies for the drinking task, and REHAB24-6 exercises are cyclic rather than discrete reaching tasks.

## License

Code: MIT. Datasets keep their own licenses and are not redistributed here.
