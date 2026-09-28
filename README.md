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
| 1 · Benchmark | MediaPipe, SAM 3D Body (later RTMW + lifting) vs. mocap: joint angles and measures (Bland-Altman, ICC) — pipeline ✅, runs 🚧 | 🚧 |
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
src/mmcul/datasets/rehab24.py  REHAB24-6 download (incl. single files from the zips), segmentation, ground truth
src/mmcul/pose/                pose model runners (MediaPipe, SAM 3D Body), resumable chunked cache
src/mmcul/align.py             one similarity transform per video (Umeyama)
src/mmcul/evaluate.py          per-repetition comparison with ground truth
src/mmcul/agreement.py         Bland-Altman, ICC(A,1)
src/mmcul/benchmark.py         phase 1: all cached videos of a model vs. REHAB24-6 ground truth
tools/run_pose.py              CLI: fetch data, run a pose model on all videos (resumable, parallel)
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
`MyDrive/mmc_upper_limb/`. MediaPipe runs on the CPU, so it is faster locally in parallel:

```bash
.venv/Scripts/pip install -e ".[pose,eval,dev]"
python tools/run_pose.py --root data/rehab24 --cache cache/rehab24 --fetch --model mediapipe-heavy --workers 4
```

Then copy `cache/rehab24/mediapipe-heavy/` to `MyDrive/mmc_upper_limb/cache/rehab24/`. SAM 3D Body runs in
notebook 02 (T4, gated checkpoint on Hugging Face).

### Evaluation protocol (phase 1)

* **Subject box** from the projected 2D ground truth (oracle box), so pose accuracy is measured separately
  from person detection — camera 17 often shows a second person.
* **Joint angles** (elbow flexion, humerothoracic elevation) are invariant to the camera frame and need no
  alignment.
* **Positions, speeds, trunk displacement** are compared after *one* similarity transform per video,
  fitted with the ground truth (oracle scale and frame). Per-frame alignment would erase the trunk and hand
  motion that the measures quantify.
* The trunk point is the shoulder midpoint for all sources (skeletons differ below the neck).

### First observation (one video, not yet a result)

On one Ex1 video, MediaPipe's 2D keypoints are smooth, but its 3D *world landmarks* jump by > 10 cm between
frames about 200 times in 1300 frames, mostly along the depth axis, and the upper-arm length varies between
16 and 32 cm. Seen from the side, it confuses the occluded arm with the visible one (wrist visibility ≈ 0.01).
Joint angles from the front are usable (shoulder elevation ICC 0.82), speed-based measures are not. The full
benchmark over all Ex1/Ex2 videos will show whether this holds.

## Limitations

Ground truth comes from healthy subjects; transfer to stroke survivors is only checked indirectly
(Toronto Rehab, skeleton data) and by simulated compensation in own recordings. Human3.6M / MoVi actions
are proxies for the drinking task, and REHAB24-6 exercises are cyclic rather than discrete reaching tasks.

## License

Code: MIT. Datasets keep their own licenses and are not redistributed here.
