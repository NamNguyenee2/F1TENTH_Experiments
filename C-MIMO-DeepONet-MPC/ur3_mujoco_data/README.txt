# UR3e MuJoCo Data Generation Pipeline

A MuJoCo-based simulation pipeline for the **Universal Robots UR3e** manipulator that generates validated trajectory datasets containing joint states, end-effector pose, and end-effector velocity. The data is suitable for MPC, imitation learning, dynamics learning, and controller validation.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup](#setup)
3. [Pipeline Overview](#pipeline-overview)
4. [Step-by-Step Usage](#step-by-step-usage)
   - [Step 1 — Inspect the Model](#step-1--inspect-the-model)
   - [Step 2 — Visualize the Robot](#step-2--visualize-the-robot)
   - [Step 3 — Run a Single Trajectory](#step-3--run-a-single-trajectory)
   - [Step 4 — Validate Kinematics](#step-4--validate-kinematics)
   - [Step 5 — Validate Velocity Logging](#step-5--validate-velocity-logging)
   - [Step 6 — Generate a Kinematics Dataset](#step-6--generate-a-kinematics-dataset)
   - [Step 7 — Inspect and Read the Dataset](#step-7--inspect-and-read-the-dataset)
5. [Data Format](#data-format)
6. [Trajectory Types](#trajectory-types)
7. [Architecture](#architecture)

---

## Project Structure

```
ur3_mujoco_data/
├── models/
│   ├── scene.xml                  # Top-level MuJoCo scene (includes robot + floor)
│   └── ur3e/
│       ├── ur3e.xml               # Official UR3e MJCF (mesh-based, real kinematics)
│       └── assets/                # 20 .obj mesh files for visual + collision geometry
├── scripts/
│   ├── inspect_model.py           # Print joints, actuators, ee_site info
│   ├── view_model.py              # Launch interactive MuJoCo viewer
│   ├── run_single_trajectory.py   # Run one trajectory and save to HDF5
│   ├── validate_kinematics.py     # Validate joint order and rotation matrices
│   ├── validate_velocity.py       # Compare Jacobian velocity vs finite-diff
│   ├── generate_dataset.py        # Legacy trajectory dataset generator
│   ├── generate_kinematics_dataset.py  # Main split-based kinematics dataset generator
│   ├── inspect_kinematics_dataset.py   # Print split structure and shapes
│   ├── prepare_jax_training.py    # Load dataset for JAX training
│   └── replay_dataset.py          # Replay a saved trajectory in the viewer
├── src/
│   └── ur3_mujoco/
│       ├── model.py               # UR3Model: joint lookups, FK, Jacobian, state
│       ├── controller.py          # PDController: position servo
│       ├── trajectory.py          # TrajectoryGenerator: sine, spline, low-amp
│       ├── logger.py              # DataLogger: buffer → HDF5
│       ├── validation.py          # Validation checks (rotation matrix, NaN, etc.)
│       └── dataset.py             # DatasetGenerator: batch wrapper
├── data/                          # Generated HDF5 datasets and validation plots
├── notebooks/
│   └── inspect_dataset.ipynb      # Jupyter notebook for dataset inspection
└── docs/
    ├── MODEL_CONVENTIONS.md       # Joint order, units, frame conventions
    └── VALIDATION_REPORT.md       # Validation results
```

---

## Setup

### Requirements

- Python 3.8+
- A display is required for the viewer scripts (`view_model.py`, `replay_dataset.py`)

### Install dependencies

```bash
pip install mujoco numpy scipy h5py matplotlib
```

### Clone the project

Copy the entire `ur3_mujoco_data/` folder to your machine. No additional downloads are needed — the UR3e mesh assets are already included under `models/ur3e/assets/`.

### Verify the installation

```bash
cd ur3_mujoco_data
python scripts/inspect_model.py
```

Expected output:
```
Gate 2 PASS: All joints found, ee_site found.
```

---

## Pipeline Overview

The data generation pipeline follows this flow:

```
Robot Model (MJCF)
       │
       ▼
Trajectory Generator
  ├── single_joint_sine   → one joint oscillates, others fixed
  ├── multi_joint_smooth  → random waypoints, cubic spline interpolation
  └── low_amplitude       → small motions near home pose
       │
       ▼
MuJoCo Simulation Loop (500 Hz)
  ├── Controller sets ctrl = q_ref   (position servo)
  ├── mj_step() advances physics
  └── Logger records state at each step
       │
       ▼
DataLogger
  ├── q, q_dot, q_dot_dot          (from qpos, qvel, qacc — by joint name)
  ├── ee_pos, ee_quat, ee_rotmat   (from site_xpos, site_xmat)
  └── ee_lin_vel, ee_ang_vel       (J_lin @ q_dot,  J_rot @ q_dot)
       │
       ▼
HDF5 Dataset
  /metadata      ← simulation and generation parameters
  /trajectory_000
  /trajectory_001
  ...
```

Each trajectory is a self-contained group in the HDF5 file with all signals time-aligned at the simulation timestep (0.002 s = 500 Hz).

---

## Step-by-Step Usage

### Step 1 — Inspect the Model

Verifies that all 6 UR3e joints, actuators, and the end-effector site load correctly.

```bash
python scripts/inspect_model.py
```

**What it prints:**

```
joint name, joint id, qpos address, qvel address, joint range
actuator name, actuator id
ee_site id, position at home, rotation matrix
Gate 2 PASS: All joints found, ee_site found.
```

This is the correctness gate before any data generation.

---

### Step 2 — Visualize the Robot

Opens an interactive MuJoCo viewer with the UR3e at its zero configuration.

```bash
python scripts/view_model.py
```

The viewer window supports mouse orbit, pan, zoom, and joint inspection. Close the window to exit.

---

### Step 3 — Run a Single Trajectory

Runs a single shoulder-pan sine sweep (5 seconds), saves it to `data/single_trajectory.h5`, then reloads and prints all field shapes.

```bash
python scripts/run_single_trajectory.py
```

**What it does internally:**

1. Loads `UR3Model` from `models/scene.xml`
2. Generates a single-joint sine trajectory for joint 0 (shoulder_pan)
3. Resets simulation to the trajectory start pose
4. Steps the simulation for 2500 steps (5 s × 500 Hz)
5. At each step: sets `ctrl = q_ref`, calls `mj_step()`, logs all signals
6. Saves to HDF5, reloads, prints shape summary

Expected output:
```
Gate 5 PASS: Trajectory saved and reloaded successfully.
```

---

### Step 4 — Validate Kinematics

Runs one trajectory per joint (6 total), verifying that only the commanded joint shows significant motion. Also checks that rotation matrices are valid (`det(R) ≈ 1`, `RᵀR ≈ I`).

```bash
python scripts/validate_kinematics.py
```

Saves a plot to `data/validate_joint_order.png` showing each joint's response.

**What to check:**
- Row `i` should show joint `i` as the most active — confirms joint name → actuator mapping is correct
- Rotation matrix errors should be `< 1e-6`

Expected output:
```
Gate 3 + 6 (kinematics): PASS
```

---

### Step 5 — Validate Velocity Logging

Runs a 10-second multi-joint trajectory and compares:
- `ee_lin_vel` logged via Jacobian (`J_lin @ q_dot`) against finite-difference of `ee_pos`
- `q_dot_dot` (from `data.qacc`) against finite-difference of `q_dot`

```bash
python scripts/validate_velocity.py
```

Saves a 4-panel plot to `data/validate_velocity.png`.

**What to check:**
- EE velocity p99 error should be `< 0.2 m/s` (mean typically `< 0.01 m/s`)
- Joint acceleration difference vs finite-diff is informational — `qacc` is the full MuJoCo dynamics result and naturally differs from a numerical derivative

Expected output:
```
Gate 6 velocity: PASS (EE vel p99=0.10 m/s | acc p99=197 rad/s² [informational])
```

---

### Step 6 — Generate a Kinematics Dataset

The current generator is `scripts/generate_kinematics_dataset.py`. It reads a YAML config, samples random robot states, generates debug trajectories, and writes a split-based HDF5 dataset.

```bash
python scripts/generate_kinematics_dataset.py --config configs\kinematics_debug.yaml --output data\debug\ur3_kinematics_debug.h5
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | YAML config under `configs\` that defines robot limits, split sizes, seeds, and trajectory settings |
| `--output` | yes | Output HDF5 path |

**Useful presets:**

```bash
# Quick sanity check
python scripts/generate_kinematics_dataset.py --config configs\kinematics_debug.yaml --output data\debug\ur3_kinematics_debug.h5

# Smaller training-ready dataset
python scripts/generate_kinematics_dataset.py --config configs\kinematics_small.yaml --output data\small\ur3_kinematics_small.h5

# Main dataset
python scripts/generate_kinematics_dataset.py --config configs\kinematics_main.yaml --output data\main\ur3_kinematics_main.h5
```

**What the generator writes:**

- `/train` — main training split
- `/val` — validation split
- `/test_random` — held-out random samples
- `/test_trajectory` — held-out spline-trajectory samples
- `/test_stress` — optional higher-speed stress split if enabled in config
- `/debug_trajectories` — trajectory groups for inspection and visualization

Each split contains both raw fields (`q`, `q_dot`, Jacobians, poses, velocities) and learning-ready tensors:

- `x` with shape `[N, 18]` = `[sin(q), cos(q), q_dot]`
- `y` with shape `[N, 15]` = `[ee_pos, ee_rot6d, ee_lin_vel, ee_ang_vel]`

---

### Step 7 — Inspect and Read the Dataset

**Quick structure check:**

```bash
python scripts/inspect_kinematics_dataset.py --dataset data\small\ur3_kinematics_small.h5
```

This prints metadata, available splits, and the shape of each dataset field.

**Prepare and sanity-check it for JAX training:**

```bash
python scripts/prepare_jax_training.py --dataset data\small\ur3_kinematics_small.h5
```

This loads the dataset through `KinematicsDataset`, fits a normalizer on the train split, saves `normalizer.json`, and prints example batch shapes.

**Notebook inspection:**

```bash
jupyter notebook notebooks\inspect_dataset.ipynb
```

The notebook is useful for plotting fields such as `q`, `q_dot`, `ee_pos_world`, and `ee_lin_vel_world`.

---

## Data Format

Datasets are saved as **HDF5** files.

### File structure

```
ur3_kinematics_*.h5
├── metadata/
│   attrs:
│     robot_name
│     mujoco_version
│     model_path
│     model_hash
│     joint_names
│     joint_limits
│     q_velocity_limits
│     ee_site_name
│     pose_frame
│     velocity_frame
│     orientation_convention
│     sampling_strategy
│     date_generated
│
├── train/
├── val/
├── test_random/
├── test_trajectory/
├── test_stress/              # optional
│   ├── q                [N, 6]
│   ├── q_dot            [N, 6]
│   ├── sin_q            [N, 6]
│   ├── cos_q            [N, 6]
│   ├── x                [N, 18]
│   ├── ee_pos_world     [N, 3]
│   ├── ee_quat_world    [N, 4]
│   ├── ee_rotmat_world  [N, 9]
│   ├── ee_rot6d_world   [N, 6]
│   ├── ee_lin_vel_world [N, 3]
│   ├── ee_ang_vel_world [N, 3]
│   ├── J_pos_world      [N, 3, 6]
│   ├── J_rot_world      [N, 3, 6]
│   ├── J_world          [N, 6, 6]
│   └── y                [N, 15]
│
└── debug_trajectories/
    ├── single_joint_sine/
    │   └── traj_000/ ...
    └── multi_joint_sine/
        └── traj_000/ ...
```

`N` is the number of samples in a split.

### Reading with `h5py`

```python
import h5py

dataset_path = r"data\small\ur3_kinematics_small.h5"

with h5py.File(dataset_path, "r") as f:
    metadata = dict(f["metadata"].attrs)
    print(metadata["joint_names"])

    x_train = f["train"]["x"][:]
    y_train = f["train"]["y"][:]
    q_train = f["train"]["q"][:]
    ee_pos = f["train"]["ee_pos_world"][:]

print(x_train.shape)  # [N, 18]
print(y_train.shape)  # [N, 15]
print(q_train.shape)  # [N, 6]
print(ee_pos.shape)   # [N, 3]
```

### Reading with the built-in JAX loader

```python
from ur3_mujoco.training.jax_dataloader import KinematicsDataset

ds = KinematicsDataset(r"data\small\ur3_kinematics_small.h5", normalize=True)
ds.fit_normalizer(split="train")

x_val, y_val = ds.get_split("val")

for x_batch, y_batch in ds.batches("train", batch_size=256, shuffle=True, rng_seed=0):
    print(x_batch.shape, y_batch.shape)
    break
```

In this format:

- `x = [sin(q), cos(q), q_dot]`
- `y = [ee_pos_world, ee_rot6d_world, ee_lin_vel_world, ee_ang_vel_world]`

---

## Trajectory Types

### `sine` — Single-joint sine sweep

Each trajectory commands one joint through a sinusoidal motion while all others hold the home pose. Cycles through joints round-robin: trajectory 0 → joint 0, trajectory 1 → joint 1, etc.

- Amplitude: ±0.4 rad
- Frequency: 0.2 Hz
- Purpose: joint order verification, actuator mapping validation

### `multi` — Multi-joint smooth (recommended for datasets)

Generates random joint-space waypoints (within 50% of joint limits) and interpolates with a **cubic spline** (clamped boundary — zero velocity at start/end). All 6 joints move simultaneously with smooth, physically plausible motion.

- Waypoints: 8 per trajectory
- Boundary condition: starts and ends at home pose
- Seed-controlled for reproducibility
- Purpose: general-purpose dataset for learning and control

### `low_amp` — Low-amplitude multi-joint

Small sinusoidal perturbations on all joints simultaneously, each at a slightly different frequency. All motion stays within ±0.1 rad of the home pose.

- Amplitude: 0.1 rad per joint
- Frequencies: 0.10 – 0.25 Hz (varied per joint)
- Purpose: numerical stability testing, linearization datasets

---

## Architecture

### How velocity is computed

End-effector velocity is **not** finite-differenced. It is computed via the **analytical Jacobian** at each timestep:

```
ee_lin_vel = J_translational(q) @ q_dot     shape: [3]
ee_ang_vel = J_rotational(q)    @ q_dot     shape: [3]
```

where the Jacobian is obtained from `mujoco.mj_jacSite()` and sliced to the 6 UR3e DOF columns.

### How joints are indexed

Joint values are **always accessed by name**, never by raw array index:

```python
q[i] = data.qpos[model.jnt_qposadr[mj_name2id(model, mjOBJ_JOINT, joint_name)]]
```

This protects against any MJCF ordering changes.

### Controller

Position actuators in the MJCF handle the servo internally. The controller simply passes `q_ref` as the control signal:

```
ctrl = q_ref      (desired joint position)
```

The MJCF actuator class (`size0/size1/size2`) applies appropriate `kp` and damping gains per joint based on UR3e torque specs.

---

## Robot Model

**Source:** [SouthColumn76/universal_robots_ur3e](https://github.com/SouthColumn76/universal_robots_ur3e) — a high-quality UR3e MJCF based on official Universal Robots geometry.

**Key properties:**

| Property | Value |
|---|---|
| DOF | 6 revolute joints |
| Payload | 3 kg |
| Reach | ~500 mm |
| Simulation timestep | 0.002 s (500 Hz) |
| Integrator | `implicitfast` |
| Actuator type | `general` (position servo with PD gains) |
| EE site | `ee_site` at flange/tool0 frame on `wrist_3_link` |
| Home pose | `[0, -π/2, π/2, -π/2, -π/2, 0]` rad |

**Joint order (canonical):**

```
[0] shoulder_pan_joint
[1] shoulder_lift_joint
[2] elbow_joint
[3] wrist_1_joint
[4] wrist_2_joint
[5] wrist_3_joint
```
