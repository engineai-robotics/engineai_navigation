# EngineAI Navigation

[中文](README.md) | [English](README.en.md)

## Introduction

This repository is an SRU-based autonomous navigation project for the EngineAI PM01 humanoid robot. It perceives the environment from ordered LiDAR range images, trains an end-to-end PPO policy, and outputs velocity commands that the low-level walking policy executes.

<table>
  <tr>
    <td align="center">
      <b>Gazebo simulation</b><br>
      <video src="https://github.com/user-attachments/assets/1eacbb77-4296-4e40-9d69-1369e5372027" width="400" controls muted loop playsinline></video>
    </td>
    <td align="center">
      <b>Real-robot deployment</b><br>
      <video src="https://github.com/user-attachments/assets/16d721af-76e7-43ad-a1bf-36f25e6ffa82" width="400" controls muted loop playsinline></video>
    </td>
  </tr>
</table>

1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Training](#training)
   - [Installation](#installation)
   - [Smoke Test](#smoke-test)
   - [Full Training](#full-training)
   - [Inference and ONNX Export](#inference-and-onnx-export)
4. [ROS 2 Deployment](#ros-2-deployment)
   - [Dependencies](#dependencies)
   - [Build](#build)
   - [Simulation](#simulation)
   - [Real Robot](#real-robot)
5. [License](#license)

## Repository Structure

The repository has two parts: `training/` covers Isaac Lab training and ONNX export; `ros/` is a ROS 2 colcon workspace, split into real-robot and simulation source trees.

| Directory | Purpose |
| --- | --- |
| [`training/`](training/) | Isaac Lab tasks, SRU training framework, train / export scripts |
| [`ros/src/deployment/`](ros/src/deployment/) | Real-robot deployment. Validated on PM01 + Livox Mid-360 |
| [`ros/src/simulation/`](ros/src/simulation/) | Gazebo simulation. `rl_nav` and `rl_walking` differ from the real-robot code |

```text
engineai_navigation_opensource/
├── training/
│   ├── scripts/nav/
│   │   ├── train.py                       # training
│   │   ├── play.py                        # inference / export ONNX
│   │   └── convert_checkpoint.py
│   ├── source/engineai_rl_lab/            # Gym task extension
│   │   └── engineai_rl_lab/
│   │       ├── tasks/
│   │       │   ├── navigation/            # PM01 navigation env and MDP
│   │       │   └── terrains/              # maze terrains
│   │       └── assets/pm01_edu_v2/        # robot URDF
│   └── sru-navigation-learning/           # SRU fork of rsl_rl (includes ActorCriticSRURangeImage)
└── ros/                                  
    ├── src/deployment/
    │   ├── livox/                         # livox_sdk2 + livox_ros_driver2
    │   ├── protocol/interface_protocol
    │   ├── rl_nav/mid360_preprocessor
    │   ├── rl_nav/rl_nav_policy           
    │   └── Super-LIO-ros2/                
    ├── src/simulation/
    │   ├── rl_nav/mid360_preprocessor
    │   ├── rl_nav/rl_nav_policy
    │   └── rl_walking/                   
    ├── build/{deployment,simulation}
    └── install/{deployment,simulation}
```

Real-robot and simulation `rl_nav_policy` / `mid360_preprocessor` share names but have different implementations. **Do not mix them into the same install.** `super_lio` and `livox_ros_driver2` are maintained only in the real-robot tree; include them via `--base-paths` when building simulation. Run all `colcon` and `ros2` commands from `ros/`.

## Training

Run the following commands from `training/`. The task name is `Isaac-Navigation-PPO-PM01-v0`.

### Installation

Clone a copy of a working Isaac Lab environment `env_isaaclab`, then install this repository's two Python packages.

`sru-navigation-learning` **must be installed**. It provides the SRU fork of `rsl_rl`, including the range-image policy `ActorCriticSRURangeImage`. The `rsl-rl-lib` that ships with Isaac Lab cannot train this task.

```bash
conda create --name sru --clone env_isaaclab
conda activate sru

cd training

# Uninstall Isaac Lab's rsl_rl so it does not shadow the SRU implementation
pip uninstall rsl-rl-lib -y

# Install the task extension (Gym task Isaac-Navigation-PPO-PM01-v0)
pip install -e source/engineai_rl_lab

# Install the SRU training framework (package name: rsl_rl)
pip install -e sru-navigation-learning
```

Verify the install:

```bash
python -c "from rsl_rl.modules import ActorCriticSRURangeImage; print('rsl_rl OK')"
pip show engineai_rl_lab rsl_rl
```

The `Editable project location` from `pip show` should point to this repo's `training/source/engineai_rl_lab` and `training/sru-navigation-learning`.

### Smoke Test

```bash
WANDB_MODE=disabled python scripts/nav/train.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 16 --headless --max_iterations 2
```

### Full Training

```bash
WANDB_MODE=disabled python scripts/nav/train.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 2048 --headless --max_iterations 6000
```

Resume training:

```bash
python scripts/nav/train.py \
  --task Isaac-Navigation-PPO-PM01-v0 \
  --resume \
  --max_iterations 8000 \
  --headless \
  --num_envs 2048 \
  --checkpoint <checkpoint.pt>
```

### Inference and ONNX Export

If `--checkpoint` is omitted, the latest model from the most recent run under `training/logs/rsl_rl/pm01_navigation_range_image_ppo/` is loaded. Log directories are `logs/rsl_rl/pm01_navigation_range_image_ppo/<timestamp>/`.

```bash
WANDB_MODE=disabled python scripts/nav/play.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 16 --export_onnx --checkpoint <checkpoint.pt>
```

## ROS 2 Deployment

### Dependencies

Shared dependencies: ROS 2 Humble, `libgoogle-glog-dev`, `libtbb-dev`, Eigen, PCL, yaml-cpp, NumPy, ONNX Runtime.

```bash
sudo apt install libgoogle-glog-dev libtbb-dev libyaml-cpp-dev
python3 -c "import numpy, onnxruntime"
```

The Livox driver builds SDK2 inside this workspace and does not depend on a preinstalled copy in `/usr/local`. If the SDK was previously installed to `/usr/local`, you can add this before running:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
```

### Build

Real robot:

```bash
cd ros
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 colcon build \
  --base-paths src/deployment \
  --build-base build/deployment \
  --install-base install/deployment \
  --symlink-install
source install/deployment/setup.bash
```

Simulation sources do not include `super_lio` / Livox. Add those two packages from the real-robot tree via `--base-paths`:

```bash
cd ros
source /opt/ros/humble/setup.bash
PYTHONNOUSERSITE=1 colcon build \
  --base-paths src/simulation src/deployment/Super-LIO-ros2 src/deployment/livox \
  --build-base build/simulation \
  --install-base install/simulation \
  --symlink-install
source install/simulation/setup.bash
```

### Simulation

Run these from `ros/`. In each new terminal, source Humble first, then the SDK workspace that provides `simulator`, then this repository's simulation overlay.

**1. Start the simulation**

```bash
cd ros
source /opt/ros/humble/setup.bash
source /path/to/sdk_ws/install/setup.bash   # EngineAI SDK simulator package
source install/simulation/setup.bash
ros2 launch rl_nav_policy pm01_rl_nav_sim.launch.py
# maze obstacle world
# ros2 launch rl_nav_policy pm01_rl_nav_sim.launch.py world:=ground.world
```

This launch starts Gazebo, the walking controller, simulated Mid360, point-cloud preprocessing, and the navigation policy. After Gazebo opens, the robot is lying down; reset it to standing first:

```bash
ros2 service call /pm01/reset_stand std_srvs/srv/Empty "{}"
```

**2. Start Super-LIO odometry**

In the same terminal, launch directly. A new terminal must repeat `cd ros` and the three `source` commands above.

```bash
ros2 launch super_lio Livox_mid360.py rviz:=true
# save the map on exit
# ros2 launch super_lio Livox_mid360.py rviz:=true save_map:=true
```

Maps are written to `ros/src/deployment/Super-LIO-ros2/src/super_lio/map/` by default.

**3. Send a navigation goal**

In RViz, use **2D Goal Pose** to publish `/goal_pose`. Or:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: 3.0, y: 0.0, z: 0.5}, orientation: {w: 1.0}}}"
```

Xbox controller: `MODE+X` for joystick control, `MODE+Y` for RL navigation. In joystick mode, the left D-pad translates forward/back and the right stick turns left/right. Without a controller:

```bash
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: true}"
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: false}"
```

### Real Robot

**1. Start the LiDAR**

```bash
cd ros
source /opt/ros/humble/setup.bash
source install/deployment/setup.bash

ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

**2. Start Super-LIO odometry**

```bash
ros2 launch super_lio Livox_mid360.py
# ros2 launch super_lio Livox_mid360.py save_map:=true
```

**3. Navigate**

```bash
ros2 launch rl_nav_policy pm01_rl_nav_deploy.launch.py
# ros2 launch rl_nav_policy pm01_rl_nav_deploy.launch.py rviz:=false
```

RViz opens by default with [`ros/src/deployment/rl_nav/rl_nav_policy/rviz/myconfig.rviz`](ros/src/deployment/rl_nav/rl_nav_policy/rviz/myconfig.rviz): Fixed Frame is `world`, and **2D Goal Pose** publishes to `/goal_pose`.

The real robot defaults to joystick control: `BACK+X` returns velocity to the onboard controller, `BACK+Y` restores RL navigation. You can also use:

```bash
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: true}"
```

## License

**EngineAI original code** in this repository is released under the [Apache License 2.0](LICENSE).

Copyright 2026 Shenzhen Zhongqing Robot Technology Co., Ltd. (EngineAI)

Vendored third-party components remain under their original licenses. See [NOTICE](NOTICE) for the full list. Summary:

| Component | Path | License |
| --- | --- | --- |
| Isaac Lab extension template | `training/source/engineai_rl_lab/` | BSD-3-Clause |
| SRU / rsl_rl | `training/sru-navigation-learning/` | BSD-3-Clause |
| Range-image navigation task / training scripts | `training/.../engineai_rl_lab/tasks/`, `training/scripts/nav/` | MIT |
| Livox SDK2, livox_ros_driver2 | `ros/src/deployment/livox/` | MIT |
| interface_protocol | `ros/src/deployment/protocol/interface_protocol/` | BSD-3-Clause |
| Super-LIO | `ros/src/deployment/Super-LIO-ros2/` | GPLv3 |
| Alibaba MNN | `ros/src/simulation/rl_walking/third_party/mnn/` | Apache-2.0 |
