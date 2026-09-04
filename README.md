# EngineAI Navigation

[中文](README.md) | [English](README.en.md)

## 项目介绍

本仓库是基于 SRU 改进的人形机器人自主导航项目，面向 EngineAI PM01：用有序 LiDAR Range Image 感知环境，PPO 训练端到端策略，输出速度指令并由底层行走策略执行。

<table>
  <tr>
    <td align="center">
      <b>Gazebo 仿真</b><br>
      <video src="https://github.com/user-attachments/assets/1eacbb77-4296-4e40-9d69-1369e5372027" width="400" controls muted loop playsinline></video>
    </td>
    <td align="center">
      <b>真机部署</b><br>
      <video src="https://github.com/user-attachments/assets/16d721af-76e7-43ad-a1bf-36f25e6ffa82" width="390" controls muted loop playsinline></video>
    </td>
  </tr>
</table>

1. [项目介绍](#项目介绍)
2. [仓库结构](#仓库结构)
3. [训练](#训练)
   - [安装](#安装)
   - [冒烟测试](#冒烟测试)
   - [正式训练](#正式训练)
   - [推理与导出 ONNX](#推理与导出-onnx)
4. [ROS 2 部署](#ros-2-部署)
   - [依赖](#依赖)
   - [编译](#编译)
   - [仿真](#仿真)
   - [真机](#真机)
5. [许可证](#许可证)

## 仓库结构

仓库分两块：`training/` 负责 Isaac Lab 训练与 ONNX 导出；`ros/` 是 ROS 2 colcon 工作空间，再分成真机与仿真两套源码。

| 目录 | 用途 |
| --- | --- |
| [`training/`](training/) | Isaac Lab 任务、SRU 训练框架、训练 / 导出脚本 |
| [`ros/src/deployment/`](ros/src/deployment/) | 真机部署。已在 PM01 + Livox Mid-360 上验证 |
| [`ros/src/simulation/`](ros/src/simulation/) | Gazebo仿真。`rl_nav`、`rl_walking` 与真机代码不同 |

```text
engineai_navigation_opensource/
├── training/
│   ├── scripts/nav/
│   │   ├── train.py                       # 训练
│   │   ├── play.py                        # 推理 / 导出 ONNX
│   │   └── convert_checkpoint.py
│   ├── source/engineai_rl_lab/            # Gym 任务扩展
│   │   └── engineai_rl_lab/
│   │       ├── tasks/
│   │       │   ├── navigation/            # PM01 导航环境与 MDP
│   │       │   └── terrains/              # 迷宫地形
│   │       └── assets/pm01_edu_v2/        # 机器人URDF
│   └── sru-navigation-learning/           # SRU 版 rsl_rl（含 ActorCriticSRURangeImage）
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

真机与仿真的 `rl_nav_policy`、`mid360_preprocessor` 同名但实现不同，**不要混进同一个 install**。`super_lio` 和 `livox_ros_driver2` 只在真机树维护，仿真编译时用 `--base-paths` 一并编进来。所有 `colcon` 与 `ros2` 命令都在 `ros/` 下执行。

## 训练

请在 `training/` 下运行以下命令。任务名为 `Isaac-Navigation-PPO-PM01-v0`。

### 安装

从已配置好的 Isaac Lab 环境 `env_isaaclab` 复制一份，再安装本仓库的两个 Python 包。

`sru-navigation-learning` **必须安装**。它提供 SRU 版 `rsl_rl`，其中包含 Range Image 策略 `ActorCriticSRURangeImage`。Isaac Lab 自带的 `rsl-rl-lib` 不能训练本任务。

```bash
conda create --name sru --clone env_isaaclab
conda activate sru

cd training

# 卸掉 Isaac Lab 自带的 rsl_rl，避免挡住 SRU 实现
pip uninstall rsl-rl-lib -y

# 安装任务扩展（Gym 任务 Isaac-Navigation-PPO-PM01-v0）
pip install -e source/engineai_rl_lab

# 安装 SRU 训练框架（包名为 rsl_rl）
pip install -e sru-navigation-learning
```

验证安装：

```bash
python -c "from rsl_rl.modules import ActorCriticSRURangeImage; print('rsl_rl OK')"
pip show engineai_rl_lab rsl_rl
```

`pip show` 中的 `Editable project location` 应分别指向本仓库的 `training/source/engineai_rl_lab` 和 `training/sru-navigation-learning`。

### 冒烟测试

```bash
WANDB_MODE=disabled python scripts/nav/train.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 16 --headless --max_iterations 2
```

### 正式训练

```bash
WANDB_MODE=disabled python scripts/nav/train.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 2048 --headless --max_iterations 6000
```

恢复训练：

```bash
python scripts/nav/train.py \
  --task Isaac-Navigation-PPO-PM01-v0 \
  --resume \
  --max_iterations 8000 \
  --headless \
  --num_envs 2048 \
  --checkpoint <checkpoint.pt>
```

### 推理与导出 ONNX

未指定 `--checkpoint` 时，会自动加载 `training/logs/rsl_rl/pm01_navigation_range_image_ppo/` 下最新一次运行的最新模型。日志目录为 `logs/rsl_rl/pm01_navigation_range_image_ppo/<timestamp>/`。

```bash
WANDB_MODE=disabled python scripts/nav/play.py --task Isaac-Navigation-PPO-PM01-v0 --num_envs 16 --export_onnx --checkpoint <checkpoint.pt>
```

## ROS 2 部署

### 依赖

共同依赖：ROS 2 Humble、`libgoogle-glog-dev`、`libtbb-dev`、Eigen、PCL、yaml-cpp、NumPy、ONNX Runtime。

```bash
sudo apt install libgoogle-glog-dev libtbb-dev libyaml-cpp-dev
python3 -c "import numpy, onnxruntime"
```

Livox 驱动会在本工作空间内编译 SDK2，不依赖 `/usr/local` 里的预装库。若本机曾把 SDK 装到 `/usr/local`，运行前可加上：

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
```

### 编译

真机：

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

仿真源码不含 `super_lio` / Livox，编译时把真机树里的这两份加进 `--base-paths`：

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

### 仿真

在 `ros/` 下执行。每个新终端都要先 source Humble，再 source 提供 `simulator` 的 SDK，最后 source 本仓库的 simulation overlay。

**1. 启动仿真环境**

```bash
cd ros
source /opt/ros/humble/setup.bash
source /path/to/sdk_ws/install/setup.bash   # EngineAI SDK 的 simulator 包
source install/simulation/setup.bash
ros2 launch rl_nav_policy pm01_rl_nav_sim.launch.py
# 迷宫障碍环境
# ros2 launch rl_nav_policy pm01_rl_nav_sim.launch.py world:=ground.world
```

该 launch 会启动 Gazebo、行走控制器、仿真 Mid360、点云预处理和导航策略。Gazebo 开启后机器人处于倒地状态，先复位到站立：

```bash
ros2 service call /pm01/reset_stand std_srvs/srv/Empty "{}"
```

**2. 开启 Super-LIO 里程计**

同一终端可直接 launch。新终端需先重复上面的 `cd ros` 和三行 `source`。

```bash
ros2 launch super_lio Livox_mid360.py rviz:=true
# 退出时保存地图
# ros2 launch super_lio Livox_mid360.py rviz:=true save_map:=true
```

地图默认写到 `ros/src/deployment/Super-LIO-ros2/src/super_lio/map/`。

**3. 发导航目标**

在 RViz 中用 **2D Goal Pose** 发布 `/goal_pose`。或：

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world}, pose: {position: {x: 3.0, y: 0.0, z: 0.5}, orientation: {w: 1.0}}}"
```

Xbox 手柄：`MODE+X` 手柄控制，`MODE+Y` 强化学习导航。手柄模式下左十字键前后平移，右摇杆左右。没有手柄时：

```bash
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: true}"
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: false}"
```

### 真机

**1. 打开雷达**

```bash
cd ros
source /opt/ros/humble/setup.bash
source install/deployment/setup.bash

ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

**2. 开启 Super-LIO 里程计**

```bash
ros2 launch super_lio Livox_mid360.py
# ros2 launch super_lio Livox_mid360.py save_map:=true
```

**3. 导航**

```bash
ros2 launch rl_nav_policy pm01_rl_nav_deploy.launch.py
# ros2 launch rl_nav_policy pm01_rl_nav_deploy.launch.py rviz:=false
```

默认用 [`ros/src/deployment/rl_nav/rl_nav_policy/rviz/myconfig.rviz`](ros/src/deployment/rl_nav/rl_nav_policy/rviz/myconfig.rviz) 打开 RViz：Fixed Frame 为 `world`，**2D Goal Pose** 发到 `/goal_pose`。

真机默认手柄控制：`BACK+X` 把速度交还本体，`BACK+Y` 恢复强化学习导航。也可用：

```bash
ros2 service call /rl_nav_policy/set_policy_enabled std_srvs/srv/SetBool "{data: true}"
```

## 许可证

本仓库 **EngineAI 原创代码** 以 [Apache License 2.0](LICENSE) 发布。

Copyright 2026 Shenzhen Zhongqing Robot Technology Co., Ltd. (EngineAI)

内嵌的第三方组件仍按其原许可证分发，完整清单见 [NOTICE](NOTICE)。要点：

| 组件 | 路径 | 许可证 |
| --- | --- | --- |
| Isaac Lab 扩展模板 | `training/source/engineai_rl_lab/` | BSD-3-Clause |
| SRU / rsl_rl | `training/sru-navigation-learning/` | BSD-3-Clause |
| Range Image 导航任务 / 训练脚本 | `training/.../engineai_rl_lab/tasks/`、`training/scripts/nav/` | MIT |
| Livox SDK2、livox_ros_driver2 | `ros/src/deployment/livox/` | MIT |
| interface_protocol | `ros/src/deployment/protocol/interface_protocol/` | BSD-3-Clause |
| Super-LIO | `ros/src/deployment/Super-LIO-ros2/` | GPLv3 |
| Alibaba MNN | `ros/src/simulation/rl_walking/third_party/mnn/` | Apache-2.0 |

