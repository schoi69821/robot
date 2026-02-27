# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Overview

This is a robotics development workstation for PiPER robot arms (AgileX Robotics) used for imitation learning. The system runs Ubuntu 22.04 with CUDA 11.8, ROS2 Humble, and ROS1 Noetic (in Docker). The primary language is Korean for the user, with Chinese/English in upstream code comments.

## Workspaces

### cobot_magic_ws (ROS1 Noetic + Docker)
Main imitation learning workspace based on the [ALOHA](https://github.com/agilexrobotics/cobot_magic) project. Contains two configurations:
- **cobot_magic_two/**: Two PiPER arms (left + right)
- **cobot_magic_four/**: Four PiPER arms (two master + two puppet), more feature-rich

Each configuration has the same internal structure:
- `collect_data/` - ROS1 data collection (subscribes to camera and arm joint topics, saves HDF5)
- `aloha-devel/act/` - ACT/Diffusion/CNNMLP policy training and inference
- `piper_ros_ws/` - ROS1 Noetic catkin workspace for PiPER arm control
- `camera_ws/` - ROS1 catkin workspace for cameras (Astra, RealSense)

**This workspace runs inside Docker** (nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 + ROS Noetic).

### piper_ws (ROS2 Humble, native)
PiPER SDK and ROS2 workspace for direct robot arm control. Built with colcon. Sourced in `.bashrc`.
- `src/piper_sdk/` - Python SDK for PiPER arms over CAN bus (interface, protocol, kinematics)
- `src/piper_ros/` - ROS2 node wrappers
- CAN configuration scripts at workspace root (`can_activate.sh`, `can_config.sh`)

### async_diffusion_policy-master
Standalone async diffusion policy project (has its own CLAUDE.md). Dual-process architecture: 15 Hz inference + 500 Hz control via shared memory. Uses `uv` for dependency management.

## Common Commands

### CAN Bus Setup (required before robot operation)
```bash
# Activate single CAN interface (default: can0, 1Mbps)
cd ~/piper_ws && bash can_activate.sh can0 1000000
# With specific USB address (when multiple CAN adapters)
bash can_activate.sh can0 1000000 1-2:1.0
# Multi-CAN configuration
bash can_config.sh
```

### piper_ws (ROS2, native host)
```bash
source ~/piper_ws/install/setup.bash  # already in .bashrc
# Build
cd ~/piper_ws && colcon build
```

### cobot_magic_ws (ROS1, inside Docker)
```bash
# Build camera workspace
cd cobot_magic_four/camera_ws && catkin_make
# Build piper ROS workspace
cd cobot_magic_four/piper_ros_ws && catkin_make
# Or use the all-in-one build script
cd cobot_magic_four && ./tools/build.sh
```

### Data Collection (inside Docker, requires roscore + arms + cameras running)
```bash
roscore  # in separate terminal
# Collect 500-frame episode
cd cobot_magic_four/collect_data
python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
```

### ACT Model Training
```bash
conda activate aloha
cd cobot_magic_four/aloha-devel
python act/train.py --dataset_dir ~/data --ckpt_dir ~/train_dir/ --num_episodes 50 --batch_size 32 --num_epochs 3000
# With pretrained checkpoint
python act/train.py --dataset_dir ~/data --ckpt_dir ~/train_dir/ --num_episodes 50 --batch_size 32 --num_epochs 600 --pretrain_ckpt ~/prev_train/policy_best.ckpt
```

### Inference (inside Docker, requires roscore + puppet arms + cameras)
```bash
cd cobot_magic_four/aloha-devel
python act/inference.py --ckpt_dir ~/train_dir/
```

### async_diffusion_policy
```bash
cd ~/async_diffusion_policy-master/async_dp
uv run pytest                              # run tests
uv run python main.py --mode train         # train
uv run python main.py --mode run           # production control
```

## Architecture: Imitation Learning Pipeline

### Master-Puppet Paradigm
- **Master arms**: Operator teleoperation input (joint positions recorded as actions)
- **Puppet arms**: Follower arms that execute commands (joint states recorded as observations)
- 3 cameras: front (`cam_high`), left wrist (`cam_left_wrist`), right wrist (`cam_right_wrist`)

### Data Format
HDF5 files with structure: `/observations/qpos` (14D), `/observations/images/{cam_name}` (480x640x3), `/action` (14D). Each arm has 7 DOF (6 joints + gripper). State dim = 14 (7 left + 7 right).

### Policy Models (`aloha-devel/act/policy.py`)
- **ACTPolicy**: CVAE-based transformer (DETR backbone), KL divergence + L1 loss
- **DiffusionPolicy**: Diffusion-based, MSE noise prediction loss
- **CNNMLPPolicy**: Simple CNN + MLP baseline
All use ResNet18 backbone for image encoding. Training normalizes qpos/actions using dataset statistics (saved as `dataset_stats.pkl`).

### ROS Topics (cobot_magic)
- Camera: `/camera_f/color/image_raw`, `/camera_l/color/image_raw`, `/camera_r/color/image_raw`
- Master arms: `/master/joint_left`, `/master/joint_right`
- Puppet arms: `/puppet/joint_left`, `/puppet/joint_right`
- Commands: published to master joint topics during inference

### PiPER SDK CAN Communication
The PiPER arms communicate via CAN bus at 1 Mbps. The SDK (`piper_sdk`) provides `C_PiperInterface` / `C_PiperInterface_V2` for arm control. CAN interface must be activated before use. The SDK supports joint control, Cartesian (MoveJ/MoveL/MoveC/MoveP), and MIT protocol (advanced, use caution).

## Key Configuration
- CUDA 11.8 with cuDNN 8.6 (path: `/usr/local/cuda-11.8/`)
- Python 3.10 with ROS2 Humble packages at `/opt/ros/humble/lib/python3.10/site-packages`
- PyTorch 2.1.1+cu118
- Docker for ROS1 Noetic environment (cobot_magic_ws)
