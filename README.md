# PiPER Robot Imitation Learning Workspace

PiPER 로봇 팔(AgileX Robotics)을 활용한 모방학습(Imitation Learning) 통합 워크스페이스입니다.

## Overview

이 레포지토리는 로봇 팔 텔레오퍼레이션, 데이터 수집, 정책 학습(ACT/Diffusion), 실시간 추론까지 전체 파이프라인을 포함합니다.

```
├── cobot_magic_ws/                  # ALOHA 기반 모방학습 (ROS1 Noetic, Docker)
├── piper_ws/                        # PiPER SDK + ROS2 Humble 제어
├── async_diffusion_policy-master/   # 비동기 확산 정책 (15Hz 추론 + 500Hz 제어)
├── Dockerfile                       # ROS1 Noetic + CUDA 11.8 Docker 환경
└── CLAUDE.md                        # Claude Code 가이드
```

## Architecture

### Master-Puppet 패러다임

```
┌──────────────┐     CAN Bus      ┌──────────────┐
│  Master Arm  │ ──────────────── │  Puppet Arm  │
│  (텔레오퍼)   │   joint states   │  (실행)       │
└──────┬───────┘                  └──────┬───────┘
       │ action (7 DOF × 2)             │ observation (7 DOF × 2)
       └──────────┬─────────────────────┘
                  ▼
          ┌───────────────┐
          │   HDF5 Dataset │  480×640 RGB × 3 cameras
          │   14D qpos/act │  cam_high, cam_left_wrist, cam_right_wrist
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │  Policy Model  │  ACT / Diffusion / CNNMLP
          │  (ResNet18 +   │  PyTorch 2.1.1 + CUDA 11.8
          │   Transformer) │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │   Inference    │  ROS topic → Puppet Arms
          └───────────────┘
```

### 워크스페이스별 역할

| 워크스페이스 | 환경 | 역할 |
|-------------|------|------|
| **cobot_magic_ws** | ROS1 Noetic (Docker) | 데이터 수집, ACT/Diffusion 학습, 추론 |
| **piper_ws** | ROS2 Humble (native) | PiPER SDK, CAN 통신, 로봇 팔 직접 제어 |
| **async_diffusion_policy** | Python (uv) | 비동기 확산 정책 (공유 메모리 IPC) |

## Requirements

- Ubuntu 22.04
- NVIDIA GPU + CUDA 11.8 + cuDNN 8.6
- Python 3.10
- PyTorch 2.1.1+cu118
- ROS2 Humble (native)
- Docker (ROS1 Noetic 환경)
- CAN USB 어댑터 (PiPER 로봇 통신)

## Quick Start

### 1. CAN 버스 설정 (로봇 연결 전 필수)

```bash
cd piper_ws
bash can_activate.sh can0 1000000

# CAN 어댑터가 여러 개일 때
bash can_activate.sh can0 1000000 1-2:1.0
bash can_config.sh
```

### 2. PiPER SDK 로봇 제어 (ROS2)

```bash
source ~/piper_ws/install/setup.bash
cd piper_ws && colcon build
```

### 3. 데이터 수집 (Docker 내부)

```bash
# Docker 빌드 & 실행
docker build -t cobot_magic .
docker run --gpus all --network host --privileged -it cobot_magic

# Docker 내부에서
roscore &
# 로봇 팔 + 카메라 시작 후:
cd cobot_magic_four/collect_data
python collect_data.py --dataset_dir ~/data --max_timesteps 500 --episode_idx 0
```

### 4. ACT 모델 학습

```bash
conda activate aloha
cd cobot_magic_four/aloha-devel

# 학습
python act/train.py \
  --dataset_dir ~/data \
  --ckpt_dir ~/train_dir/ \
  --num_episodes 50 \
  --batch_size 32 \
  --num_epochs 3000

# 사전학습 모델 기반 파인튜닝
python act/train.py \
  --dataset_dir ~/data \
  --ckpt_dir ~/train_dir/ \
  --num_episodes 50 \
  --batch_size 32 \
  --num_epochs 600 \
  --pretrain_ckpt ~/prev_train/policy_best.ckpt
```

### 5. 추론 (로봇 실행)

```bash
cd cobot_magic_four/aloha-devel
python act/inference.py --ckpt_dir ~/train_dir/
```

### 6. 비동기 확산 정책 (async_diffusion_policy)

```bash
cd async_diffusion_policy-master/async_dp
uv run pytest                          # 테스트
uv run python main.py --mode train     # 학습
uv run python main.py --mode run       # 실시간 제어
```

## Policy Models

| 모델 | 설명 | 손실 함수 |
|------|------|----------|
| **ACT** | CVAE 기반 Transformer (DETR backbone) | KL Divergence + L1 |
| **Diffusion** | 확산 모델 기반 정책 | MSE (noise prediction) |
| **CNNMLP** | CNN + MLP 베이스라인 | L1 / L2 / Smooth L1 |

- 공통: ResNet18 이미지 인코더, 14D 상태 공간 (7 DOF × 2 arms)
- 데이터: HDF5 (480×640×3 RGB × 3 cameras + 14D qpos + 14D action)

## Data Format

```
episode_0.hdf5
├── observations/
│   ├── qpos          (N, 14)     # 7 joints × 2 arms
│   ├── qvel          (N, 14)
│   ├── effort        (N, 14)
│   └── images/
│       ├── cam_high           (N, 480, 640, 3)  uint8
│       ├── cam_left_wrist     (N, 480, 640, 3)  uint8
│       └── cam_right_wrist    (N, 480, 640, 3)  uint8
├── action            (N, 14)     # master arm joint positions
└── base_action       (N, 2)      # mobile base (optional)
```

## ROS Topics

| Topic | Type | 설명 |
|-------|------|------|
| `/camera_f/color/image_raw` | sensor_msgs/Image | 전방 카메라 |
| `/camera_l/color/image_raw` | sensor_msgs/Image | 좌측 손목 카메라 |
| `/camera_r/color/image_raw` | sensor_msgs/Image | 우측 손목 카메라 |
| `/master/joint_left` | sensor_msgs/JointState | 마스터 좌측 팔 |
| `/master/joint_right` | sensor_msgs/JointState | 마스터 우측 팔 |
| `/puppet/joint_left` | sensor_msgs/JointState | 퍼펫 좌측 팔 |
| `/puppet/joint_right` | sensor_msgs/JointState | 퍼펫 우측 팔 |

## References

- [ALOHA (cobot_magic)](https://github.com/agilexrobotics/cobot_magic) - AgileX Robotics
- [PiPER SDK](https://github.com/agilexrobotics/piper_sdk) - AgileX Robotics
- [ACT (Action Chunking with Transformers)](https://github.com/tonyzhaozh/act) - Tony Zhao et al.
