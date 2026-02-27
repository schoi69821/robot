# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Async DP (Asynchronous Diffusion Policy) is a robotics control system for wafer inspection using the Interbotix VX300s robot arm. It implements a dual-process architecture separating AI inference (15 Hz "brain") from real-time control (500 Hz "muscle") via shared memory IPC.

## Commands

```bash
# Install dependencies (requires uv package manager)
cd async_dp
python install_deps.py

# Run tests
cd async_dp
uv run pytest

# Run application modes
uv run python main.py --mode train     # Train diffusion model
uv run python main.py --mode run       # Production control loop
uv run python main.py --mode optimize  # Gripper optimization

# Visualize control logic
uv run python scripts/visualize_traj.py

# Run a single test
uv run pytest tests/test_model.py -v
```

## Architecture

### Dual-Process Control System
- **Inference Process** (`src/core/async_engine.py:run_inference_process`): Runs AI model at 15 Hz, writes predicted action trajectories to shared memory
- **Control Process** (`src/core/async_engine.py:run_control_process`): Runs at 500 Hz, reads trajectories from shared memory, applies interpolation and EMA smoothing, sends commands to robot
- **Shared Memory** (`src/core/shared_mem.py`): Lock-protected numpy arrays for action trajectories (16x14), observation state (14), and update counter

### Diffusion Policy Model
- Uses UNet1D from diffusers library with pad-crop architecture to handle horizon 16
- Input is padded from 16 to 32 before UNet, then cropped back to 16
- Configuration: 14 DOF (7 per arm), 16-step prediction horizon, 2-step observation horizon

### Robot Driver
- `src/drivers/robot_driver.py`: Lazy-imports Interbotix SDK, falls back to dummy mode if hardware unavailable
- Supports `mode='real'` for actual hardware, `mode='dummy'` for testing

### Key Configuration (`config/settings.py`)
- `CONTROL_FREQ = 500`: Real-time control loop frequency
- `INFERENCE_FREQ = 15`: AI inference frequency
- `PRED_HORIZON = 16`: Action prediction steps
- `ACTION_DIM = 14`: Joint dimensions (7 DOF x 2 arms)

## Directory Structure

```
async_dp/
├── config/settings.py    # Central configuration
├── src/
│   ├── core/             # Async engine, shared memory
│   ├── drivers/          # Robot hardware interface
│   ├── models/           # Diffusion policy, scheduler, training
│   └── utils/            # Math utilities, dataset loader
├── scripts/              # Visualization tools
├── tests/                # Pytest tests
└── main.py               # Entry point with train/run/optimize modes
```
