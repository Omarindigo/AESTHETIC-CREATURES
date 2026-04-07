# AESTHETIC CREATURES

Transform robotic motion into visual art. Train reinforcement learning agents on MuJoCo physics and convert their movement trajectories into mesmerizing abstract animations.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![RL](https://img.shields.io/badge/Reinforcement-Learning-PPO-blue)

## Overview

A modular pipeline combining MuJoCo physics simulation with artistic visualization:

1. **Train** RL agents (PPO) on MuJoCo environments
2. **Record** full trajectory data (positions, velocities, forces)
3. **Generate** abstract art from movement trajectories

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List all environments
python -m aesthetic_creatures.scripts.train --list-envs

# Train on standard environment
python -m aesthetic_creatures.scripts.train --env-id Humanoid-v5 --total-timesteps 1000000

# Train on Menagerie robot
python -m aesthetic_creatures.scripts.train_menagerie --robot unitree_go2

# Generate art
python -m aesthetic_creatures.scripts.replay make-art-video \
    --rollout-npz runs/...npz \
    --output-path art.mp4 \
    --style trail --palette aurora
```

## Project Structure

```
src/aesthetic_creatures/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclasses
├── envs/
│   ├── __init__.py
│   ├── gymnasium_envs.py    # Standard Gymnasium environments
│   └── menagerie.py        # MuJoCo Menagerie robots
├── models/
│   ├── __init__.py
│   └── ppo.py              # PPO model builder
├── recording/
│   ├── __init__.py
│   ├── recorder.py         # Episode recording
│   └── saver.py            # File saving utilities
├── rendering/
│   ├── __init__.py
│   ├── video.py             # Video export
│   └── art.py               # Abstract visualization
└── scripts/
    ├── __init__.py
    ├── train.py             # Training entry point
    ├── train_menagerie.py   # Menagerie recording
    └── replay.py           # Replay & art generation
```

## Supported Environments

### Standard Gymnasium MuJoCo

| Category | Environments |
|----------|-------------|
| **Quadrupeds** | Ant-v5, Ant-v4 |
| **Humanoids** | Humanoid-v5, Humanoid-v4, HumanoidStandup-v5, HumanoidStandup-v4 |
| **Walkers** | Hopper-v5/v4, Walker2d-v5/v4 |
| **Swimmers** | Swimmer-v5/v4/v3 |
| **Cheetah** | HalfCheetah-v5/v4/v3 |
| **Manipulation** | Pusher-v5/v4/v2, Reacher-v5/v4/v2 |
| **Pendulums** | InvertedPendulum-v5/v4/v2, InvertedDoublePendulum-v5/v4/v3 |
| **Fetch** | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace |
| **Hand** | HandReach, HandManipulateBlock/Egg/Pen |

### MuJoCo Menagerie (50+ Real Robots)

**Requires:** `pip install mujoco_menagerie` or `git clone https://github.com/google-deepmind/mujoco_menagerie.git`

| Category | Examples |
|----------|----------|
| **Quadrupeds** | Unitree Go2, Go1, A1, ANYmal B/C, Spot, Barkour |
| **Humanoids** | Unitree H1, G1, Cassie, Apollo, TALOS, OP3 |
| **Arms** | Franka Panda/FR3, KUKA iiwa, UR5e/UR10e, Kinova Gen3 |
| **Hands** | Allegro, Shadow Hand, LEAP Hand |
| **Mobile** | Google Robot, Stretch 2/3, ALOHA 2 |

## Art Generation

### Styles
- `trail` - Glowing trail with dynamic coloring
- `multi_trail` - Multiple offset trails for depth
- `particle` - Particle explosion effect

### Palettes
`aurora`, `fire`, `ocean`, `neon`, `matrix`, `sunset`

```bash
# Basic art
python -m aesthetic_creatures.scripts.replay make-art-video \
    --rollout-npz runs/humanoid_v5/rollouts/humanoid_v5_step_001000000.npz \
    --output-path art.mp4

# Custom style
python -m aesthetic_creatures.scripts.replay make-art-video \
    --rollout-npz runs/go2/go2_rollout.npz \
    --output-path art.mp4 \
    --style particle --palette neon
```

## Training Examples

```bash
# Train Humanoid (walking)
python -m aesthetic_creatures.scripts.train --env-id Humanoid-v5 --total-timesteps 5000000

# Train Humanoid Standup (get up from ground)
python -m aesthetic_creatures.scripts.train --env-id HumanoidStandup-v5 --total-timesteps 2000000

# Train Walker
python -m aesthetic_creatures.scripts.train --env-id Walker2d-v5 --total-timesteps 1000000

# Train HalfCheetah
python -m aesthetic_creatures.scripts.train --env-id HalfCheetah-v5 --total-timesteps 2000000
```

## Requirements

- Python 3.8+
- MuJoCo physics engine
- PyTorch (for Stable-Baselines3)
- imageio + imageio-ffmpeg (for video export)

## License

MIT License - see [LICENSE](LICENSE)
