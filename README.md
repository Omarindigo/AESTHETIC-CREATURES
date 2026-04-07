# AESTHETIC CREATURES

Transform robotic motion into visual art. Train reinforcement learning agents on MuJoCo physics and convert their movement trajectories into mesmerizing abstract animations.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![RL](https://img.shields.io/badge/Reinforcement-Learning-PPO-blue)

## Overview

A pipeline combining MuJoCo physics simulation with artistic visualization:

1. **Train** reinforcement learning agents (PPO) on MuJoCo environments
2. **Record** full trajectory data (positions, velocities, forces)
3. **Render** simulation videos or generate abstract art from movement

Currently configured for the **Ant-v5** environment, but extensible to any MuJoCo task.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train an Ant agent (1M timesteps)
python -m aesthetic_creatures.train

# Generate art from a trained model
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/ant_art/rollouts/step_001000000.npz \
    --output-path my_art.mp4

# Or replay the trained model as video
python -m aesthetic_creatures.replay replay \
    --model-path runs/ant_art/models/ppo_ant_final.zip \
    --output-path replay.mp4
```

## Project Structure

```
src/aesthetic_creatures/
├── __init__.py      # Package exports
├── config.py        # Configuration dataclasses
├── envs.py          # Gymnasium environment factories
├── model.py         # PPO model builder
├── recorder.py      # Trajectory data capture
├── render.py        # Video export
├── art.py           # Abstract visualization
├── train.py         # Training script
└── replay.py        # Replay & art generation
```

## Training

```bash
python -m aesthetic_creatures.train --total-timesteps 1000000 --output-dir runs/my_ant
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--total-timesteps` | 1,000,000 | Total training steps |
| `--chunk-timesteps` | 50,000 | Steps per eval checkpoint |
| `--n-envs` | 8 | Parallel environments |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--hidden-size` | 256 | Policy network size |
| `--video-fps` | 30 | Output video framerate |

### Output Structure

```
runs/my_ant/
├── config.json         # Training config
├── metrics.csv         # Reward/length per checkpoint
├── models/             # Saved PPO models (.zip)
├── videos/             # Rendered episodes
└── rollouts/           # Trajectory data (.npz)
```

## Generating Art

The art module converts physics trajectories into abstract visualizations:

```bash
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/my_ant/rollouts/step_001000000.npz \
    --output-path art_video.mp4 \
    --width 1920 \
    --height 1080 \
    --fps 30 \
    --history 60
```

### Art Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--width` | 1080 | Output width |
| `--height` | 1080 | Output height |
| `--history` | 60 | Trail length in frames |
| `--fps` | 30 | Output framerate |

## Technical Details

### Environment
- **Gymnasium** (formerly Gym) with MuJoCo physics
- **Stable-Baselines3** for PPO implementation
- Vectorized training with `DummyVecEnv`

### Trajectory Data
The `.npz` files contain:
- `torso_com`: 3D position of the torso (for art)
- `qpos`: Generalized positions
- `qvel`: Generalized velocities  
- `actions`: Control inputs applied
- `rewards`: Per-step rewards
- `frames`: Rendered RGB frames

### Art Algorithm
1. Extract torso position trajectory
2. Normalize to canvas coordinates
3. Draw trails with color mapped to reward/action magnitude
4. Render disk at current position with dynamic size
5. Add overlay bars showing current reward/action intensity

## Requirements

- Python 3.8+
- MuJoCo physics engine (included in `mujoco` package)
- PyTorch (for Stable-Baselines3)
- imageio + imageio-ffmpeg (for video export)

## License

MIT License - see [LICENSE](LICENSE)
