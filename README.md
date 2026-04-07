# AESTHETIC CREATURES

Transform robotic motion into visual art. Train reinforcement learning agents on MuJoCo physics and convert their movement trajectories into mesmerizing abstract animations.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![RL](https://img.shields.io/badge/Reinforcement-Learning-PPO-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-orange)

## Overview

A pipeline combining MuJoCo physics simulation with artistic visualization:

1. **Train** reinforcement learning agents (PPO) on any Gymnasium MuJoCo environment
2. **Record** full trajectory data (positions, velocities, forces, body parts)
3. **Render** simulation videos or generate abstract art from movement

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List available environments
python -m aesthetic_creatures.train --list-envs

# Train an Ant agent
python -m aesthetic_creatures.train --env-id Ant-v5 --total-timesteps 1000000

# Generate art with different styles
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/ant_v5/rollouts/ant_v5_step_001000000.npz \
    --output-path my_art.mp4 \
    --style trail --palette aurora
```

## Supported Environments

### Quadrupeds
| Environment | Body Parts |
|------------|------------|
| `Ant-v5`, `Ant-v4` | torso, legs |

### Bipeds
| Environment | Body Parts |
|------------|------------|
| `Humanoid-v5`, `Humanoid-v4` | torso, head, hands, feet |
| `HumanoidStandup-v5`, `HumanoidStandup-v4` | torso, head, hands, feet |

### Walkers
| Environment | Body Parts |
|------------|------------|
| `Hopper-v5`, `Hopper-v4` | torso, foot |
| `Walker2d-v5`, `Walker2d-v4` | torso, foot |

### Swimmers
| Environment | Body Parts |
|------------|------------|
| `Swimmer-v5`, `Swimmer-v4`, `Swimmer-v3` | torso, head |

### Cheetah
| Environment | Body Parts |
|------------|------------|
| `HalfCheetah-v5`, `HalfCheetah-v4`, `HalfCheetah-v3` | torso, foot |

### Robots
| Environment | Body Parts |
|------------|------------|
| `Pusher-v5`, `Pusher-v4`, `Pusher-v2` | arm links |
| `Reacher-v5`, `Reacher-v4`, `Reacher-v2` | tip |

### Pendulums
| Environment | Body Parts |
|------------|------------|
| `InvertedPendulum-v5`, `InvertedPendulum-v4` | cart, pole |
| `InvertedDoublePendulum-v5`, `InvertedDoublePendulum-v4` | cart, poles |

### Fetch Robotics
| Environment | Body Parts |
|------------|------------|
| `FetchReach-v5`, `FetchPush-v5`, `FetchSlide-v5` | gripper |
| `FetchPickAndPlace-v5` | gripper |

### Shadow Hand
| Environment | Body Parts |
|------------|------------|
| `HandReach-v5`, `HandManipulateBlock-v5` | finger tips |
| `HandManipulateEgg-v5`, `HandManipulatePen-v5` | finger tips |

## Project Structure

```
src/aesthetic_creatures/
├── __init__.py      # Package exports
├── config.py        # Configuration dataclasses
├── envs.py          # Environment registry (50+ environments)
├── model.py         # PPO model builder
├── recorder.py      # Trajectory data capture
├── render.py        # Video export
├── art.py           # Abstract visualization (3 styles, 6 palettes)
├── train.py         # Training script
└── replay.py        # Replay & art generation
```

## Training

```bash
# Train on any environment
python -m aesthetic_creatures.train --env-id Humanoid-v5 --total-timesteps 5000000

# Custom hyperparameters
python -m aesthetic_creatures.train \
    --env-id HalfCheetah-v5 \
    --total-timesteps 2000000 \
    --learning-rate 1e-3 \
    --n-envs 16 \
    --hidden-size 512

# Quick test run
python -m aesthetic_creatures.train \
    --env-id Ant-v5 \
    --total-timesteps 50000 \
    --chunk-timesteps 50000
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env-id` | Ant-v5 | Gymnasium environment ID |
| `--total-timesteps` | 1,000,000 | Total training steps |
| `--chunk-timesteps` | 50,000 | Steps per eval checkpoint |
| `--n-envs` | 8 | Parallel environments |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--hidden-size` | 256 | Policy network size |
| `--video-fps` | 30 | Output video framerate |

### Output Structure

```
runs/{env_id}/
├── config.json         # Training config
├── metrics.csv         # Reward/length per checkpoint
├── models/             # Saved PPO models (.zip)
├── videos/             # Rendered episodes
└── rollouts/           # Trajectory data (.npz)
```

## Generating Art

### Art Styles

| Style | Description |
|-------|-------------|
| `trail` | Glowing trail with dynamic coloring |
| `multi_trail` | Multiple offset trails for depth |
| `particle` | Particle explosion effect |

### Color Palettes

| Palette | Best For |
|---------|----------|
| `aurora` | Smooth, ethereal animations |
| `fire` | Dramatic, energetic movements |
| `ocean` | Fluid, underwater feel |
| `neon` | Cyberpunk aesthetic |
| `matrix` | Digital, tech vibes |
| `sunset` | Warm, nostalgic tones |

### Examples

```bash
# Basic art
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/ant_v5/rollouts/ant_v5_step_001000000.npz \
    --output-path art.mp4

# Custom size and style
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/humanoid_v5/rollouts/humanoid_v5_step_005000000.npz \
    --output-path art.mp4 \
    --width 1920 --height 1080 \
    --style particle --palette neon

# Long trails
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/ant_v5/rollouts/ant_v5_step_001000000.npz \
    --output-path art.mp4 \
    --history 120 --fps 30
```

### Art Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--width` | 1080 | Output width |
| `--height` | 1080 | Output height |
| `--history` | 60 | Trail length in frames |
| `--fps` | 30 | Output framerate |
| `--style` | trail | Art style |
| `--palette` | aurora | Color palette |

## Replay

Render a trained model as video:

```bash
python -m aesthetic_creatures.replay replay \
    --env-id Ant-v5 \
    --model-path runs/ant_v5/models/ppo_ant_v5_final.zip \
    --output-path replay.mp4
```

Also saves rollout data for art generation.

## Technical Details

### Environment
- **Gymnasium** with MuJoCo physics
- **Stable-Baselines3** for PPO implementation
- Vectorized training with `DummyVecEnv`
- Environment-specific body part tracking

### Trajectory Data
The `.npz` files contain:
- `env_id`: Environment name
- `body_parts`: List of tracked body parts
- `{body}_com`: 3D position of each body part
- `qpos`: Generalized positions
- `qvel`: Generalized velocities
- `actions`: Control inputs applied
- `rewards`: Per-step rewards
- `frames`: Rendered RGB frames

### Art Algorithm
1. Extract body position trajectory from rollout
2. Normalize to canvas coordinates
3. Draw trails with color mapped to reward/action magnitude
4. Render glow effect at current position
5. Add overlay bars showing current intensity

## Requirements

- Python 3.8+
- MuJoCo physics engine
- PyTorch (for Stable-Baselines3)
- imageio + imageio-ffmpeg (for video export)

## License

MIT License - see [LICENSE](LICENSE)
