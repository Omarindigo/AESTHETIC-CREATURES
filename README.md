# AESTHETIC CREATURES

Transform robotic motion into visual art. Train reinforcement learning agents on MuJoCo physics and convert their movement trajectories into mesmerizing abstract animations.

![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-purple)
![RL](https://img.shields.io/badge/Reinforcement-Learning-PPO-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-MuJoCo-orange)

## Overview

A pipeline combining MuJoCo physics simulation with artistic visualization:

1. **Train** reinforcement learning agents (PPO) on MuJoCo environments
2. **Record** full trajectory data (positions, velocities, forces)
3. **Generate** abstract art from movement trajectories

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# List all environments
python -m aesthetic_creatures.train --list-envs

# Train on standard Gymnasium environment
python -m aesthetic_creatures.train --env-id Ant-v5 --total-timesteps 1000000

# Record real robot from MuJoCo Menagerie
python -m aesthetic_creatures.train_menagerie --robot unitree_go2 --xml-path mujoco_menagerie/unitree_go2/scene.xml

# Generate art
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/...npz \
    --output-path my_art.mp4 \
    --style trail --palette aurora
```

## Supported Environments

### Standard Gymnasium MuJoCo

| Category | Environments |
|----------|-------------|
| Quadrupeds | Ant-v5, Ant-v4 |
| Bipeds | Humanoid-v5, Humanoid-v4, HumanoidStandup |
| Walkers | Hopper-v5/v4, Walker2d-v5/v4 |
| Swimmers | Swimmer-v5/v4/v3 |
| Robots | HalfCheetah-v5/v4/v3, Pusher, Reacher |
| Pendulums | InvertedPendulum, InvertedDoublePendulum |
| Fetch | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace |
| Hand | HandReach, HandManipulateBlock/Egg/Pen |

### MuJoCo Menagerie (Real Robot Models)

Real-world robots from Google DeepMind's curated collection.

**Requires:** `pip install mujoco_menagerie` or `git clone https://github.com/google-deepmind/mujoco_menagerie.git`

#### Quadrupeds
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `unitree_go2` | Unitree | 12 | Advanced quadruped |
| `unitree_go1` | Unitree | 12 | Budget quadruped |
| `unitree_a1` | Unitree | 12 | High-speed |
| `anymal_b` | ANYbotics | 12 | Industrial |
| `anymal_c` | ANYbotics | 12 | Enhanced industrial |
| `boston_dynamics_spot` | Boston Dynamics | 12 | Spot with arm |
| `google_barkour_v0` | Google | 12 | Variable stiffness |
| `google_barkour_vb` | Google | 12 | Barbell design |

#### Humanoids & Bipeds
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `agility_cassie` | Agility Robotics | 28 | Dynamic biped |
| `unitree_h1` | Unitree | 19 | Full-size humanoid |
| `unitree_g1` | Unitree | 37 | General-purpose humanoid |
| `apptronik_apollo` | Apptronik | 32 | Industrial |
| `booster_t1` | Booster | 23 | Full-size |
| `fourier_n1` | Fourier | 30 | Expressive hands |
| `robotis_op3` | ROBOTIS | 20 | Education |
| `pal_talos` | PAL Robotics | 32 | Research |

#### Robotic Arms
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `franka_panda` | Frania | 7 | Popular research arm |
| `franka_fr3` | Frania | 7 | Latest Frania arm |
| `kuka_iiwa_14` | KUKA | 7 | Industrial |
| `ur5e` | Universal Robots | 6 | Collaborative |
| `ur10e` | Universal Robots | 6 | Large collaborative |
| `kinova_gen3` | Kinova | 7 | Compact service |
| `xarm7` | UFACTORY | 7 | Budget 7-dof |
| `lite6` | UFACTORY | 6 | Compact |
| `sawyer` | Rethink | 7 | Cobot with face |

#### Hands
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `allegro_hand` | Wonik | 16 | Dexterous hand |
| `shadow_hand` | Shadow | 24 | DEX-EE |
| `leap_hand` | CMU | 16 | Low-profile |

#### Mobile Manipulators
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `google_robot` | Google | 9 | Mobile arm |
| `stretch_2` | Hello Robot | 17 | Home assistant |
| `stretch_3` | Hello Robot | 17 | Updated |
| `aloha_2` | Trossen/DM | 16 | Bimanual |

#### Drones
| Robot | Maker | DoF | Description |
|-------|-------|-----|-------------|
| `crazyflie_2` | Bitcraze | 0 | Nano quadcopter |
| `skydio_x2` | Skydio | 0 | Autonomous |

## Project Structure

```
src/aesthetic_creatures/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── envs.py               # Environment registry (60+ environments)
├── model.py              # PPO model builder
├── recorder.py           # Trajectory data capture
├── render.py             # Video export
├── art.py                # Abstract visualization
├── train.py              # Gymnasium training script
├── train_menagerie.py     # Menagerie recording script
└── replay.py             # Replay & art generation
```

## Training (Gymnasium)

```bash
# Standard environments
python -m aesthetic_creatures.train --env-id Ant-v5 --total-timesteps 1000000
python -m aesthetic_creatures.train --env-id Humanoid-v5 --total-timesteps 5000000

# Custom hyperparameters
python -m aesthetic_creatures.train \
    --env-id HalfCheetah-v5 \
    --total-timesteps 2000000 \
    --learning-rate 1e-3 \
    --n-envs 16
```

## Recording (Menagerie)

```bash
# First, clone MuJoCo Menagerie
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Record a robot (random policy for now)
python -m aesthetic_creatures.train_menagerie \
    --robot unitree_go2 \
    --xml-path mujoco_menagerie/unitree_go2/scene.xml \
    --num-episodes 3 \
    --output-dir runs/go2

# List available robots
python -m aesthetic_creatures.train_menagerie --list-robots
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
| `aurora` | Smooth, ethereal |
| `fire` | Dramatic, energetic |
| `ocean` | Fluid, underwater |
| `neon` | Cyberpunk |
| `matrix` | Digital, tech |
| `sunset` | Warm, nostalgic |

```bash
# Basic art
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/ant_v5/rollouts/...npz \
    --output-path art.mp4

# Custom style
python -m aesthetic_creatures.replay make-art-video \
    --rollout-npz runs/go2/go2_rollout.npz \
    --output-path art.mp4 \
    --style particle --palette neon
```

## Requirements

- Python 3.8+
- MuJoCo physics engine
- PyTorch (for Stable-Baselines3)
- imageio + imageio-ffmpeg (for video export)
- mujoco_menagerie (optional, for real robot models)

## License

MIT License - see [LICENSE](LICENSE)
