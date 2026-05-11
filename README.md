# Aesthetic Creatures

Transform robotic motion into visual art. Train reinforcement learning agents on MuJoCo physics, render their movement at 1080p, and watch skill progression across hundreds of thousands of timesteps.

---

## Methodology

### Why PPO → SAC

The project began with **PPO (Proximal Policy Optimization)** but migrated to **SAC (Soft Actor-Critic)** for the final training pipeline. The transition was driven by five key factors:

**1. Sample Efficiency** — SAC is off-policy. It stores every experience in a 1M-capacity replay buffer and reuses them multiple times for gradient updates. PPO discards data after a single update pass. Result: SAC learns to walk in far fewer environment steps with the same GPU budget.

**2. GPU Utilization** — SAC samples training batches from the replay buffer and performs repeated dense tensor operations on each batch. This keeps an NVIDIA GPU saturated with parallel computation. PPO's on-policy design (collect-once, update-once) is more CPU-bound and leaves GPU throughput on the table.

**3. Exploration** — SAC incorporates entropy maximization directly into the reward function. The agent is rewarded for maintaining uncertainty, which forces wide, sustained exploration of the action space. PPO relies on fixed noise schedules and can converge to premature local optima — e.g., repeatedly falling in the same direction.

**4. Continuous Control** — SAC's twin-Q architecture mitigates value overestimation bias, a well-known failure mode in continuous action spaces. This matters critically for Humanoid's 17-degree-of-freedom action space. PPO's clipped surrogate objective is simpler but empirically less effective for high-DoF locomotion tasks.

**5. Empirical Results** — PPO Humanoid-v5 plateaued at 250K timesteps with marginal walking quality. SAC Humanoid-v4 scaled cleanly to 800K timesteps with visibly more fluid gait at every checkpoint — from stumbling (200K) to stable walking (800K).

### SAC Architecture

```
┌──────────────────────────────────────────┐
│              SAC Agent                    │
├──────────────────────────────────────────┤
│  Actor (π)        │  Twin Critics (Q₁,Q₂) │
│  Policy Network   │  Value Networks       │
│  ┌───────────┐    │  ┌──────┐ ┌──────┐    │
│  │ observ →  │    │  │ obs  │ │ obs  │    │
│  │ action    │    │  │ act  │ │ act  │    │
│  └───────────┘    │  │→ Q₁  │ │→ Q₂  │    │
│                   │  └──────┘ └──────┘    │
│  Entropy: α ∙ H(π(·|s))  (automated α)   │
│  Replay Buffer: 1,000,000 transitions     │
└──────────────────────────────────────────┘
```

- **Actor (Policy Network)** — Maps observation to action distribution. Updated to maximize expected return + entropy.
- **Twin Critics** — Two Q-networks reduce overestimation bias. Target networks updated via soft Polyak averaging (τ = 0.005).
- **Entropy Regularization** — Temperature α is learned automatically. High entropy early → exploration. Lower entropy late → exploitation.
- **Replay Buffer** — FIFO buffer of 1M transitions (s, a, r, s′). Batches of 256 sampled uniformly.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Environment | Humanoid-v4 |
| Algorithm | SAC |
| Total Timesteps | 800,000 |
| Save Interval | every 25,000 steps |
| Batch Size | 256 |
| Learning Rate | 3 × 10⁻⁴ |
| Replay Buffer | 1,000,000 |
| Discount (γ) | 0.99 |
| Soft Update (τ) | 0.005 |
| Architecture | MLP (256, 256) |
| Hardware | NVIDIA GPU (CUDA) |
| Resolution | 1920 × 1080 @ 30 fps |

---

## Code Structure

```
src/
├── aesthetic_creatures/       # v3.0 modular pipeline
│   ├── config.py              # TrainConfig, ArtConfig dataclasses
│   ├── envs/
│   │   ├── gymnasium_envs.py  # 50+ env specs with body-part definitions
│   │   └── menagerie.py       # 30+ real-robot models (Unitree, Spot, etc.)
│   ├── models/ppo.py          # PPO builder via Stable-Baselines3
│   ├── recording/
│   │   ├── recorder.py        # Episode trajectory capture
│   │   └── saver.py           # NPZ save + CSV metrics
│   ├── rendering/
│   │   ├── art.py             # Abstract art generation (trail/particle/multi_trail)
│   │   └── video.py           # MP4 export via imageio
│   └── scripts/
│       ├── train.py           # PPO training entry point
│       └── replay.py          # Replay models or generate art
├── models/                     # 69 SAC checkpoints (25K–800K)
├── render_checkpoints.py       # Render any checkpoint to HD video
├── train_sac.py               # SAC training loop (original)
├── train_interactive.py       # Interactive training (pause/resume/restart)
├── train_continue.py          # Resume from checkpoint
├── render_*.py                # Various render scripts
└── runs/                      # Output logs, metrics, rollouts
```

---

## Results: Training Progression

Four checkpoints rendered at 1920×1080, 1000 frames each (approx. 33 seconds at 30 fps).

| Checkpoint | Description |
|------------|-------------|
| `humanoid_200k.mp4` | Early gait — rough, unstable, limited coordination |
| `humanoid_400k.mp4` | Mid-training — improved balance, more fluid motion |
| `humanoid_600k.mp4` | Late training — stable walking, smoother transitions |
| `humanoid_800k.mp4` | Final — most refined locomotion |

> The full progression is best viewed in sequence — each step shows measurable improvement in gait stability and efficiency.

---

## How to Run

### Installation

```bash
pip install -r requirements.txt
```

### Training (SAC — Humanoid-v4)

```bash
cd src
python train_sac.py
```

### Render Checkpoints to Video

```bash
cd src
python render_checkpoints.py
```

### PPO Training (alternative environments)

```bash
python -m aesthetic_creatures.scripts.train --env-id Walker2d-v5 --total-timesteps 250000
```

### Generate Art from Rollouts

```bash
python -m aesthetic_creatures.scripts.replay make-art-video \
    --rollout-npz runs/...npz \
    --output-path art.mp4 \
    --style trail --palette aurora
```

---

## Built With

- **MuJoCo** — Physics simulation engine
- **Gymnasium** — RL environment API
- **Stable-Baselines3** — PPO / SAC implementations
- **PyTorch** — Neural network backend
- **imageio-ffmpeg** — Video export
- **NVIDIA CUDA** — GPU-accelerated training

---

## License

MIT
