from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class TrainConfig:
    env_id: str = "Ant-v5"
    output_dir: str = "runs/ant_art"
    total_timesteps: int = 1_000_000
    chunk_timesteps: int = 50_000
    n_envs: int = 8
    eval_max_steps: int = 1000
    seed: int = 42

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    device: str = "auto"
    policy_net: Tuple[int, int] = (256, 256)

    save_video: bool = True
    video_fps: int = 30
    frame_stride: int = 2
    deterministic_eval: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReplayConfig:
    env_id: str = "Ant-v5"
    model_path: str = ""
    output_path: str = "replay.mp4"
    rollout_npz: str = ""
    max_steps: int = 1000
    fps: int = 30
    frame_stride: int = 1
    seed: int = 42
    stochastic: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ArtConfig:
    rollout_npz: str = ""
    output_path: str = "art_video.mp4"
    width: int = 1080
    height: int = 1080
    fps: int = 30
    history: int = 60

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunPaths:
    root: Path
    models: Path
    videos: Path
    rollouts: Path
    metrics_csv: Path
    config_json: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_run_dirs(output_dir: str) -> RunPaths:
    root = Path(output_dir)
    models = root / "models"
    videos = root / "videos"
    rollouts = root / "rollouts"
    ensure_dir(root)
    ensure_dir(models)
    ensure_dir(videos)
    ensure_dir(rollouts)
    return RunPaths(
        root=root,
        models=models,
        videos=videos,
        rollouts=rollouts,
        metrics_csv=root / "metrics.csv",
        config_json=root / "config.json",
    )


def save_config(config: TrainConfig, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
