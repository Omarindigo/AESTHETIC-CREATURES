from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .envs import get_mujoco_state, safe_array


def run_episode_and_record(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    capture_frames: bool,
    frame_stride: int,
) -> Dict[str, Any]:
    obs, info = env.reset()

    observations: List[np.ndarray] = [safe_array(obs)]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    infos: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    qpos: List[np.ndarray] = []
    qvel: List[np.ndarray] = []
    torso_com: List[np.ndarray] = []
    cfrc_ext: List[np.ndarray] = []

    state0 = get_mujoco_state(env)
    if "qpos" in state0:
        qpos.append(state0["qpos"])
    if "qvel" in state0:
        qvel.append(state0["qvel"])
    if "torso_com" in state0:
        torso_com.append(state0["torso_com"])
    if "cfrc_ext" in state0:
        cfrc_ext.append(state0["cfrc_ext"])

    if capture_frames:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    episode_reward = 0.0
    episode_length = 0

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)

        observations.append(safe_array(next_obs))
        actions.append(safe_array(action))
        rewards.append(float(reward))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        infos.append(info)

        episode_reward += float(reward)
        episode_length += 1

        st = get_mujoco_state(env)
        if "qpos" in st:
            qpos.append(st["qpos"])
        if "qvel" in st:
            qvel.append(st["qvel"])
        if "torso_com" in st:
            torso_com.append(st["torso_com"])
        if "cfrc_ext" in st:
            cfrc_ext.append(st["cfrc_ext"])

        if capture_frames and ((step_idx + 1) % max(frame_stride, 1) == 0):
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        obs = next_obs
        if terminated or truncated:
            break

    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32) if actions else np.zeros((0,), dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "terminations": np.array(terminations, dtype=np.bool_),
        "truncations": np.array(truncations, dtype=np.bool_),
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "frames": np.stack(frames).astype(np.uint8) if frames else np.zeros((0,), dtype=np.uint8),
        "qpos": np.stack(qpos).astype(np.float32) if qpos else np.zeros((0,), dtype=np.float32),
        "qvel": np.stack(qvel).astype(np.float32) if qvel else np.zeros((0,), dtype=np.float32),
        "torso_com": np.stack(torso_com).astype(np.float32) if torso_com else np.zeros((0,), dtype=np.float32),
        "cfrc_ext": np.stack(cfrc_ext).astype(np.float32) if cfrc_ext else np.zeros((0,), dtype=np.float32),
        "final_info": infos[-1] if infos else {},
    }


def save_rollout_npz(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(output_path, **serializable)


def append_metrics_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
