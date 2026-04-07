from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from aesthetic_creatures.envs import get_mujoco_state, safe_array


def run_episode_and_record(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    capture_frames: bool,
    frame_stride: int,
    body_parts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    obs, info = env.reset()

    env_id = getattr(env.spec, "id", "unknown")
    if body_parts is None:
        from aesthetic_creatures.envs import get_env_spec
        body_parts = get_env_spec(env_id).body_parts

    observations: List[np.ndarray] = [safe_array(obs)]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    terminations: List[bool] = []
    truncations: List[bool] = []
    infos: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    qpos: List[np.ndarray] = []
    qvel: List[np.ndarray] = []
    cfrc_ext: List[np.ndarray] = []
    
    body_positions: Dict[str, List[np.ndarray]] = {bp: [] for bp in body_parts}

    state0 = get_mujoco_state(env, body_parts)
    if "qpos" in state0:
        qpos.append(state0["qpos"])
    if "qvel" in state0:
        qvel.append(state0["qvel"])
    if "cfrc_ext" in state0:
        cfrc_ext.append(state0["cfrc_ext"])
    
    for bp in body_parts:
        key = f"{bp}_com"
        if key in state0:
            body_positions[bp].append(state0[key])

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

        st = get_mujoco_state(env, body_parts)
        if "qpos" in st:
            qpos.append(st["qpos"])
        if "qvel" in st:
            qvel.append(st["qvel"])
        if "cfrc_ext" in st:
            cfrc_ext.append(st["cfrc_ext"])
        
        for bp in body_parts:
            key = f"{bp}_com"
            if key in st:
                body_positions[bp].append(st[key])

        if capture_frames and ((step_idx + 1) % max(frame_stride, 1) == 0):
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        obs = next_obs
        if terminated or truncated:
            break

    result = {
        "env_id": env_id,
        "body_parts": body_parts,
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
        "cfrc_ext": np.stack(cfrc_ext).astype(np.float32) if cfrc_ext else np.zeros((0,), dtype=np.float32),
        "final_info": infos[-1] if infos else {},
    }
    
    for bp, positions in body_positions.items():
        if positions:
            result[f"{bp}_com"] = np.stack(positions).astype(np.float32)
        else:
            result[f"{bp}_com"] = np.zeros((0, 3), dtype=np.float32)
    
    if "torso_com" in result:
        result["primary_com"] = result["torso_com"]
    elif body_parts and f"{body_parts[0]}_com" in result:
        result["primary_com"] = result[f"{body_parts[0]}_com"]

    return result
