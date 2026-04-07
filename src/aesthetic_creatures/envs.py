from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def env_factory(env_id: str, seed: int, rank: int, render_mode: Optional[str] = None):
    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        env = RecordEpisodeStatistics(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _make


def make_training_env(env_id: str, n_envs: int, seed: int):
    env_fns = [env_factory(env_id, seed, rank=i, render_mode=None) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    return vec_env


def make_eval_env(env_id: str, seed: int, render_mode: Optional[str] = "rgb_array"):
    return env_factory(env_id, seed, rank=10_000, render_mode=render_mode)()


def get_mujoco_state(env) -> Dict[str, np.ndarray]:
    base = env.unwrapped
    state: Dict[str, np.ndarray] = {}

    if hasattr(base, "data"):
        data = base.data
        if hasattr(data, "qpos"):
            state["qpos"] = np.array(data.qpos, dtype=np.float32).copy()
        if hasattr(data, "qvel"):
            state["qvel"] = np.array(data.qvel, dtype=np.float32).copy()
        if hasattr(data, "cfrc_ext"):
            state["cfrc_ext"] = np.array(data.cfrc_ext, dtype=np.float32).copy()

    try:
        torso = base.get_body_com("torso")
        state["torso_com"] = np.array(torso, dtype=np.float32).copy()
    except Exception:
        pass

    return state


def safe_array(x: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(x, dtype=dtype)
