from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


@dataclass
class EnvSpec:
    env_id: str
    body_parts: List[str]
    default_camera: str = "track"
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None


ENVIRONMENTS: Dict[str, EnvSpec] = {
    # Quadrupeds
    "Ant-v5": EnvSpec(
        env_id="Ant-v5",
        body_parts=["torso", "front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg"],
    ),
    "Ant-v4": EnvSpec(
        env_id="Ant-v4",
        body_parts=["torso", "front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg"],
    ),
    
    # Bipeds
    "Humanoid-v5": EnvSpec(
        env_id="Humanoid-v5",
        body_parts=["torso", "head", "left_hand", "right_foot"],
    ),
    "Humanoid-v4": EnvSpec(
        env_id="Humanoid-v4",
        body_parts=["torso", "head", "left_hand", "right_foot"],
    ),
    "HumanoidStandup-v5": EnvSpec(
        env_id="HumanoidStandup-v5",
        body_parts=["torso", "head", "left_hand", "right_foot"],
    ),
    "HumanoidStandup-v4": EnvSpec(
        env_id="HumanoidStandup-v4",
        body_parts=["torso", "head", "left_hand", "right_foot"],
    ),
    
    # walkers
    "Hopper-v5": EnvSpec(
        env_id="Hopper-v5",
        body_parts=["torso", "foot"],
    ),
    "Hopper-v4": EnvSpec(
        env_id="Hopper-v4",
        body_parts=["torso", "foot"],
    ),
    "Walker2d-v5": EnvSpec(
        env_id="Walker2d-v5",
        body_parts=["torso", "foot"],
    ),
    "Walker2d-v4": EnvSpec(
        env_id="Walker2d-v4",
        body_parts=["torso", "foot"],
    ),
    
    # swimmers
    "Swimmer-v5": EnvSpec(
        env_id="Swimmer-v5",
        body_parts=["torso", "head"],
    ),
    "Swimmer-v4": EnvSpec(
        env_id="Swimmer-v4",
        body_parts=["torso", "head"],
    ),
    "Swimmer-v3": EnvSpec(
        env_id="Swimmer-v3",
        body_parts=["torso", "head"],
    ),
    
    # cheetah
    "HalfCheetah-v5": EnvSpec(
        env_id="HalfCheetah-v5",
        body_parts=["torso", "foot"],
    ),
    "HalfCheetah-v4": EnvSpec(
        env_id="HalfCheetah-v4",
        body_parts=["torso", "foot"],
    ),
    "HalfCheetah-v3": EnvSpec(
        env_id="HalfCheetah-v3",
        body_parts=["torso", "foot"],
    ),
    
    # manipulation
    "Pusher-v5": EnvSpec(
        env_id="Pusher-v5",
        body_parts=["r_elbow_flex_link", "r_wrist_flex_link"],
    ),
    "Pusher-v4": EnvSpec(
        env_id="Pusher-v4",
        body_parts=["r_elbow_flex_link", "r_wrist_flex_link"],
    ),
    "Pusher-v2": EnvSpec(
        env_id="Pusher-v2",
        body_parts=["r_elbow_flex_link", "r_wrist_flex_link"],
    ),
    "Reacher-v5": EnvSpec(
        env_id="Reacher-v5",
        body_parts=["tip"],
    ),
    "Reacher-v4": EnvSpec(
        env_id="Reacher-v4",
        body_parts=["tip"],
    ),
    "Reacher-v2": EnvSpec(
        env_id="Reacher-v2",
        body_parts=["tip"],
    ),
    
    # inverted pendulum
    "InvertedPendulum-v5": EnvSpec(
        env_id="InvertedPendulum-v5",
        body_parts=["cart", "pole"],
    ),
    "InvertedPendulum-v4": EnvSpec(
        env_id="InvertedPendulum-v4",
        body_parts=["cart", "pole"],
    ),
    "InvertedPendulum-v2": EnvSpec(
        env_id="InvertedPendulum-v2",
        body_parts=["cart", "pole"],
    ),
    "InvertedDoublePendulum-v5": EnvSpec(
        env_id="InvertedDoublePendulum-v5",
        body_parts=["cart", "pole_a", "pole_b"],
    ),
    "InvertedDoublePendulum-v4": EnvSpec(
        env_id="InvertedDoublePendulum-v4",
        body_parts=["cart", "pole_a", "pole_b"],
    ),
    "InvertedDoublePendulum-v3": EnvSpec(
        env_id="InvertedDoublePendulum-v3",
        body_parts=["cart", "pole_a", "pole_b"],
    ),
    
    # Fetch robotics
    "FetchReach-v5": EnvSpec(
        env_id="FetchReach-v5",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchReach-v4": EnvSpec(
        env_id="FetchReach-v4",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchReach-v2": EnvSpec(
        env_id="FetchReach-v2",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchSlide-v5": EnvSpec(
        env_id="FetchSlide-v5",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchSlide-v4": EnvSpec(
        env_id="FetchSlide-v4",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchSlide-v2": EnvSpec(
        env_id="FetchSlide-v2",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPush-v5": EnvSpec(
        env_id="FetchPush-v5",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPush-v4": EnvSpec(
        env_id="FetchPush-v4",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPush-v2": EnvSpec(
        env_id="FetchPush-v2",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPickAndPlace-v5": EnvSpec(
        env_id="FetchPickAndPlace-v5",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPickAndPlace-v4": EnvSpec(
        env_id="FetchPickAndPlace-v4",
        body_parts=["robot0:gripper_link"],
    ),
    "FetchPickAndPlace-v2": EnvSpec(
        env_id="FetchPickAndPlace-v2",
        body_parts=["robot0:gripper_link"],
    ),
    
    # Hand manipulation
    "HandReach-v5": EnvSpec(
        env_id="HandReach-v5",
        body_parts=["hand"],
    ),
    "HandReach-v4": EnvSpec(
        env_id="HandReach-v4",
        body_parts=["hand"],
    ),
    "HandReach-v0": EnvSpec(
        env_id="HandReach-v0",
        body_parts=["hand"],
    ),
    "HandManipulateBlock-v5": EnvSpec(
        env_id="HandManipulateBlock-v5",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulateBlock-v4": EnvSpec(
        env_id="HandManipulateBlock-v4",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulateBlock-v0": EnvSpec(
        env_id="HandManipulateBlock-v0",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulateEgg-v5": EnvSpec(
        env_id="HandManipulateEgg-v5",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulateEgg-v4": EnvSpec(
        env_id="HandManipulateEgg-v4",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulateEgg-v0": EnvSpec(
        env_id="HandManipulateEgg-v0",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulatePen-v5": EnvSpec(
        env_id="HandManipulatePen-v5",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulatePen-v4": EnvSpec(
        env_id="HandManipulatePen-v4",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
    "HandManipulatePen-v0": EnvSpec(
        env_id="HandManipulatePen-v0",
        body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"],
    ),
}


def get_available_environments() -> List[str]:
    return list(ENVIRONMENTS.keys())


def get_env_spec(env_id: str) -> EnvSpec:
    if env_id in ENVIRONMENTS:
        return ENVIRONMENTS[env_id]
    return EnvSpec(env_id=env_id, body_parts=["torso"])


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


def get_mujoco_state(env, body_parts: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
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

    if body_parts is None:
        body_parts = get_env_spec(env.spec.id).body_parts

    for body_name in body_parts:
        try:
            pos = base.get_body_com(body_name)
            state[f"{body_name}_com"] = np.array(pos, dtype=np.float32).copy()
        except Exception:
            try:
                xpos = base.data.get_xpos(body_name)
                if xpos is not None:
                    state[f"{body_name}_com"] = np.array(xpos, dtype=np.float32).copy()
            except Exception:
                pass

    if "torso_com" not in state and "root" in dir(base):
        try:
            state["torso_com"] = np.array(base.data.qpos[:3], dtype=np.float32).copy()
        except Exception:
            pass

    return state


def get_primary_body_position(state: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["torso_com", "hand_com", "foot_com", "tip_com", "robot0:gripper_link_com"]:
        if key in state:
            return state[key]
    
    for k, v in state.items():
        if k.endswith("_com") and len(v) >= 2:
            return v
    
    if "qpos" in state and len(state["qpos"]) >= 3:
        return state["qpos"][:3]
    
    return np.zeros(3, dtype=np.float32)


def safe_array(x: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(x, dtype=dtype)
