from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import gymnasium as gym
import numpy as np
import mujoco
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
    xml_path: Optional[str] = None
    is_menagerie: bool = False


@dataclass
class MenagerieSpec:
    xml_path: str
    body_parts: List[str]
    maker: str
    dofs: int
    description: str = ""


MENAGERIE_ROBOTS: Dict[str, MenagerieSpec] = {
    # ===== QUADRUPEDS =====
    "unitree_go2": MenagerieSpec(
        xml_path="mujoco_menagerie/unitree_go2/scene.xml",
        body_parts=["base", "trunk", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        maker="Unitree Robotics",
        dofs=12,
        description="Advanced quadruped robot with enhanced agility",
    ),
    "unitree_go1": MenagerieSpec(
        xml_path="mujoco_menagerie/unitree_go1/scene.xml",
        body_parts=["base", "trunk", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        maker="Unitree Robotics",
        dofs=12,
        description="Budget quadruped with good mobility",
    ),
    "unitree_a1": MenagerieSpec(
        xml_path="mujoco_menagerie/unitree_a1/scene.xml",
        body_parts=["base", "trunk", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        maker="Unitree Robotics",
        dofs=12,
        description="High-speed quadruped",
    ),
    "anymal_b": MenagerieSpec(
        xml_path="mujoco_menagerie/anybotics_anymal_b/scene.xml",
        body_parts=["base", "trunk", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        maker="ANYbotics",
        dofs=12,
        description="Industrial quadruped for harsh environments",
    ),
    "anymal_c": MenagerieSpec(
        xml_path="mujoco_menagerie/anybotics_anymal_c/scene.xml",
        body_parts=["base", "trunk", "FL_hip", "FR_hip", "RL_hip", "RR_hip"],
        maker="ANYbotics",
        dofs=12,
        description="Enhanced industrial quadruped",
    ),
    "boston_dynamics_spot": MenagerieSpec(
        xml_path="mujoco_menagerie/boston_dynamics_spot/scene.xml",
        body_parts=["body", "fl_hx", "fr_hx", "hl_hx", "hr_hx"],
        maker="Boston Dynamics",
        dofs=12,
        description="FamousSpot robot with arm",
    ),
    "google_barkour_v0": MenagerieSpec(
        xml_path="mujoco_menagerie/google_barkour_v0/scene.xml",
        body_parts=["base_link", "torso_link", "fl Hip", "fr Hip", "hl Hip", "hr Hip"],
        maker="Google DeepMind",
        dofs=12,
        description="Variable stiffness quadruped",
    ),
    "google_barkour_vb": MenagerieSpec(
        xml_path="mujoco_menagerie/google_barkour_vb/scene.xml",
        body_parts=["base_link", "torso_link", "fl Hip", "fr Hip", "hl Hip", "hr Hip"],
        maker="Google DeepMind",
        dofs=12,
        description="Barbell quadruped design",
    ),
    
    # ===== BIPEDS =====
    "agility_cassie": MenagerieSpec(
        xml_path="mujoco_menagerie/agility_cassie/scene.xml",
        body_parts=["cassie", "pelvis", "left", "right"],
        maker="Agility Robotics",
        dofs=28,
        description="Dynamic bipedal robot",
    ),
    "unitree_h1": MenagerieSpec(
        xml_path="mujoco_menagerie/unitree_h1/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Unitree Robotics",
        dofs=19,
        description="Full-size humanoid robot",
    ),
    "unitree_g1": MenagerieSpec(
        xml_path="mujoco_menagerie/unitree_g1/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Unitree Robotics",
        dofs=37,
        description="General-purpose humanoid",
    ),
    "apptronik_apollo": MenagerieSpec(
        xml_path="mujoco_menagerie/apptronik_apollo/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Apptronik",
        dofs=32,
        description="Industrial humanoid",
    ),
    "berkeley_humanoid": MenagerieSpec(
        xml_path="mujoco_menagerie/berkeley_humanoid/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="UC Berkeley",
        dofs=12,
        description="Research humanoid",
    ),
    "booster_t1": MenagerieSpec(
        xml_path="mujoco_menagerie/booster_t1/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Booster Robotics",
        dofs=23,
        description="Full-size humanoid",
    ),
    "fourier_n1": MenagerieSpec(
        xml_path="mujoco_menagerie/fourier_n1/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Fourier Robotics",
        dofs=30,
        description="Humanoid with expressive hands",
    ),
    "robotis_op3": MenagerieSpec(
        xml_path="mujoco_menagerie/robotis_op3/scene.xml",
        body_parts=["base", "body", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="ROBOTIS",
        dofs=20,
        description="Humanoid robot for education",
    ),
    "pal_talos": MenagerieSpec(
        xml_path="mujoco_menagerie/pal_talos/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="PAL Robotics",
        dofs=32,
        description="Research humanoid",
    ),
    "toddlerbot_2xc": MenagerieSpec(
        xml_path="mujoco_menagerie/toddlerbot_2xc/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Stanford",
        dofs=30,
        description="Bi-manual toddler robot",
    ),
    "toddlerbot_2xm": MenagerieSpec(
        xml_path="mujoco_menagerie/toddlerbot_2xm/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="Stanford",
        dofs=30,
        description="Mobile toddler robot",
    ),
    "pndbotics_adam_lite": MenagerieSpec(
        xml_path="mujoco_menagerie/pndbotics_adam_lite/scene.xml",
        body_parts=["root", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="PNDbotics",
        dofs=25,
        description="Lightweight humanoid",
    ),
    
    # ===== ROBOTIC ARMS =====
    "franka_panda": MenagerieSpec(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        body_parts=["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand"],
        maker="Franka Robotics",
        dofs=7,
        description="Popular research arm",
    ),
    "franka_fr3": MenagerieSpec(
        xml_path="mujoco_menagerie/franka_fr3/scene.xml",
        body_parts=["link1", "link2", "link3", "link4", "link5", "link6", "link7"],
        maker="Franka Robotics",
        dofs=7,
        description="Latest Franka arm",
    ),
    "panda": MenagerieSpec(
        xml_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        body_parts=["panda_link0", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand"],
        maker="Franka Robotics",
        dofs=7,
        description="Panda arm (same as franka_panda)",
    ),
    "kuka_iiwa_14": MenagerieSpec(
        xml_path="mujoco_menagerie/kuka_iiwa_14/scene.xml",
        body_parts=["base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7"],
        maker="KUKA",
        dofs=7,
        description="Heavy-duty industrial arm",
    ),
    "ur5e": MenagerieSpec(
        xml_path="mujoco_menagerie/universal_robots_ur5e/scene.xml",
        body_parts=["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link"],
        maker="Universal Robots",
        dofs=6,
        description="Collaborative arm",
    ),
    "ur10e": MenagerieSpec(
        xml_path="mujoco_menagerie/universal_robots_ur10e/scene.xml",
        body_parts=["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist1_link", "wrist2_link", "wrist3_link"],
        maker="Universal Robots",
        dofs=6,
        description="Large collaborative arm",
    ),
    "kinova_gen3": MenagerieSpec(
        xml_path="mujoco_menagerie/kinova_gen3/scene.xml",
        body_parts=["base_link", "shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link", "wrist_link", "end_effector_link"],
        maker="Kinova Robotics",
        dofs=7,
        description="Compact service arm",
    ),
    "xarm7": MenagerieSpec(
        xml_path="mujoco_menagerie/ufactory_xarm7/scene.xml",
        body_parts=["base_link", "link1", "link2", "link3", "link4", "link5", "link6"],
        maker="UFACTORY",
        dofs=7,
        description="Budget 7-dof arm",
    ),
    "lite6": MenagerieSpec(
        xml_path="mujoco_menagerie/ufactory_lite6/scene.xml",
        body_parts=["base_link", "link1", "link2", "link3", "link4", "link5"],
        maker="UFACTORY",
        dofs=6,
        description="Compact 6-dof arm",
    ),
    "sawyer": MenagerieSpec(
        xml_path="mujoco_menagerie/rethink_robotics_sawyer/scene.xml",
        body_parts=["base_link", "right_arm_link0", "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4", "right_arm_link5", "right_arm_link6"],
        maker="Rethink Robotics",
        dofs=7,
        description="Cobot with expressive face",
    ),
    "vx300s": MenagerieSpec(
        xml_path="mujoco_menagerie/trossen_vx300s/scene.xml",
        body_parts=["base_link", "link1", "link2", "link3", "link4", "link5", "link6"],
        maker="Trossen Robotics",
        dofs=6,
        description="ViperX 300S arm",
    ),
    "wx250s": MenagerieSpec(
        xml_path="mujoco_menagerie/trossen_wx250s/scene.xml",
        body_parts=["base_link", "link1", "link2", "link3", "link4", "link5", "link6"],
        maker="Trossen Robotics",
        dofs=6,
        description="WidowX 250S arm",
    ),
    "allegro_hand": MenagerieSpec(
        xml_path="mujoco_menagerie/wonik_allegro/scene_right.xml",
        body_parts=["palm", "thumb", "index", "middle", "ring"],
        maker="Wonik Robotics",
        dofs=16,
        description="Allegro Hand v3",
    ),
    "shadow_hand": MenagerieSpec(
        xml_path="mujoco_menagerie/shadow_hand/scene_right.xml",
        body_parts=["palm", "thumb_0", "index_0", "middle_0", "ring_0", "pinky_0"],
        maker="Shadow Robot",
        dofs=24,
        description="DEX-EE Shadow Hand",
    ),
    "leap_hand": MenagerieSpec(
        xml_path="mujoco_menagerie/leap_hand/scene_left.xml",
        body_parts=["palm", "thumb", "index", "middle", "ring"],
        maker="Carnegie Mellon",
        dofs=16,
        description="Low-profile dexterous hand",
    ),
    
    # ===== MOBILE MANIPULATORS =====
    "google_robot": MenagerieSpec(
        xml_path="mujoco_menagerie/google_robot/scene.xml",
        body_parts=["base_link", "arm_0_link", "arm_1_link", "arm_2_link", "arm_3_link", "arm_4_link", "arm_5_link", "gripper"],
        maker="Google DeepMind",
        dofs=9,
        description="Mobile manipulator",
    ),
    "stretch_2": MenagerieSpec(
        xml_path="mujoco_menagerie/hello_robot_stretch/scene.xml",
        body_parts=["base_link", "lift_link", "arm_latch", "stretch_arm", "gripper"],
        maker="Hello Robot",
        dofs=17,
        description="Mobile manipulator for home",
    ),
    "stretch_3": MenagerieSpec(
        xml_path="mujoco_menagerie/hello_robot_stretch_3/scene.xml",
        body_parts=["base_link", "lift_link", "arm_latch", "stretch_arm", "gripper"],
        maker="Hello Robot",
        dofs=17,
        description="Updated mobile manipulator",
    ),
    "aloha_2": MenagerieSpec(
        xml_path="mujoco_menagerie/aloha/scene_bimanual.xml",
        body_parts=["base_link", "left_arm_0", "right_arm_0", "left_gripper", "right_gripper"],
        maker="Trossen/DeepMind",
        dofs=16,
        description="Bimanual teleoperation platform",
    ),
    
    # ===== DRONES =====
    "crazyflie_2": MenagerieSpec(
        xml_path="mujoco_menagerie/bitcraze_crazyflie_2/scene.xml",
        body_parts=["base"],
        maker="Bitcraze",
        dofs=0,
        description="Nano quadcopter",
    ),
    "skydio_x2": MenagerieSpec(
        xml_path="mujoco_menagerie/skydio_x2/scene.xml",
        body_parts=["base"],
        maker="Skydio",
        dofs=0,
        description="Autonomous drone",
    ),
    
    # ===== BIOMECHANICAL =====
    "iit_softfoot": MenagerieSpec(
        xml_path="mujoco_menagerie/iit_softfoot/scene.xml",
        body_parts=["pelvis", "torso", "head", "left_leg", "right_leg"],
        maker="IIT",
        dofs=92,
        description="Soft foot humanoid",
    ),
    "flybody": MenagerieSpec(
        xml_path="mujoco_menagerie/flybody/scene.xml",
        body_parts=["thorax", "head", "abdomen", "wing_left", "wing_right"],
        maker="Janelia/DeepMind",
        dofs=102,
        description="Fruit fly model",
    ),
    "ms_human_700": MenagerieSpec(
        xml_path="mujoco_menagerie/ms_human_700/scene.xml",
        body_parts=["pelvis", "torso", "head", "left_arm", "right_arm", "left_leg", "right_leg"],
        maker="LNS Group",
        dofs=157,
        description="Musculoskeletal human model",
    ),
}


ENVIRONMENTS: Dict[str, EnvSpec] = {
    # ===== STANDARD GYMNASIUM MUJOCO =====
    
    # Quadrupeds
    "Ant-v5": EnvSpec(env_id="Ant-v5", body_parts=["torso", "front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg"]),
    "Ant-v4": EnvSpec(env_id="Ant-v4", body_parts=["torso", "front_left_leg", "front_right_leg", "back_left_leg", "back_right_leg"]),
    
    # Bipeds
    "Humanoid-v5": EnvSpec(env_id="Humanoid-v5", body_parts=["torso", "head", "left_hand", "right_foot"]),
    "Humanoid-v4": EnvSpec(env_id="Humanoid-v4", body_parts=["torso", "head", "left_hand", "right_foot"]),
    "HumanoidStandup-v5": EnvSpec(env_id="HumanoidStandup-v5", body_parts=["torso", "head", "left_hand", "right_foot"]),
    "HumanoidStandup-v4": EnvSpec(env_id="HumanoidStandup-v4", body_parts=["torso", "head", "left_hand", "right_foot"]),
    
    # Walkers
    "Hopper-v5": EnvSpec(env_id="Hopper-v5", body_parts=["torso", "foot"]),
    "Hopper-v4": EnvSpec(env_id="Hopper-v4", body_parts=["torso", "foot"]),
    "Walker2d-v5": EnvSpec(env_id="Walker2d-v5", body_parts=["torso", "foot"]),
    "Walker2d-v4": EnvSpec(env_id="Walker2d-v4", body_parts=["torso", "foot"]),
    
    # Swimmers
    "Swimmer-v5": EnvSpec(env_id="Swimmer-v5", body_parts=["torso", "head"]),
    "Swimmer-v4": EnvSpec(env_id="Swimmer-v4", body_parts=["torso", "head"]),
    "Swimmer-v3": EnvSpec(env_id="Swimmer-v3", body_parts=["torso", "head"]),
    
    # Cheetah
    "HalfCheetah-v5": EnvSpec(env_id="HalfCheetah-v5", body_parts=["torso", "foot"]),
    "HalfCheetah-v4": EnvSpec(env_id="HalfCheetah-v4", body_parts=["torso", "foot"]),
    "HalfCheetah-v3": EnvSpec(env_id="HalfCheetah-v3", body_parts=["torso", "foot"]),
    
    # Manipulation
    "Pusher-v5": EnvSpec(env_id="Pusher-v5", body_parts=["r_elbow_flex_link", "r_wrist_flex_link"]),
    "Pusher-v4": EnvSpec(env_id="Pusher-v4", body_parts=["r_elbow_flex_link", "r_wrist_flex_link"]),
    "Pusher-v2": EnvSpec(env_id="Pusher-v2", body_parts=["r_elbow_flex_link", "r_wrist_flex_link"]),
    "Reacher-v5": EnvSpec(env_id="Reacher-v5", body_parts=["tip"]),
    "Reacher-v4": EnvSpec(env_id="Reacher-v4", body_parts=["tip"]),
    "Reacher-v2": EnvSpec(env_id="Reacher-v2", body_parts=["tip"]),
    
    # Pendulums
    "InvertedPendulum-v5": EnvSpec(env_id="InvertedPendulum-v5", body_parts=["cart", "pole"]),
    "InvertedPendulum-v4": EnvSpec(env_id="InvertedPendulum-v4", body_parts=["cart", "pole"]),
    "InvertedPendulum-v2": EnvSpec(env_id="InvertedPendulum-v2", body_parts=["cart", "pole"]),
    "InvertedDoublePendulum-v5": EnvSpec(env_id="InvertedDoublePendulum-v5", body_parts=["cart", "pole_a", "pole_b"]),
    "InvertedDoublePendulum-v4": EnvSpec(env_id="InvertedDoublePendulum-v4", body_parts=["cart", "pole_a", "pole_b"]),
    "InvertedDoublePendulum-v3": EnvSpec(env_id="InvertedDoublePendulum-v3", body_parts=["cart", "pole_a", "pole_b"]),
    
    # Fetch
    "FetchReach-v5": EnvSpec(env_id="FetchReach-v5", body_parts=["robot0:gripper_link"]),
    "FetchReach-v4": EnvSpec(env_id="FetchReach-v4", body_parts=["robot0:gripper_link"]),
    "FetchReach-v2": EnvSpec(env_id="FetchReach-v2", body_parts=["robot0:gripper_link"]),
    "FetchSlide-v5": EnvSpec(env_id="FetchSlide-v5", body_parts=["robot0:gripper_link"]),
    "FetchSlide-v4": EnvSpec(env_id="FetchSlide-v4", body_parts=["robot0:gripper_link"]),
    "FetchSlide-v2": EnvSpec(env_id="FetchSlide-v2", body_parts=["robot0:gripper_link"]),
    "FetchPush-v5": EnvSpec(env_id="FetchPush-v5", body_parts=["robot0:gripper_link"]),
    "FetchPush-v4": EnvSpec(env_id="FetchPush-v4", body_parts=["robot0:gripper_link"]),
    "FetchPush-v2": EnvSpec(env_id="FetchPush-v2", body_parts=["robot0:gripper_link"]),
    "FetchPickAndPlace-v5": EnvSpec(env_id="FetchPickAndPlace-v5", body_parts=["robot0:gripper_link"]),
    "FetchPickAndPlace-v4": EnvSpec(env_id="FetchPickAndPlace-v4", body_parts=["robot0:gripper_link"]),
    "FetchPickAndPlace-v2": EnvSpec(env_id="FetchPickAndPlace-v2", body_parts=["robot0:gripper_link"]),
    
    # Hand
    "HandReach-v5": EnvSpec(env_id="HandReach-v5", body_parts=["hand"]),
    "HandReach-v4": EnvSpec(env_id="HandReach-v4", body_parts=["hand"]),
    "HandReach-v0": EnvSpec(env_id="HandReach-v0", body_parts=["hand"]),
    "HandManipulateBlock-v5": EnvSpec(env_id="HandManipulateBlock-v5", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulateBlock-v4": EnvSpec(env_id="HandManipulateBlock-v4", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulateBlock-v0": EnvSpec(env_id="HandManipulateBlock-v0", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulateEgg-v5": EnvSpec(env_id="HandManipulateEgg-v5", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulateEgg-v4": EnvSpec(env_id="HandManipulateEgg-v4", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulateEgg-v0": EnvSpec(env_id="HandManipulateEgg-v0", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulatePen-v5": EnvSpec(env_id="HandManipulatePen-v5", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulatePen-v4": EnvSpec(env_id="HandManipulatePen-v4", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
    "HandManipulatePen-v0": EnvSpec(env_id="HandManipulatePen-v0", body_parts=["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal"]),
}


def get_available_environments() -> List[str]:
    return list(ENVIRONMENTS.keys()) + list(MENAGERIE_ROBOTS.keys())


def get_env_spec(env_id: str) -> EnvSpec:
    if env_id in ENVIRONMENTS:
        return ENVIRONMENTS[env_id]
    if env_id in MENAGERIE_ROBOTS:
        menagerie = MENAGERIE_ROBOTS[env_id]
        return EnvSpec(
            env_id=env_id,
            body_parts=menagerie.body_parts,
            xml_path=menagerie.xml_path,
            is_menagerie=True
        )
    return EnvSpec(env_id=env_id, body_parts=["torso"])


def get_menagerie_spec(robot_id: str) -> Optional[MenagerieSpec]:
    return MENAGERIE_ROBOTS.get(robot_id)


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
        try:
            body_parts = get_env_spec(env.spec.id).body_parts
        except:
            body_parts = ["torso"]

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
    for key in ["torso_com", "hand_com", "foot_com", "tip_com", "base_com", "panda_hand_com", "gripper_com"]:
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


def list_menagerie_by_category() -> Dict[str, List[Tuple[str, str]]]:
    categories = {
        "Quadrupeds": [],
        "Bipeds": [],
        "Humanoids": [],
        "Arms": [],
        "Hands": [],
        "Mobile Manipulators": [],
        "Drones": [],
        "Biomechanical": [],
    }
    
    quadrupeds = ["unitree_go2", "unitree_go1", "unitree_a1", "anymal_b", "anymal_c", "boston_dynamics_spot", "google_barkour_v0", "google_barkour_vb"]
    bipeds = ["agility_cassie"]
    humanoids = ["unitree_h1", "unitree_g1", "apptronik_apollo", "berkeley_humanoid", "booster_t1", "fourier_n1", "robotis_op3", "pal_talos", "toddlerbot_2xc", "toddlerbot_2xm", "pndbotics_adam_lite"]
    arms = ["franka_panda", "franka_fr3", "kuka_iiwa_14", "ur5e", "ur10e", "kinova_gen3", "xarm7", "lite6", "sawyer", "vx300s", "wx250s"]
    hands = ["allegro_hand", "shadow_hand", "leap_hand"]
    mobile = ["google_robot", "stretch_2", "stretch_3", "aloha_2"]
    drones = ["crazyflie_2", "skydio_x2"]
    biomech = ["iit_softfoot", "flybody", "ms_human_700"]
    
    for r in quadrupeds:
        if r in MENAGERIE_ROBOTS:
            categories["Quadrupeds"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in bipeds:
        if r in MENAGERIE_ROBOTS:
            categories["Bipeds"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in humanoids:
        if r in MENAGERIE_ROBOTS:
            categories["Humanoids"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in arms:
        if r in MENAGERIE_ROBOTS:
            categories["Arms"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in hands:
        if r in MENAGERIE_ROBOTS:
            categories["Hands"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in mobile:
        if r in MENAGERIE_ROBOTS:
            categories["Mobile Manipulators"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in drones:
        if r in MENAGERIE_ROBOTS:
            categories["Drones"].append((r, MENAGERIE_ROBOTS[r].maker))
    for r in biomech:
        if r in MENAGERIE_ROBOTS:
            categories["Biomechanical"].append((r, MENAGERIE_ROBOTS[r].maker))
    
    return categories
