from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mujoco
import numpy as np
from gymnasium import spaces

from aesthetic_creatures.envs import get_menagerie_spec, list_menagerie_by_category, MENAGERIE_ROBOTS
from aesthetic_creatures.rendering import make_art_video


class MenagerieEnv:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, xml_path: str, render_mode: str = "rgb_array"):
        self.xml_path = xml_path
        self.render_mode = render_mode
        
        if not Path(xml_path).exists():
            raise FileNotFoundError(f"XML not found: {xml_path}. Run: git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        self.nq = self.model.nq
        self.nu = self.model.nu
        self._max_episode_steps = 1000
        self._elapsed_steps = 0
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nq + self.nu,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
        
        self.viewer = None
    
    def reset(self, *, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._elapsed_steps = 0
        obs = np.concatenate([self.data.qpos, self.data.qvel])[:self.observation_space.shape[0]]
        return obs.astype(np.float32), {}
    
    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        obs = np.concatenate([self.data.qpos, self.data.qvel])[:self.observation_space.shape[0]]
        reward = 0.0
        terminated = False
        truncated = self._elapsed_steps >= self._max_episode_steps
        self._elapsed_steps += 1
        return obs.astype(np.float32), reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, width=640, height=480)
            renderer.update_scene(self.data, camera="track")
            img = renderer.render()
            renderer.free()
            return img
        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return None
        return None
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, obs, deterministic=False):
        return self.action_space.sample(), None


def record_menagerie(xml_path: str, robot_id: str, output_path: str, num_episodes: int = 1, max_steps: int = 1000, fps: int = 30) -> list:
    spec = get_menagerie_spec(robot_id)
    if spec is None:
        raise ValueError(f"Unknown robot: {robot_id}")
    
    print(f"\nRecording {robot_id} ({spec.maker}, {spec.dofs} DoF)")
    print(f"XML: {xml_path}")
    print(f"Body parts: {spec.body_parts}")
    print("-" * 50)
    
    env = MenagerieEnv(xml_path, render_mode="rgb_array")
    policy = RandomPolicy(env.action_space)
    all_data = []
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        obs, _ = env.reset()
        
        frames = []
        qpos_history = []
        qvel_history = []
        actions_history = []
        rewards_history = []
        torso_positions = []
        
        for step in range(max_steps):
            action, _ = policy.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            qpos_history.append(env.data.qpos.copy())
            qvel_history.append(env.data.qvel.copy())
            actions_history.append(action.copy())
            rewards_history.append(reward)
            
            body_positions = {}
            for bp in spec.body_parts:
                try:
                    body_positions[f"{bp}_com"] = env.data.body(bp).xpos.copy()
                except:
                    pass
            
            if body_positions:
                torso_positions.append(list(body_positions.values())[0])
            
            if terminated or truncated:
                break
            
            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}")
        
        episode_data = {
            "env_id": robot_id,
            "body_parts": spec.body_parts,
            "frames": np.array(frames, dtype=np.uint8) if frames else np.zeros((0,), dtype=np.uint8),
            "qpos": np.array(qpos_history, dtype=np.float32),
            "qvel": np.array(qvel_history, dtype=np.float32),
            "actions": np.array(actions_history, dtype=np.float32),
            "rewards": np.array(rewards_history, dtype=np.float32),
            "episode_reward": sum(rewards_history),
            "episode_length": len(rewards_history),
        }
        
        if torso_positions:
            episode_data["torso_com"] = np.array(torso_positions, dtype=np.float32)
        
        all_data.append(episode_data)
        print(f"  Episode complete! Reward: {episode_data['episode_reward']:.2f}, Length: {episode_data['episode_length']}")
        
        if frames:
            import imageio.v2 as imageio
            video_path = Path(output_path).parent / f"{robot_id}_ep{ep+1}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(video_path), frames, fps=fps)
            print(f"  Saved video: {video_path}")
    
    env.close()
    return all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MuJoCo Menagerie robots for art generation.")
    parser.add_argument("--robot", type=str, required=True, help="Robot ID (e.g., unitree_go2, franka_panda)")
    parser.add_argument("--xml-path", type=str, default=None, help="Path to scene.xml")
    parser.add_argument("--output-dir", type=str, default="runs/menagerie")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--list-robots", action="store_true", help="List available robots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.list_robots:
        print("\n" + "=" * 70)
        print(" AVAILABLE MUJOCO MENAGERIE ROBOTS")
        print("=" * 70)
        print("\nInstall: pip install mujoco_menagerie")
        print("Or clone: git clone https://github.com/google-deepmind/mujoco_menagerie.git\n")
        
        cats = list_menagerie_by_category()
        for cat, robots in cats.items():
            if robots:
                print(f"\n  {cat}:")
                for robot_id, maker in robots:
                    spec = MENAGERIE_ROBOTS[robot_id]
                    print(f"    {robot_id:<25} {spec.dofs:>3} DoF  - {spec.description}")
        
        print("\n" + "=" * 70)
        return
    
    robot_id = args.robot
    spec = get_menagerie_spec(robot_id)
    
    if spec is None:
        print(f"Unknown robot: {robot_id}")
        print("Use --list-robots to see available options")
        return
    
    xml_path = args.xml_path or spec.xml_path
    
    if not Path(xml_path).exists():
        print(f"\nXML not found: {xml_path}")
        print("\nPlease install MuJoCo Menagerie:")
        print("  Option 1: pip install mujoco_menagerie")
        print("  Option 2: git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        return
    
    output_dir = Path(args.output_dir) / robot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = record_menagerie(xml_path=xml_path, robot_id=robot_id, output_path=str(output_dir), num_episodes=args.num_episodes, max_steps=args.max_steps, fps=args.fps)
    
    if data and "torso_com" in data[0]:
        rollout_path = output_dir / f"{robot_id}_rollout.npz"
        np.savez_compressed(rollout_path, **data[0])
        print(f"\nSaved rollout: {rollout_path}")
        
        art_path = output_dir / f"{robot_id}_art.mp4"
        make_art_video(rollout_npz=str(rollout_path), output_path=str(art_path), width=1080, height=1080, fps=args.fps)
        print(f"Generated art: {art_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
