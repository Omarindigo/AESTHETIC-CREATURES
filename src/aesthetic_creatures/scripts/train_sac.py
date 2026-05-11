from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aesthetic_creatures.config import prepare_run_dirs, save_config, TrainConfig
from aesthetic_creatures.envs import make_eval_env, make_training_env, get_env_spec, get_mujoco_state
from aesthetic_creatures.recording import append_metrics_row, run_episode_and_record, save_rollout_npz
from aesthetic_creatures.rendering import save_video

from stable_baselines3 import SAC


def evaluate_and_export(model, config, step_count: int, paths) -> dict:
    try:
        eval_env = make_eval_env(config.env_id, config.seed, render_mode="rgb_array")
        env_spec = get_env_spec(config.env_id)
        body_parts = env_spec.body_parts

        obs, _ = eval_env.reset()
        
        frames = []
        observations = []
        actions = []
        rewards = []
        body_positions = {bp: [] for bp in body_parts}
        
        episode_reward = 0
        episode_length = 0
        
        for step_idx in range(config.eval_max_steps):
            action, _ = model.predict(obs, deterministic=config.deterministic_eval)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            episode_reward += reward
            episode_length += 1
            
            state = get_mujoco_state(eval_env, body_parts)
            for bp in body_parts:
                key = f"{bp}_com"
                if key in state:
                    body_positions[bp].append(state[key])
            
            if config.save_video and step_idx % config.frame_stride == 0:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)
            
            if terminated or truncated:
                break
        
        eval_env.close()
        
        import numpy as np
        env_name = config.env_id.replace("-", "_").lower()
        tag = f"step_{step_count:09d}"
        
        rollout_path = paths.rollouts / f"{env_name}_{tag}.npz"
        np.savez_compressed(
            rollout_path,
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            **{f"{bp}_com": np.array(body_positions[bp], dtype=np.float32) for bp in body_parts}
        )
        
        video_path = paths.videos / f"{env_name}_{tag}.mp4"
        if config.save_video and frames:
            from aesthetic_creatures.rendering import save_video
            save_video(frames, video_path, fps=config.video_fps)
        
        model_path = paths.models / f"sac_{env_name}_{tag}.zip"
        model.save(model_path)
        
        metrics = {
            "env_id": config.env_id,
            "timesteps": step_count,
            "eval_reward": float(episode_reward),
            "eval_length": int(episode_length),
            "saved_rollout": rollout_path.name,
            "saved_video": video_path.name if config.save_video and frames else "",
            "body_parts": body_parts,
            "algorithm": "SAC",
        }
        
        with open(paths.metrics_csv, "a") as f:
            if f.tell() == 0:
                f.write("env_id,timesteps,eval_reward,eval_length,saved_rollout,saved_video\n")
            f.write(f"{metrics['env_id']},{metrics['timesteps']},{metrics['eval_reward']:.2f},{metrics['eval_length']},{metrics['saved_rollout']},{metrics['saved_video']}\n")
        
        print(f"\n  ✓ Saved: {video_path.name if video_path.exists() else 'rollout only'}")
        return metrics
        
    except Exception as e:
        print(f"\n  ✗ Eval failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "env_id": config.env_id,
            "timesteps": step_count,
            "eval_reward": 0,
            "eval_length": 0,
            "error": str(e),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuJoCo agents with SAC (better for Humanoid).")
    
    envs = parser.add_argument_group("Environment")
    envs.add_argument("--env-id", type=str, default="Humanoid-v5", help="Gymnasium environment ID")
    envs.add_argument("--list-envs", action="store_true", help="List all available environments")
    
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--output-dir", type=str, default=None, help="Output directory")
    
    training = parser.add_argument_group("Training")
    training.add_argument("--total-timesteps", type=int, default=5_000_000)
    training.add_argument("--chunk-timesteps", type=int, default=500_000)
    training.add_argument("--eval-freq", type=int, default=500_000)
    training.add_argument("--n-envs", type=int, default=32)
    training.add_argument("--eval-max-steps", type=int, default=1000)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--buffer-size", type=int, default=1_000_000)

    sac = parser.add_argument_group("SAC Hyperparameters")
    sac.add_argument("--learning-rate", type=float, default=3e-4)
    sac.add_argument("--batch-size", type=int, default=256)
    sac.add_argument("--gamma", type=float, default=0.99)
    sac.add_argument("--tau", type=float, default=0.005)
    sac.add_argument("--ent-coef", type=str, default="auto")
    sac.add_argument("--device", type=str, default="auto")
    sac.add_argument("--hidden-size", type=int, default=256)
    sac.add_argument("--gradient-steps", type=int, default=1)

    output = parser.add_argument_group("Output")
    output.add_argument("--video-fps", type=int, default=30)
    output.add_argument("--frame-stride", type=int, default=2)
    output.add_argument("--no-video", action="store_true")
    output.add_argument("--deterministic-eval", action="store_true", default=True)

    return parser.parse_args()


def list_environments():
    from aesthetic_creatures.envs import ENVIRONMENTS, list_menagerie_by_category, MENAGERIE_ROBOTS
    
    print("\n" + "=" * 70)
    print(" GYMNASIUM MUJOCO ENVIRONMENTS")
    print("=" * 70)
    
    categories = {
        "Humanoids": [],
        "Walkers": [],
        "Quadrupeds": [],
    }
    
    for env_id in sorted(ENVIRONMENTS.keys()):
        if "Humanoid" in env_id:
            categories["Humanoids"].append(env_id)
        elif "Walker" in env_id or "Hopper" in env_id:
            categories["Walkers"].append(env_id)
        elif "Ant" in env_id:
            categories["Quadrupeds"].append(env_id)
    
    for category, envs in categories.items():
        if envs:
            print(f"\n  {category}:")
            for env_id in envs:
                print(f"    {env_id}")
    
    print("\n" + "=" * 70)


def main() -> None:
    args = parse_args()
    
    if args.list_envs:
        list_environments()
        return

    if args.output_dir is None:
        env_name = args.env_id.replace("-", "_").lower()
        args.output_dir = f"runs/{env_name}_sac"

    config = TrainConfig(
        env_id=args.env_id,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        chunk_timesteps=args.chunk_timesteps,
        n_envs=args.n_envs,
        eval_max_steps=args.eval_max_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=args.device,
        policy_net=(args.hidden_size, args.hidden_size),
        save_video=not args.no_video,
        video_fps=args.video_fps,
        frame_stride=args.frame_stride,
        deterministic_eval=args.deterministic_eval,
    )

    print(f"\n{'='*60}")
    print(f" Training {args.env_id} with SAC")
    print(f" (SAC is better for Humanoid - off-policy sample efficient)")
    print(f"{'='*60}")
    print(f"Output dir: {config.output_dir}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Eval frequency: {args.eval_freq:,}")
    print(f"Body parts: {get_env_spec(config.env_id).body_parts}")
    print("-" * 40)

    paths = prepare_run_dirs(config.output_dir)
    save_config(config, paths.config_json)

    env = make_training_env(config.env_id, config.n_envs, config.seed)
    
    policy_kwargs = dict(net_arch=[config.policy_net[0], config.policy_net[0]])
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        ent_coef=args.ent_coef,
        gradient_steps=args.gradient_steps,
        seed=config.seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=config.device,
        tensorboard_log=str(paths.root / "tensorboard"),
    )

    timesteps_done = 0
    while timesteps_done < config.total_timesteps:
        learn_steps = min(args.eval_freq, config.total_timesteps - timesteps_done)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False, progress_bar=True)
        timesteps_done += learn_steps
        
        metrics = evaluate_and_export(model, config, timesteps_done, paths)
        print(json.dumps(metrics, indent=2))

    env_name = config.env_id.replace("-", "_").lower()
    model.save(paths.models / f"sac_{env_name}_final.zip")
    env.close()
    print("\nSAC Training complete!")


if __name__ == "__main__":
    main()
