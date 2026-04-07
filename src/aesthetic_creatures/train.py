from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aesthetic_creatures.config import TrainConfig, prepare_run_dirs, save_config
from aesthetic_creatures.envs import make_eval_env, make_training_env, get_env_spec
from aesthetic_creatures.model import build_model
from aesthetic_creatures.recorder import append_metrics_row, run_episode_and_record, save_rollout_npz
from aesthetic_creatures.render import save_video


def evaluate_and_export(model, config: TrainConfig, step_count: int, paths) -> dict:
    eval_env = make_eval_env(config.env_id, config.seed, render_mode="rgb_array")
    
    env_spec = get_env_spec(config.env_id)
    body_parts = env_spec.body_parts

    episode_data = run_episode_and_record(
        env=eval_env,
        model=model,
        max_steps=config.eval_max_steps,
        deterministic=config.deterministic_eval,
        capture_frames=config.save_video,
        frame_stride=config.frame_stride,
        body_parts=body_parts,
    )
    eval_env.close()

    env_name = config.env_id.replace("-", "_").lower()
    tag = f"step_{step_count:09d}"

    rollout_path = paths.rollouts / f"{env_name}_{tag}.npz"
    save_rollout_npz(episode_data, rollout_path)

    video_path = paths.videos / f"{env_name}_{tag}.mp4"
    if config.save_video:
        save_video(episode_data["frames"], video_path, fps=config.video_fps)

    model_path = paths.models / f"ppo_{env_name}_{tag}.zip"
    model.save(model_path)

    metrics = {
        "env_id": config.env_id,
        "timesteps": step_count,
        "eval_reward": episode_data["episode_reward"],
        "eval_length": episode_data["episode_length"],
        "saved_rollout": rollout_path.name,
        "saved_video": video_path.name if config.save_video else "",
        "body_parts": body_parts,
    }
    append_metrics_row(paths.metrics_csv, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MuJoCo agents with PPO and export rollout data.")
    
    envs = parser.add_argument_group("Environment")
    envs.add_argument("--env-id", type=str, default="Ant-v5", 
                      help="Gymnasium environment ID (see list below)")
    envs.add_argument("--list-envs", action="store_true", 
                      help="List all available environments")
    
    paths = parser.add_argument_group("Paths")
    paths.add_argument("--output-dir", type=str, default=None,
                      help="Output directory (default: runs/{env_id})")
    
    training = parser.add_argument_group("Training")
    training.add_argument("--total-timesteps", type=int, default=1_000_000)
    training.add_argument("--chunk-timesteps", type=int, default=50_000)
    training.add_argument("--n-envs", type=int, default=8)
    training.add_argument("--eval-max-steps", type=int, default=1000)
    training.add_argument("--seed", type=int, default=42)

    ppo = parser.add_argument_group("PPO Hyperparameters")
    ppo.add_argument("--learning-rate", type=float, default=3e-4)
    ppo.add_argument("--n-steps", type=int, default=2048)
    ppo.add_argument("--batch-size", type=int, default=256)
    ppo.add_argument("--n-epochs", type=int, default=10)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-range", type=float, default=0.2)
    ppo.add_argument("--ent-coef", type=float, default=0.0)
    ppo.add_argument("--vf-coef", type=float, default=0.5)
    ppo.add_argument("--max-grad-norm", type=float, default=0.5)
    ppo.add_argument("--device", type=str, default="auto")
    ppo.add_argument("--hidden-size", type=int, default=256)

    output = parser.add_argument_group("Output")
    output.add_argument("--video-fps", type=int, default=30)
    output.add_argument("--frame-stride", type=int, default=2)
    output.add_argument("--no-video", action="store_true")
    output.add_argument("--stochastic-eval", action="store_true")

    return parser.parse_args()


def list_environments():
    from aesthetic_creatures.envs import get_available_environments, ENVIRONMENTS
    
    print("\nAvailable MuJoCo Environments:")
    print("=" * 60)
    
    categories = {
        "Quadrupeds": [],
        "Bipeds": [],
        "Walkers": [],
        "Swimmers": [],
        "Robots": [],
        "Pendulums": [],
        "Fetch": [],
        "Hand": [],
    }
    
    for env_id in sorted(get_available_environments()):
        if "Ant" in env_id:
            categories["Quadrupeds"].append(env_id)
        elif "Humanoid" in env_id:
            categories["Bipeds"].append(env_id)
        elif "Hopper" in env_id or "Walker2d" in env_id:
            categories["Walkers"].append(env_id)
        elif "Swimmer" in env_id:
            categories["Swimmers"].append(env_id)
        elif "Cheetah" in env_id:
            categories["Robots"].append(env_id)
        elif "Pendulum" in env_id or "InvertedPendulum" in env_id:
            categories["Pendulums"].append(env_id)
        elif "Fetch" in env_id:
            categories["Fetch"].append(env_id)
        elif "Hand" in env_id or "Manipulate" in env_id:
            categories["Hand"].append(env_id)
        elif "Pusher" in env_id or "Reacher" in env_id:
            categories["Robots"].append(env_id)
    
    for category, envs in categories.items():
        if envs:
            print(f"\n{category}:")
            for env_id in envs:
                spec = ENVIRONMENTS[env_id]
                print(f"  {env_id}")
    
    print("\n")


def main() -> None:
    args = parse_args()
    
    if args.list_envs:
        list_environments()
        return

    if args.output_dir is None:
        env_name = args.env_id.replace("-", "_").lower()
        args.output_dir = f"runs/{env_name}"

    config = TrainConfig(
        env_id=args.env_id,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        chunk_timesteps=args.chunk_timesteps,
        n_envs=args.n_envs,
        eval_max_steps=args.eval_max_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        policy_net=(args.hidden_size, args.hidden_size),
        save_video=not args.no_video,
        video_fps=args.video_fps,
        frame_stride=args.frame_stride,
        deterministic_eval=not args.stochastic_eval,
    )

    print(f"\nTraining on: {config.env_id}")
    print(f"Output dir: {config.output_dir}")
    print(f"Body parts: {get_env_spec(config.env_id).body_parts}")
    print("-" * 40)

    paths = prepare_run_dirs(config.output_dir)
    save_config(config, paths.config_json)

    env = make_training_env(config.env_id, config.n_envs, config.seed)
    model = build_model(config, env)

    timesteps_done = 0
    while timesteps_done < config.total_timesteps:
        learn_steps = min(config.chunk_timesteps, config.total_timesteps - timesteps_done)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False, progress_bar=True)
        timesteps_done += learn_steps

        metrics = evaluate_and_export(model, config, timesteps_done, paths)
        print(json.dumps(metrics, indent=2))

    env_name = config.env_id.replace("-", "_").lower()
    model.save(paths.models / f"ppo_{env_name}_final.zip")
    env.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
