from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aesthetic_creatures.config import TrainConfig, prepare_run_dirs, save_config
from aesthetic_creatures.envs import make_eval_env, make_training_env
from aesthetic_creatures.model import build_model
from aesthetic_creatures.recorder import append_metrics_row, run_episode_and_record, save_rollout_npz
from aesthetic_creatures.render import save_video


def evaluate_and_export(model, config: TrainConfig, step_count: int, paths) -> dict:
    eval_env = make_eval_env(config.env_id, config.seed, render_mode="rgb_array")

    episode_data = run_episode_and_record(
        env=eval_env,
        model=model,
        max_steps=config.eval_max_steps,
        deterministic=config.deterministic_eval,
        capture_frames=config.save_video,
        frame_stride=config.frame_stride,
    )
    eval_env.close()

    tag = f"step_{step_count:09d}"

    rollout_path = paths.rollouts / f"{tag}.npz"
    save_rollout_npz(episode_data, rollout_path)

    video_path = paths.videos / f"{tag}.mp4"
    if config.save_video:
        save_video(episode_data["frames"], video_path, fps=config.video_fps)

    model_path = paths.models / f"ppo_ant_{tag}.zip"
    model.save(model_path)

    metrics = {
        "timesteps": step_count,
        "eval_reward": episode_data["episode_reward"],
        "eval_length": episode_data["episode_length"],
        "saved_rollout": rollout_path.name,
        "saved_video": video_path.name if config.save_video else "",
    }
    append_metrics_row(paths.metrics_csv, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ant-v5 with PPO and export rollout data.")
    parser.add_argument("--env-id", type=str, default="Ant-v5")
    parser.add_argument("--output-dir", type=str, default="runs/ant_art")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--chunk-timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--stochastic-eval", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    model.save(paths.models / "ppo_ant_final.zip")
    env.close()


if __name__ == "__main__":
    main()
