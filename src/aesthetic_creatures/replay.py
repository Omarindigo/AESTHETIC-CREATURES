from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aesthetic_creatures.art import make_art_video
from aesthetic_creatures.config import ArtConfig, ReplayConfig
from aesthetic_creatures.envs import make_eval_env
from aesthetic_creatures.model import load_model
from aesthetic_creatures.recorder import run_episode_and_record, save_rollout_npz
from aesthetic_creatures.render import save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Ant model or turn a rollout into an art video.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="Render a saved PPO Ant model to MP4.")
    replay_parser.add_argument("--env-id", type=str, default="Ant-v5")
    replay_parser.add_argument("--model-path", type=str, required=True)
    replay_parser.add_argument("--output-path", type=str, default="replay.mp4")
    replay_parser.add_argument("--rollout-npz", type=str, default="")
    replay_parser.add_argument("--max-steps", type=int, default=1000)
    replay_parser.add_argument("--fps", type=int, default=30)
    replay_parser.add_argument("--frame-stride", type=int, default=1)
    replay_parser.add_argument("--seed", type=int, default=42)
    replay_parser.add_argument("--stochastic", action="store_true")

    art_parser = subparsers.add_parser("make-art-video", help="Turn a rollout NPZ into an abstract animation.")
    art_parser.add_argument("--rollout-npz", type=str, required=True)
    art_parser.add_argument("--output-path", type=str, default="art_video.mp4")
    art_parser.add_argument("--width", type=int, default=1080)
    art_parser.add_argument("--height", type=int, default=1080)
    art_parser.add_argument("--fps", type=int, default=30)
    art_parser.add_argument("--history", type=int, default=60)

    return parser.parse_args()


def run_replay(config: ReplayConfig) -> None:
    env = make_eval_env(config.env_id, config.seed, render_mode="rgb_array")
    model = load_model(config.model_path)

    data = run_episode_and_record(
        env=env,
        model=model,
        max_steps=config.max_steps,
        deterministic=not config.stochastic,
        capture_frames=True,
        frame_stride=max(config.frame_stride, 1),
    )
    env.close()

    out_path = Path(config.output_path)
    save_video(data["frames"], out_path, fps=config.fps)

    if config.rollout_npz:
        save_rollout_npz(data, Path(config.rollout_npz))

    print(
        json.dumps(
            {
                "episode_reward": data["episode_reward"],
                "episode_length": data["episode_length"],
                "video": str(out_path),
                "rollout_npz": config.rollout_npz,
            },
            indent=2,
        )
    )


def run_art(config: ArtConfig) -> None:
    make_art_video(
        rollout_npz=config.rollout_npz,
        output_path=config.output_path,
        width=config.width,
        height=config.height,
        fps=config.fps,
        history=config.history,
    )
    print(json.dumps({"art_video": config.output_path}, indent=2))


def main() -> None:
    args = parse_args()

    if args.command == "replay":
        config = ReplayConfig(
            env_id=args.env_id,
            model_path=args.model_path,
            output_path=args.output_path,
            rollout_npz=args.rollout_npz,
            max_steps=args.max_steps,
            fps=args.fps,
            frame_stride=args.frame_stride,
            seed=args.seed,
            stochastic=args.stochastic,
        )
        run_replay(config)

    elif args.command == "make-art-video":
        config = ArtConfig(
            rollout_npz=args.rollout_npz,
            output_path=args.output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            history=args.history,
        )
        run_art(config)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
