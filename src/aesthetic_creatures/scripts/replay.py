from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aesthetic_creatures.config import ArtConfig, ReplayConfig
from aesthetic_creatures.envs import make_eval_env, get_env_spec
from aesthetic_creatures.models import load_model
from aesthetic_creatures.recording import run_episode_and_record, save_rollout_npz
from aesthetic_creatures.rendering import make_art_video, list_available_styles, list_available_palettes, save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay trained models or generate art from rollouts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="Render a trained model to MP4.")
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
    art_parser.add_argument("--style", type=str, default="trail", choices=list_available_styles() + ["list"])
    art_parser.add_argument("--palette", type=str, default="aurora", choices=list_available_palettes() + ["list"])

    list_parser = subparsers.add_parser("list", help="List available styles and palettes.")
    list_parser.add_argument("--styles", action="store_true")
    list_parser.add_argument("--palettes", action="store_true")

    return parser.parse_args()


def run_replay(config: ReplayConfig) -> None:
    print(f"\nReplaying {config.env_id} from {config.model_path}")
    
    eval_env = make_eval_env(config.env_id, config.seed, render_mode="rgb_array")
    model = load_model(config.model_path)
    env_spec = get_env_spec(config.env_id)
    body_parts = env_spec.body_parts

    data = run_episode_and_record(
        env=eval_env,
        model=model,
        max_steps=config.max_steps,
        deterministic=not config.stochastic,
        capture_frames=True,
        frame_stride=max(config.frame_stride, 1),
        body_parts=body_parts,
    )
    eval_env.close()

    out_path = Path(config.output_path)
    save_video(data["frames"], out_path, fps=config.fps)

    if config.rollout_npz:
        save_rollout_npz(data, Path(config.rollout_npz))

    print(json.dumps({
        "env_id": config.env_id,
        "episode_reward": data["episode_reward"],
        "episode_length": data["episode_length"],
        "video": str(out_path),
        "rollout_npz": config.rollout_npz,
        "body_parts": body_parts,
    }, indent=2))


def run_art(config: ArtConfig, style: str, palette: str) -> None:
    make_art_video(
        rollout_npz=config.rollout_npz,
        output_path=config.output_path,
        width=config.width,
        height=config.height,
        fps=config.fps,
        history=config.history,
        style=style,
        palette=palette,
    )
    print(json.dumps({"art_video": config.output_path, "style": style, "palette": palette}, indent=2))


def list_all():
    print("\nArt Styles:")
    print("=" * 50)
    for style in list_available_styles():
        print(f"  {style}")

    print("\n\nColor Palettes:")
    print("=" * 50)
    for pal in list_available_palettes():
        print(f"  {pal}")
    print("\n")


def main() -> None:
    args = parse_args()

    if args.command == "list":
        list_all()
        return

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
        style = args.style
        palette = args.palette
        
        if style == "list":
            print("Available styles:", list_available_styles())
            return
        if palette == "list":
            print("Available palettes:", list_available_palettes())
            return
        
        config = ArtConfig(
            rollout_npz=args.rollout_npz,
            output_path=args.output_path,
            width=args.width,
            height=args.height,
            fps=args.fps,
            history=args.history,
        )
        run_art(config, style, palette)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
