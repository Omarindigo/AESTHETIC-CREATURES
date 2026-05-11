from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aesthetic_creatures.config import prepare_run_dirs, save_config, TrainConfig
from aesthetic_creatures.envs import make_eval_env, make_training_env, get_env_spec
from aesthetic_creatures.recording import append_metrics_row, run_episode_and_record, save_rollout_npz
from aesthetic_creatures.rendering import save_video

from stable_baselines3 import SAC
import numpy as np


def train_single_agent(env_id: str, output_dir: str, total_timesteps: int, seed: int, eval_freq: int) -> dict:
    config = TrainConfig(
        env_id=env_id,
        output_dir=output_dir,
        total_timesteps=total_timesteps,
        n_envs=1,
        eval_max_steps=1000,
        seed=seed,
        device="cpu",
        policy_net=(256, 256),
        save_video=True,
        video_fps=30,
        frame_stride=2,
        deterministic_eval=True,
    )

    paths = prepare_run_dirs(output_dir)
    
    print(f"\n{'='*50}")
    print(f"Training {env_id} (seed={seed})")
    print(f"{'='*50}")

    env = make_training_env(env_id, 1, seed)
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        gradient_steps=1,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
    )

    timesteps_done = 0
    best_reward = -float('inf')
    
    while timesteps_done < total_timesteps:
        learn_steps = min(eval_freq, total_timesteps - timesteps_done)
        model.learn(total_timesteps=learn_steps, reset_num_timesteps=False, progress_bar=False)
        timesteps_done += learn_steps
        
        eval_env = make_eval_env(env_id, seed, render_mode="rgb_array")
        obs = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        
        for _ in range(config.eval_max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            if done:
                break
        
        eval_env.close()
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            env_name = env_id.replace("-", "_").lower()
            model.save(paths.models / f"sac_{env_name}_best.zip")

        print(f"  [{timesteps_done:,}/{total_timesteps:,}] Reward: {episode_reward:.1f} (best: {best_reward:.1f})")
    
    env.close()
    
    return {
        "env_id": env_id,
        "seed": seed,
        "best_reward": best_reward,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel training with multiple agents")
    parser.add_argument("--envs", nargs="+", default=["Walker2d-v5", "HalfCheetah-v5", "Ant-v5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 999])
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# PARALLEL TRAINING: {len(args.envs)} envs x {len(args.seeds)} seeds = {len(args.envs) * len(args.seeds)} agents")
    print(f"# Parallel workers: {args.parallel}")
    print(f"# Total timesteps per agent: {args.total_timesteps:,}")
    print(f"{'#'*60}\n")

    tasks = []
    for env_id in args.envs:
        for seed in args.seeds:
            output_dir = f"runs/parallel_{env_id.lower()}_seed{seed}"
            tasks.append((env_id, output_dir, args.total_timesteps, seed, args.eval_freq))

    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(train_single_agent, *task): task 
            for task in tasks
        }
        
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Completed: {result['env_id']} seed={result['seed']} - Best: {result['best_reward']:.1f}")
            except Exception as e:
                print(f"\n✗ Failed: {task[0]} - {e}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE - Summary:")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x['best_reward']):
        print(f"  {r['env_id']:<25} seed={r['seed']:<4} → Best reward: {r['best_reward']:.1f}")
    
    best = max(results, key=lambda x: x['best_reward'])
    print(f"\nBest agent: {best['env_id']} seed={best['seed']}")
    print(f"Model saved: {best['output_dir']}/models/")


if __name__ == "__main__":
    main()
