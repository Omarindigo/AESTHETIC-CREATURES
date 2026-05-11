import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from stable_baselines3 import SAC
import gymnasium as gym
import imageio
import numpy as np

CHECKPOINTS = [200000, 400000, 600000, 800000]

for ts in CHECKPOINTS:
    print(f"\n=== Rendering humanoid_{ts} ===")
    env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1920, height=1080)
    model = SAC.load(f"models/humanoid_{ts}", env=env, device="cpu")

    frames = []
    obs = env.reset()[0]
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        frames.append(env.render())
        if step % 200 == 0:
            print(f"  Frame {step}")
        if done:
            obs = env.reset()[0]

    out = f"../video/humanoid_{ts}k.mp4"
    imageio.mimsave(out, frames, fps=30, quality=8)
    print(f"Saved {out} ({len(frames)} frames, 1920x1080)")
    env.close()

print("\nAll done!")
