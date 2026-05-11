import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gymnasium as gym
from stable_baselines3 import SAC
import imageio
import numpy as np
import sys

env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1280, height=720)
model = SAC.load("models/humanoid_400000", env=env, device="cpu")

frames = []
obs = env.reset()[0]
target_frames = 5400  # 3 min at 30fps
chunk_size = 1800     # save every 1800 frames (1 min)

for i in range(target_frames):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    frames.append(env.render())
    if done:
        obs = env.reset()[0]
    
    if len(frames) == chunk_size:
        print(f"Saving chunk {i//chunk_size}...")
        with open(f"runs/humanoid_chunk_{i//chunk_size}.mp4", "wb") as f:
            imageio.mimsave(f, frames, fps=30, quality=7)
        frames = []
        print(f"Saved chunk {i//chunk_size}")

if frames:
    print("Saving final chunk...")
    imageio.mimsave("runs/humanoid_walk_3min.mp4", frames, fps=30, quality=7)
    
print("Done!")
