import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gymnasium as gym
from stable_baselines3 import SAC
import imageio
import numpy as np

env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1280, height=720)
model = SAC.load("models/humanoid_400000", env=env, device="cpu")

frames = []
obs = env.reset()[0]
target_frames = 5400  # 3 min at 30fps

for i in range(target_frames):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    frames.append(env.render())
    if done:
        obs = env.reset()[0]
    if i % 500 == 0:
        print(f"Frame {i}/{target_frames}")

imageio.mimsave("runs/humanoid_walk_3min.mp4", frames, fps=30, quality=8)
print("Saved runs/humanoid_walk_3min.mp4")