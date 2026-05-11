import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gymnasium as gym
from stable_baselines3 import SAC
import imageio

env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1280, height=720)
model = SAC.load("models/humanoid_400000", env=env, device="cpu")

frames = []
obs = env.reset()[0]
for i in range(1800):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    frames.append(env.render())
    if done:
        obs = env.reset()[0]
    if i % 500 == 0:
        print(f"Frame {i}")

imageio.mimsave("runs/humanoid_part1.mp4", frames, fps=30, quality=8)
print("Saved part 1")
