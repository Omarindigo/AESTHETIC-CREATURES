import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gymnasium as gym
from stable_baselines3 import SAC
import imageio

env = gym.make("Humanoid-v4", render_mode="rgb_array", width=1280, height=720)
model = SAC.load("models/humanoid_400000", env=env, device="cpu")

print("Rendering 1-minute video...")

frames = []
obs = env.reset()[0]

for i in range(1800):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    frame = env.render()
    frames.append(frame)
    
    if done:
        obs = env.reset()[0]
    
    if i % 300 == 0:
        print(f"Frame {i}/1800")

imageio.mimsave("runs/humanoid_1min.mp4", frames, fps=30, quality=8)
print("Saved: runs/humanoid_1min.mp4")
