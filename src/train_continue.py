import gymnasium as gym
from stable_baselines3 import SAC
import imageio
import numpy as np
import os

os.makedirs("models", exist_ok=True)

CONTINUE_FROM = 400000

env = gym.make("Humanoid-v4", render_mode=None)

model = SAC.load(f"models/humanoid_{CONTINUE_FROM}", env=env)

TOTAL = 800000
SAVE_EVERY = 25000

start = CONTINUE_FROM
print(f"Resuming from {start} timesteps...")

for i in range(start // SAVE_EVERY, TOTAL // SAVE_EVERY):
    model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False, progress_bar=True)
    current = SAVE_EVERY * (i + 1)
    model.save(f"models/humanoid_{current}")
    print(f"Saved: {current} timesteps")

print("Training complete! Rendering video...")

env = gym.make("Humanoid-v4", render_mode="rgb_array")
model = SAC.load(f"models/humanoid_{TOTAL}")
frames = []
obs = env.reset()[0]
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    frames.append(env.render())
    if done:
        obs = env.reset()[0]

imageio.mimsave("runs/humanoid_walk.mp4", frames, fps=30)
print("Saved runs/humanoid_walk.mp4")
