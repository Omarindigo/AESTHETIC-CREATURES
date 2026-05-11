import gymnasium as gym
from stable_baselines3 import SAC
import imageio
import numpy as np
import os

os.makedirs("models", exist_ok=True)
os.makedirs("runs/humanoid_sac/tensorboard", exist_ok=True)

env = gym.make("Humanoid-v4", render_mode=None)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    buffer_size=1_000_000,
    learning_rate=3e-4,
    tensorboard_log="runs/humanoid_sac/tensorboard",
)

TOTAL = 400_000
SAVE_EVERY = 25_000

print(f"Training for {TOTAL} timesteps...")
for i in range(TOTAL // SAVE_EVERY):
    model.learn(total_timesteps=SAVE_EVERY, reset_num_timesteps=False, progress_bar=True)
    model.save(f"models/humanoid_{SAVE_EVERY * (i+1)}")
    print(f"Saved: {SAVE_EVERY * (i+1)} timesteps")

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