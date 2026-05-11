import gymnasium as gym
from stable_baselines3 import SAC
import os

os.makedirs("runs/humanoid_sac/models", exist_ok=True)
os.makedirs("runs/humanoid_sac/tensorboard", exist_ok=True)

env = gym.make("Humanoid-v4", render_mode=None)

model = SAC(
    'MlpPolicy', 
    env, 
    verbose=1, 
    device='cuda',
    tensorboard_log="runs/humanoid_sac/tensorboard",
    buffer_size=1_000_000,
)

print("Training Humanoid-v4 with SAC...")
for i in range(10):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True)
    model.save(f"runs/humanoid_sac/models/sac_humanoid_{10000*(i+1)}")
    print(f"Saved: {10000*(i+1)} timesteps")

print("DONE! Model saved.")