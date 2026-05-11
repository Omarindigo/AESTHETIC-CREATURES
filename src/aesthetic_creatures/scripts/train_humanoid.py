import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
from pathlib import Path

model_dir = "runs/humanoid_sac/models"
log_dir = "runs/humanoid_sac/tensorboard"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(env_id, timesteps=25000):
    env = gym.make(env_id, render_mode=None)
    
    model = SAC(
        'MlpPolicy', 
        env, 
        verbose=1, 
        device='cuda',
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
    )
    
    iters = 0
    total = 0
    while True:
        iters += 1
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False, progress_bar=True)
        total += timesteps
        model.save(f"{model_dir}/sac_humanoid_{total}")
        print(f"Saved: {total} timesteps")


def test(env_id, path_to_model):
    env = gym.make(env_id, render_mode='human')
    model = SAC.load(path_to_model, env=env)
    
    obs = env.reset()[0]
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Humanoid-v4')
    parser.add_argument('--timesteps', type=int, default=25000)
    parser.add_argument('--test', type=str, default=None)
    args = parser.parse_args()
    
    if args.test:
        test(args.env, args.test)
    else:
        train(args.env, args.timesteps)
