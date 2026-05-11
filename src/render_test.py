from stable_baselines3 import SAC
import gymnasium as gym
import imageio

env = gym.make("Humanoid-v4", render_mode="rgb_array")
model = SAC.load("runs/humanoid_sac/models/sac_humanoid_6276") if __import__("os").path.exists("runs/humanoid_sac/models/sac_humanoid_6276") else None

if model is None:
    print("No saved model. Train first with quick_train.py")
else:
    frames = []
    obs = env.reset()[0]
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        frames.append(env.render())
        if done:
            obs = env.reset()[0]
    
    imageio.mimsave("runs/humanoid_walk.mp4", frames, fps=30)
    print("Saved to runs/humanoid_walk.mp4")