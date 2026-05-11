import gymnasium as gym
from stable_baselines3 import SAC
import imageio
import numpy as np
import os
import sys
import threading
import time
from datetime import datetime

os.makedirs("models", exist_ok=True)
os.makedirs("runs", exist_ok=True)

TOTAL = 1_000_000
SAVE_EVERY = 25_000
CONTINUE_FROM = 0
RENDER_INTERVAL = 5000

ctrl = {"paused": False, "stop": False, "restart": False}

def input_loop():
    if sys.platform == "win32":
        import msvcrt
        print("\nControls: P=pause Q=quit R=restart VIEW=render frame\n")
        while not ctrl["stop"]:
            if msvcrt.kbhit():
                c = msvcrt.getch().decode().lower()
                if c == 'q':
                    print("\nQuit..."); ctrl["stop"] = True
                elif c == 'p':
                    ctrl["paused"] = not ctrl["paused"]
                    print("\nPAUSED" if ctrl["paused"] else "\nRESUMED")
                elif c == 'r':
                    print("\nRestart..."); ctrl["restart"] = True
    else:
        import tty, termios
        print("\nControls: P=pause Q=quit R=restart\n")
        fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
        tty.setraw(fd)
        try:
            while not ctrl["stop"]:
                c = sys.stdin.read(1).lower()
                if c == 'q': print("\nQuit..."); ctrl["stop"] = True
                elif c == 'p': ctrl["paused"] = not ctrl["paused"]; print("\nPAUSED" if ctrl["paused"] else "\nRESUMED")
                elif c == 'r': print("\nRestart..."); ctrl["restart"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

env = gym.make("Humanoid-v4", render_mode="human")
view_env = gym.make("Humanoid-v4", render_mode="rgb_array")

model = SAC.load(f"models/humanoid_{CONTINUE_FROM}", env=env) if CONTINUE_FROM > 0 else SAC(
    "MlpPolicy", env, verbose=0, device="cuda",
    buffer_size=1_000_000, learning_rate=3e-4,
)

start = CONTINUE_FROM
print(f"\n{'='*50}")
print(f"Training SAC Humanoid from {start:,} timesteps")
print(f"Target: {TOTAL:,} | Save every: {SAVE_EVERY:,}")
print(f"{'='*50}")
print("\nControls: P=pause Q=quit R=restart\n")

threading.Thread(target=input_loop, daemon=True).start()

def render_preview(save_video=True, steps=200):
    obs = view_env.reset()[0]
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = view_env.step(action)
        if done:
            obs = view_env.reset()[0]
    if save_video:
        frames = []
        obs = view_env.reset()[0]
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = view_env.step(action)
            frames.append(view_env.render())
            if done:
                obs = view_env.reset()[0]
        return frames
    return None

def evaluate(episodes=3):
    print(f"\nEvaluating ({episodes} episodes)...")
    for ep in range(episodes):
        obs = env.reset()[0]
        total_rew = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _, _ = env.step(action)
            total_rew += rew
        print(f"  Episode {ep+1}: reward = {total_rew:.1f}")

last_render = start

while start < TOTAL and not ctrl["stop"]:
    if ctrl["restart"]:
        print(">>> RESTARTING <<<")
        env.close()
        view_env.close()
        env = gym.make("Humanoid-v4", render_mode="human")
        view_env = gym.make("Humanoid-v4", render_mode="rgb_array")
        model = SAC.load(f"models/humanoid_{start}", env=env)
        ctrl["restart"] = False

    for i in range(start // SAVE_EVERY, TOTAL // SAVE_EVERY):
        if ctrl["stop"] or ctrl["restart"]: break
        
        while ctrl["paused"] and not ctrl["stop"] and not ctrl["restart"]:
            time.sleep(0.1)
        
        evaluate()
        
        print(f"\n--- Training batch {i+1} ({SAVE_EVERY:,} steps) ---")
        model.learn(
            total_timesteps=SAVE_EVERY,
            reset_num_timesteps=False,
            progress_bar=True,
            log_interval=1
        )
        
        current = SAVE_EVERY * (i + 1)
        model.save(f"models/humanoid_{current}")
        start = current
        
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Saved: {current:,}/{TOTAL:,} ({100*current//TOTAL}%)")
        
        if current - last_render >= RENDER_INTERVAL:
            print(f"\n>>> Rendering preview <<<")
            frames = render_preview(save_video=True, steps=200)
            if frames:
                vid_path = f"runs/humanoid_{current}.mp4"
                imageio.mimsave(vid_path, frames, fps=30)
                print(f"Saved: {vid_path}")
            last_render = current
        
        if ctrl["stop"]: break

env.close()
view_env.close()

final = start if ctrl["stop"] else TOTAL
print(f"\n{'='*50}")
print(f"Training {'STOPPED' if ctrl['stop'] else 'COMPLETE'} at {final:,} timesteps")
print(f"{'='*50}")
print("Rendering final video...")

view_env = gym.make("Humanoid-v4", render_mode="rgb_array")
model = SAC.load(f"models/humanoid_{final}")

frames = []
obs = view_env.reset()[0]
for _ in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = view_env.step(action)
    frames.append(view_env.render())
    if done:
        obs = view_env.reset()[0]
view_env.close()

imageio.mimsave("runs/humanoid_walk.mp4", frames, fps=30)
print("Saved runs/humanoid_walk.mp4")
print("\nDone!")