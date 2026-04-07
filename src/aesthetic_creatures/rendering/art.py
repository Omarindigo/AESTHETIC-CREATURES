from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np


class ArtStyle(Enum):
    TRAIL = "trail"
    MULTI_TRAIL = "multi_trail"
    PARTICLE = "particle"


PALETTES: Dict[str, List[Tuple[int, int, int]]] = {
    "fire": [(20, 0, 0), (180, 50, 0), (255, 120, 0), (255, 200, 50), (255, 255, 200)],
    "ocean": [(10, 20, 60), (20, 60, 120), (40, 140, 180), (100, 200, 220), (200, 240, 255)],
    "neon": [(20, 0, 40), (100, 0, 120), (180, 0, 180), (255, 50, 255), (255, 200, 255)],
    "matrix": [(10, 30, 10), (20, 80, 20), (50, 150, 50), (100, 220, 100), (200, 255, 200)],
    "sunset": [(40, 10, 60), (120, 40, 80), (200, 80, 60), (255, 150, 50), (255, 220, 150)],
    "aurora": [(20, 60, 80), (40, 140, 120), (80, 200, 160), (140, 220, 200), (200, 255, 240)],
}


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) == 0 or k <= 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")


def normalize_xy(points: np.ndarray, width: int, height: int, padding: int = 40) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    xy = pts[:, :2]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    coords = (xy - mins) * scale
    coords[:, 1] = (height - 2 * padding) - coords[:, 1]
    coords += padding
    return coords.astype(np.int32)


def get_interpolated_color(t: float, palette: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if len(palette) == 0:
        return (255, 255, 255)
    if len(palette) == 1:
        return palette[0]
    t = max(0, min(1, t))
    idx = t * (len(palette) - 1)
    i = int(idx)
    f = idx - i
    if i >= len(palette) - 1:
        return palette[-1]
    c1 = palette[i]
    c2 = palette[i + 1]
    return (int(c1[0] + f * (c2[0] - c1[0])), int(c1[1] + f * (c2[1] - c1[1])), int(c1[2] + f * (c2[2] - c1[2])))


def draw_disk(img: np.ndarray, cx: int, cy: int, radius: int, color: Tuple[int, int, int]) -> None:
    h, w, _ = img.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][mask] = color


def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int], thickness: int = 1) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        draw_disk(img, x0, y0, thickness, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def norm_signal(sig: np.ndarray) -> np.ndarray:
    if len(sig) == 0:
        return sig
    smin = float(sig.min())
    smax = float(sig.max())
    if abs(smax - smin) < 1e-8:
        return np.zeros_like(sig)
    return (sig - smin) / (smax - smin)


def make_trail_art(trajectory: np.ndarray, rewards: Optional[np.ndarray], actions: Optional[np.ndarray], width: int, height: int, history: int, palette: str) -> List[np.ndarray]:
    xy = normalize_xy(trajectory, width, height, padding=60)
    pal = PALETTES.get(palette, PALETTES["aurora"])
    
    if actions is not None and len(actions):
        action_mag = np.linalg.norm(actions, axis=1)
    else:
        action_mag = np.zeros(len(xy), dtype=np.float32)

    if rewards is not None and len(rewards):
        reward_curve = np.abs(np.diff(np.concatenate([[0], rewards])))
    else:
        reward_curve = np.zeros(len(xy), dtype=np.float32)

    action_mag = np.pad(action_mag, (0, max(0, len(xy) - len(action_mag))), mode="edge")[:len(xy)]
    reward_curve = np.pad(reward_curve, (0, max(0, len(xy) - len(reward_curve))), mode="edge")[:len(xy)]

    action_mag = moving_average(action_mag, 7)
    reward_curve = moving_average(reward_curve, 7)

    a_norm = norm_signal(action_mag)
    r_norm = norm_signal(reward_curve)

    frames: List[np.ndarray] = []
    hist = max(10, history)

    for i in range(len(xy)):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = int(pal[0][0] * 0.3)
        img[:, :, 1] = int(pal[0][1] * 0.3)
        img[:, :, 2] = int(pal[0][2] * 0.3)

        start = max(0, i - hist)
        for j in range(start + 1, i + 1):
            p0 = xy[j - 1]
            p1 = xy[j]
            t = (j - start) / max(1, i - start)
            intensity = 0.3 + 0.7 * t
            color = get_interpolated_color(r_norm[j] * 0.8 + a_norm[j] * 0.2, pal)
            color = tuple(int(c * intensity) for c in color)
            draw_line(img, int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), color, thickness=max(1, int(2 + 3 * t)))

        cx, cy = int(xy[i, 0]), int(xy[i, 1])
        radius = int(4 + 10 * (0.5 + 0.5 * a_norm[i]))
        glow_color = get_interpolated_color(r_norm[i], pal)
        core_color = tuple(min(255, c + 50) for c in glow_color)
        for r in range(3, 0, -1):
            alpha = 0.3 / r
            glow = tuple(int(c * alpha) for c in glow_color)
            draw_disk(img, cx, cy, radius + r * 4, glow)
        draw_disk(img, cx, cy, radius, core_color)

        bar_w = max(1, width // 4)
        left = width // 2 - bar_w // 2
        bottom = height - 30
        action_fill = int(bar_w * float(a_norm[i]))
        reward_fill = int(bar_w * float(r_norm[i]))
        img[bottom - 12:bottom - 6, left:left + action_fill] = pal[-2]
        img[bottom - 6:bottom, left:left + reward_fill] = pal[-1]

        frames.append(img)

    return frames


def make_multi_trail_art(trajectory: np.ndarray, rewards: Optional[np.ndarray], actions: Optional[np.ndarray], width: int, height: int, history: int, palette: str) -> List[np.ndarray]:
    pal = PALETTES.get(palette, PALETTES["aurora"])
    xy = normalize_xy(trajectory, width, height, padding=80)
    
    if rewards is not None and len(rewards):
        reward_curve = np.abs(np.diff(np.concatenate([[0], rewards])))
    else:
        reward_curve = np.zeros(len(xy), dtype=np.float32)
    
    reward_curve = np.pad(reward_curve, (0, max(0, len(xy) - len(reward_curve))), mode="edge")[:len(xy)]
    reward_curve = moving_average(reward_curve, 7)
    r_norm = norm_signal(reward_curve)

    frames: List[np.ndarray] = []
    hist = max(20, history)

    for i in range(len(xy)):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 2] = 20

        for offset in range(-2, 3):
            trail_hist = max(0, i - abs(offset) * (hist // 5))
            start = max(0, trail_hist)
            trail_pal = PALETTES.get(list(PALETTES.keys())[(list(PALETTES.keys()).index(palette) + offset) % len(PALETTES)], pal)
            for j in range(start + 1, trail_hist + 1):
                if j < len(xy):
                    p0 = xy[j - 1]
                    p1 = xy[j]
                    t = (j - start) / max(1, trail_hist - start)
                    intensity = 0.2 + 0.3 * (1 - abs(offset) / 3)
                    color = get_interpolated_color(r_norm[j] * 0.7 + (1 - abs(offset) / 5), trail_pal)
                    color = tuple(int(c * intensity) for c in color)
                    draw_line(img, int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), color, thickness=1)

        cx, cy = int(xy[i, 0]), int(xy[i, 1])
        for r in range(5, 0, -1):
            alpha = 0.15 / r
            color = get_interpolated_color(r_norm[i], pal)
            glow = tuple(int(c * alpha) for c in color)
            draw_disk(img, cx, cy, r * 3, glow)
        draw_disk(img, cx, cy, 6, (255, 255, 255))

        frames.append(img)

    return frames


def make_particle_art(trajectory: np.ndarray, rewards: Optional[np.ndarray], actions: Optional[np.ndarray], width: int, height: int, history: int, palette: str) -> List[np.ndarray]:
    pal = PALETTES.get(palette, PALETTES["neon"])
    xy = normalize_xy(trajectory, width, height, padding=60)
    
    if rewards is not None and len(rewards):
        reward_curve = np.abs(np.diff(np.concatenate([[0], rewards])))
    else:
        reward_curve = np.zeros(len(xy), dtype=np.float32)
    
    if actions is not None and len(actions):
        action_mag = np.linalg.norm(actions, axis=1)
    else:
        action_mag = np.zeros(len(xy), dtype=np.float32)
    
    reward_curve = np.pad(reward_curve, (0, max(0, len(xy) - len(reward_curve))), mode="edge")[:len(xy)]
    action_mag = np.pad(action_mag, (0, max(0, len(xy) - len(action_mag))), mode="edge")[:len(xy)]
    
    reward_curve = moving_average(reward_curve, 5)
    action_mag = moving_average(action_mag, 5)
    
    r_norm = norm_signal(reward_curve)
    a_norm = norm_signal(action_mag)

    frames: List[np.ndarray] = []
    hist = max(30, history)

    for i in range(len(xy)):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        for j in range(max(0, i - hist), i + 1):
            t = (j - max(0, i - hist)) / max(1, hist)
            base_color = get_interpolated_color(r_norm[j], pal)
            cx = int(xy[j, 0]) + int(np.random.randn() * 3)
            cy = int(xy[j, 1]) + int(np.random.randn() * 3)
            radius = int(1 + 4 * t)
            alpha = 0.3 + 0.7 * t
            if j == i:
                radius = int(8 + 12 * a_norm[i])
                alpha = 1.0
            particle_color = tuple(int(c * alpha) for c in base_color)
            draw_disk(img, cx, cy, radius, particle_color)
            if j == i:
                for r in range(3, 0, -1):
                    glow_alpha = 0.2 / r
                    glow_color = tuple(int(c * glow_alpha) for c in base_color)
                    draw_disk(img, cx, cy, radius + r * 5, glow_color)

        frames.append(img)

    return frames


def list_available_styles() -> List[str]:
    return [s.value for s in ArtStyle]


def list_available_palettes() -> List[str]:
    return list(PALETTES.keys())


def make_art_video(
    rollout_npz: str,
    output_path: str,
    width: int = 1080,
    height: int = 1080,
    fps: int = 30,
    history: int = 60,
    style: str = "trail",
    palette: str = "aurora",
) -> None:
    data = np.load(rollout_npz)
    
    trajectory = None
    for key in ["primary_com", "torso_com", "hand_com", "foot_com", "tip_com"]:
        if key in data:
            trajectory = data[key]
            break
    
    if trajectory is None:
        for key in data.files:
            if key.endswith("_com") and len(data[key].shape) == 2 and data[key].shape[1] >= 2:
                if trajectory is None:
                    trajectory = data[key]
                break
    
    if trajectory is None:
        raise ValueError("No trajectory found in rollout data.")
    
    rewards = data.get("rewards")
    actions = data.get("actions")
    
    print(f"Generating art with {len(trajectory)} frames")
    print(f"Style: {style}, Palette: {palette}")
    
    if style == "multi_trail":
        frames = make_multi_trail_art(trajectory, rewards, actions, width, height, history, palette)
    elif style == "particle":
        frames = make_particle_art(trajectory, rewards, actions, width, height, history, palette)
    else:
        frames = make_trail_art(trajectory, rewards, actions, width, height, history, palette)
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Saved art video to {out_path}")
