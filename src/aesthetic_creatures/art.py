from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np


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


def draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
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


def make_art_video(
    rollout_npz: str,
    output_path: str,
    width: int = 1080,
    height: int = 1080,
    fps: int = 30,
    history: int = 60,
) -> None:
    data = np.load(rollout_npz)
    torso = data.get("torso_com")
    actions = data.get("actions")
    rewards = data.get("rewards")

    if torso is None or len(torso) == 0:
        raise ValueError("The rollout file does not contain torso_com data.")

    xy = normalize_xy(torso, width, height, padding=60)

    if actions is not None and len(actions):
        action_mag = np.linalg.norm(actions, axis=1)
    else:
        action_mag = np.zeros(len(xy), dtype=np.float32)

    if rewards is not None and len(rewards):
        reward_curve = rewards
    else:
        reward_curve = np.zeros(len(xy), dtype=np.float32)

    action_mag = np.pad(action_mag, (0, max(0, len(xy) - len(action_mag))), mode="edge")[: len(xy)]
    reward_curve = np.pad(reward_curve, (0, max(0, len(xy) - len(reward_curve))), mode="edge")[: len(xy)]

    action_mag = moving_average(action_mag, 7)
    reward_curve = moving_average(reward_curve, 7)

    a_norm = norm_signal(action_mag)
    r_norm = norm_signal(reward_curve)

    frames: List[np.ndarray] = []
    hist = max(10, history)

    for i in range(len(xy)):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        start = max(0, i - hist)
        for j in range(start + 1, i + 1):
            p0 = xy[j - 1]
            p1 = xy[j]
            t = (j - start) / max(1, i - start)
            color = (
                int(30 + 150 * t),
                int(80 + 100 * (1.0 - r_norm[j])),
                int(120 + 100 * a_norm[j]),
            )
            draw_line(img, int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]), color, thickness=2)

        cx, cy = int(xy[i, 0]), int(xy[i, 1])
        radius = int(6 + 12 * a_norm[i])

        core = (
            int(180 + 70 * r_norm[i]),
            int(120 + 100 * a_norm[i]),
            int(200 + 50 * (1.0 - r_norm[i])),
        )
        halo = (
            int(60 + 80 * a_norm[i]),
            int(40 + 120 * r_norm[i]),
            int(100 + 80 * a_norm[i]),
        )

        draw_disk(img, cx, cy, radius + 8, halo)
        draw_disk(img, cx, cy, radius, core)

        bar_w = max(1, width // 3)
        left = width // 2 - bar_w // 2
        bottom = height - 30
        action_fill = int(bar_w * float(a_norm[i]))
        reward_fill = int(bar_w * float(r_norm[i]))

        img[bottom - 18 : bottom - 10, left : left + action_fill] = (200, 120, 40)
        img[bottom - 8 : bottom, left : left + reward_fill] = (80, 200, 180)

        frames.append(img)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
