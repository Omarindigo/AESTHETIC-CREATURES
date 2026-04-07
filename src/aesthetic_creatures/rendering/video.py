from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_video(frames: np.ndarray, output_path: Path, fps: int) -> None:
    if frames.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames), fps=fps)
