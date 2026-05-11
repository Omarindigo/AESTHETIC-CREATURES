from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def save_video(frames, output_path: Path, fps: int) -> None:
    if isinstance(frames, list):
        if len(frames) == 0:
            return
        frames_array = np.array(frames)
    else:
        frames_array = frames
    if frames_array.size == 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, list(frames_array), fps=fps)
