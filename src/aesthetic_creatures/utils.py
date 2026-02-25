from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SimConfig:
    time_step: float = 1.0 / 240.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    substeps: int = 1


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def quat_mul(q1, q2):
    # (x,y,z,w) * (x,y,z,w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return (x, y, z, w)


def quat_from_axis_angle(axis, angle):
    ax = np.array(axis, dtype=np.float64)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    ax = ax / n
    s = math.sin(angle * 0.5)
    return (float(ax[0] * s), float(ax[1] * s), float(ax[2] * s), float(math.cos(angle * 0.5)))