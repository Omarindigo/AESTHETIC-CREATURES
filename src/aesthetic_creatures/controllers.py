from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p


@dataclass
class OscillatorParams:
    mode: int = 0            # 0=position target (sine), 1=velocity target (sine)
    amp: float = 0.6
    freq: float = 1.0        # Hz
    phase_stride: float = 0.6
    bias: float = 0.0


class OscillatorController:
    def __init__(self, body_id: int, joint_indices: List[int], params: OscillatorParams):
        self.body_id = int(body_id)
        self.joint_indices = [int(j) for j in joint_indices]
        self.params = params
        self.t = 0.0

    def reset(self):
        self.t = 0.0

    def step(self, dt: float):
        self.t += float(dt)
        w = 2.0 * np.pi * float(self.params.freq)

        for k, j in enumerate(self.joint_indices):
            phase = float(self.params.phase_stride) * float(k)
            s = np.sin(w * self.t + phase)

            if int(self.params.mode) == 0:
                target_pos = float(self.params.bias) + float(self.params.amp) * float(s)
                p.setJointMotorControl2(
                    self.body_id,
                    j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=60.0,
                    maxVelocity=20.0,
                    positionGain=0.05,
                    velocityGain=1.0,
                )
            else:
                target_vel = float(self.params.bias) + float(self.params.amp) * float(s)
                p.setJointMotorControl2(
                    self.body_id,
                    j,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_vel,
                    force=60.0,
                )

    @staticmethod
    def from_defaults(defaults: Dict[str, float]) -> OscillatorParams:
        return OscillatorParams(
            mode=int(defaults.get("mode", 0.0)),
            amp=float(defaults.get("amp", 0.6)),
            freq=float(defaults.get("freq", 1.0)),
            phase_stride=float(defaults.get("phase_stride", 0.6)),
            bias=float(defaults.get("bias", 0.0)),
        )