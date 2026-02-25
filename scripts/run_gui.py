from __future__ import annotations

import argparse
import time

import numpy as np
import pybullet as p
import pybullet_data

from aesthetic_creatures.creatures import get_creature_spec, build_creature
from aesthetic_creatures.controllers import OscillatorController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", type=str, default="starfish", choices=["starfish", "swimmer", "tumbler"])
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    p.loadURDF("plane.urdf")

    spec = get_creature_spec(args.creature)
    body_id, _ = build_creature(spec, base_pos=(0.0, 0.0, 0.65))

    joint_indices = list(range(p.getNumJoints(body_id)))
    controller = OscillatorController(body_id, joint_indices, OscillatorController.from_defaults(spec.controller_defaults))

    # Simple interactive sliders for live tuning.
    amp_id = p.addUserDebugParameter("amp", 0.0, 2.0, float(controller.params.amp))
    freq_id = p.addUserDebugParameter("freq", 0.1, 4.0, float(controller.params.freq))
    phs_id = p.addUserDebugParameter("phase_stride", 0.0, 2.5, float(controller.params.phase_stride))
    bias_id = p.addUserDebugParameter("bias", -2.0, 2.0, float(controller.params.bias))
    mode_id = p.addUserDebugParameter("mode(0=pos,1=vel)", 0.0, 1.0, float(controller.params.mode))

    steps = int(args.seconds * 240)
    for _ in range(steps):
        controller.params.amp = float(p.readUserDebugParameter(amp_id))
        controller.params.freq = float(p.readUserDebugParameter(freq_id))
        controller.params.phase_stride = float(p.readUserDebugParameter(phs_id))
        controller.params.bias = float(p.readUserDebugParameter(bias_id))
        controller.params.mode = 0 if float(p.readUserDebugParameter(mode_id)) < 0.5 else 1

        controller.step(1 / 240)
        p.stepSimulation()
        time.sleep(1 / 240)

    p.disconnect(cid)


if __name__ == "__main__":
    main()