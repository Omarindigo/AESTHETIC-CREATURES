from __future__ import annotations

import argparse
import os
import math
import random

import pybullet as p
import pybullet_data

from aesthetic_creatures.creatures import get_creature_spec, build_creature
from aesthetic_creatures.controllers import OscillatorController
from aesthetic_creatures.recording import TrajectoryRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", type=str, default="starfish", choices=["starfish", "swimmer", "tumbler"])
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out_dir", type=str, default="specimens")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    time_step = 1 / 240
    steps = int(args.seconds / time_step)

    spec = get_creature_spec(args.creature)

    for i in range(args.count):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        p.loadURDF("plane.urdf")

        body_id, _ = build_creature(spec, base_pos=(0.0, 0.0, 0.65))
        joint_indices = list(range(p.getNumJoints(body_id)))
        controller = OscillatorController(body_id, joint_indices, OscillatorController.from_defaults(spec.controller_defaults))

        # Randomize controller a bit for “specimen” variation.
        controller.params.amp *= random.uniform(0.6, 1.4)
        controller.params.freq *= random.uniform(0.7, 1.5)
        controller.params.phase_stride *= random.uniform(0.6, 1.6)
        controller.params.bias += random.uniform(-0.2, 0.2)
        controller.params.mode = 0 if random.random() < 0.6 else 1

        rec = TrajectoryRecorder(body_id, creature_name=args.creature, time_step=time_step)
        for _ in range(steps):
            controller.step(time_step)
            p.stepSimulation()
            rec.capture()

        out_path = os.path.join(args.out_dir, f"{args.creature}_{i:03d}.json")
        rec.save_json(out_path)

    p.disconnect(cid)


if __name__ == "__main__":
    main()