from __future__ import annotations

import argparse

import pybullet as p
import pybullet_data

from aesthetic_creatures.creatures import get_creature_spec, build_creature
from aesthetic_creatures.controllers import OscillatorController
from aesthetic_creatures.recording import TrajectoryRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", type=str, default="starfish", choices=["starfish", "swimmer", "tumbler"])
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    time_step = 1 / 240
    p.setTimeStep(time_step)

    p.loadURDF("plane.urdf")

    spec = get_creature_spec(args.creature)
    body_id, _ = build_creature(spec, base_pos=(0.0, 0.0, 0.65))
    joint_indices = list(range(p.getNumJoints(body_id)))
    controller = OscillatorController(body_id, joint_indices, OscillatorController.from_defaults(spec.controller_defaults))

    rec = TrajectoryRecorder(body_id, creature_name=args.creature, time_step=time_step)

    steps = int(args.seconds / time_step)
    for _ in range(steps):
        controller.step(time_step)
        p.stepSimulation()
        rec.capture()

    rec.save_json(args.out)
    p.disconnect(cid)


if __name__ == "__main__":
    main()