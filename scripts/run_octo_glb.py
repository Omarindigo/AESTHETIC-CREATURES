from __future__ import annotations

import argparse
import os
import time
import traceback
import sys

import pybullet as p
import pybullet_data

from aesthetic_creatures.octo_from_glb import make_octo_from_glb
from aesthetic_creatures.creatures import build_creature
from aesthetic_creatures.controllers import OscillatorController


def _try_build(spec, label: str):
    print(f"\nTRY BUILD: {label}")
    sys.stdout.flush()
    sys.stderr.flush()
    body_id, extra = build_creature(spec, base_pos=(0.0, 0.0, 0.65), use_self_collision=False)
    print("BUILD OK:", label, "body_id:", body_id, "numJoints:", p.getNumJoints(body_id))
    return body_id, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=str, default="assets/Octo.glb")
    ap.add_argument("--arms", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()

    print("RUN_OCTO_GLB: starting")
    print("cwd:", os.getcwd())
    print("glb arg:", args.glb)
    print("glb exists:", os.path.exists(args.glb))

    cid = p.connect(p.GUI)
    print("pybullet connected:", cid)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)

    plane_id = p.loadURDF("plane.urdf")
    print("plane loaded:", plane_id)

    print("building spec from glb...")
    spec_full = make_octo_from_glb(args.glb, arms=args.arms)
    print("spec built. links:", len(spec_full.links), "joints:", len(spec_full.joints))

    # IMPORTANT: Bullet is hard-crashing inside build_creature for the full model.
    # We isolate the first crash point by progressively building larger subsets.
    sizes = [4, 8, 16, len(spec_full.links)]

    built_body_id = None

    for n in sizes:
        # Make a sliced copy (dataclass-ish, but we rebuild a new CreatureSpec)
        spec = type(spec_full)(
            spec_full.name,
            spec_full.base,
            spec_full.links[:n],
            spec_full.joints[:n],
            spec_full.controller_defaults,
        )

        print("calling build_creature... (links/joints =", n, ")")
        sys.stdout.flush()
        sys.stderr.flush()

        try:
            built_body_id, _ = _try_build(spec, f"{n}_links")
            # If we succeeded, clear and reset sim before next attempt
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.setTimeStep(1 / 240)
            p.loadURDF("plane.urdf")
        except Exception as e:
            print("Python exception during build (rare):", repr(e))
            traceback.print_exc()
            input("Press Enter to close...")
            return

    # If we got here, every size succeeded (rare). Build full once more and run.
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.loadURDF("plane.urdf")

    print("\nFINAL BUILD (full creature)")
    sys.stdout.flush()
    sys.stderr.flush()

    body_id, _ = build_creature(spec_full, base_pos=(0.0, 0.0, 0.65), use_self_collision=False)
    print("body built:", body_id, "numJoints:", p.getNumJoints(body_id))

    joint_indices = list(range(p.getNumJoints(body_id)))
    controller = OscillatorController(body_id, joint_indices, OscillatorController.from_defaults(spec_full.controller_defaults))

    amp_id = p.addUserDebugParameter("amp", 0.0, 2.0, float(controller.params.amp))
    freq_id = p.addUserDebugParameter("freq", 0.1, 4.0, float(controller.params.freq))
    phs_id = p.addUserDebugParameter("phase_stride", 0.0, 2.5, float(controller.params.phase_stride))
    bias_id = p.addUserDebugParameter("bias", -2.0, 2.0, float(controller.params.bias))
    mode_id = p.addUserDebugParameter("mode(0=pos,1=vel)", 0.0, 1.0, float(controller.params.mode))

    print("entering sim loop for seconds:", args.seconds)
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

    print("done. disconnecting.")
    p.disconnect(cid)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("ERROR occurred. Press Enter to close...")