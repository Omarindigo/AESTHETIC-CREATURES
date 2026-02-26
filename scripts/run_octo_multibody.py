from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from aesthetic_creatures.glb_topology import load_spheres_from_glb, infer_radial_topology


_ARM_RE = re.compile(r"arm_ring_(\d+)", re.IGNORECASE)


def _get_name(obj) -> str:
    if isinstance(obj, dict):
        return str(obj.get("name", ""))
    return str(getattr(obj, "name", ""))


def _get_center(obj) -> np.ndarray:
    if isinstance(obj, dict):
        return np.array(obj["center"], dtype=np.float64)
    return np.array(getattr(obj, "center"), dtype=np.float64)


def _get_radius(obj) -> float:
    if isinstance(obj, dict):
        return float(obj["radius"])
    return float(getattr(obj, "radius"))


def _arm_index_from_name(name: str) -> int | None:
    m = _ARM_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _seg_counts_from_names(spheres, arms: int, base_center: np.ndarray) -> List[int]:
    groups: Dict[int, List[float]] = {i + 1: [] for i in range(arms)}  # 1..arms

    for s in spheres:
        name = _get_name(s)
        a = _arm_index_from_name(name)
        if a is None:
            continue
        c = _get_center(s)
        d = float(np.linalg.norm(c - base_center))
        groups.setdefault(a, []).append(d)

    seg_counts = []
    for a in range(1, arms + 1):
        ds = groups.get(a, [])
        ds.sort()
        seg_counts.append(len(ds))

    return seg_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=str, default="assets/Octo.glb")
    ap.add_argument("--arms", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--target_base_radius", type=float, default=0.18)
    ap.add_argument("--amp", type=float, default=0.55)
    ap.add_argument("--freq", type=float, default=1.0)
    ap.add_argument("--phase_stride", type=float, default=0.6)
    args = ap.parse_args()

    print("RUN_OCTO_MULTIBODY: starting")
    print("cwd:", os.getcwd())
    print("glb:", args.glb, "exists:", os.path.exists(args.glb))

    spheres = load_spheres_from_glb(args.glb)
    topo = infer_radial_topology(spheres, arms=args.arms)

    base_center = np.array(topo.base.center, dtype=np.float64)
    raw_base_r = float(topo.base.radius)
    raw_seg_r = float(topo.small_radius)

    scale = (float(args.target_base_radius) / raw_base_r) if raw_base_r > 1e-9 else 1.0
    base_r = float(np.clip(raw_base_r * scale, 0.05, 0.60))
    seg_r = float(np.clip(raw_seg_r * scale, 0.01, 0.20))

    print("GLB radii:", "raw_base", raw_base_r, "raw_seg", raw_seg_r, "scale", scale)
    print("Sim radii:", "base_r", base_r, "seg_r", seg_r)

    # Try named grouping first (may fail if GLB doesn't preserve names)
    seg_counts_named = _seg_counts_from_names(spheres, args.arms, base_center)

    # FORCE 8 ARMS: distribute total segments evenly across arms.
    # We count all non-base spheres as segments.
    total_segs = 0
    for s in spheres:
        name = _get_name(s).lower()
        if "base" in name:
            continue
        if "arm_ring" in name:
            total_segs += 1

    # If names weren't present or didn't count anything, fall back to "all spheres minus 1 base"
    if total_segs <= 0:
        total_segs = max(0, len(spheres) - 1)

    per = total_segs // args.arms
    rem = total_segs % args.arms
    seg_counts_forced = [per + (1 if i < rem else 0) for i in range(args.arms)]

    print("segments per arm (named):", seg_counts_named, "total:", sum(seg_counts_named))
    print("segments per arm (forced):", seg_counts_forced, "total:", sum(seg_counts_forced))

    # Use forced counts so you ALWAYS get 8 arms
    seg_counts = seg_counts_forced

    # Constant spacing for stability
    link_len = float(np.clip((seg_r * 2.0) * 0.98, 0.03, 0.20))

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.loadURDF("plane.urdf")

    base_col = p.createCollisionShape(p.GEOM_SPHERE, radius=base_r)
    base_vis = p.createVisualShape(p.GEOM_SPHERE, radius=base_r, rgbaColor=[0.75, 0.75, 0.75, 1.0])

    seg_col = p.createCollisionShape(p.GEOM_SPHERE, radius=seg_r)
    seg_vis = p.createVisualShape(p.GEOM_SPHERE, radius=seg_r, rgbaColor=[0.85, 0.85, 0.85, 1.0])

    link_masses: List[float] = []
    link_col: List[int] = []
    link_vis: List[int] = []
    link_pos: List[List[float]] = []
    link_orn: List[List[float]] = []
    link_in_pos: List[List[float]] = []
    link_in_orn: List[List[float]] = []
    link_parents: List[int] = []
    link_joint_types: List[int] = []
    link_joint_axis: List[List[float]] = []

    link_meta: List[Tuple[int, int]] = []

    for a in range(args.arms):
        nseg = int(seg_counts[a])
        if nseg <= 0:
            continue

        ang = (2.0 * math.pi * a) / float(args.arms)
        dx = math.cos(ang)
        dy = math.sin(ang)
        dir_xy = np.array([dx, dy, 0.0], dtype=np.float64)

        for j in range(nseg):
            this_index = len(link_masses)
            link_meta.append((a, j))

            link_masses.append(0.12)
            link_col.append(seg_col)
            link_vis.append(seg_vis)

            if j == 0:
                rel = dir_xy * float(base_r + seg_r * 0.95)
                parent_link = -1
            else:
                rel = dir_xy * float(link_len)
                parent_link = this_index - 1

            bullet_parent = 0 if parent_link < 0 else (int(parent_link) + 1)
            link_parents.append(bullet_parent)

            link_pos.append([float(rel[0]), float(rel[1]), float(rel[2])])
            link_orn.append([0.0, 0.0, 0.0, 1.0])
            link_in_pos.append([0.0, 0.0, 0.0])
            link_in_orn.append([0.0, 0.0, 0.0, 1.0])

            link_joint_types.append(p.JOINT_REVOLUTE)

            ax = 0.03 * dx
            ay = 0.03 * dy
            link_joint_axis.append([float(ax), float(ay), 1.0])

    start_pos = [0.0, 0.0, 0.65]
    start_orn = [0.0, 0.0, 0.0, 1.0]

    body_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=start_pos,
        baseOrientation=start_orn,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_col,
        linkVisualShapeIndices=link_vis,
        linkPositions=link_pos,
        linkOrientations=link_orn,
        linkInertialFramePositions=link_in_pos,
        linkInertialFrameOrientations=link_in_orn,
        linkParentIndices=link_parents,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axis,
    )

    print("CREATED body:", body_id, "numJoints:", p.getNumJoints(body_id))

    nj = p.getNumJoints(body_id)
    for j in range(nj):
        p.changeDynamics(body_id, j, lateralFriction=1.0, restitution=0.0, linearDamping=0.05, angularDamping=0.05)

    for i in range(nj):
        for k in range(i + 1, nj):
            p.setCollisionFilterPair(body_id, body_id, i, k, enableCollision=0)

    amp_id = p.addUserDebugParameter("amp", 0.0, 2.0, float(args.amp))
    freq_id = p.addUserDebugParameter("freq", 0.1, 4.0, float(args.freq))
    phs_id = p.addUserDebugParameter("phase_stride", 0.0, 2.5, float(args.phase_stride))

    t0 = time.time()
    steps = int(float(args.seconds) * 240.0)
    for _ in range(steps):
        amp = float(p.readUserDebugParameter(amp_id))
        freq = float(p.readUserDebugParameter(freq_id))
        phase_stride = float(p.readUserDebugParameter(phs_id))

        t = time.time() - t0
        w = 2.0 * math.pi * freq

        for joint_index in range(nj):
            a, seg_i = link_meta[joint_index]
            arm_phase = (2.0 * math.pi * a) / float(args.arms)
            target = amp * math.sin(w * t + phase_stride * seg_i + arm_phase)

            p.setJointMotorControl2(
                body_id,
                joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=4.0,
                positionGain=0.20,
                velocityGain=0.75,
            )

        p.stepSimulation()
        time.sleep(1 / 240)

    p.disconnect(cid)


if __name__ == "__main__":
    main()