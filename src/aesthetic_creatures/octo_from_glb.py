from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .creatures import LinkSpec, JointSpec, CreatureSpec
from .glb_topology import load_spheres_from_glb, infer_radial_topology


def make_octo_from_glb(
    glb_path: str,
    arms: int = 8,
    base_mass: float = 1.3,
    seg_mass: float = 0.18,
    target_base_radius: float = 0.18,
) -> CreatureSpec:
    spheres = load_spheres_from_glb(glb_path)
    topo = infer_radial_topology(spheres, arms=arms)

    raw_base_r = float(topo.base.radius)
    raw_seg_r = float(topo.small_radius)

    scale = (target_base_radius / raw_base_r) if raw_base_r > 1e-9 else 1.0

    base_r = float(np.clip(raw_base_r * scale, 0.05, 0.60))
    seg_r = float(np.clip(raw_seg_r * scale, 0.01, 0.20))

    print("OCTO_FROM_GLB:", "raw_base_r", raw_base_r, "raw_seg_r", raw_seg_r, "scale", scale)
    print("OCTO_FROM_GLB:", "base_r", base_r, "seg_r", seg_r)

    # Use a fixed link spacing for stability
    link_len = float(np.clip((base_r + seg_r) * 0.95, 0.03, 0.25))

    base = LinkSpec("base", "sphere", (base_r, 0.0), base_mass, (0.75, 0.75, 0.75, 1.0))

    links: List[LinkSpec] = []
    joints: List[JointSpec] = []

    # Determine segment count per arm from GLB bins (we keep your varying arm lengths)
    seg_counts: Dict[int, int] = {}
    for arm in topo.arms:
        seg_counts[arm.arm_index] = len(arm.segments)

    # Build links in arm order
    link_index_map: Dict[Tuple[int, int], int] = {}
    for a in range(arms):
        nseg = seg_counts.get(a, 0)
        for j in range(nseg):
            li = len(links)
            link_index_map[(a, j)] = li
            links.append(
                LinkSpec(
                    f"arm{a}_seg{j}",
                    "sphere",
                    (seg_r, 0.0),
                    seg_mass,
                    (0.85, 0.85, 0.85, 1.0),
                )
            )

    # Build joints with constant offsets:
    # First segment attaches to base at radius (base_r + seg_r) along arm direction.
    # Subsequent segments attach along the same arm direction with fixed link_len.
    for a in range(arms):
        nseg = seg_counts.get(a, 0)
        if nseg == 0:
            continue

        ang = (2.0 * math.pi * a) / float(arms)
        dir_xy = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=np.float64)

        for j in range(nseg):
            child = link_index_map[(a, j)]

            if j == 0:
                parent = -1
                rel = dir_xy * float(base_r + seg_r)  # stable base attachment
            else:
                parent = link_index_map[(a, j - 1)]
                rel = dir_xy * float(link_len)        # stable chain spacing

            parent_frame_pos = (float(rel[0]), float(rel[1]), float(rel[2]))

            # Revolute axis mostly Z; slight tilt per arm to break symmetry
            ax = 0.05 * math.cos(ang)
            ay = 0.05 * math.sin(ang)
            axis = (float(ax), float(ay), 1.0)

            joints.append(
                JointSpec(
                    name=f"joint_arm{a}_seg{j}",
                    parent=int(parent),
                    child=int(child),
                    joint_type="revolute",
                    axis=axis,
                    parent_frame_pos=parent_frame_pos,
                    child_frame_pos=(0.0, 0.0, 0.0),
                    lower=-0.8,
                    upper=0.8,
                    max_force=6.0,     # lower forces = more stable with many links
                    max_vel=10.0,
                    damping=0.7,
                )
            )

    controller_defaults = {
        "mode": 0.0,
        "amp": 0.55,
        "freq": 1.0,
        "phase_stride": 0.6,
        "bias": 0.0,
    }

    return CreatureSpec("octo_glb", base, links, joints, controller_defaults)