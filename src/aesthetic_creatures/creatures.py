from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pybullet as p


@dataclass
class LinkSpec:
    name: str
    shape: str  # "sphere" or "capsule"
    dims: Tuple[float, float]  # sphere: (radius, _), capsule: (radius, length)
    mass: float
    rgba: Tuple[float, float, float, float]


@dataclass
class JointSpec:
    name: str
    parent: int
    child: int
    joint_type: str  # "revolute"
    axis: Tuple[float, float, float]
    parent_frame_pos: Tuple[float, float, float]
    child_frame_pos: Tuple[float, float, float]
    lower: float
    upper: float
    max_force: float
    max_vel: float
    damping: float


@dataclass
class CreatureSpec:
    name: str
    base: LinkSpec
    links: List[LinkSpec]
    joints: List[JointSpec]
    controller_defaults: Dict[str, float]


def _col_shape(shape: str, dims: Tuple[float, float]) -> int:
    if shape == "sphere":
        radius = float(dims[0])
        return p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    if shape == "capsule":
        radius = float(dims[0])
        length = float(dims[1])
        return p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length)
    raise ValueError(f"Unknown shape: {shape}")


def _vis_shape(shape: str, dims: Tuple[float, float], rgba) -> int:
    if shape == "sphere":
        radius = float(dims[0])
        return p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    if shape == "capsule":
        radius = float(dims[0])
        length = float(dims[1])
        return p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=rgba)
    raise ValueError(f"Unknown shape: {shape}")


def build_creature(
    creature: CreatureSpec,
    base_pos=(0.0, 0.0, 0.55),
    base_orn=(0.0, 0.0, 0.0, 1.0),
    use_self_collision: bool = True,
) -> Tuple[int, Dict[str, int]]:
    base_col = _col_shape(creature.base.shape, creature.base.dims)
    base_vis = _vis_shape(creature.base.shape, creature.base.dims, creature.base.rgba)

    link_masses = []
    link_col = []
    link_vis = []
    link_pos = []
    link_orn = []
    link_inertial_pos = []
    link_inertial_orn = []
    link_parent_indices = []
    link_joint_types = []
    link_joint_axes = []

    name_to_link_index: Dict[str, int] = {"base": -1}

    # We construct links as children; each link spec becomes a Bullet "link".
    # Joint specs determine connectivity; Bullet needs per-link parent and joint params.
    for i, link in enumerate(creature.links):
        link_masses.append(float(link.mass))
        link_col.append(_col_shape(link.shape, link.dims))
        link_vis.append(_vis_shape(link.shape, link.dims, link.rgba))
        link_pos.append((0.0, 0.0, 0.0))  # overridden via joint frames (parent/child)
        link_orn.append((0.0, 0.0, 0.0, 1.0))
        link_inertial_pos.append((0.0, 0.0, 0.0))
        link_inertial_orn.append((0.0, 0.0, 0.0, 1.0))
        name_to_link_index[link.name] = i

    # Map joint specs to per-link params.
    # Bullet expects: each link i has a parent index and a joint definition to attach it to its parent.
    joints_by_child = {j.child: j for j in creature.joints}
    for i in range(len(creature.links)):
        j = joints_by_child.get(i, None)
        if j is None:
            raise ValueError(f"Link index {i} has no joint spec; every non-base link must have one.")

        parent = j.parent
        link_parent_indices.append(int(parent))
        link_joint_types.append(p.JOINT_REVOLUTE if j.joint_type == "revolute" else p.JOINT_FIXED)
        link_joint_axes.append(tuple(map(float, j.axis)))

        # The link frame is expressed relative to the parent frame; we pass the parentFramePos as link position.
        link_pos[i] = tuple(map(float, j.parent_frame_pos))
        # Child frame position is used by Bullet to place the joint relative to the child COM.
        link_inertial_pos[i] = (0.0, 0.0, 0.0)

    flags = 0
    if use_self_collision:
        flags |= p.URDF_USE_SELF_COLLISION

    body_id = p.createMultiBody(
        baseMass=float(creature.base.mass),
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=base_pos,
        baseOrientation=base_orn,
        linkMasses=link_masses,
        linkCollisionShapeIndices=link_col,
        linkVisualShapeIndices=link_vis,
        linkPositions=link_pos,
        linkOrientations=link_orn,
        linkInertialFramePositions=link_inertial_pos,
        linkInertialFrameOrientations=link_inertial_orn,
        linkParentIndices=link_parent_indices,
        linkJointTypes=link_joint_types,
        linkJointAxis=link_joint_axes,
        flags=flags,
    )

    # Configure joint limits and dynamics.
    for j_idx in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, j_idx, linearDamping=0.04, angularDamping=0.04)
        spec = joints_by_child[j_idx]
        p.setJointMotorControl2(body_id, j_idx, controlMode=p.VELOCITY_CONTROL, force=0)
        p.changeDynamics(body_id, j_idx, jointDamping=float(spec.damping))

    # Apply joint limits via user constraints? Bullet’s multibody limits are best handled in control logic.
    return body_id, name_to_link_index


def make_starfish(
    arms: int = 5,
    base_radius: float = 0.16,
    arm_radius: float = 0.05,
    arm_length: float = 0.26,
    arm_mass: float = 0.25,
    base_mass: float = 1.2,
) -> CreatureSpec:
    base = LinkSpec("base", "sphere", (base_radius, 0.0), base_mass, (0.85, 0.55, 0.25, 1.0))

    links: List[LinkSpec] = []
    joints: List[JointSpec] = []

    # Each arm is a capsule attached via a revolute joint around a tangential axis to allow “pulsing”.
    # We place arms radially around the base.
    for i in range(arms):
        links.append(LinkSpec(f"arm_{i}", "capsule", (arm_radius, arm_length), arm_mass, (0.9, 0.35, 0.2, 1.0)))

    for i in range(arms):
        angle = (2.0 * np.pi * i) / arms
        # Parent frame: point on base surface, slightly outward.
        px = float(np.cos(angle) * (base_radius * 0.95))
        py = float(np.sin(angle) * (base_radius * 0.95))
        pz = 0.0

        # Arm capsule is aligned with its local Z by default (Bullet capsule axis); for expressive motion we keep it simple.
        # Put the arm COM offset away from joint along radial direction.
        cx = float(np.cos(angle) * (arm_length * 0.5))
        cy = float(np.sin(angle) * (arm_length * 0.5))
        cz = 0.0

        # Axis: rotate about an axis perpendicular to radial vector in the XY plane, producing up/down “flap”.
        axis = (-np.sin(angle), np.cos(angle), 0.0)

        joints.append(
            JointSpec(
                name=f"joint_arm_{i}",
                parent=-1,  # base
                child=i,
                joint_type="revolute",
                axis=(float(axis[0]), float(axis[1]), float(axis[2])),
                parent_frame_pos=(px + cx, py + cy, pz),
                child_frame_pos=(0.0, 0.0, 0.0),
                lower=-0.9,
                upper=0.9,
                max_force=8.0,
                max_vel=12.0,
                damping=0.6,
            )
        )

    controller_defaults = {
        "mode": 0.0,          # 0=position-sine, 1=velocity-sine
        "amp": 0.55,
        "freq": 1.0,          # Hz
        "phase_stride": 0.7,  # radians per arm index
        "bias": 0.0,
    }
    return CreatureSpec("starfish", base, links, joints, controller_defaults)


def make_swimmer(
    segments: int = 8,
    radius: float = 0.06,
    seg_length: float = 0.14,
    mass_per: float = 0.22,
    base_mass: float = 0.6,
) -> CreatureSpec:
    base = LinkSpec("base", "sphere", (radius * 1.1, 0.0), base_mass, (0.25, 0.7, 0.95, 1.0))

    links: List[LinkSpec] = []
    joints: List[JointSpec] = []

    for i in range(segments):
        links.append(LinkSpec(f"seg_{i}", "capsule", (radius, seg_length), mass_per, (0.2, 0.6, 0.9, 1.0)))

    # Chain along +X; axis around Z to create lateral undulation.
    for i in range(segments):
        if i == 0:
            parent = -1
            parent_pos = (radius * 1.4, 0.0, 0.0)
        else:
            parent = i - 1
            parent_pos = (seg_length, 0.0, 0.0)

        joints.append(
            JointSpec(
                name=f"joint_seg_{i}",
                parent=parent,
                child=i,
                joint_type="revolute",
                axis=(0.0, 0.0, 1.0),
                parent_frame_pos=parent_pos,
                child_frame_pos=(0.0, 0.0, 0.0),
                lower=-0.9,
                upper=0.9,
                max_force=6.0,
                max_vel=14.0,
                damping=0.45,
            )
        )

    controller_defaults = {
        "mode": 0.0,
        "amp": 0.7,
        "freq": 1.3,
        "phase_stride": 0.55,  # travelling wave
        "bias": 0.0,
    }
    return CreatureSpec("swimmer", base, links, joints, controller_defaults)


def make_tumbler(
    limbs: int = 4,
    base_radius: float = 0.14,
    limb_radius: float = 0.05,
    limb_length: float = 0.22,
    limb_mass: float = 0.3,
    base_mass: float = 1.0,
) -> CreatureSpec:
    base = LinkSpec("base", "sphere", (base_radius, 0.0), base_mass, (0.75, 0.3, 0.8, 1.0))
    links: List[LinkSpec] = []
    joints: List[JointSpec] = []

    # Asymmetric “unstable rolling” by placing limbs in a non-uniform pattern.
    offsets = [
        (1.0, 0.2),
        (-0.8, 0.6),
        (0.2, -1.0),
        (-0.4, -0.6),
    ]
    for i in range(limbs):
        links.append(LinkSpec(f"limb_{i}", "capsule", (limb_radius, limb_length), limb_mass, (0.9, 0.35, 0.95, 1.0)))

    for i in range(limbs):
        ox, oy = offsets[i % len(offsets)]
        v = np.array([ox, oy], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        angle = np.arctan2(v[1], v[0])

        px = float(v[0] * (base_radius * 0.95))
        py = float(v[1] * (base_radius * 0.95))
        pz = 0.0

        cx = float(v[0] * (limb_length * 0.45))
        cy = float(v[1] * (limb_length * 0.45))
        cz = 0.0

        # Axis mixes Z and a tangential component so it “kicks” into tumbling.
        axis = (float(-np.sin(angle) * 0.6), float(np.cos(angle) * 0.6), 0.8)

        joints.append(
            JointSpec(
                name=f"joint_limb_{i}",
                parent=-1,
                child=i,
                joint_type="revolute",
                axis=axis,
                parent_frame_pos=(px + cx, py + cy, pz),
                child_frame_pos=(0.0, 0.0, 0.0),
                lower=-1.2,
                upper=1.2,
                max_force=10.0,
                max_vel=18.0,
                damping=0.5,
            )
        )

    controller_defaults = {
        "mode": 1.0,          # velocity sine tends to look more chaotic
        "amp": 8.0,           # used as velocity amplitude if mode=1
        "freq": 1.0,
        "phase_stride": 1.1,
        "bias": 0.0,
    }
    return CreatureSpec("tumbler", base, links, joints, controller_defaults)


def get_creature_spec(name: str) -> CreatureSpec:
    name = name.strip().lower()
    if name == "starfish":
        return make_starfish()
    if name == "swimmer":
        return make_swimmer()
    if name == "tumbler":
        return make_tumbler()
    raise ValueError(f"Unknown creature '{name}'. Use: starfish, swimmer, tumbler.")