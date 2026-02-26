from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh


@dataclass
class SphereItem:
    name: str
    center: np.ndarray  # (3,)
    radius: float


@dataclass
class ArmChain:
    arm_index: int
    segments: List[SphereItem]  # ordered from near base -> outward


@dataclass
class Topology:
    base: SphereItem
    arms: List[ArmChain]
    small_radius: float


def _approx_sphere_from_mesh(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    # Use bounding sphere approximation for radius and center.
    # For your low-poly spheres, this is accurate enough.
    center, radius = mesh.bounding_sphere.primitive.center, float(mesh.bounding_sphere.primitive.radius)
    return np.array(center, dtype=np.float64), radius


def load_spheres_from_glb(glb_path: str) -> List[SphereItem]:
    scene_or_mesh = trimesh.load(glb_path, force="scene")

    spheres: List[SphereItem] = []

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        c, r = _approx_sphere_from_mesh(scene_or_mesh)
        spheres.append(SphereItem("mesh_0", c, r))
        return spheres

    scene: trimesh.Scene = scene_or_mesh
    # scene.geometry is a dict of meshes (not transformed), scene.graph stores transforms per node
    for node_name in scene.graph.nodes_geometry:
        geom_name = scene.graph[node_name][1]
        geom = scene.geometry.get(geom_name, None)
        if geom is None:
            continue

        # get world transform for this node
        T = scene.graph.get(node_name)[0]
        mesh_world = geom.copy()
        mesh_world.apply_transform(T)

        c, r = _approx_sphere_from_mesh(mesh_world)

        # filter out junk/empties: skip extremely tiny meshes
        if r < 1e-6:
            continue

        spheres.append(SphereItem(str(node_name), c, r))

    return spheres


def infer_radial_topology(
    spheres: List[SphereItem],
    arms: int = 8,
    z_tolerance: float = 1e-2,
) -> Topology:
    # Choose base as largest radius
    spheres_sorted = sorted(spheres, key=lambda s: s.radius, reverse=True)
    base = spheres_sorted[0]
    others = spheres_sorted[1:]

    base_center = base.center

    # Keep mostly planar items (your model is in XY plane)
    planar: List[SphereItem] = []
    for s in others:
        if abs(float(s.center[2] - base_center[2])) <= float(z_tolerance):
            planar.append(s)
        else:
            planar.append(s)  # keep anyway; you can tighten later if needed

    # Estimate "small sphere radius" as median of remaining radii
    small_r = float(np.median([s.radius for s in planar])) if planar else float(base.radius * 0.25)

    # Compute angle and distance for each sphere relative to base center
    items = []
    for s in planar:
        v = s.center - base_center
        ang = math.atan2(float(v[1]), float(v[0]))  # [-pi, pi]
        dist = float(np.linalg.norm(v[:2]))
        items.append((s, ang, dist))

    # Quantize into arm bins
    # Map angle to [0, 2pi)
    def ang01(a: float) -> float:
        x = a % (2.0 * math.pi)
        return x

    bin_size = (2.0 * math.pi) / float(arms)

    bins: Dict[int, List[Tuple[SphereItem, float]]] = {i: [] for i in range(arms)}
    for s, a, d in items:
        a2 = ang01(a)
        idx = int(round(a2 / bin_size)) % arms
        bins[idx].append((s, d))

    # Sort each bin by radial distance (near base -> outward)
    arm_chains: List[ArmChain] = []
    for idx in range(arms):
        segs = sorted(bins[idx], key=lambda t: t[1])
        arm_chains.append(ArmChain(idx, [t[0] for t in segs]))

    return Topology(base=base, arms=arm_chains, small_radius=small_r)