"""Sim2Real-calibrated analytic capsule/sphere collision profile for Tiangong3.

Ported verbatim from ``MLAB4_Fast`` (``mjlab.utils.tiangong_capsule_collision``,
profile ``tiangong3_capsule_v1``) so Retarget_X Stage 1/2/4 (kinematic scoring,
contact-aware ground projection, audit), Stage 3 (DTMR MuJoCo Warp MPC), and
downstream ``MLAB4_Fast`` RL training share the exact same 39-primitive
zero-mesh collision geometry:

  - Exact CAD 1.35-deg heel-to-toe rocker sole (4 tapered r=10 mm capsules per
    foot, ``condim=3``) matching ``sdk_a_scene.xml`` to 0.27-0.32 mm MAE
    (replacing the legacy flat 3-capsule sole that had 3.7-6.3 mm ball/toe
    error and up to 15.6 mm bias on plantarflexed BONES-SEED clips).
  - Calibrated whole-body recovery & self-collision envelope: 3 pelvis capsules,
    2 lateral torso capsules + head crown + head sphere, hip pitch/roll
    trochanters, dual-segment thigh, anterior kneecap + lower shin, deltoid/
    upper-arm/elbow/forearm/wrist, and exact TCP tip hand capsules
    (``z_min = -0.1046 m``, ``condim=3``).
  - Zero ``mjGEOM_MESH`` collision geoms (``nmesh == 0``), keeping MuJoCo Warp
    100% inside closed-form O(1) ``_primitive_narrowphase`` without launching
    ``plane_convex`` or GJK/EPA/MultiCCD kernels.
  - Preserves 100% of body masses, inertias, CoMs, joint limits, armatures,
    dampings, and actuator torque ranges, plus all 20 compound-joint
    ``<exclude>`` pairs (3 base + 17 2-hop gimbal exclusions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TIANGONG3_CAPSULE_PROFILE_VERSION = "tiangong3_capsule_v1"

DEFAULT_FRICTION: tuple[float, float, float] = (1.0, 0.01, 0.01)

TIANGONG3_CAPSULE_FOOT_GEOM_NAMES: tuple[str, ...] = (
    "foot_l_1_collision",
    "foot_l_2_collision",
    "foot_l_3_collision",
    "foot_l_4_collision",
    "foot_r_1_collision",
    "foot_r_2_collision",
    "foot_r_3_collision",
    "foot_r_4_collision",
)

TIANGONG3_CAPSULE_HAND_GEOM_NAMES: tuple[str, ...] = (
    "left_tcp_link_collision",
    "right_tcp_link_collision",
)

# 3 base exclusions from sdk_a_scene.xml
TIANGONG3_BASE_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("knee_pitch_l_link", "ankle_roll_l_link"),
    ("knee_pitch_r_link", "ankle_roll_r_link"),
    ("pelvis", "waist_roll_link"),
)

# 17 2-hop compound-joint self-collision exclusions from MLAB4_Fast (eliminating
# false pelvis<->hip_roll/yaw & gimbal overlaps while keeping all cross-limb
# leg-leg, arm-leg, arm-torso, arm-pelvis, and hand-hand self-collisions active).
TIANGONG3_CAPSULE_EXTRA_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("pelvis", "hip_roll_l_link"),
    ("pelvis", "hip_roll_r_link"),
    ("pelvis", "hip_yaw_l_link"),
    ("pelvis", "hip_yaw_r_link"),
    ("pelvis", "waist_pitch_link"),
    ("hip_pitch_l_link", "hip_yaw_l_link"),
    ("hip_pitch_r_link", "hip_yaw_r_link"),
    ("waist_pitch_link", "shoulder_roll_l_link"),
    ("waist_pitch_link", "shoulder_roll_r_link"),
    ("shoulder_roll_l_link", "elbow_pitch_l_link"),
    ("shoulder_roll_r_link", "elbow_pitch_r_link"),
    ("elbow_pitch_l_link", "wrist_pitch_l_link"),
    ("elbow_pitch_r_link", "wrist_pitch_r_link"),
    ("elbow_yaw_l_link", "left_tcp_link"),
    ("elbow_yaw_r_link", "right_tcp_link"),
    ("wrist_pitch_l_link", "left_tcp_link"),
    ("wrist_pitch_r_link", "right_tcp_link"),
)

TIANGONG3_CAPSULE_ALL_EXCLUDES: tuple[tuple[str, str], ...] = (
    TIANGONG3_BASE_EXCLUDES + TIANGONG3_CAPSULE_EXTRA_EXCLUDES
)

# Removed internal / sensor / intermediate gimbal bodies (14 bodies -> 0 collision geoms):
TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES: tuple[str, ...] = (
    "imu_waist_link",
    "imu_head_link",
    "camera_body_front_link",
    "camera_head_link",
    "radar_head_link",
    "head_pitch_link",
    "waist_yaw_link",
    "waist_roll_link",
    "ankle_pitch_l_link",
    "ankle_pitch_r_link",
    "shoulder_pitch_l_link",
    "shoulder_pitch_r_link",
    "wrist_roll_l_link",
    "wrist_roll_r_link",
)


@dataclass(frozen=True)
class CollisionPrimitiveSpec:
    """Single analytic capsule or sphere collision geom in body-local coordinates."""

    body: str
    name: str
    geom_type: str  # "capsule" | "sphere"
    radius: float
    p1: tuple[float, float, float]
    p2: tuple[float, float, float] | None = None
    condim: int = 1
    css_class: str = "collision"


def _fmt_floats(vals: tuple[float, ...]) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _build_capsule_specs() -> tuple[CollisionPrimitiveSpec, ...]:
    specs: list[CollisionPrimitiveSpec] = [
        # 1. Pelvis: CAD bbox [-0.1080..+0.1060, -0.1251..+0.1250, -0.1824..+0.0575]
        CollisionPrimitiveSpec("pelvis", "pelvis_l_collision", "capsule", 0.080, (-0.028, 0.045, -0.036), (0.026, 0.045, -0.036)),
        CollisionPrimitiveSpec("pelvis", "pelvis_r_collision", "capsule", 0.080, (-0.028, -0.045, -0.036), (0.026, -0.045, -0.036)),
        CollisionPrimitiveSpec("pelvis", "pelvis_lower_collision", "capsule", 0.074, (-0.002, -0.050, -0.108), (-0.002, 0.050, -0.108)),
        # 2. Torso (waist_pitch_link): CAD bbox [-0.1097..+0.1145, -0.1636..+0.1641, -0.0340..+0.5278]
        CollisionPrimitiveSpec("waist_pitch_link", "torso_lower_collision", "capsule", 0.108, (0.002, -0.055, 0.105), (0.002, 0.055, 0.105)),
        CollisionPrimitiveSpec("waist_pitch_link", "torso_upper_collision", "capsule", 0.106, (-0.001, -0.055, 0.220), (-0.001, 0.055, 0.220)),
        CollisionPrimitiveSpec("waist_pitch_link", "head_crown_collision", "sphere", 0.064, (0.0002, 0.0, 0.464)),
        # 3. Head (head_yaw_link + fixed head_pitch_link + camera_head_link)
        CollisionPrimitiveSpec("head_yaw_link", "head_yaw_link_collision", "sphere", 0.085, (0.001, 0.0, 0.008)),
    ]

    for side, sy in (("l", 1.0), ("r", -1.0)):
        # 4. Hip (buttock / lateral trochanter envelope)
        specs.append(CollisionPrimitiveSpec(
            f"hip_pitch_{side}_link", f"hip_pitch_{side}_link_collision", "capsule", 0.049,
            (-0.001, 0.052 * sy, -0.042), (0.002, 0.090 * sy, -0.024),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"hip_roll_{side}_link", f"hip_roll_{side}_link_collision", "capsule", 0.058,
            (0.001, 0.0, 0.002), (0.001, 0.0, -0.030),
        ))
        # 5. Thigh (hip_yaw_link)
        specs.append(CollisionPrimitiveSpec(
            f"hip_yaw_{side}_link", f"hip_yaw_{side}_link_collision", "capsule", 0.066,
            (0.001, -0.006 * sy, -0.060), (0.001, -0.006 * sy, -0.205),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"hip_yaw_{side}_link", f"thigh_distal_{side}_collision", "capsule", 0.051,
            (-0.002, 0.0, -0.220), (-0.006, 0.0, -0.292),
        ))
        # 6. Shin & anterior kneecap (knee_pitch_link)
        specs.append(CollisionPrimitiveSpec(
            f"knee_pitch_{side}_link", f"knee_cap_{side}_collision", "capsule", 0.054,
            (0.017, 0.0, -0.015), (0.016, 0.0, -0.175),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"knee_pitch_{side}_link", f"shin_lower_{side}_collision", "capsule", 0.038,
            (0.011, 0.0, -0.195), (0.007, 0.0, -0.355),
        ))
        # 7. Foot sole (ankle_roll_link): 4 tapered capsules matching exact CAD 1.35-deg rocker slope
        foot_defs = (
            (f"foot_{side}_1_collision", (-0.084, -0.0335, -0.04677), (0.085, -0.0455, -0.04305)),
            (f"foot_{side}_2_collision", (-0.084, 0.0335, -0.04677), (0.085, 0.0455, -0.04305)),
            (f"foot_{side}_3_collision", (0.085, -0.0455, -0.04305), (0.1688, -0.0180, -0.04050)),
            (f"foot_{side}_4_collision", (0.085, 0.0455, -0.04305), (0.1688, 0.0180, -0.04050)),
        )
        for fg_name, p1, p2 in foot_defs:
            specs.append(CollisionPrimitiveSpec(
                f"ankle_roll_{side}_link", fg_name, "capsule", 0.010, p1, p2,
                condim=3, css_class="foot_capsule",
            ))
        # 8. Deltoid / Upper arm / Elbow / Forearm / Wrist / Hand TCP
        specs.append(CollisionPrimitiveSpec(
            f"shoulder_roll_{side}_link", f"shoulder_roll_{side}_link_collision", "capsule", 0.048,
            (-0.001, 0.0, -0.009), (-0.001, 0.0, -0.056),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"shoulder_yaw_{side}_link", f"shoulder_yaw_{side}_link_collision", "capsule", 0.041,
            (0.001, 0.0, -0.020), (0.012, -0.002 * sy, -0.154),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"elbow_pitch_{side}_link", f"elbow_pitch_{side}_link_collision", "capsule", 0.045,
            (-0.009, -0.001 * sy, 0.0), (-0.018, 0.0, -0.075),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"elbow_yaw_{side}_link", f"elbow_yaw_{side}_link_collision", "capsule", 0.036,
            (0.0, 0.003 * sy, -0.015), (0.0, 0.0, -0.092),
        ))
        specs.append(CollisionPrimitiveSpec(
            f"wrist_pitch_{side}_link", f"wrist_pitch_{side}_link_collision", "capsule", 0.034,
            (0.0, -0.002 * sy, 0.0), (-0.001, 0.0, -0.060),
        ))
        tcp_body = "left_tcp_link" if side == "l" else "right_tcp_link"
        specs.append(CollisionPrimitiveSpec(
            tcp_body, f"{tcp_body}_collision", "capsule", 0.047,
            (0.005, -0.006 * sy, -0.018), (0.006, -0.008 * sy, -0.0576),
            condim=3, css_class="hand_capsule",
        ))

    return tuple(specs)


TIANGONG3_CAPSULE_SPECS: tuple[CollisionPrimitiveSpec, ...] = _build_capsule_specs()

TIANGONG3_CAPSULE_SPECS_BY_BODY: dict[str, tuple[CollisionPrimitiveSpec, ...]] = {}
for _spec in TIANGONG3_CAPSULE_SPECS:
    TIANGONG3_CAPSULE_SPECS_BY_BODY.setdefault(_spec.body, ())
    TIANGONG3_CAPSULE_SPECS_BY_BODY[_spec.body] = (
        *TIANGONG3_CAPSULE_SPECS_BY_BODY[_spec.body],
        _spec,
    )

TIANGONG3_GEOM_TO_BODY: dict[str, str] = {s.name: s.body for s in TIANGONG3_CAPSULE_SPECS}


def _build_explicit_contact_pairs() -> tuple[tuple[str, str, str], ...]:
    """Build explicit <contact><pair> list for MJX / MuJoCo Warp (contype=0 mode).

    Includes:
      - All 39 analytic collision geoms vs ``floor`` (``condim=3``; high-friction
        ``solref="0.005 1" friction="1 1"`` on the 8 foot sole and 2 hand TCP capsules).
      - Curated cross-limb self-collision pairs (``condim=1``) covering foot-foot,
        foot-shin, shin-shin, shin-thigh, thigh-thigh, arm-torso, arm-pelvis,
        arm-leg, arm-head, and arm-arm interactions while strictly respecting
        ``TIANGONG3_CAPSULE_ALL_EXCLUDES``.
    """
    pairs: list[tuple[str, str, str]] = []
    foot_set = set(TIANGONG3_CAPSULE_FOOT_GEOM_NAMES)
    hand_set = set(TIANGONG3_CAPSULE_HAND_GEOM_NAMES)

    # 1. All 8 feet -> floor first (canonical ordering for DTMR / Retarget_X)
    for fg in TIANGONG3_CAPSULE_FOOT_GEOM_NAMES:
        pairs.append((fg, "floor", 'condim="3" solref="0.005 1" friction="1 1"'))
    # 2. Both hand TCPs -> floor (high friction for hand-support / ground recovery)
    for hg in TIANGONG3_CAPSULE_HAND_GEOM_NAMES:
        pairs.append((hg, "floor", 'condim="3" solref="0.005 1" friction="1 1"'))
    # 3. Remaining 29 body capsules/spheres -> floor (full 3D friction)
    for spec in TIANGONG3_CAPSULE_SPECS:
        if spec.name in foot_set or spec.name in hand_set:
            continue
        pairs.append((spec.name, "floor", 'condim="3"'))

    # 4. Cross-limb self-collisions (condim=1, frictionless normal repulsion)
    self_pairs: list[tuple[str, str]] = [
        # Foot <-> opposite foot (medial-to-medial, toe-to-toe, heel-ball cross)
        ("foot_l_1_collision", "foot_r_2_collision"),
        ("foot_l_1_collision", "foot_r_4_collision"),
        ("foot_l_3_collision", "foot_r_2_collision"),
        ("foot_l_3_collision", "foot_r_4_collision"),
        ("foot_l_2_collision", "foot_r_1_collision"),
        ("foot_l_4_collision", "foot_r_3_collision"),
        # Foot <-> opposite lower shin
        ("foot_l_1_collision", "shin_lower_r_collision"),
        ("foot_l_3_collision", "shin_lower_r_collision"),
        ("foot_r_2_collision", "shin_lower_l_collision"),
        ("foot_r_4_collision", "shin_lower_l_collision"),
        # Shin / kneecap <-> opposite shin / kneecap
        ("shin_lower_l_collision", "shin_lower_r_collision"),
        ("knee_cap_l_collision", "knee_cap_r_collision"),
        ("knee_cap_l_collision", "shin_lower_r_collision"),
        ("knee_cap_r_collision", "shin_lower_l_collision"),
        # Shin / kneecap <-> opposite thigh
        ("shin_lower_l_collision", "thigh_distal_r_collision"),
        ("shin_lower_r_collision", "thigh_distal_l_collision"),
        ("knee_cap_l_collision", "thigh_distal_r_collision"),
        ("knee_cap_r_collision", "thigh_distal_l_collision"),
        ("knee_cap_l_collision", "hip_yaw_r_link_collision"),
        ("knee_cap_r_collision", "hip_yaw_l_link_collision"),
        # Thigh <-> opposite thigh
        ("hip_yaw_l_link_collision", "hip_yaw_r_link_collision"),
        ("thigh_distal_l_collision", "thigh_distal_r_collision"),
        ("hip_yaw_l_link_collision", "thigh_distal_r_collision"),
        ("hip_yaw_r_link_collision", "thigh_distal_l_collision"),
        # Torso <-> arms (note: shoulder_roll is excluded via TIANGONG3_CAPSULE_EXTRA_EXCLUDES)
        ("torso_upper_collision", "shoulder_yaw_l_link_collision"),
        ("torso_upper_collision", "shoulder_yaw_r_link_collision"),
        ("torso_upper_collision", "elbow_pitch_l_link_collision"),
        ("torso_upper_collision", "elbow_pitch_r_link_collision"),
        ("torso_upper_collision", "elbow_yaw_l_link_collision"),
        ("torso_upper_collision", "elbow_yaw_r_link_collision"),
        ("torso_upper_collision", "wrist_pitch_l_link_collision"),
        ("torso_upper_collision", "wrist_pitch_r_link_collision"),
        ("torso_upper_collision", "left_tcp_link_collision"),
        ("torso_upper_collision", "right_tcp_link_collision"),
        ("torso_lower_collision", "elbow_pitch_l_link_collision"),
        ("torso_lower_collision", "elbow_pitch_r_link_collision"),
        ("torso_lower_collision", "elbow_yaw_l_link_collision"),
        ("torso_lower_collision", "elbow_yaw_r_link_collision"),
        ("torso_lower_collision", "wrist_pitch_l_link_collision"),
        ("torso_lower_collision", "wrist_pitch_r_link_collision"),
        ("torso_lower_collision", "left_tcp_link_collision"),
        ("torso_lower_collision", "right_tcp_link_collision"),
        # Pelvis <-> distal arms
        ("pelvis_l_collision", "elbow_yaw_l_link_collision"),
        ("pelvis_r_collision", "elbow_yaw_r_link_collision"),
        ("pelvis_l_collision", "wrist_pitch_l_link_collision"),
        ("pelvis_r_collision", "wrist_pitch_r_link_collision"),
        ("pelvis_l_collision", "left_tcp_link_collision"),
        ("pelvis_r_collision", "right_tcp_link_collision"),
        # Head <-> hands
        ("head_yaw_link_collision", "left_tcp_link_collision"),
        ("head_yaw_link_collision", "right_tcp_link_collision"),
        # Arms <-> thighs / hips
        ("left_tcp_link_collision", "hip_yaw_l_link_collision"),
        ("right_tcp_link_collision", "hip_yaw_r_link_collision"),
        ("left_tcp_link_collision", "thigh_distal_l_collision"),
        ("right_tcp_link_collision", "thigh_distal_r_collision"),
        ("wrist_pitch_l_link_collision", "hip_yaw_l_link_collision"),
        ("wrist_pitch_r_link_collision", "hip_yaw_r_link_collision"),
        # Arm <-> opposite arm
        ("left_tcp_link_collision", "right_tcp_link_collision"),
        ("left_tcp_link_collision", "wrist_pitch_r_link_collision"),
        ("right_tcp_link_collision", "wrist_pitch_l_link_collision"),
        ("left_tcp_link_collision", "elbow_yaw_r_link_collision"),
        ("right_tcp_link_collision", "elbow_yaw_l_link_collision"),
        ("wrist_pitch_l_link_collision", "wrist_pitch_r_link_collision"),
        ("elbow_yaw_l_link_collision", "elbow_yaw_r_link_collision"),
    ]

    excluded_body_pairs = {tuple(sorted(pair)) for pair in TIANGONG3_CAPSULE_ALL_EXCLUDES}
    for g1, g2 in self_pairs:
        b1, b2 = TIANGONG3_GEOM_TO_BODY[g1], TIANGONG3_GEOM_TO_BODY[g2]
        if b1 == b2 or tuple(sorted((b1, b2))) in excluded_body_pairs:
            raise ValueError(f"Self-collision pair ({g1}, {g2}) violates body exclusion ({b1}, {b2})")
        pairs.append((g1, g2, 'condim="1"'))

    return tuple(pairs)


TIANGONG3_EXPLICIT_CONTACT_PAIRS: tuple[tuple[str, str, str], ...] = _build_explicit_contact_pairs()


def apply_tiangong3_capsule_collisions(
    spec: mujoco.MjSpec,
    *,
    friction: tuple[float, float, float] = DEFAULT_FRICTION,
) -> dict[str, Any]:
    """Replace mesh/legacy collisions on a Tiangong3 ``MjSpec`` with ``tiangong3_capsule_v1``.

    API-compatible with ``MLAB4_Fast``'s ``mjlab.utils.tiangong_capsule_collision``.
    """
    target_names: set[str] = {s.name for s in TIANGONG3_CAPSULE_SPECS}
    paired: set[str] = {
        name for pair in spec.pairs for name in (pair.geomname1, pair.geomname2)
    }
    for geom in list(spec.geoms):
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        # Replace existing tiangong3_capsule_v1 geoms in-place (idempotent even when referenced by <pair>).
        if geom.name in target_names:
            spec.delete(geom)
            continue
        if geom.name in paired:
            continue
        is_collision_geom = (
            geom.density == 0
            or bool(geom.contype)
            or bool(geom.conaffinity)
            or geom.group == 3
            or (geom.name or "").endswith("_collision")
        )
        if is_collision_geom and not (np.isfinite(geom.mass) and geom.mass > 0):
            spec.delete(geom)

    referenced_meshes: set[str] = {
        geom.meshname
        for geom in spec.geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.meshname
    }
    for mesh in list(spec.meshes):
        if mesh.name not in referenced_meshes:
            spec.delete(mesh)

    for pspec in TIANGONG3_CAPSULE_SPECS:
        body = spec.body(pspec.body)
        if pspec.geom_type == "capsule":
            assert pspec.p2 is not None
            body.add_geom(
                name=pspec.name,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[pspec.radius, 0.0, 0.0],
                fromto=[*pspec.p1, *pspec.p2],
                group=1,
                condim=pspec.condim,
                friction=list(friction),
                rgba=[0.752941, 0.752941, 0.752941, 1.0],
            )
        elif pspec.geom_type == "sphere":
            body.add_geom(
                name=pspec.name,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[pspec.radius, 0.0, 0.0],
                pos=list(pspec.p1),
                group=1,
                condim=pspec.condim,
                friction=list(friction),
                rgba=[0.752941, 0.752941, 0.752941, 1.0],
            )
        else:
            raise ValueError(f"Unsupported geom_type: {pspec.geom_type}")

    existing_excludes: set[tuple[str, ...]] = {
        tuple(sorted((ex.bodyname1, ex.bodyname2))) for ex in spec.excludes
    }
    added_excludes: list[list[str]] = []
    for b1, b2 in TIANGONG3_CAPSULE_ALL_EXCLUDES:
        key = tuple(sorted((b1, b2)))
        if key not in existing_excludes:
            spec.add_exclude(bodyname1=b1, bodyname2=b2)
            existing_excludes.add(key)
            added_excludes.append([b1, b2])

    return {
        "profile": "capsule",
        "version": TIANGONG3_CAPSULE_PROFILE_VERSION,
        "collision_preserved": False,
        "mesh_collision_geoms": 0,
        "analytic_collision_geoms": len(TIANGONG3_CAPSULE_SPECS),
        "foot_collision_geoms": list(TIANGONG3_CAPSULE_FOOT_GEOM_NAMES),
        "removed_internal_bodies": list(TIANGONG3_CAPSULE_REMOVED_INTERNAL_BODIES),
        "added_compound_joint_excludes": added_excludes,
    }


def _primitive_to_xml_attrib(pspec: CollisionPrimitiveSpec) -> dict[str, str]:
    attrib: dict[str, str] = {"name": pspec.name, "class": pspec.css_class}
    if pspec.css_class == "foot_capsule":
        assert pspec.p2 is not None
        attrib["fromto"] = _fmt_floats((*pspec.p1, *pspec.p2))
        return attrib
    if pspec.css_class == "hand_capsule":
        assert pspec.p2 is not None
        attrib["size"] = f"{pspec.radius:.6g}"
        attrib["fromto"] = _fmt_floats((*pspec.p1, *pspec.p2))
        return attrib

    attrib["type"] = pspec.geom_type
    attrib["size"] = f"{pspec.radius:.6g}"
    if pspec.geom_type == "capsule":
        assert pspec.p2 is not None
        attrib["fromto"] = _fmt_floats((*pspec.p1, *pspec.p2))
    else:
        attrib["pos"] = _fmt_floats(pspec.p1)
    return attrib


def build_tiangong3_mjx_robot_tree(
    source_xml: Path,
    *,
    headless: bool = True,
    meshdir: str = "assets",
) -> ET.ElementTree:
    """Build Tiangong3 MJX/MuJoCo-Warp torque robot XML using ``tiangong3_capsule_v1``.

    Accepts either an existing ``tiangong3_mjx_torque*.xml`` or the original
    ``sdk_a_scene.xml`` as ``source_xml``. Preserves 100% of body kinematics,
    explicit ``<inertial>`` blocks, joints, and actuators while replacing all
    collision geoms with the 39 CAD-calibrated ``MLAB4_Fast`` analytic capsules/
    spheres and writing all 20 compound-joint ``<exclude>`` entries.
    """
    tree = ET.parse(source_xml)
    root = tree.getroot()
    root.attrib["model"] = (
        "tiangong3_29 mjx torque headless" if headless else "tiangong3_29 mjx torque"
    )

    for tag in ("default", "option", "contact"):
        for elem in list(root.findall(tag)):
            root.remove(elem)

    compiler = root.find("compiler")
    if headless:
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
        for asset in list(root.findall("asset")):
            root.remove(asset)
    else:
        if compiler is not None:
            compiler.attrib["meshdir"] = meshdir

    default_elem = ET.fromstring(
        """
    <default>
      <default class="collision">
        <geom group="3" rgba=".2 .6 .2 .3" type="capsule" condim="1" contype="0" conaffinity="0" friction="1 0.01 0.01" solimp="0.99 0.999 0.001 0.5 2" solref="0.005 1" />
        <default class="foot_capsule">
          <geom size="0.01" condim="3" />
        </default>
        <default class="hand_capsule">
          <geom condim="3" />
        </default>
      </default>
    </default>
    """
    )
    option_elem = ET.fromstring(
        """
    <option timestep="0.005" tolerance="1e-9" impratio="1" integrator="implicitfast" solver="Newton">
      <flag eulerdamp="disable" />
    </option>
    """
    )
    insert_idx = list(root).index(compiler) + 1 if compiler is not None else 0
    root.insert(insert_idx, default_elem)
    root.insert(insert_idx + 1, option_elem)

    wb = root.find("worldbody")
    assert wb is not None, f"Missing <worldbody> in {source_xml}"
    for g in list(wb.findall("geom")):
        if g.attrib.get("name") == "floor":
            wb.remove(g)

    pelvis = wb.find("body[@name='pelvis']")
    assert pelvis is not None, f"Missing pelvis body in {source_xml}"
    fj = pelvis.find("freejoint")
    if fj is not None:
        pelvis.remove(fj)
    existing_free = [
        j for j in pelvis.findall("joint") if j.attrib.get("type") == "free"
    ]
    if not existing_free:
        pelvis.insert(
            1,
            ET.Element(
                "joint",
                {
                    "name": "floating_base_joint",
                    "type": "free",
                    "limited": "false",
                    "actuatorfrclimited": "false",
                },
            ),
        )

    present_bodies: set[str] = set()
    for body in wb.iter("body"):
        bname = body.attrib.get("name", "")
        present_bodies.add(bname)
        for g in list(body.findall("geom")):
            gname = g.attrib.get("name", "")
            is_visual = g.attrib.get("group") == "2" or gname.endswith("_visual")
            if headless or not is_visual:
                body.remove(g)
        for pspec in TIANGONG3_CAPSULE_SPECS_BY_BODY.get(bname, ()):
            body.append(ET.Element("geom", _primitive_to_xml_attrib(pspec)))

    missing_bodies = sorted(set(TIANGONG3_CAPSULE_SPECS_BY_BODY) - present_bodies)
    if missing_bodies:
        raise ValueError(f"Expected Tiangong3 bodies missing from {source_xml}: {missing_bodies}")

    contact_elem = ET.Element("contact")
    for b1, b2 in TIANGONG3_CAPSULE_ALL_EXCLUDES:
        ET.SubElement(contact_elem, "exclude", {"body1": b1, "body2": b2})
    actuator_elem = root.find("actuator")
    if actuator_elem is not None:
        root.insert(list(root).index(actuator_elem), contact_elem)
    else:
        root.append(contact_elem)

    ET.indent(tree, space="  ")
    return tree


def build_tiangong3_mjx_scene_xml(
    include_file: str = "tiangong3_mjx_torque_headless.xml",
) -> str:
    """Build Tiangong3 flat-terrain explicit-pair scene XML for ``tiangong3_capsule_v1``."""
    pairs_xml = "\n".join(
        f'    <pair name="{g1}__{g2}" geom1="{g1}" geom2="{g2}" {extra}/>'
        for g1, g2, extra in TIANGONG3_EXPLICIT_CONTACT_PAIRS
    )
    home_qpos = (
        "0 0 0.99914  1 0 0 0  "
        "0 0 0 0 0 0  "
        "0 0 0 0 0 0  "
        "0 0 0  "
        "0 0.15 0 -0.2 0 0 0  "
        "0 -0.15 0 -0.2 0 0 0"
    )
    knees_qpos = (
        "0 0 0.97694  1 0 0 0  "
        "-0.25 0 0 0.5 -0.25 0  "
        "-0.25 0 0 0.5 -0.25 0  "
        "0 0 0  "
        "0 0.3 0 -0.3 0 0 0  "
        "0 -0.3 0 -0.3 0 0 0"
    )
    zero_ctrl = " ".join(["0"] * 29)
    return f"""<mujoco model="tiangong3 flat terrain scene torque ({TIANGONG3_CAPSULE_PROFILE_VERSION})">
  <include file="{include_file}"/>

  <default>
    <geom solimp="0.99 0.999 0.001 0.5 2" solref="0.005 1"/>
    <pair solref="0.005 1" solimp="0.99 0.999 0.001 0.5 2"/>
  </default>

  <statistic center="0 0 0.8" extent="1.4" meansize="0.02"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba force="1 0 0 1" haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
    <map force="0.01"/>
    <scale forcewidth="0.3" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>
  </worldbody>

  <contact>
{pairs_xml}
  </contact>

  <keyframe>
    <key name="home" qpos="{home_qpos}" ctrl="{zero_ctrl}"/>
    <key name="knees_bent" qpos="{knees_qpos}" ctrl="{zero_ctrl}"/>
  </keyframe>
</mujoco>
"""


def audit_tiangong3_capsule_scene(scene_xml: Path) -> dict[str, Any]:
    """Compile and audit a Tiangong3 ``tiangong3_capsule_v1`` scene XML."""
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    floor_id = model.geom("floor").id

    robot_geoms = [
        model.geom(i).name
        for i in range(model.ngeom)
        if i != floor_id and model.geom_group[i] != 2
    ]
    mesh_collision_geoms = [
        model.geom(i).name
        for i in range(model.ngeom)
        if i != floor_id
        and model.geom_group[i] != 2
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    ]

    keyframe_audits: dict[str, dict[str, float | int]] = {}
    for kf in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, kf)
        mujoco.mj_forward(model, data)
        foot_dist = min(
            mujoco.mj_geomDistance(model, data, floor_id, model.geom(g).id, 3.0, None)
            for g in TIANGONG3_CAPSULE_FOOT_GEOM_NAMES
        )
        n_floor_con = 0
        n_self_con = 0
        for c in data.contact[: data.ncon]:
            if floor_id in (c.geom1, c.geom2):
                n_floor_con += 1
            else:
                n_self_con += 1
        keyframe_audits[model.key(kf).name] = {
            "foot_min_dist_mm": round(float(foot_dist) * 1000.0, 4),
            "n_floor_contacts": n_floor_con,
            "n_self_contacts": n_self_con,
        }

    return {
        "profile_version": TIANGONG3_CAPSULE_PROFILE_VERSION,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "ngeom_total": int(model.ngeom),
        "n_collision_geoms": len(robot_geoms),
        "n_mesh_collision_geoms": len(mesh_collision_geoms),
        "npair": int(model.npair),
        "nexclude": int(model.nexclude),
        "keyframes": keyframe_audits,
    }
