"""Retarget_X: Unified Kinodynamic & Dense Temporal Motion Retargeting for Humanoid Robots."""
from .config import RobotSpec, load_robot_spec
from .stage2_conditioning import ConditioningReport, MotionDynamismProfile, condition_reference
from .stage4_audit_export import AuditReport, audit_trajectory
from .tiangong3_capsule_collision import (
    TIANGONG3_CAPSULE_FOOT_GEOM_NAMES,
    TIANGONG3_CAPSULE_PROFILE_VERSION,
    apply_tiangong3_capsule_collisions,
    audit_tiangong3_capsule_scene,
)

__version__ = "0.1.0"
__all__ = [
    "RobotSpec",
    "load_robot_spec",
    "ConditioningReport",
    "MotionDynamismProfile",
    "condition_reference",
    "AuditReport",
    "audit_trajectory",
    "TIANGONG3_CAPSULE_PROFILE_VERSION",
    "TIANGONG3_CAPSULE_FOOT_GEOM_NAMES",
    "apply_tiangong3_capsule_collisions",
    "audit_tiangong3_capsule_scene",
]
