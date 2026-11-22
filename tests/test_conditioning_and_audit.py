"""Unit tests for Retarget_X Stage 2 conditioning and Stage 4 auditing."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from retarget_x.config import load_robot_spec
from retarget_x.stage2_conditioning import condition_reference
from retarget_x.stage4_audit_export import audit_trajectory


def test_tiangong3_spec_loads():
    spec = load_robot_spec("tiangong3")
    assert spec.name == "tiangong3"
    assert spec.dof == 29
    assert len(spec.joint_names) == 29
    assert len(spec.kp) == 29
    assert len(spec.kd) == 29
    assert spec.headless_scene_path.exists()


def test_stage2_eliminates_penetration_and_smooths_jitter():
    spec = load_robot_spec("tiangong3")
    rng = np.random.default_rng(42)
    T = 80
    q = np.zeros((T, 7 + spec.dof), dtype=np.float64)
    # Deliberately penetrate floor by 25mm and inject high-frequency joint jitter
    q[:, 2] = spec.standing_root_height_m - 0.025
    q[:, 3] = 1.0
    t = np.linspace(0, 2 * np.pi, T)
    for j in range(spec.dof):
        q[:, 7 + j] = 0.2 * np.sin(t + 0.1 * j) + rng.normal(0.0, 0.03, size=T)

    q_cond, qvel_cond, rep = condition_reference(q, spec, fps=50.0)
    assert q_cond.shape == q.shape
    assert qvel_cond.shape == (T, 6 + spec.dof)
    assert rep.raw_floor_pen_p95_mm > 20.0
    assert rep.cond_floor_pen_p95_mm < 2.0
    assert rep.cond_joint_acc_p95_rad_s2 < rep.raw_joint_acc_p95_rad_s2 * 0.7
    assert rep.profile.difficulty in ("low", "medium", "high")

    audit = audit_trajectory(q_cond, spec, n_ref=T, fps=50.0)
    assert audit.floor_pen_p95_mm < 2.0
    assert audit.frames == T
