"""Stage 2: Kinodynamic Reference Conditioning (Retarget_X Core Bridge).

Bridges fast kinematic IK outputs (hhtools / GMR / SOMA) and physics-based temporal MPC (DTMR):
  1. Conformal zero-phase Savitzky-Golay smoothing on root position, continuous quaternions, and joints
     to remove high-frequency LM-IK acceleration spikes without introducing phase lag.
  2. Strict physical joint-limit clamping with safety margin.
  3. Contact-aware ground projection using the target robot's exact MuJoCo foot collision geometries,
     eliminating systematic floor penetration while preserving true aerial flight phases.
  4. Motion dynamism profiling to classify clip difficulty ('low' / 'medium' / 'high') and select
     optimal adaptive MPC horizons, sample budgets, and temporal elasticity weights for Stage 3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from .config import RobotSpec


@dataclass
class MotionDynamismProfile:
    difficulty: str                 # 'low' | 'medium' | 'high'
    max_root_speed_m_s: float
    p95_root_speed_m_s: float
    max_root_acc_m_s2: float
    flight_ratio: float             # fraction of frames where both feet > threshold above floor
    max_yaw_rate_rad_s: float
    min_pelvis_height_ratio: float  # min pelvis_z / standing_root_height
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditioningReport:
    frames: int
    fps: float
    raw_floor_pen_p95_mm: float
    raw_floor_pen_max_mm: float
    cond_floor_pen_p95_mm: float
    cond_floor_pen_max_mm: float
    raw_joint_acc_p95_rad_s2: float
    cond_joint_acc_p95_rad_s2: float
    raw_joint_step_p99_deg: float
    cond_joint_step_p99_deg: float
    mean_z_lift_mm: float
    profile: MotionDynamismProfile

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["profile"] = self.profile.to_dict()
        return d


def _safe_savgol(arr: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    n = len(arr)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < polyorder + 2 or w < 3:
        return arr.copy()
    return savgol_filter(arr, window_length=w, polyorder=polyorder, axis=0, mode="interp")


def _smooth_quaternions_wxyz(quats: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Sign-continuous Savitzky-Golay smoothing on unit quaternions (wxyz)."""
    q = quats.astype(np.float64).copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    q_s = _safe_savgol(q, window, polyorder)
    norms = np.linalg.norm(q_s, axis=1, keepdims=True)
    q_s /= np.maximum(norms, 1e-12)
    return q_s


def _measure_foot_and_floor_distances(
    model: mujoco.MjModel, qpos: np.ndarray, foot_geom_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min_foot_dist[T], max_floor_pen[T]) in metres.
    min_foot_dist < 0 means foot penetrates floor; > 0 means clearance above floor.
    """
    d = mujoco.MjData(model)
    floor_id = model.geom("floor").id
    n = len(qpos)
    min_foot_dist = np.zeros(n, dtype=np.float64)
    max_floor_pen = np.zeros(n, dtype=np.float64)

    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(model, d)
        min_foot_dist[i] = min(
            mujoco.mj_geomDistance(model, d, floor_id, gid, 3.0, None) for gid in foot_geom_ids
        )
        fp = 0.0
        for c in d.contact[: d.ncon]:
            if floor_id in (c.geom1, c.geom2):
                fp = max(fp, -float(c.dist))
        max_floor_pen[i] = fp
    return min_foot_dist, max_floor_pen


def _compute_qvel(model: mujoco.MjModel, qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 1.0 / fps
    for i in range(len(qpos)):
        lo, hi = max(0, i - 1), min(len(qpos) - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) * dt, qpos[lo], qpos[hi])
    return qvel


def _profile_dynamism(
    qpos: np.ndarray,
    qvel: np.ndarray,
    min_foot_dist: np.ndarray,
    robot_spec: RobotSpec,
    fps: float,
) -> MotionDynamismProfile:
    dt = 1.0 / fps
    horiz_speed = np.linalg.norm(qvel[:, 0:2], axis=1)
    root_acc = np.linalg.norm(np.diff(qvel[:, 0:3], axis=0) / dt, axis=1) if len(qvel) > 1 else np.zeros(1)
    yaw_rate = np.abs(qvel[:, 5])

    flight_thresh = float(robot_spec.conditioning.get("flight_clearance_threshold_m", 0.035))
    flight_ratio = float(np.mean(min_foot_dist > flight_thresh))
    min_pelvis_ratio = float(np.min(qpos[:, 2]) / robot_spec.standing_root_height_m)

    max_spd = float(np.max(horiz_speed))
    p95_spd = float(np.quantile(horiz_speed, 0.95))
    max_acc = float(np.quantile(root_acc, 0.98))
    max_yaw = float(np.quantile(yaw_rate, 0.98))

    # Composite difficulty score calibrated on BONES-SEED Tiangong3 runs
    score = (
        0.8 * max_spd
        + 0.12 * max_acc
        + 6.0 * flight_ratio
        + 0.35 * max_yaw
        + 2.5 * max(0.0, 0.72 - min_pelvis_ratio)
    )
    if flight_ratio >= 0.05 or max_spd >= 2.1 or max_yaw >= 3.2 or min_pelvis_ratio < 0.65 or score >= 3.2:
        difficulty = "high"
    elif max_spd >= 0.9 or max_yaw >= 1.6 or score >= 1.6:
        difficulty = "medium"
    else:
        difficulty = "low"

    return MotionDynamismProfile(
        difficulty=difficulty,
        max_root_speed_m_s=round(max_spd, 3),
        p95_root_speed_m_s=round(p95_spd, 3),
        max_root_acc_m_s2=round(max_acc, 3),
        flight_ratio=round(flight_ratio, 4),
        max_yaw_rate_rad_s=round(max_yaw, 3),
        min_pelvis_height_ratio=round(min_pelvis_ratio, 3),
        score=round(score, 3),
    )


def condition_reference(
    qpos_raw: np.ndarray,
    robot_spec: RobotSpec,
    *,
    fps: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, ConditioningReport]:
    """Run Stage 2 kinodynamic conditioning on raw kinematic qpos (T, 7+dof, wxyz)."""
    model = mujoco.MjModel.from_xml_path(str(robot_spec.headless_scene_path))
    foot_gids = [model.geom(g).id for g in robot_spec.foot_collision_geoms]

    q_raw = np.asarray(qpos_raw, dtype=np.float64).copy()
    assert q_raw.shape[1] == model.nq, f"Expected nq={model.nq}, got {q_raw.shape[1]}"

    # Baseline kinematic diagnostics before conditioning
    _, raw_floor_pen = _measure_foot_and_floor_distances(model, q_raw, foot_gids)
    raw_dq = np.diff(q_raw[:, 7:], axis=0)
    raw_acc = np.diff(q_raw[:, 7:], n=2, axis=0) * (fps ** 2)

    # 1. Zero-phase Savitzky-Golay smoothing
    win = int(robot_spec.conditioning.get("savgol_window", 7))
    poly = int(robot_spec.conditioning.get("savgol_polyorder", 3))
    q_cond = q_raw.copy()
    q_cond[:, 0:3] = _safe_savgol(q_raw[:, 0:3], win, poly)
    q_cond[:, 3:7] = _smooth_quaternions_wxyz(q_raw[:, 3:7], win, poly)
    q_cond[:, 7:] = _safe_savgol(q_raw[:, 7:], win, poly)

    # 2. Physical joint limit clamping (with tiny 1e-3 rad safety margin)
    for j in range(robot_spec.dof):
        jnt_id = model.actuator_trnid[j, 0]
        if model.jnt_limited[jnt_id]:
            lo, hi = model.jnt_range[jnt_id]
            q_cond[:, 7 + j] = np.clip(q_cond[:, 7 + j], lo + 1e-3, hi - 1e-3)

    # 3. Contact-aware ground projection: remove penetration smoothly while preserving aerial flight
    min_foot_dist, floor_pen_pre = _measure_foot_and_floor_distances(model, q_cond, foot_gids)
    spawn_clear = float(robot_spec.conditioning.get("spawn_clearance_m", 0.002))
    # Needed lift per frame to eliminate floor penetration (including non-foot body penetration)
    pen_needed = np.maximum(0.0, np.maximum(-min_foot_dist, floor_pen_pre)) + spawn_clear
    # Zero out lift on clearly aerial frames (foot > flight_thresh and no body penetration)
    flight_thresh = float(robot_spec.conditioning.get("flight_clearance_threshold_m", 0.035))
    aerial_mask = (min_foot_dist > flight_thresh) & (floor_pen_pre <= 1e-4)
    pen_needed[aerial_mask] = 0.0

    g_win = int(robot_spec.conditioning.get("ground_smooth_window", 9))
    z_lift = _safe_savgol(pen_needed, g_win, min(2, poly))
    # Ensure we never push the robot deeper into the floor than pen_needed on contact frames
    z_lift = np.where(pen_needed > 0.0, np.maximum(z_lift, pen_needed * 0.95), np.maximum(z_lift, 0.0))
    q_cond[:, 2] += z_lift

    # Post-conditioning diagnostics
    post_foot_dist, cond_floor_pen = _measure_foot_and_floor_distances(model, q_cond, foot_gids)
    qvel_cond = _compute_qvel(model, q_cond, fps)
    cond_dq = np.diff(q_cond[:, 7:], axis=0)
    cond_acc = np.diff(q_cond[:, 7:], n=2, axis=0) * (fps ** 2)

    profile = _profile_dynamism(q_cond, qvel_cond, post_foot_dist, robot_spec, fps)

    report = ConditioningReport(
        frames=len(q_cond),
        fps=fps,
        raw_floor_pen_p95_mm=round(float(np.quantile(raw_floor_pen, 0.95)) * 1000.0, 2),
        raw_floor_pen_max_mm=round(float(np.max(raw_floor_pen)) * 1000.0, 2),
        cond_floor_pen_p95_mm=round(float(np.quantile(cond_floor_pen, 0.95)) * 1000.0, 2),
        cond_floor_pen_max_mm=round(float(np.max(cond_floor_pen)) * 1000.0, 2),
        raw_joint_acc_p95_rad_s2=round(float(np.quantile(np.abs(raw_acc), 0.95)), 2),
        cond_joint_acc_p95_rad_s2=round(float(np.quantile(np.abs(cond_acc), 0.95)), 2),
        raw_joint_step_p99_deg=round(float(np.degrees(np.quantile(np.abs(raw_dq), 0.99))), 2),
        cond_joint_step_p99_deg=round(float(np.degrees(np.quantile(np.abs(cond_dq), 0.99))), 2),
        mean_z_lift_mm=round(float(np.mean(z_lift)) * 1000.0, 2),
        profile=profile,
    )
    return q_cond, qvel_cond, report
