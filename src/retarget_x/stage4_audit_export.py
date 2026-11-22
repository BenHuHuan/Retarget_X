"""Stage 4: Physics-Consistent Audit & Unified Deployment Export."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from .config import RobotSpec


@dataclass
class AuditReport:
    frames: int
    n_ref: int
    time_stretch_ratio: float
    completed: bool
    success: bool
    floor_pen_p95_mm: float
    floor_pen_max_mm: float
    self_pen_p95_mm: float
    joint_step_p99_deg: float
    joint_acc_p95_rad_s2: float
    mpjpe_world_mm: float | None = None
    mpjpe_relcom_mm: float | None = None
    orient_err_deg: float | None = None
    base_pos_err_mm: float | None = None
    base_orient_err_deg: float | None = None
    joint_mae_rad: float | None = None
    dtw_mean_cost: float | None = None
    mean_reward: float | None = None
    phi_final: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_qpos_and_phi(log_npz_path: Path) -> tuple[np.ndarray, np.ndarray | None, float | None]:
    """Robustly extract qpos and phase trajectory from DTMR's *_log.npz regardless of key naming."""
    with np.load(log_npz_path, allow_pickle=True) as z:
        files = set(z.files)
        qpos = None
        for k in ("qpos", "qpos_log", "xs", "state", "states"):
            if k in files:
                arr = np.asarray(z[k], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 7:
                    qpos = arr
                    break
        if qpos is None:
            raise KeyError(f"Could not locate qpos array in {log_npz_path} (keys={sorted(files)})")

        phi_arr = None
        for k in ("phi", "phis", "phi_log", "phase", "phases"):
            if k in files:
                phi_arr = np.asarray(z[k], dtype=np.float64).ravel()
                break
        phi_final = float(phi_arr[-1]) if phi_arr is not None and len(phi_arr) > 0 else None
    return qpos, phi_arr, phi_final


def audit_trajectory(
    qpos: np.ndarray,
    robot_spec: RobotSpec,
    *,
    n_ref: int | None = None,
    fps: float = 50.0,
    phi_metrics_path: Path | None = None,
    dtw_metrics_path: Path | None = None,
    phi_final: float | None = None,
) -> AuditReport:
    """Evaluate kinematic/contact cleanliness and parse DTMR physical tracking metrics."""
    model = mujoco.MjModel.from_xml_path(str(robot_spec.headless_scene_path))
    d = mujoco.MjData(model)
    floor_id = model.geom("floor").id

    q = np.asarray(qpos, dtype=np.float64)
    if q.shape[1] > model.nq:
        q = q[:, : model.nq]

    fp_list, sp_list = [], []
    for i in range(len(q)):
        d.qpos[:] = q[i]
        mujoco.mj_forward(model, d)
        fp, sp = 0.0, 0.0
        for c in d.contact[: d.ncon]:
            if floor_id in (c.geom1, c.geom2):
                fp = max(fp, -float(c.dist))
            else:
                sp = max(sp, -float(c.dist))
        fp_list.append(fp)
        sp_list.append(sp)

    dq = np.diff(q[:, 7:], axis=0) if len(q) > 1 else np.zeros((1, robot_spec.dof))
    acc = np.diff(q[:, 7:], n=2, axis=0) * (fps ** 2) if len(q) > 2 else np.zeros((1, robot_spec.dof))

    mpjpe_w = mpjpe_c = ori_deg = base_mm = base_ori = j_mae = dtw_cost = mean_rew = None
    if phi_metrics_path and phi_metrics_path.exists():
        pm = yaml.safe_load(phi_metrics_path.read_text(encoding="utf-8"))
        mpjpe_w = round(float(pm["phi/mpjpe_world_m"]) * 1000.0, 2)
        mpjpe_c = round(float(pm["phi/mpjpe_relcom_m"]) * 1000.0, 2)
        ori_deg = round(float(pm["phi/mean_orient_err_deg"]), 2)
        base_mm = round(float(pm["phi/qpos_base_pos_err_m"]) * 1000.0, 2)
        base_ori = round(float(pm["phi/qpos_base_orient_err_deg"]), 2)
        j_mae = round(float(pm["phi/qpos_joint_mae_rad"]), 4)
    if dtw_metrics_path and dtw_metrics_path.exists():
        dm = yaml.safe_load(dtw_metrics_path.read_text(encoding="utf-8"))
        dtw_cost = round(float(dm["mean_cost"]), 4)
        mean_rew = round(float(dm["mean_reward"]), 4)
        if n_ref is None and "n_ref" in dm:
            n_ref = int(dm["n_ref"])

    n_ref_val = int(n_ref) if n_ref is not None else len(q)
    stretch = round(len(q) / max(n_ref_val, 1), 3)
    # If phi_final not explicitly in log, infer completion from reaching reference end (< n_steps cap)
    completed = bool((phi_final is not None and phi_final >= 0.95) or (phi_final is None and len(q) < 695))
    success = bool(
        completed
        and (base_mm is None or base_mm < 250.0)
        and (mpjpe_w is None or mpjpe_w < 250.0)
    )

    return AuditReport(
        frames=len(q),
        n_ref=n_ref_val,
        time_stretch_ratio=stretch,
        completed=completed,
        success=success,
        floor_pen_p95_mm=round(float(np.quantile(fp_list, 0.95)) * 1000.0, 2),
        floor_pen_max_mm=round(float(np.max(fp_list)) * 1000.0, 2),
        self_pen_p95_mm=round(float(np.quantile(sp_list, 0.95)) * 1000.0, 2),
        joint_step_p99_deg=round(float(np.degrees(np.quantile(np.abs(dq), 0.99))), 2),
        joint_acc_p95_rad_s2=round(float(np.quantile(np.abs(acc), 0.95)), 2),
        mpjpe_world_mm=mpjpe_w,
        mpjpe_relcom_mm=mpjpe_c,
        orient_err_deg=ori_deg,
        base_pos_err_mm=base_mm,
        base_orient_err_deg=base_ori,
        joint_mae_rad=j_mae,
        dtw_mean_cost=dtw_cost,
        mean_reward=mean_rew,
        phi_final=round(phi_final, 4) if phi_final is not None else None,
    )


def export_unified_npz(
    log_npz_path: Path,
    robot_spec: RobotSpec,
    out_npz_path: Path,
    *,
    fps: float = 50.0,
) -> Path:
    """Export a clean deployment/RL-ready NPZ (root_pos_w, root_quat_w, joint_pos, joint_vel, body_pos_w)."""
    qpos_full, phi_arr, _ = _extract_qpos_and_phi(log_npz_path)
    model = mujoco.MjModel.from_xml_path(str(robot_spec.headless_scene_path))
    d = mujoco.MjData(model)
    q = qpos_full[:, : model.nq]
    n = len(q)

    qvel = np.zeros((n, model.nv), dtype=np.float64)
    dt = 1.0 / fps
    body_pos = np.zeros((n, model.nbody, 3), dtype=np.float64)
    body_quat = np.zeros((n, model.nbody, 4), dtype=np.float64)
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], (hi - lo) * dt, q[lo], q[hi])
        d.qpos[:] = q[i]
        mujoco.mj_forward(model, d)
        body_pos[i] = d.xpos
        body_quat[i] = d.xquat

    payload: dict[str, Any] = {
        "fps": np.asarray(fps, dtype=np.float64),
        "robot": np.asarray(robot_spec.name),
        "joint_names": np.asarray(robot_spec.joint_names),
        "body_names": np.asarray([model.body(i).name for i in range(model.nbody)]),
        "root_pos_w": q[:, 0:3].astype(np.float32),
        "root_quat_w": q[:, 3:7].astype(np.float32),
        "root_lin_vel_w": qvel[:, 0:3].astype(np.float32),
        "root_ang_vel_w": qvel[:, 3:6].astype(np.float32),
        "joint_pos": q[:, 7:].astype(np.float32),
        "joint_vel": qvel[:, 6:].astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
    }
    if phi_arr is not None:
        payload["phi"] = phi_arr.astype(np.float32)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz_path, **payload)
    return out_npz_path
