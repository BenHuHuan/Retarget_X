"""Stage 3: Difficulty-Adaptive DTMR Physics & Temporal Retargeting (MuJoCo Warp MPPI).

Takes a Stage-2 conditioned reference trajectory + MotionDynamismProfile, builds a clip-tailored
DTMR YAML configuration (adapting Nsample, Hsample, Hnode, temporal elasticity w_dphi, ankle PD
authority, and exploration scale to the clip's physical difficulty), and executes the GPU MPC solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

from .config import RobotSpec
from .stage2_conditioning import MotionDynamismProfile


@dataclass
class PhysicsMPCResult:
    output_dir: Path
    config_path: Path
    log_npz_path: Path | None
    phi_metrics_path: Path | None
    dtw_metrics_path: Path | None
    difficulty: str
    elapsed_s: float
    returncode: int


def export_dtmr_reference_npz(
    qpos: np.ndarray,
    robot_spec: RobotSpec,
    out_dir: Path,
    name: str,
    *,
    fps: float = 50.0,
) -> Path:
    """Export (T, 7+dof) wxyz qpos into DTMR's exact <name>_50Hz.npz format."""
    from dtmr.data.robot_motion import RobotMotion, write_reference

    rm = RobotMotion(
        root_pos=qpos[:, 0:3],
        root_quat_wxyz=qpos[:, 3:7],
        joint_pos=qpos[:, 7:],
        joint_names=list(robot_spec.joint_names),
        fps=fps,
    )
    return write_reference(
        rm,
        robot=robot_spec.name,
        out_dir=out_dir,
        name=name,
        model_file=robot_spec.headless_scene_path.name,
        target_fps=fps,
        tag="retarget_x",
    )


def build_adaptive_dtmr_config(
    robot_spec: RobotSpec,
    profile: MotionDynamismProfile,
    motion_path: Path,
    n_ref_frames: int,
    *,
    seed: int = 0,
    budget_scale: float = 1.0,
) -> dict[str, Any]:
    """Synthesize a clip-tailored DTMR configuration from the robot spec and dynamism profile."""
    prof = robot_spec.mpc_profiles.get(profile.difficulty) or robot_spec.mpc_profiles["medium"]
    n_sample = max(512, int(round(int(prof["Nsample"]) * budget_scale / 256.0)) * 256)
    n_steps = min(700, max(250, int(n_ref_frames * 2.2)))

    cfg: dict[str, Any] = {
        "robot": robot_spec.dtmr_robot_name,
        "seed": seed,
        "Nsample": n_sample,
        "Hsample": int(prof["Hsample"]),
        "Hnode": int(prof["Hnode"]),
        "Ndiffuse": int(prof.get("Ndiffuse", 5)),
        "Ndiffuse_init": int(prof.get("Ndiffuse_init", 5)),
        "sigma_scale": float(prof["sigma_scale"]),
        "temp_sample": 0.3,
        "horizon_diffuse_factor": 0.95,
        "traj_diffuse_factor": 0.6,
        "dt": 0.02,
        "timestep": 0.005,
        "action_scale": 1.0,
        "motion_fps": 50,
        "dphi_exp_base": 2.0,
        "inner_rate_pd": True,
        "auto_joint_range": True,
        "joint_range_margin_abs": 0.5,
        "joint_range_margin_ratio": 0.05,
        "joint_range_round_to": 0.1,
        "joint_range_min_width": 0.05,
        "sigma_scale_phi": float(prof["sigma_scale_phi"]),
        "Ndiffuse_steady": 2,
        "Ndiffuse_dyn": 3,
        "dyn_lo": 1.5,
        "dyn_hi": 3.0,
        "elite_frac": 0.1,
        "Ndiffuse_warmup_steps": int(prof.get("Ndiffuse_warmup_steps", 12)),
        "dyn_entry_floor_exp": 0,
        "dyn_end_exp": 2,
        "Ndiffuse_init_start_exp": -2,
        "Ndiffuse_warmup_start_exp": -2,
        "sigma_cert": 0,
        "sigma_cert_flat_ess": 0.55,
        "sigma_cert_skip_imp": -3.0,
        "sigma_cert_entry_mode": 1,
        "sigma_cert_rew_bad": -0.3,
        "sigma_cert_rew_good": -0.25,
        "sigma_cert_rew_ema": 10,
        "sigma_cert_max_iters": 6,
        "cost_config": {
            "w_pos_relcom": 0.0,
            "w_com": 0.0,
            "w_orient_pure": float(prof["w_orient_pure"]),
            "w_orient_offset": 0.0,
            "w_energy": 0.04,
            "w_dphi": float(prof["w_dphi"]),
            "w_joint_vel": 1.0e-5,
            "w_action_rate": 1.0,
            "w_joint_jerk": 0.0,
            "w_dphi_accel": 0.0,
            "w_foot_slip": float(prof.get("w_foot_slip", 0.5)),
            "cost_link_scale": {
                "head": 16,
                "waist_pitch_link": 16,
            },
            "w_joint_limit": 0.0,
            "joint_limit_margin": 0.1,
            "joint_limit_use_physical": False,
            "foot_slip_dist_threshold": 0.0,
            "w_pelvis": float(prof["w_pelvis"]),
            "w_pos_relpelvis": float(prof["w_pos_relpelvis"]),
            "w_joint_vel_ref": 0.01,
        },
        "motion_path": str(motion_path),
        "n_steps": n_steps,
        "kp": list(robot_spec.kp),
    }
    return cfg


def run_adaptive_dtmr(
    motion_path: Path,
    robot_spec: RobotSpec,
    profile: MotionDynamismProfile,
    out_dir: Path,
    *,
    dtmr_bin: str = "dtmr",
    seed: int = 0,
    budget_scale: float = 1.0,
) -> PhysicsMPCResult:
    """Execute DTMR with the synthesized difficulty-adaptive configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(motion_path) as z:
        n_ref = len(z["qpos"]) if "qpos" in z.files else len(z["states"])

    cfg = build_adaptive_dtmr_config(
        robot_spec, profile, motion_path, n_ref, seed=seed, budget_scale=budget_scale
    )
    cfg_path = out_dir / "retarget_x_dtmr_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        dtmr_bin,
        "--config", str(cfg_path),
        "--motion_path", str(motion_path),
        "--output_dir", str(out_dir),
        "--no_viz",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - t0
    (out_dir / "dtmr_run.log").write_text(proc.stdout[-10000:], encoding="utf-8")

    logs = sorted(out_dir.glob("*_log.npz"))
    phis = sorted(out_dir.glob("*_phi_tracking_metrics.yaml"))
    dtws = sorted(out_dir.glob("*_dtw_metrics.yaml"))
    return PhysicsMPCResult(
        output_dir=out_dir,
        config_path=cfg_path,
        log_npz_path=logs[-1] if logs else None,
        phi_metrics_path=phis[-1] if phis else None,
        dtw_metrics_path=dtws[-1] if dtws else None,
        difficulty=profile.difficulty,
        elapsed_s=round(elapsed, 2),
        returncode=proc.returncode,
    )
