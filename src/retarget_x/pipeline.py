"""Unified end-to-end Retarget_X pipeline orchestrating Stages 1 -> 2 -> 3 -> 4."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import RobotSpec, load_robot_spec
from .stage2_conditioning import condition_reference
from .stage3_physics_mpc import export_dtmr_reference_npz, run_adaptive_dtmr
from .stage4_audit_export import _extract_qpos_and_phi, audit_trajectory, export_unified_npz


class RetargetXPipeline:
    """Four-stage kinodynamic + dense temporal motion retargeting pipeline."""

    def __init__(self, robot: str | Path = "tiangong3", dtmr_bin: str = "dtmr"):
        self.robot_spec: RobotSpec = load_robot_spec(robot)
        self.dtmr_bin = dtmr_bin

    def run_from_kinematic_npz(
        self,
        kinematic_npz: Path,
        out_dir: Path,
        *,
        name: str | None = None,
        run_mpc: bool = True,
        seed: int = 0,
        budget_scale: float = 1.0,
    ) -> dict[str, Any]:
        """Run Stage 2 (Conditioning) -> Stage 3 (Adaptive DTMR) -> Stage 4 (Audit & Export)
        starting from a kinematic reference NPZ (e.g., hhtools output).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_name = name or kinematic_npz.stem.replace("_50Hz", "")

        with np.load(kinematic_npz, allow_pickle=True) as z:
            if "qpos" in z.files:
                q_raw = np.asarray(z["qpos"], dtype=np.float64)
            elif "states" in z.files:
                q_raw = np.asarray(z["states"], dtype=np.float64)[:, : 7 + self.robot_spec.dof]
            else:
                raise KeyError(f"Unsupported NPZ format: {z.files}")
            fps = float(np.asarray(z["fps"]).ravel()[0]) if "fps" in z.files else 50.0

        # Stage 2: Kinodynamic Conditioning + Difficulty Profiling
        q_cond, _, cond_report = condition_reference(q_raw, self.robot_spec, fps=fps)
        cond_dir = out_dir / "conditioned_ref"
        cond_npz = export_dtmr_reference_npz(q_cond, self.robot_spec, cond_dir, clip_name, fps=fps)

        result: dict[str, Any] = {
            "clip": clip_name,
            "robot": self.robot_spec.name,
            "input_npz": str(kinematic_npz),
            "conditioned_npz": str(cond_npz),
            "stage2_conditioning": cond_report.to_dict(),
        }

        if not run_mpc:
            (out_dir / "retarget_x_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        # Stage 3: Difficulty-Adaptive DTMR Physics & Temporal Retargeting
        mpc_dir = out_dir / "mpc"
        mpc_res = run_adaptive_dtmr(
            cond_npz,
            self.robot_spec,
            cond_report.profile,
            mpc_dir,
            dtmr_bin=self.dtmr_bin,
            seed=seed,
            budget_scale=budget_scale,
        )
        result["stage3_mpc"] = {
            "difficulty": mpc_res.difficulty,
            "elapsed_s": mpc_res.elapsed_s,
            "returncode": mpc_res.returncode,
            "config_path": str(mpc_res.config_path),
        }

        # Stage 4: Physics-Consistent Audit & Unified Export
        if mpc_res.log_npz_path and mpc_res.log_npz_path.exists():
            q_out, _, phi_final = _extract_qpos_and_phi(mpc_res.log_npz_path)
            audit = audit_trajectory(
                q_out,
                self.robot_spec,
                n_ref=len(q_cond),
                fps=fps,
                phi_metrics_path=mpc_res.phi_metrics_path,
                dtw_metrics_path=mpc_res.dtw_metrics_path,
                phi_final=phi_final,
            )
            unified_npz = export_unified_npz(
                mpc_res.log_npz_path,
                self.robot_spec,
                out_dir / f"{clip_name}_retarget_x_50Hz.npz",
                fps=fps,
            )
            result["stage4_audit"] = audit.to_dict()
            result["output_npz"] = str(unified_npz)
        else:
            result["stage4_audit"] = {"status": "failed", "returncode": mpc_res.returncode}

        (out_dir / "retarget_x_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
