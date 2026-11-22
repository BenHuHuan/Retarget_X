"""Command-line interface for Retarget_X."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RetargetXPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="retarget-x",
        description="Retarget_X: Unified Kinodynamic & Dense Temporal Motion Retargeting (hhtools + DTMR)",
    )
    parser.add_argument("--robot", default="tiangong3", help="Robot name or YAML path (default: tiangong3)")
    parser.add_argument("--input-npz", type=Path, required=True, help="Kinematic reference NPZ (e.g. from hhtools)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--name", default=None, help="Optional clip name override")
    parser.add_argument("--condition-only", action="store_true", help="Run Stage 2 conditioning only (skip GPU MPC)")
    parser.add_argument("--budget-scale", type=float, default=1.0, help="Scale factor for MPC Nsample budget")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MPC")
    parser.add_argument("--dtmr-bin", default="dtmr", help="Path to dtmr executable")
    args = parser.parse_args()

    pipe = RetargetXPipeline(robot=args.robot, dtmr_bin=args.dtmr_bin)
    report = pipe.run_from_kinematic_npz(
        args.input_npz,
        args.out_dir,
        name=args.name,
        run_mpc=not args.condition_only,
        seed=args.seed,
        budget_scale=args.budget_scale,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
