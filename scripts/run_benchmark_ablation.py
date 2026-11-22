"""Run head-to-head Retarget_X evaluation on Tiangong3 BONES-SEED non-HOI clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from retarget_x.pipeline import RetargetXPipeline

DEFAULT_CLIPS = [
    "egypt_dance_R_003__A275",
    "high_jump_R_001__A277",
    "jog_ff_loop_180_R_slow_002__A447",
    "walk_ff_stop_225_002__A096",
    "jog_arc_cw_start_002__A181",
    "walk_ff_loop_225_R_003__A235",
    "turn_start_jog_0045_003__A026",
    "crouch_ff_loop_225_R_001__A245",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-dir", type=Path, required=True, help="Directory with hhtools *_50Hz.npz files")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for Retarget_X runs")
    p.add_argument("--robot", default="tiangong3")
    p.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    p.add_argument("--budget-scale", type=float, default=1.0)
    p.add_argument("--dtmr-bin", default="dtmr")
    args = p.parse_args()

    pipe = RetargetXPipeline(robot=args.robot, dtmr_bin=args.dtmr_bin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for idx, fn in enumerate(args.clips):
        npz_path = args.ref_dir / f"{fn}_50Hz.npz"
        clip_out = args.out_dir / fn
        rep_file = clip_out / "retarget_x_report.json"
        if rep_file.exists():
            rep = json.loads(rep_file.read_text(encoding="utf-8"))
            if rep.get("stage4_audit", {}).get("mpjpe_world_mm") is not None:
                summary.append(rep)
                a = rep["stage4_audit"]
                print(
                    f"[{idx+1:02d}/{len(args.clips)}] CACHED {fn:35s} diff={rep['stage3_mpc']['difficulty']:6s} "
                    f"mpjpe={a['mpjpe_world_mm']:5.1f}mm base={a['base_pos_err_mm']:5.1f}mm "
                    f"ori={a['orient_err_deg']:4.1f}deg pen95={a['floor_pen_p95_mm']:4.1f}mm",
                    flush=True,
                )
                continue

        t0 = time.monotonic()
        rep = pipe.run_from_kinematic_npz(
            npz_path,
            clip_out,
            name=fn,
            run_mpc=True,
            budget_scale=args.budget_scale,
        )
        summary.append(rep)
        a = rep.get("stage4_audit", {})
        if a.get("mpjpe_world_mm") is not None:
            print(
                f"[{idx+1:02d}/{len(args.clips)}] {fn:35s} diff={rep['stage3_mpc']['difficulty']:6s} "
                f"{rep['stage3_mpc']['elapsed_s']:5.1f}s mpjpe={a['mpjpe_world_mm']:5.1f}mm "
                f"relcom={a['mpjpe_relcom_mm']:5.1f}mm base={a['base_pos_err_mm']:5.1f}mm "
                f"ori={a['orient_err_deg']:4.1f}deg pen95={a['floor_pen_p95_mm']:4.1f}mm "
                f"stretch={a['time_stretch_ratio']:.2f}x",
                flush=True,
            )
        else:
            print(f"[{idx+1:02d}/{len(args.clips)}] FAILED {fn} in {time.monotonic()-t0:.1f}s: {a}", flush=True)

    out_json = args.out_dir / "retarget_x_ablation_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL DONE ({len(summary)} clips) -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
