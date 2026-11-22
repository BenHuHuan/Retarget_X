"""Compare hhtools vs Vanilla DTMR vs Retarget_X on Tiangong3 BONES-SEED non-HOI clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hht", type=Path, required=True, help="hhtools_summary.json")
    p.add_argument("--dtmr", type=Path, required=True, help="vanilla DTMR summary json")
    p.add_argument("--rx", type=Path, required=True, help="retarget_x_ablation_summary.json")
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    hht_map = {r["filename"]: r for r in json.loads(args.hht.read_text(encoding="utf-8"))}
    dtmr_map = {r["filename"]: r for r in json.loads(args.dtmr.read_text(encoding="utf-8"))}
    rx_rows = json.loads(args.rx.read_text(encoding="utf-8"))

    lines = [
        "| Clip | Difficulty | `hhtools` PD Surv | `hhtools` FloorPen95 | Vanilla DTMR MPJPE / Base / Ori | **Retarget_X MPJPE / Base / Ori** | MPJPE Delta | Base Delta |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    d_mpjpe, r_mpjpe = [], []
    d_base, r_base = [], []
    d_ori, r_ori = [], []
    h_pen, r_pen = [], []
    h_surv = []

    for rx in rx_rows:
        fn = rx["clip"]
        h = hht_map.get(fn, {})
        d = dtmr_map.get(fn, {})
        a = rx.get("stage4_audit", {})
        c2 = rx.get("stage2_conditioning", {})
        if not d or a.get("mpjpe_world_mm") is None:
            continue

        dm, db, do = float(d["mpjpe_world_mm"]), float(d["base_pos_err_mm"]), float(d["orient_err_deg"])
        rm, rb, ro = float(a["mpjpe_world_mm"]), float(a["base_pos_err_mm"]), float(a["orient_err_deg"])
        d_mpjpe.append(dm); r_mpjpe.append(rm)
        d_base.append(db); r_base.append(rb)
        d_ori.append(do); r_ori.append(ro)
        h_pen.append(float(h.get("kin_floor_pen_p95_mm", c2.get("raw_floor_pen_p95_mm", 0.0))))
        r_pen.append(float(c2.get("cond_floor_pen_p95_mm", 0.0)))
        h_surv.append(float(h.get("pd_survival_ratio", 0.0)) * 100.0)

        m_pct = (rm - dm) / dm * 100.0
        b_pct = (rb - db) / db * 100.0
        lines.append(
            f"| `{fn}` | {rx['stage3_mpc']['difficulty']} | {h_surv[-1]:.1f}% | {h_pen[-1]:.1f} mm | "
            f"{dm:.1f} mm / {db:.1f} mm / {do:.1f}° | **{rm:.1f} mm / {rb:.1f} mm / {ro:.1f}°** | "
            f"{m_pct:+.1f}% | {b_pct:+.1f}% |"
        )

    if d_mpjpe:
        dm_avg, rm_avg = np.mean(d_mpjpe), np.mean(r_mpjpe)
        db_avg, rb_avg = np.mean(d_base), np.mean(r_base)
        do_avg, ro_avg = np.mean(d_ori), np.mean(r_ori)
        lines.append(
            f"| **Mean ({len(d_mpjpe)} clips)** | — | **{np.mean(h_surv):.1f}%** (0% no-fall) | "
            f"**{np.mean(h_pen):.1f} mm → {np.mean(r_pen):.2f} mm** | "
            f"{dm_avg:.1f} mm / {db_avg:.1f} mm / {do_avg:.2f}° | "
            f"**{rm_avg:.1f} mm / {rb_avg:.1f} mm / {ro_avg:.2f}°** | "
            f"**{(rm_avg-dm_avg)/dm_avg*100:+.1f}%** | **{(rb_avg-db_avg)/db_avg*100:+.1f}%** |"
        )

    md = "\n".join(lines) + "\n"
    print(md)
    if args.out_md:
        args.out_md.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
