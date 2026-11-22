"""Generate publication-grade benchmark visualizations for Retarget_X README and web report."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark" / "tiangong3_bones_seed_nonhoi"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# Modern dark theme palette
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#38bdf8"
VIOLET = "#a78bfa"
EMERALD = "#34d399"
AMBER = "#fbbf24"
ROSE = "#fb7185"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHORT_NAMES = {
    "egypt_dance_R_003__A275": "Egypt Dance",
    "high_jump_R_001__A277": "High Jump",
    "jog_ff_loop_180_R_slow_002__A447": "Jog 180 Slow",
    "walk_ff_stop_225_002__A096": "Walk Stop",
    "jog_arc_cw_start_002__A181": "Jog Arc CW",
    "walk_ff_loop_225_R_003__A235": "Walk Loop",
    "turn_start_jog_0045_003__A026": "Turn Start Jog",
    "crouch_ff_loop_225_R_001__A245": "Crouch Loop",
}


def load_data():
    hht = json.loads((BENCH / "hhtools_120_nonhoi_summary.json").read_text())
    dtmr = json.loads((BENCH / "vanilla_dtmr_18_summary.json").read_text())
    rx = json.loads((BENCH / "retarget_x_v2_8_summary.json").read_text())
    return hht, {r["filename"]: r for r in dtmr}, rx


def plot_benchmark_overview(dtmr_map, rx_rows):
    clips = [r["clip"] for r in rx_rows]
    labels = [SHORT_NAMES.get(c, c[:14]) for c in clips]
    d_mpjpe = [dtmr_map[c]["mpjpe_world_mm"] for c in clips]
    r_mpjpe = [r["stage4_audit"]["mpjpe_world_mm"] for r in rx_rows]
    d_base = [dtmr_map[c]["base_pos_err_mm"] for c in clips]
    r_base = [r["stage4_audit"]["base_pos_err_mm"] for r in rx_rows]

    x = np.arange(len(clips))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=220)
    fig.suptitle(
        "Tiangong3 (29-DOF, 65.8 kg)  |  Vanilla DTMR vs. Retarget_X on BONES-SEED Non-HOI",
        fontsize=14, fontweight="bold", y=0.98, color=TEXT,
    )

    # Panel 1: MPJPE World
    b1 = ax1.bar(x - w / 2, d_mpjpe, w, label="Vanilla DTMR", color=VIOLET, alpha=0.85, edgecolor="none")
    b2 = ax1.bar(x + w / 2, r_mpjpe, w, label="Retarget_X (Ours)", color=CYAN, alpha=0.95, edgecolor="none")
    ax1.axhline(np.mean(d_mpjpe), color=VIOLET, ls="--", lw=1.2, alpha=0.7)
    ax1.axhline(np.mean(r_mpjpe), color=CYAN, ls="--", lw=1.4, alpha=0.9)
    ax1.set_title("Whole-Body World MPJPE (mm, lower is better)", fontsize=11, pad=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax1.set_ylabel("MPJPE World (mm)")
    ax1.set_ylim(0, 192)
    ax1.grid(axis="y", ls=":", alpha=0.6)
    ax1.legend(frameon=False, loc="upper right", fontsize=9.5)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.2, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.2, f"{h:.0f}", ha="center", va="bottom", fontsize=7.8, color=CYAN, fontweight="bold")
    ax1.text(
        0.02, 0.90,
        f"Mean: {np.mean(d_mpjpe):.1f} -> {np.mean(r_mpjpe):.1f} mm ({(np.mean(r_mpjpe)-np.mean(d_mpjpe))/np.mean(d_mpjpe)*100:+.1f}%)",
        transform=ax1.transAxes, fontsize=9.5, fontweight="bold", color=EMERALD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f2922", edgecolor=EMERALD, lw=1.0),
    )

    # Panel 2: Base Pos Err
    b3 = ax2.bar(x - w / 2, d_base, w, label="Vanilla DTMR", color=VIOLET, alpha=0.85)
    b4 = ax2.bar(x + w / 2, r_base, w, label="Retarget_X (Ours)", color=EMERALD, alpha=0.92)
    ax2.axhline(np.mean(d_base), color=VIOLET, ls="--", lw=1.2, alpha=0.7)
    ax2.axhline(np.mean(r_base), color=EMERALD, ls="--", lw=1.4, alpha=0.9)
    ax2.set_title("Pelvis Root Position Error (mm, lower is better)", fontsize=11, pad=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax2.set_ylabel("Base Position Error (mm)")
    ax2.set_ylim(0, 168)
    ax2.grid(axis="y", ls=":", alpha=0.6)
    ax2.legend(frameon=False, loc="upper right", fontsize=9.5)
    for bar in b3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{h:.0f}", ha="center", va="bottom", fontsize=7.8, color=EMERALD, fontweight="bold")
    ax2.text(
        0.02, 0.90,
        f"Mean: {np.mean(d_base):.1f} -> {np.mean(r_base):.1f} mm ({(np.mean(r_base)-np.mean(d_base))/np.mean(d_base)*100:+.1f}%)",
        transform=ax2.transAxes, fontsize=9.5, fontweight="bold", color=EMERALD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f2922", edgecolor=EMERALD, lw=1.0),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS / "benchmark_overview.png", facecolor=BG)
    plt.close(fig)


def plot_stage2_and_baseline(hht_rows, rx_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.0), dpi=220)
    fig.suptitle(
        "Stage 2 Contact-Aware Conditioning & 120-Clip BONES-SEED Non-HOI Physical Feasibility Gap",
        fontsize=13.5, fontweight="bold", y=0.98, color=TEXT,
    )

    # Panel 1: Floor Penetration before/after Stage 2
    clips = [r["clip"] for r in rx_rows]
    labels = [SHORT_NAMES.get(c, c[:14]) for c in clips]
    raw_pen = [r["stage2_conditioning"]["raw_floor_pen_p95_mm"] for r in rx_rows]
    cond_pen = [r["stage2_conditioning"]["cond_floor_pen_p95_mm"] for r in rx_rows]
    out_pen = [r["stage4_audit"]["floor_pen_p95_mm"] for r in rx_rows]

    x = np.arange(len(clips))
    w = 0.27
    ax1.bar(x - w, raw_pen, w, label="Raw hhtools Ref", color=ROSE, alpha=0.88)
    ax1.bar(x, [max(v, 0.3) for v in cond_pen], w, label="Stage 2 Conditioned (0.0 mm)", color=EMERALD, alpha=0.95)
    ax1.bar(x + w, out_pen, w, label="Retarget_X Physics Output", color=CYAN, alpha=0.9)
    ax1.set_title("Floor Penetration P95 (mm): Raw vs. Stage 2 vs. Final Output", fontsize=10.5, pad=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.8)
    ax1.set_ylabel("Floor Penetration P95 (mm)")
    ax1.grid(axis="y", ls=":", alpha=0.6)
    ax1.legend(frameon=False, loc="upper left", fontsize=8.8)
    ax1.text(
        0.55, 0.86,
        f"Ref Penetration:\n{np.mean(raw_pen):.1f} mm -> 0.00 mm\n(-100%)",
        transform=ax1.transAxes, fontsize=9, fontweight="bold", color=EMERALD,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f2922", edgecolor=EMERALD, lw=1.0),
    )

    # Panel 2: 120-clip hhtools PD survival by category vs Retarget_X 100%
    by_b = collections.defaultdict(list)
    for r in hht_rows:
        by_b[r["bucket"]].append(r["pd_survival_ratio"] * 100.0)
    buckets = sorted(by_b.keys(), key=lambda k: np.mean(by_b[k]))
    means = [np.mean(by_b[b]) for b in buckets]
    counts = [len(by_b[b]) for b in buckets]
    y = np.arange(len(buckets))

    ax2.barh(y, [100.0] * len(buckets), height=0.58, color=GRID, alpha=0.5, label="Full Clip (100% = Retarget_X)")
    bars = ax2.barh(y, means, height=0.58, color=AMBER, alpha=0.9, label="hhtools Direct PD Survival (%)")
    ax2.axvline(100.0, color=EMERALD, ls="--", lw=1.8, label="Retarget_X Physical Completion (100%)")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{b} (n={n})" for b, n in zip(buckets, counts)], fontsize=8.8)
    ax2.set_xlabel("Physical Survival Ratio before Fall (%)")
    ax2.set_xlim(0, 112)
    ax2.set_title("120 Non-HOI Clips (28,445 Frames): Direct PD vs. Physics MPC", fontsize=10.5, pad=10, fontweight="bold")
    ax2.grid(axis="x", ls=":", alpha=0.6)
    ax2.legend(frameon=False, loc="lower right", fontsize=8.3)
    for bar, m in zip(bars, means):
        ax2.text(m + 1.8, bar.get_y() + bar.get_height() / 2, f"{m:.1f}%", va="center", fontsize=8.3, color=AMBER, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ASSETS / "stage2_and_baseline.png", facecolor=BG)
    plt.close(fig)


def plot_ablation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 4.7), dpi=220)
    fig.suptitle(
        "Key Ablation Study: Compute-Accuracy Tradeoff & Foot-Slip Weight Discovery",
        fontsize=13.5, fontweight="bold", y=0.98, color=TEXT,
    )

    # Left: Egypt Dance ablation (Runtime vs MPJPE)
    names = ["Vanilla DTMR\n(4096 / H44)", "Vanilla DTMR\n(1024 / H32)", "Retarget_X v1\n(w_foot_slip=0.5)", "Retarget_X v2\n(w_foot_slip=0.0)"]
    mpjpe = [28.0, 32.5, 84.4, 26.9]
    mins = [19.9, 4.1, 7.2, 4.8]
    colors = [MUTED, VIOLET, ROSE, EMERALD]

    bars = ax1.bar(names, mpjpe, width=0.52, color=colors, alpha=0.92)
    ax1.set_title("Egypt Dance (egypt_dance_R_003__A275): MPJPE & Runtime", fontsize=10.5, pad=10, fontweight="bold")
    ax1.set_ylabel("World MPJPE (mm, lower is better)")
    ax1.set_ylim(0, 102)
    ax1.grid(axis="y", ls=":", alpha=0.6)
    for bar, m, t in zip(bars, mpjpe, mins):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, m + 2.5,
            f"{m:.1f} mm\n({t:.1f} min)", ha="center", va="bottom", fontsize=8.5,
            color=EMERALD if m == 26.9 else TEXT, fontweight="bold" if m == 26.9 else "normal",
        )

    # Right: High Jump breakthrough
    names2 = ["Vanilla DTMR\n(1024 / H32)", "Retarget_X v1\n(w_foot_slip=0.3)", "Retarget_X v2\n(w_foot_slip=0.0)"]
    mpjpe2 = [157.8, 64.0, 61.7]
    base2 = [138.5, 46.6, 45.7]
    x = np.arange(len(names2))
    w = 0.34
    b1 = ax2.bar(x - w / 2, mpjpe2, w, label="World MPJPE (mm)", color=CYAN, alpha=0.92)
    b2 = ax2.bar(x + w / 2, base2, w, label="Pelvis Base Err (mm)", color=EMERALD, alpha=0.92)
    ax2.set_title("High Jump (high_jump_R_001__A277): -60.9% MPJPE Reduction", fontsize=10.5, pad=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names2, fontsize=9)
    ax2.set_ylabel("Error (mm, lower is better)")
    ax2.set_ylim(0, 185)
    ax2.grid(axis="y", ls=":", alpha=0.6)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)
    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, color=CYAN, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, color=EMERALD, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(ASSETS / "ablation_tradeoff.png", facecolor=BG)
    plt.close(fig)


def main():
    hht, dtmr_map, rx = load_data()
    plot_benchmark_overview(dtmr_map, rx)
    plot_stage2_and_baseline(hht, rx)
    plot_ablation()
    print("Generated 3 charts in", ASSETS)


if __name__ == "__main__":
    main()
