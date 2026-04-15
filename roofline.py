"""
ME759 Course Participation — Roofline Plot Generator
Run on your laptop after getting benchmark numbers from Euler.

Fill in YOUR MEASURED RESULTS below, then run:
    pip install matplotlib numpy
    python roofline.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════
# 1.  GPU SPECS  (hardware constants — do not change)
# ══════════════════════════════════════════════════════════════
GPUS = {
    "K20\n(Kepler 2012)":   {"peak_flops": 3.52e12,  "peak_bw": 208e9,  "color": "#5577AA", "ls": ":"},
    "V100\n(Volta 2017)":   {"peak_flops": 14.0e12,  "peak_bw": 900e9,  "color": "#7799CC", "ls": "--"},
    "A100\n(Ampere 2020)":  {"peak_flops": 19.5e12,  "peak_bw": 2000e9, "color": "#00B4D8", "ls": "-"},
    "H100\n(Hopper 2022)":  {"peak_flops": 60.0e12,  "peak_bw": 3350e9, "color": "#FFFFFF", "ls": "-"},
}

# ══════════════════════════════════════════════════════════════
# 2.  YOUR MEASURED RESULTS — fill these in from Euler output
# ══════════════════════════════════════════════════════════════
#
# Arithmetic Intensity (AI) = FLOPs / bytes transferred
#   Naive:  AI = 2(2R+1) / [(2R+1)*4 + 4]  ≈ 0.5 FLOP/byte
#   Tiled:  AI = 2(2R+1) / [2*4]  = (2R+1)/4 FLOP/byte
#
# Achieved performance (GFLOP/s) = FLOPs_per_element * N / time_s
#   FLOPs per element = 2*(2R+1)   (one mult + one add per weight)

N = 1_048_576  # problem size from benchmark

# ── Fill in your measured times (milliseconds) from stencil_results_*.txt ──

RESULTS = {
    # "label": (R, time_naive_ms, time_tiled_ms)
    "R=1":  (1,  None, None),   # <-- replace None with your numbers
    "R=4":  (4,  None, None),
    "R=8":  (8,  None, None),
    "R=16": (16, None, None),
}

# ── Which GPU did Euler give you? ──
# Set to one of: "K20\n(Kepler 2012)", "V100\n(Volta 2017)",
#                "A100\n(Ampere 2020)", "H100\n(Hopper 2022)"
YOUR_GPU = "A100\n(Ampere 2020)"

# ══════════════════════════════════════════════════════════════
# 3.  PLOT
# ══════════════════════════════════════════════════════════════

BG      = "#0D1B2A"
PANEL   = "#0F2035"
CYAN    = "#00B4D8"
WHITE   = "#FFFFFF"
MUTED   = "#7A9BB5"
ORANGE  = "#FF9944"
GREEN   = "#44DD88"

fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=BG)
ax.set_facecolor(BG)

# Log-log axes
ax.set_xscale("log")
ax.set_yscale("log")

ai_range = np.logspace(-2, 2, 400)   # 0.01 → 100 FLOP/byte

# ── Draw roofline for each GPU ────────────────────────────────
for name, g in GPUS.items():
    bw   = g["peak_bw"]
    fp   = g["peak_flops"]
    col  = g["color"]
    ls   = g["ls"]
    ridge = fp / bw   # FLOP/byte

    roof = np.minimum(bw * ai_range, fp) / 1e12   # TFLOP/s
    ax.plot(ai_range, roof, color=col, lw=1.8, ls=ls, alpha=0.85, zorder=2)

    # Label at the flat ceiling (right side)
    ax.text(80, fp / 1e12 * 1.05, name.replace("\n", " "),
            color=col, fontsize=8, va="bottom", ha="right",
            fontfamily="monospace")

    # Ridge point marker
    ax.axvline(ridge, color=col, lw=0.5, ls=":", alpha=0.3, zorder=1)

# ── Stencil operating points ──────────────────────────────────
markers_naive = []
markers_tiled = []

for label, (R, t_naive, t_tiled) in RESULTS.items():
    flops_per_elem = 2 * (2 * R + 1)
    total_flops    = flops_per_elem * N

    # Arithmetic intensity
    bytes_naive = N * ((2*R + 1) + 1) * 4   # (2R+1) reads + 1 write
    bytes_tiled = N * 2 * 4                  # 1 read + 1 write (shmem reuse)
    ai_naive = total_flops / bytes_naive
    ai_tiled = total_flops / bytes_tiled

    # Achieved performance (if measured)
    if t_naive is not None:
        perf_naive = total_flops / (t_naive * 1e-3) / 1e12  # TFLOP/s
        p = ax.scatter(ai_naive, perf_naive, marker="o", s=80,
                       color=ORANGE, edgecolors=WHITE, lw=0.8, zorder=5)
        ax.annotate(f"{label}\nnaive", (ai_naive, perf_naive),
                    textcoords="offset points", xytext=(6, -14),
                    color=ORANGE, fontsize=7.5, fontfamily="monospace")
        markers_naive.append(p)

    if t_tiled is not None:
        perf_tiled = total_flops / (t_tiled * 1e-3) / 1e12
        p = ax.scatter(ai_tiled, perf_tiled, marker="^", s=80,
                       color=GREEN, edgecolors=WHITE, lw=0.8, zorder=5)
        ax.annotate(f"{label}\ntiled", (ai_tiled, perf_tiled),
                    textcoords="offset points", xytext=(6, 4),
                    color=GREEN, fontsize=7.5, fontfamily="monospace")
        markers_tiled.append(p)

    # Even without measured times, mark the AI positions with vertical lines
    if t_naive is None:
        ax.axvline(ai_naive, color=ORANGE, lw=1.0, ls="--", alpha=0.5)
        ax.text(ai_naive * 1.05, 0.015,
                f"{label}\nnaive\nAI={ai_naive:.2f}",
                color=ORANGE, fontsize=7, va="bottom", fontfamily="monospace")

    if t_tiled is None:
        ax.axvline(ai_tiled, color=GREEN, lw=1.0, ls="--", alpha=0.5)
        ax.text(ai_tiled * 1.05, 0.015,
                f"{label}\ntiled\nAI={ai_tiled:.2f}",
                color=GREEN, fontsize=7, va="bottom", fontfamily="monospace",
                rotation=0)

# ── Axes styling ──────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", color=WHITE,
              fontsize=12, labelpad=8)
ax.set_ylabel("Performance  (TFLOP/s)", color=WHITE,
              fontsize=12, labelpad=8)
ax.set_title("Roofline Model — 1D Stencil across GPU Generations",
             color=WHITE, fontsize=14, fontweight="bold", pad=12)

ax.set_xlim(0.1, 100)
ax.set_ylim(0.01, 200)

ax.tick_params(colors=MUTED, labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor("#1A3858")
ax.grid(True, which="both", color="#1A3858", lw=0.5, alpha=0.7)

# ── Legend ────────────────────────────────────────────────────
legend_gpu = [
    Line2D([0], [0], color=g["color"], lw=2, ls=g["ls"], label=name.replace("\n", " "))
    for name, g in GPUS.items()
]
legend_pts = [
    Line2D([0], [0], marker="o", color=ORANGE, lw=0, ms=8,
           markeredgecolor=WHITE, label="Naive (global mem)"),
    Line2D([0], [0], marker="^", color=GREEN, lw=0, ms=8,
           markeredgecolor=WHITE, label="Tiled (shared mem)"),
]
leg = ax.legend(handles=legend_gpu + legend_pts,
                loc="upper left", framealpha=0.2,
                facecolor=PANEL, edgecolor="#1A3858",
                labelcolor=WHITE, fontsize=8.5)

# ── Memory-bound / Compute-bound labels ───────────────────────
ax.text(0.13, 50, "memory-bound", color=MUTED, fontsize=9,
        rotation=42, va="center", fontstyle="italic")
ax.text(30, 0.05, "compute-bound", color=MUTED, fontsize=9,
        va="center", fontstyle="italic")

plt.tight_layout(pad=0.8)

out = "roofline.png"
plt.savefig(out, dpi=180, facecolor=BG, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
