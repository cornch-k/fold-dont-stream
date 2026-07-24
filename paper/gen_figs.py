#!/usr/bin/env python3
"""Generate fig1_recovery.png, fig2_cliff.png, fig3_policy.png for arXiv paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))  # 그림은 스크립트와 같은 디렉터리에 생성

# ---------------------------------------------------------------- palette --
BLUE = "#0072B2"       # primary series
ORANGE = "#E69F00"     # accent
VERMILLION = "#D55E00" # warning / cliff
GRAY = "#666666"       # baseline / reference
SPINE_GRAY = "#888888"
BG = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": BG,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": SPINE_GRAY,
    "axes.linewidth": 0.6,
    "xtick.color": SPINE_GRAY,
    "ytick.color": SPINE_GRAY,
    "xtick.labelcolor": "black",
    "ytick.labelcolor": "black",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def strip_spines(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        if side not in keep:
            spine.set_visible(False)
        else:
            spine.set_color(SPINE_GRAY)
            spine.set_linewidth(0.6)


# =============================================================================
# Fig 1 -- recovery rate vs loop count R
# =============================================================================
def fig1():
    x = [1, 2, 3, 4, 6]
    y = [0, 27.6, 41.7, 48.5, 51.7]

    fig, ax = plt.subplots(figsize=(3.4, 2.55))

    ax.plot(x, y, "-o", color=BLUE, linewidth=1.3, markersize=4.2,
            markerfacecolor=BLUE, markeredgecolor=BLUE, zorder=3,
            label="Recovery (measured)")

    # direct labels for main series (selective, skip R=1 which is 0)
    label_offsets = {1: (0, 6), 2: (0, 7), 3: (2, -11), 4: (0, 7), 6: (-2, 7)}
    for xi, yi in zip(x, y):
        dx, dy = label_offsets[xi]
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=7, color="black",
                    zorder=4)

    # fp32 cross-check point at R=3
    ax.plot(3, 40.9, marker="D", markersize=5, color=GRAY, zorder=3,
            markeredgecolor=GRAY)
    ax.annotate("fp32 cross-check", (3, 40.9), textcoords="offset points",
                xytext=(10, -14), ha="left", fontsize=6.6, color=GRAY)

    # upper bound line at y=100
    ax.axhline(100, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.annotate("2× memory upper bound (large)", xy=(6, 100),
                xytext=(6, 100), textcoords="data", ha="right", va="bottom",
                fontsize=6.6, color=GRAY)

    # saturation band near R=6
    ax.axhspan(53, 55, xmin=(4.3 - 0.6) / (6.6 - 0.6), xmax=1.0,
               color=GRAY, alpha=0.12, zorder=0)
    ax.plot([4.3, 6.6], [53, 53], linestyle=":", linewidth=0.8, color=GRAY, zorder=1)
    ax.plot([4.3, 6.6], [55, 55], linestyle=":", linewidth=0.8, color=GRAY, zorder=1)
    ax.annotate("saturation ≈ 53–55%", xy=(4.3, 54),
                xytext=(3.55, 65), fontsize=6.6, color=GRAY,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6,
                                 shrinkA=0, shrinkB=0))

    ax.set_xlabel("Loop count $R$")
    ax.set_ylabel("Gap recovered (%)")
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.set_xlim(0.6, 6.6)
    ax.set_ylim(-5, 112)

    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    strip_spines(ax)
    ax.tick_params(direction="out")
    ax.grid(False)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig1_recovery.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Fig 2 -- storage cliff removal (log-x dot plot)
# =============================================================================
def fig2():
    fig, ax = plt.subplots(figsize=(5.0, 3.05))

    rows = [
        ("Llama-3.3-70B (dense, 42.5GB) — streamed", 0.02, VERMILLION, "o", 4),
        ("— folded R=2 (21GB resident)", 2.40, BLUE, "o", 3),
        ("— IQ2_XXS quantized (17.8GB) — streamed", 1.7, GRAY, "o", 2),
        ("Laguna-S-2.1 (118B-A8B MoE, 57.6GB) — streamed", 1.4, VERMILLION, "o", 1),
        ("— folded R=3 (17GB resident)", 2.37, BLUE, "o", 0),
    ]

    for label, xv, color, marker, yv in rows:
        ax.scatter([xv], [yv], color=color, marker=marker, s=42, zorder=3,
                   edgecolor="none")

    ax.set_xscale("log")
    ax.set_xlim(0.01, 6)
    ax.set_ylim(-0.8, 4.8)
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Decode speed (tokens/s, log scale)")

    # connecting arrows streamed -> folded, same model
    def connect(y0, x0, y1, x1, note, note_xy_frac=0.55, dy=0.0):
        arr = FancyArrowPatch((x0, y0), (x1, y1),
                               arrowstyle="-|>", mutation_scale=8,
                               color=GRAY, linewidth=0.8, shrinkA=6, shrinkB=6,
                               zorder=1, alpha=0.85)
        ax.add_patch(arr)
        # midpoint in log-x space
        xm = 10 ** (np.log10(x0) * (1 - note_xy_frac) + np.log10(x1) * note_xy_frac)
        ym = y0 * (1 - note_xy_frac) + y1 * note_xy_frac + dy
        ax.annotate(note, xy=(xm, ym), ha="center", va="center",
                    fontsize=7.2, color=GRAY,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85))

    connect(4, 0.02, 3, 2.40, "≈125×", note_xy_frac=0.5, dy=0.18)
    connect(1, 1.4, 0, 2.37, "≈1.7×", note_xy_frac=0.5, dy=0.18)

    # value labels at each point
    value_labels = {4: (0.02, "0.02"), 3: (2.40, "2.40 ± 0.10"),
                    2: (1.7, "1.7 (coherent)"), 1: (1.4, "1.4"),
                    0: (2.37, "2.37 ± 0.42")}
    label_dx = {4: (10, 4), 3: (0, 10), 2: (0, 10), 1: (0, 10), 0: (0, 10)}
    for yv, (xv, txt) in value_labels.items():
        dx, dy = label_dx[yv]
        ax.annotate(txt, (xv, yv), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if yv == 4 else "center", fontsize=7, color="black",
                    zorder=4)

    # annotation: memory-bound read volume near Llama streamed point
    ax.annotate("~44 GB read per token", xy=(0.02, 4), xytext=(0.013, 4.55),
                fontsize=6.6, color=GRAY, ha="left")

    strip_spines(ax)
    ax.tick_params(direction="out")
    ax.grid(axis="x", which="major", color="#dddddd", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig2_cliff.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Fig 3 -- MoE loop routing policy: quality vs touch cost
# =============================================================================
def fig3():
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    pts = {
        "baseline": (2.000, 1.7655, GRAY, "o", 30),
        "sticky": (2.000, 1.7129, BLUE, "o", 46),
        "re-route": (2.369, 1.7081, ORANGE, "o", 34),
        "upper bound (2×mem)": (2.000, 1.6596, GRAY, "s", 32),
    }

    for name, (xv, yv, color, marker, size) in pts.items():
        ax.scatter([xv], [yv], color=color, marker=marker, s=size,
                   zorder=4, edgecolor="none")

    # direct labels, offset to the left of the x=2.000 column to avoid overlap
    ax.annotate("baseline (no loop)\n1.7655", xy=(2.000, 1.7655),
                xytext=(-32, 6), textcoords="offset points",
                ha="right", va="center", fontsize=6.6, color="black")
    ax.annotate("sticky\n1.7129", xy=(2.000, 1.7129),
                xytext=(-32, -2), textcoords="offset points",
                ha="right", va="center", fontsize=6.8, color="black",
                fontweight="bold")
    ax.annotate("upper bound (2×mem)\n1.6596", xy=(2.000, 1.6596),
                xytext=(-32, -10), textcoords="offset points",
                ha="right", va="center", fontsize=6.6, color="black")
    ax.annotate("re-route\n1.7081", xy=(2.369, 1.7081),
                xytext=(8, 8), textcoords="offset points",
                ha="left", va="center", fontsize=6.6, color="black")

    # sticky -> re-route dotted connector + annotation
    ax.plot([2.000, 2.369], [1.7129, 1.7081], linestyle=":", linewidth=0.9,
            color=GRAY, zorder=2)
    ax.annotate("$\\Delta$loss +0.003 ± 0.004 (3 seeds): parity\n$\\Delta$touches +18.5%",
                xy=(2.19, 1.7105), xytext=(2.19, 1.7175),
                ha="center", va="bottom", fontsize=6.4, color=GRAY)

    # sticky -> baseline vertical arrow: loop gain at zero extra traffic
    ax.annotate("", xy=(2.000, 1.7135), xytext=(2.000, 1.7649),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.8,
                                 shrinkA=2, shrinkB=2))
    ax.annotate("loop gain 0.053 nats\nat zero extra traffic",
                xy=(2.045, 1.740), xytext=(2.045, 1.740),
                ha="left", va="center", fontsize=6.4, color=GRAY)

    ax.set_xlabel("Experts touched per (token, layer-slot)")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(1.9, 2.5)
    ax.set_ylim(1.648, 1.775)

    strip_spines(ax)
    ax.tick_params(direction="out")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig3_policy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# Fig 4 -- KV-context regime map: total resident (folded weights + KV cache)
# =============================================================================
def fig4():
    # Llama-3.3-70B: 80 layers, GQA kv_heads=8, head_dim=128, fp16 KV
    # KV/token = 2 * 8 * 128 * 80 * 2 bytes = 327680 bytes = 320 KiB/token
    KV_GIB_PER_TOK = 320.0 / (1024.0 ** 2)  # GiB per token, fp16

    WEIGHTS = {"R=2": 21.2, "R=3": 14.2, "R=4": 10.6}
    RAM_LINE = 24.0

    def total_gb(weight_gb, ctx_tok, kv_scale=1.0):
        return weight_gb + kv_scale * KV_GIB_PER_TOK * ctx_tok

    def crossing_ctx(weight_gb, kv_scale=1.0):
        budget = RAM_LINE - weight_gb
        return budget / (kv_scale * KV_GIB_PER_TOK)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    ctx = np.exp2(np.linspace(10, 16, 300))  # 1024 .. 65536 tokens

    series = [
        # key,        weight,           kv_scale, color,      ls,     label
        ("R2",        WEIGHTS["R=2"],   1.0,      BLUE,       "-",    "R=2"),
        ("R3",        WEIGHTS["R=3"],   1.0,      ORANGE,     "-",    "R=3"),
        ("R4",        WEIGHTS["R=4"],   1.0,      VERMILLION, "-",    "R=4"),
        ("R2_kvq8",   WEIGHTS["R=2"],   0.5,      BLUE,       "--",   "R=2+KV q8"),
    ]

    for key, weight, kv_scale, color, ls, label in series:
        y = total_gb(weight, ctx, kv_scale)
        ax.plot(ctx, y, linestyle=ls, color=color, linewidth=1.3, zorder=3,
                clip_on=True)

    # usable-RAM reference line
    ax.axhline(RAM_LINE, color=GRAY, linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    ax.annotate("usable RAM (32GB device)", xy=(1024, RAM_LINE),
                xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.6, color=GRAY)

    # crossing markers: max ctx each R sustains under the RAM line
    marker_label_offsets = {
        "R2":      (-4, 9),
        "R3":      (-15, -11),
        "R4":      (9, 5),
        "R2_kvq8": (4, 9),
    }
    for key, weight, kv_scale, color, ls, label in series:
        xc = crossing_ctx(weight, kv_scale)
        ax.plot([xc], [RAM_LINE], marker="o", markersize=4.2,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)
        dx, dy = marker_label_offsets[key]
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        ctx_k = xc / 1024.0
        ax.annotate(f"{ctx_k:.1f}k", xy=(xc, RAM_LINE),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=6.2, color=color, zorder=6)

    # direct curve labels, placed at distinct x to avoid clutter
    label_specs = [
        ("R4",      8192,  VERMILLION),
        ("R3",      11500, ORANGE),
        ("R2_kvq8", 41000, BLUE),
        ("R2",      50000, BLUE),
    ]
    label_text = {"R4": "R=4", "R3": "R=3", "R2_kvq8": "R=2+KV q8", "R2": "R=2"}
    label_offset = {"R4": (4, -9), "R3": (-4, 7), "R2_kvq8": (4, 6), "R2": (4, 4)}
    weight_of = {"R2": WEIGHTS["R=2"], "R3": WEIGHTS["R=3"], "R4": WEIGHTS["R=4"], "R2_kvq8": WEIGHTS["R=2"]}
    kvscale_of = {"R2": 1.0, "R3": 1.0, "R4": 1.0, "R2_kvq8": 0.5}
    for key, xpos, color in label_specs:
        yv = total_gb(weight_of[key], xpos, kvscale_of[key])
        dx, dy = label_offset[key]
        ax.annotate(label_text[key], xy=(xpos, yv), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, color=color,
                    fontweight="bold", ha="left", va="center", zorder=6)

    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k", "32k", "64k"])
    ax.set_xlim(1024, 65536)
    ax.set_ylim(0, 40)

    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Total resident (GB)")

    strip_spines(ax)
    ax.tick_params(direction="out")
    ax.grid(False)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "fig4_kvmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    p1 = fig1()
    p2 = fig2()
    p3 = fig3()
    p4 = fig4()
    for p in (p1, p2, p3, p4):
        print(p)
