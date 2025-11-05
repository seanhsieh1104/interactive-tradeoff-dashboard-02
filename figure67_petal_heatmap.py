# make_fig6_petals_and_heatmap.py
# Generates 7 figures for your IEEE manuscript:
#   - 6 "petal" composition charts (Option 3)
#   - 1 combined heatmap (Option 1)
#
# Requirements: matplotlib>=3.4, numpy

import os
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ---------------------------
# Synthetic scenarios (10 metrics)
# ---------------------------
METRICS = [
    "Error", "Coverage", "Latency", "Energy", "Battery Days",
    "EDP", "EEP", "Norm Error", "Stress Uplift", "Cloud Boost"
]

CASES = {
    "Battery-Dominant": np.array([0.05, 0.10, 0.05, 0.25, 0.20, 0.05, 0.05, 0.05, 0.05, 0.15]),
    "Latency-Sensitive": np.array([0.15, 0.10, 0.25, 0.10, 0.05, 0.15, 0.05, 0.05, 0.03, 0.07]),
    "Cloud-Assisted": np.array([0.10, 0.05, 0.10, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.35]),
    "Stress-Robust": np.array([0.20, 0.05, 0.10, 0.10, 0.05, 0.10, 0.05, 0.10, 0.20, 0.05]),
    "Throughput-Optimized": np.array([0.10, 0.25, 0.20, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.10]),
    "Balanced-Clinical": np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
}

# ---------------------------
# Helpers
# ---------------------------
def ensure_outdir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def make_petal(case_name: str, values: np.ndarray, out_path: str):
    """
    Petal composition (radial bars). Bar length encodes weighted contribution.
    One figure per scenario.
    """
    vals = values.astype(float)
    vmax = vals.max() if np.isfinite(vals.max()) and vals.max() > 0 else 1.0
    vals_norm = vals / vmax  # scale for display only (relative petals)

    N = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)

    # orient 12 o'clock at top, clockwise
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # category labels around the circle
    ax.set_xticks(angles)
    ax.set_xticklabels(METRICS, fontsize=8)

    # hide radial tick labels (purely relative lengths)
    ax.set_yticklabels([])

    # title
    #ax.set_title(f"Fig. 5b - {case_name}\nMetric-Contribution Petal Composition", fontsize=12, pad=18)

    # draw petals (let Matplotlib default color cycle handle colors)
    bar_width = (2 * np.pi / N) * 0.8
    for ang, v in zip(angles, vals_norm):
        ax.bar(ang, v, width=bar_width, alpha=0.85, edgecolor='k', linewidth=0.4)

    # --- NEW: percentage labels ---
    ax.set_ylim(0, 1.25)  # headroom for labels
    for ang, v_norm, v_true in zip(angles, vals_norm, vals):
        r_text = min(v_norm + 0.06, 1.2)     # label radius just above the bar
        label = f"{v_true*100:.0f}%"         # show 0-decimal %; switch to .1f if you prefer
        if 0 < v_true*100 < 1:
            label = f"{v_true*100:.1f}%"
        ax.text(ang, r_text, label, ha='center', va='bottom', fontsize=8)
    # --- end NEW ---

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def make_combined_heatmap(cases: dict, out_path: str):
    """
    Combined heatmap (metrics x scenarios) using matplotlib only.
    Color shows weighted contribution fraction (no specific cmap set).
    """
    scenario_names = list(cases.keys())
    data = np.stack([cases[name] for name in scenario_names], axis=1)  # shape: (metrics, scenarios)

    fig, ax = plt.subplots(figsize=(8, 6))
    #im = ax.imshow(data, aspect='auto')  # default colormap
    
    # Use a lighter colormap such as 'YlGnBu' or 'PuBuGn'
    im = ax.imshow(data, aspect='auto', cmap='YlGnBu', vmin=0, vmax=0.35)


    # axes labels/ticks
    ax.set_xticks(np.arange(len(scenario_names)))
    ax.set_xticklabels(scenario_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(METRICS)))
    ax.set_yticklabels(METRICS, fontsize=9)

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted Contribution", rotation=90)

    # optional annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    #ax.set_title("Fig. 5b - Combined\nMetric-Contribution Heatmap Summary (All Six Scenarios)", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(out_dir: str = "outputs/figs_petal_heatmap"):
    ensure_outdir(out_dir)

    # Generate 6 petal figures
    ordered_cases = [
        "Battery-Dominant",
        "Latency-Sensitive",
        "Cloud-Assisted",
        "Stress-Robust",
        "Throughput-Optimized",
        "Balanced-Clinical",
    ]
    for idx, case in enumerate(ordered_cases, start=1):
        out_path = os.path.join(out_dir, f"Fig6_Petal_{idx}_{case.replace('-', '')}.png")
        make_petal(case, CASES[case], out_path)

    # Generate combined heatmap
    heatmap_out = os.path.join(out_dir, "Fig7_Combined_Heatmap_AllSixScenarios.png")
    make_combined_heatmap({k: CASES[k] for k in ordered_cases}, heatmap_out)

    print("Done. Saved to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
