"""
Simple 3-line profile plot: H_total, D_cond, D_conv vs latitude.
Each quantity gets its own line with uncertainty band. No stacking, no inversion.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from pub_style import apply_style, PAL, label_panel

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 250

SCENARIOS = [
    ("uniform_transport", "Uniform transport", "Ashkenazy & Tziperman (2021)"),
    ("soderlund2014_equator", "Equator-enhanced", "Soderlund et al. (2014)"),
    ("lemasquerier2023_polar", "Polar-enhanced", "Lemasquerier et al. (2023)"),
]


def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, (name, title, citation) in enumerate(SCENARIOS):
        ax = axes[idx]
        path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
        d = dict(np.load(path, allow_pickle=True))
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Dv = d["D_conv_profiles"]

        quantities = [
            (H,  PAL.BLACK,  1.8, r"H$_{\rm total}$"),
            (Dc, PAL.BLUE,   1.8, r"D$_{\rm cond}$"),
            (Dv, PAL.ORANGE, 1.5, r"D$_{\rm conv}$"),
        ]

        for arr, color, lw, label in quantities:
            med = np.median(arr, axis=0)
            lo = np.percentile(arr, 16, axis=0)
            hi = np.percentile(arr, 84, axis=0)
            ax.fill_between(lat, lo, hi, color=color, alpha=0.15)
            ax.plot(lat, med, color=color, lw=lw, label=label)

        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_xlabel("Latitude (deg)", fontsize=8)
        ax.set_title(f"{title}\n{citation}", fontsize=7.5, loc="left")
        label_panel(ax, chr(97 + idx))

        # Stats annotation
        dH = np.median(H, axis=0)[-1] - np.median(H, axis=0)[0]
        ax.text(
            0.97, 0.97, f"dH = {dH:+.1f} km\nN = {len(H)}",
            transform=ax.transAxes, fontsize=6, va="top", ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"),
        )

    axes[0].set_ylabel("Thickness (km)", fontsize=10)
    axes[0].legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Europa ice shell structure: 250 MC samples, 19 columns",
        fontsize=10, fontweight="bold", y=1.02,
    )

    save_path = os.path.join(FIGURES_DIR, "profiles_simple_250mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
