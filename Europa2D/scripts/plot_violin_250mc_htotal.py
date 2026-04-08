"""
Total shell thickness violin plot: equatorial (0-30 deg) vs polar (60-90 deg).
500 MC samples, global grain mode, clean split violins.
"""
import os
import sys

import numpy as np
import matplotlib as mpl

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from profile_diagnostics import band_mean_samples
from pub_style import PAL

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 500

EQ_BAND = (0.0, 30.0)
POLAR_BAND = (60.0, 90.0)

SCENARIOS = [
    ("uniform_transport", "Uniform"),
    ("soderlund2014_equator", r"Equator-enhanced"),
    ("lemasquerier2023_polar", r"Polar-enhanced"),
]

C_EQ = PAL.CYAN
C_POLAR = PAL.ORANGE

MAX_PHYSICAL_H = 200.0  # km


def load_bands(name):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    d = np.load(path, allow_pickle=True)
    lats = d["latitudes_deg"]
    h_profiles = d["H_profiles"]
    eq = band_mean_samples(lats, h_profiles, EQ_BAND)
    polar = band_mean_samples(lats, h_profiles, POLAR_BAND)
    mask = (eq < MAX_PHYSICAL_H) & (polar < MAX_PHYSICAL_H)
    return eq[mask], polar[mask]


def draw_half_violin(ax, datasets, positions, side, color):
    """Draw clean half-violins with median/IQR marks, no scatter dots."""
    parts = ax.violinplot(
        datasets, positions=positions, widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        m = np.mean(verts[:, 0])
        if side == "left":
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, m)
        else:
            verts[:, 0] = np.clip(verts[:, 0], m, np.inf)
        body.set_facecolor(color)
        body.set_edgecolor("k")
        body.set_linewidth(0.6)
        body.set_alpha(0.55)

    for i, d in enumerate(datasets):
        med = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        x = positions[i]
        dx = 0.20

        if side == "left":
            # Median bar
            ax.hlines(med, x - dx, x, color="k", lw=2.0, zorder=6)
            # IQR bars
            ax.hlines([q25, q75], x - dx * 0.55, x,
                      color="k", lw=0.7, ls="--", alpha=0.7, zorder=6)
            # Thin line connecting IQR
            ax.vlines(x - dx * 0.27, q25, q75,
                      color="k", lw=0.6, alpha=0.4, zorder=5)
            # Label
            ax.text(
                x - dx - 0.06, med, f"{med:.0f}",
                ha="right", va="center", fontsize=8, fontweight="bold", zorder=7,
            )
        else:
            ax.hlines(med, x, x + dx, color="k", lw=2.0, zorder=6)
            ax.hlines([q25, q75], x, x + dx * 0.55,
                      color="k", lw=0.7, ls="--", alpha=0.7, zorder=6)
            ax.vlines(x + dx * 0.27, q25, q75,
                      color="k", lw=0.6, alpha=0.4, zorder=5)
            ax.text(
                x + dx + 0.04, med, f"{med:.0f}",
                ha="left", va="center", fontsize=8, fontweight="bold", zorder=7,
            )


def main():
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    positions = np.arange(len(SCENARIOS))
    eq_data, polar_data = [], []
    for name, _ in SCENARIOS:
        eq, polar = load_bands(name)
        eq_data.append(eq)
        polar_data.append(polar)

    draw_half_violin(ax, eq_data, positions, "left", C_EQ)
    draw_half_violin(ax, polar_data, positions, "right", C_POLAR)

    # Delta-H labels centred below each scenario
    for i in range(len(SCENARIOS)):
        eq_med = np.median(eq_data[i])
        polar_med = np.median(polar_data[i])
        dH = polar_med - eq_med
        ax.text(
            positions[i], -6.0,
            rf"$\Delta = {dH:+.1f}$ km",
            fontsize=8, color="0.25", va="top", ha="center",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in SCENARIOS])
    ax.set_ylabel(r"Total shell thickness $H_{\mathrm{total}}$ (km)")
    ax.set_ylim(-1, None)
    ax.set_clip_on(False)

    ax.set_title(
        r"\textbf{Total shell thickness by ocean transport scenario}"
        "\n"
        r"{\small 500 MC samples $\cdot$ area-weighted band means $\cdot$ global grain mode}",
    )

    eq_patch = mpatches.Patch(
        facecolor=C_EQ, alpha=0.55, edgecolor="k", lw=0.5,
        label=r"Equatorial ($0$--$30^{\circ}$)",
    )
    polar_patch = mpatches.Patch(
        facecolor=C_POLAR, alpha=0.55, edgecolor="k", lw=0.5,
        label=r"Polar ($60$--$90^{\circ}$)",
    )
    ax.legend(handles=[eq_patch, polar_patch], loc="upper right",
              frameon=True, framealpha=0.85, edgecolor="0.8", fancybox=False)

    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "violin_htotal_500mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight",
                transparent=True)
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
