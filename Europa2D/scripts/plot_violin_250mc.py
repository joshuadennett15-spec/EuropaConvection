"""
D_cond violin plot: equatorial (0-30 deg) vs polar (60-90 deg).
500 MC samples, global grain mode, split violins with strip overlay.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from profile_diagnostics import band_mean_samples
from pub_style import apply_style, PAL, label_panel

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 500

EQ_BAND = (0.0, 30.0)
POLAR_BAND = (60.0, 90.0)

JUNO_D_COND = 29.0
JUNO_ERR = 10.0

SCENARIOS = [
    ("uniform_transport", "Uniform"),
    ("soderlund2014_equator", "Equator-\nenhanced"),
    ("lemasquerier2023_polar", "Polar-\nenhanced"),
]

C_EQ = PAL.CYAN
C_POLAR = PAL.ORANGE


MAX_PHYSICAL_D_COND = 150.0  # km — reject numerical blowups


def load_bands(name):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    d = np.load(path, allow_pickle=True)
    lats = d["latitudes_deg"]
    Dc = d["D_cond_profiles"]
    eq = band_mean_samples(lats, Dc, EQ_BAND)
    polar = band_mean_samples(lats, Dc, POLAR_BAND)
    mask = (eq < MAX_PHYSICAL_D_COND) & (polar < MAX_PHYSICAL_D_COND)
    return eq[mask], polar[mask]


def draw_half_violin(ax, datasets, positions, side, color, alpha=0.6):
    parts = ax.violinplot(
        datasets, positions=positions, widths=0.7,
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
        body.set_linewidth(0.5)
        body.set_alpha(alpha)

    rng = np.random.default_rng(42)
    for i, d in enumerate(datasets):
        med = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        x = positions[i]
        dx = 0.18

        # Strip overlay — individual MC samples as visible dots
        jitter = rng.uniform(0.02, dx * 0.95, len(d))
        if side == "left":
            ax.scatter(x - jitter, d,
                       s=10, color=color, alpha=0.45, edgecolors="k",
                       linewidths=0.15, zorder=3, rasterized=True)
            ax.hlines(med, x - dx, x, color="k", lw=1.5, zorder=6)
            ax.hlines([q25, q75], x - dx * 0.6, x, color="k", lw=0.6, ls="--", zorder=6)
            ax.text(x - dx - 0.04, med, f"{med:.0f}",
                    ha="right", va="center", fontsize=6.5, fontweight="bold", zorder=7)
        else:
            ax.scatter(x + jitter, d,
                       s=10, color=color, alpha=0.45, edgecolors="k",
                       linewidths=0.15, zorder=3, rasterized=True)
            ax.hlines(med, x, x + dx, color="k", lw=1.5, zorder=6)
            ax.hlines([q25, q75], x, x + dx * 0.6, color="k", lw=0.6, ls="--", zorder=6)
            ax.text(x + dx + 0.04, med, f"{med:.0f}",
                    ha="left", va="center", fontsize=6.5, fontweight="bold", zorder=7)


def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    positions = np.arange(len(SCENARIOS))
    eq_data, polar_data = [], []
    for name, _ in SCENARIOS:
        eq, polar = load_bands(name)
        eq_data.append(eq)
        polar_data.append(polar)

    draw_half_violin(ax, eq_data, positions, "left", C_EQ)
    draw_half_violin(ax, polar_data, positions, "right", C_POLAR)

    # Delta-H brackets
    for i in range(len(SCENARIOS)):
        eq_med = np.median(eq_data[i])
        polar_med = np.median(polar_data[i])
        x_bracket = positions[i] + 0.30
        ax.annotate(
            "", xy=(x_bracket, polar_med), xytext=(x_bracket, eq_med),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8),
        )
        dH = polar_med - eq_med
        ax.text(x_bracket + 0.05, (eq_med + polar_med) / 2,
                f"{dH:+.1f} km", fontsize=6, color="0.3", va="center", ha="left")

    # Juno constraint
    ax.axhspan(JUNO_D_COND - JUNO_ERR, JUNO_D_COND + JUNO_ERR,
               color=PAL.GREEN, alpha=0.08, zorder=0)
    ax.axhline(JUNO_D_COND, color=PAL.GREEN, lw=0.7, ls="--", zorder=0)
    ax.text(
        len(SCENARIOS) - 0.5, JUNO_D_COND + 1.0,
        "Juno MWR: 29 +/- 10 km",
        fontsize=6.5, color=PAL.GREEN, ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in SCENARIOS], fontsize=9)
    ax.set_ylabel(r"D$_{\rm cond}$ (km)", fontsize=11)
    ax.set_ylim(0, None)

    ax.set_title(
        "Conductive lid thickness by ocean transport scenario\n"
        "500 MC samples, area-weighted band means, global grain mode",
        fontsize=9, fontweight="bold",
    )

    eq_patch = mpatches.Patch(facecolor=C_EQ, alpha=0.6, edgecolor="k", lw=0.5,
                               label="Equatorial (0-30 deg)")
    polar_patch = mpatches.Patch(facecolor=C_POLAR, alpha=0.6, edgecolor="k", lw=0.5,
                                  label="Polar (60-90 deg)")
    ax.legend(handles=[eq_patch, polar_patch], loc="upper left", fontsize=8)

    save_path = os.path.join(FIGURES_DIR, "violin_dcond_500mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
