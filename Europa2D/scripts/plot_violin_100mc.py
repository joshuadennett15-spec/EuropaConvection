"""Latitude-band violin plots for 100-iteration MC results."""
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

EQ_BAND = (0.0, 30.0)
MID_BAND = (30.0, 60.0)
POLAR_BAND = (60.0, 90.0)
N_ITER = 100

SCENARIOS = [
    ("uniform_transport", "Uniform"),
    ("soderlund2014_equator", "Equator-\nenhanced"),
    ("lemasquerier2023_polar", "Polar-\nenhanced"),
]

C_EQ = PAL.CYAN
C_MID = PAL.PURPLE
C_POLAR = PAL.ORANGE


def load_bands(name, key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    d = np.load(path, allow_pickle=True)
    lats = d["latitudes_deg"]
    profiles = d[key]
    eq = band_mean_samples(lats, profiles, EQ_BAND)
    mid = band_mean_samples(lats, profiles, MID_BAND)
    polar = band_mean_samples(lats, profiles, POLAR_BAND)
    return eq, mid, polar


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

    for i, d in enumerate(datasets):
        med = np.median(d)
        q25, q75 = np.percentile(d, [25, 75])
        x = positions[i]
        dx = 0.18
        if side == "left":
            ax.hlines(med, x - dx, x, color="k", lw=1.3, zorder=5)
            ax.hlines([q25, q75], x - dx * 0.6, x, color="k", lw=0.5, ls="--", zorder=5)
        else:
            ax.hlines(med, x, x + dx, color="k", lw=1.3, zorder=5)
            ax.hlines([q25, q75], x, x + dx * 0.6, color="k", lw=0.5, ls="--", zorder=5)


def draw_grouped_violins(ax, eq_list, mid_list, polar_list, scenario_labels):
    """Draw three side-by-side violins per scenario (eq, mid, polar)."""
    n_scen = len(scenario_labels)
    group_width = 0.75  # total width per scenario group
    vw = group_width / 3  # width per violin
    offsets = np.array([-vw, 0, vw])

    for i in range(n_scen):
        for data, offset, color in [
            (eq_list[i], offsets[0], C_EQ),
            (mid_list[i], offsets[1], C_MID),
            (polar_list[i], offsets[2], C_POLAR),
        ]:
            pos = [i + offset]
            parts = ax.violinplot(
                [data], positions=pos, widths=vw * 0.9,
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("k")
                body.set_linewidth(0.4)
                body.set_alpha(0.6)

            # Median + IQR
            med = np.median(data)
            q25, q75 = np.percentile(data, [25, 75])
            hw = vw * 0.3
            ax.hlines(med, pos[0] - hw, pos[0] + hw, color="k", lw=1.2, zorder=5)
            ax.hlines([q25, q75], pos[0] - hw * 0.6, pos[0] + hw * 0.6,
                      color="k", lw=0.5, ls="--", zorder=5)


def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.25)

    panels = [
        (ax1, "D_cond_profiles", r"D$_{\rm cond}$ (km)", "Conductive lid thickness", "a"),
        (ax2, "H_profiles", r"H$_{\rm total}$ (km)", "Total shell thickness", "b"),
    ]

    scenario_labels = [label for _, label in SCENARIOS]

    for ax, key, ylabel, title_str, letter in panels:
        eq_data, mid_data, polar_data = [], [], []
        for name, _ in SCENARIOS:
            eq, mid, polar = load_bands(name, key)
            eq_data.append(eq)
            mid_data.append(mid)
            polar_data.append(polar)

        draw_grouped_violins(ax, eq_data, mid_data, polar_data, scenario_labels)

        if key == "D_cond_profiles":
            ax.axhspan(19.0, 39.0, color=PAL.GREEN, alpha=0.10, zorder=0)
            ax.axhline(29.0, color=PAL.GREEN, lw=0.8, ls="--", zorder=0)
            ax.text(2.7, 30.5, "Juno MWR: 29 +/- 10 km",
                    fontsize=6.5, color=PAL.GREEN, ha="right")

        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels(scenario_labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title_str, fontsize=9, fontweight="bold")
        ax.set_ylim(0, None)
        label_panel(ax, letter)

    eq_patch = mpatches.Patch(
        facecolor=C_EQ, alpha=0.6, edgecolor="k", lw=0.5,
        label="Equatorial (0-30 deg)",
    )
    mid_patch = mpatches.Patch(
        facecolor=C_MID, alpha=0.6, edgecolor="k", lw=0.5,
        label="Mid-latitude (30-60 deg)",
    )
    polar_patch = mpatches.Patch(
        facecolor=C_POLAR, alpha=0.6, edgecolor="k", lw=0.5,
        label="Polar (60-90 deg)",
    )
    fig.legend(
        handles=[eq_patch, mid_patch, polar_patch], loc="lower center",
        ncol=3, fontsize=7.5, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Shell structure by ocean transport scenario  |  100 MC samples, 19 columns\n"
        "Area-weighted band means: equatorial (0-30), mid-latitude (30-60), polar (60-90 deg)",
        fontsize=9, fontweight="bold", y=1.05,
    )

    save_path = os.path.join(FIGURES_DIR, "violin_dcond_htotal_100mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
