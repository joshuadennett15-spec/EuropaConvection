"""
Combined figure for 250-iteration MC results:
  - Top row (a,b,c): Cross-section profiles (D_cond + D_conv stacked) per scenario
  - Bottom row (d): Violin plot of D_cond across all three scenarios with strip overlay

Uses area-weighted band means for violins, full latitude profiles for cross-sections.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from profile_diagnostics import band_mean_samples
from pub_style import apply_style, PAL, label_panel

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")

N_ITER = 250

# Band definitions
EQ_BAND = (0.0, 30.0)
MID_BAND = (30.0, 60.0)
POLAR_BAND = (60.0, 90.0)

# Juno
JUNO_D_COND = 29.0
JUNO_ERR = 10.0

SCENARIOS = [
    ("uniform_transport", "Uniform transport", "Ashkenazy & Tziperman (2021)"),
    ("soderlund2014_equator", "Equator-enhanced", "Soderlund et al. (2014)"),
    ("lemasquerier2023_polar", "Polar-enhanced", "Lemasquerier et al. (2023)"),
]

# Cross-section colours
C_LID = "#4A90D9"
C_CONV = "#E8645A"
C_LID_DARK = "#2C5F99"
C_CONV_DARK = "#B83B31"
C_TOTAL = "#333333"

# Violin colours
C_EQ = PAL.CYAN
C_MID = PAL.PURPLE
C_POLAR = PAL.ORANGE


def pct_band(arr, lo=16, hi=84):
    return (
        np.median(arr, axis=0),
        np.percentile(arr, lo, axis=0),
        np.percentile(arr, hi, axis=0),
    )


def load_data(name):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


# =========================================================================
# TOP ROW: Cross-section panels
# =========================================================================
def plot_cross_section(ax, name, title, citation, letter):
    d = load_data(name)
    lat = d["latitudes_deg"]
    H = d["H_profiles"]
    Dc = d["D_cond_profiles"]
    n_valid = len(H)

    H_med, H_lo, H_hi = pct_band(H)
    Dc_med, Dc_lo, Dc_hi = pct_band(Dc)

    # Conductive lid band + line
    ax.fill_between(lat, Dc_lo, Dc_hi, color=C_LID, alpha=0.25, zorder=2)
    ax.plot(lat, Dc_med, color=C_LID_DARK, lw=1.8, zorder=4)

    # Convective sublayer fill
    ax.fill_between(lat, Dc_med, H_med, color=C_CONV, alpha=0.30, zorder=2)

    # Total shell band + line
    ax.fill_between(lat, H_lo, H_hi, color=C_TOTAL, alpha=0.10, zorder=1)
    ax.plot(lat, H_med, color=C_TOTAL, lw=1.8, zorder=4)

    # Surface
    ax.axhline(0, color="k", lw=0.8, zorder=5)

    # Stats box
    eq_H = H_med[0]
    pole_H = H_med[-1]
    eq_Dc = Dc_med[0]
    pole_Dc = Dc_med[-1]
    dH = pole_H - eq_H
    stats = (
        f"N = {n_valid}\n"
        f"Eq:   H={eq_H:.1f}, Dc={eq_Dc:.1f}\n"
        f"Pole: H={pole_H:.1f}, Dc={pole_Dc:.1f}\n"
        f"dH = {dH:+.1f} km"
    )
    ax.text(
        0.97, 0.03, stats,
        transform=ax.transAxes, fontsize=6,
        va="bottom", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"),
    )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("Latitude (deg)", fontsize=8)
    ax.invert_yaxis()
    label_panel(ax, letter)
    ax.set_title(f"{title}\n{citation}", fontsize=7.5, loc="left")


# =========================================================================
# BOTTOM ROW: Violin plot with strip overlay
# =========================================================================
def draw_violin_group(ax, datasets, positions, color, width=0.22):
    """Draw violins with jittered strip overlay."""
    for i, data in enumerate(datasets):
        pos = positions[i]

        # Violin body
        parts = ax.violinplot(
            [data], positions=[pos], widths=width * 0.85,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("k")
            body.set_linewidth(0.4)
            body.set_alpha(0.45)

        # Strip overlay (jittered dots)
        jitter = np.random.default_rng(42 + i).uniform(-width * 0.25, width * 0.25, len(data))
        ax.scatter(
            pos + jitter, data,
            s=3, color=color, alpha=0.35, edgecolors="none", zorder=3,
        )

        # Median + IQR
        med = np.median(data)
        q25, q75 = np.percentile(data, [25, 75])
        hw = width * 0.35
        ax.hlines(med, pos - hw, pos + hw, color="k", lw=1.5, zorder=6)
        ax.hlines([q25, q75], pos - hw * 0.7, pos + hw * 0.7,
                  color="k", lw=0.6, ls="--", zorder=6)

        # Annotate median value
        ax.text(pos, med - 1.2, f"{med:.0f}",
                ha="center", va="top", fontsize=5.5, fontweight="bold", zorder=7)


def plot_violins(ax):
    """D_cond violin plot with three latitude bands per scenario."""
    n_scen = len(SCENARIOS)
    group_centres = np.arange(n_scen) * 1.0
    vw = 0.25
    offsets = np.array([-vw, 0, vw])

    eq_data, mid_data, polar_data = [], [], []
    for name, _, _ in SCENARIOS:
        d = load_data(name)
        lats = d["latitudes_deg"]
        Dc = d["D_cond_profiles"]
        eq_data.append(band_mean_samples(lats, Dc, EQ_BAND))
        mid_data.append(band_mean_samples(lats, Dc, MID_BAND))
        polar_data.append(band_mean_samples(lats, Dc, POLAR_BAND))

    draw_violin_group(ax, eq_data, group_centres + offsets[0], C_EQ, width=vw)
    draw_violin_group(ax, mid_data, group_centres + offsets[1], C_MID, width=vw)
    draw_violin_group(ax, polar_data, group_centres + offsets[2], C_POLAR, width=vw)

    # Juno constraint
    ax.axhspan(JUNO_D_COND - JUNO_ERR, JUNO_D_COND + JUNO_ERR,
               color=PAL.GREEN, alpha=0.08, zorder=0)
    ax.axhline(JUNO_D_COND, color=PAL.GREEN, lw=0.7, ls="--", zorder=0)
    ax.text(
        n_scen * 1.0 - 0.55, JUNO_D_COND + 0.8,
        "Juno MWR: 29 +/- 10 km",
        fontsize=6, color=PAL.GREEN, ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
    )

    # Delta-H brackets between eq and polar medians
    for i, (name, label, _) in enumerate(SCENARIOS):
        eq_med = np.median(eq_data[i])
        polar_med = np.median(polar_data[i])
        x_bracket = group_centres[i] + vw + 0.12
        ax.annotate(
            "", xy=(x_bracket, polar_med), xytext=(x_bracket, eq_med),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8),
        )
        dH = polar_med - eq_med
        ax.text(x_bracket + 0.04, (eq_med + polar_med) / 2, f"{dH:+.0f}",
                fontsize=5.5, color="0.3", va="center", ha="left")

    ax.set_xticks(group_centres)
    ax.set_xticklabels(["Uniform", "Equator-\nenhanced", "Polar-\nenhanced"], fontsize=8)
    ax.set_ylabel(r"D$_{\rm cond}$ (km)", fontsize=10)
    ax.set_ylim(0, None)
    ax.set_title("Conductive lid thickness by latitude band", fontsize=9, fontweight="bold")
    label_panel(ax, "d")


# =========================================================================
# MAIN
# =========================================================================
def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1, 1.1],
        hspace=0.38, wspace=0.22,
    )

    # Top row: cross-section panels
    letters = ["a", "b", "c"]
    for idx, (name, title, citation) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[0, idx])
        plot_cross_section(ax, name, title, citation, letters[idx])
        if idx == 0:
            ax.set_ylabel("Depth from surface (km)", fontsize=8)

    # Shared cross-section legend
    cs_handles = [
        mpatches.Patch(facecolor=C_LID, alpha=0.5, edgecolor=C_LID_DARK,
                       label=r"D$_{\rm cond}$ (lid)"),
        mpatches.Patch(facecolor=C_CONV, alpha=0.5, edgecolor=C_CONV_DARK,
                       label=r"D$_{\rm conv}$ (sublayer)"),
        mpatches.Patch(facecolor=C_TOTAL, alpha=0.3, edgecolor=C_TOTAL,
                       label=r"H$_{\rm total}$ (16-84th pctl)"),
    ]
    fig.legend(
        handles=cs_handles, loc="upper center",
        ncol=3, fontsize=7, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.52),
    )

    # Bottom row: violin panel spanning all three columns
    ax_violin = fig.add_subplot(gs[1, :])
    plot_violins(ax_violin)

    # Violin legend
    v_handles = [
        mpatches.Patch(facecolor=C_EQ, alpha=0.5, edgecolor="k", lw=0.5,
                       label="Equatorial (0-30 deg)"),
        mpatches.Patch(facecolor=C_MID, alpha=0.5, edgecolor="k", lw=0.5,
                       label="Mid-latitude (30-60 deg)"),
        mpatches.Patch(facecolor=C_POLAR, alpha=0.5, edgecolor="k", lw=0.5,
                       label="Polar (60-90 deg)"),
    ]
    ax_violin.legend(
        handles=v_handles, loc="upper left",
        fontsize=7, frameon=True, framealpha=0.9,
    )

    fig.suptitle(
        "Europa ice shell structure: 250 MC samples, 19 latitude columns\n"
        "Andrade rheology, audited priors, q_tidal_scale = 1.20",
        fontsize=10, fontweight="bold", y=0.98,
    )

    save_path = os.path.join(FIGURES_DIR, "combined_250mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
