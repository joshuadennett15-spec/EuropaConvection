"""
Side-by-side comparison: global vs strain grain mode.
Top row: cross-section profiles (D_cond + D_conv stacked, depth from surface)
         global = solid, strain = dashed overlay.
Bottom row: D_cond violin plot, global vs strain.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from profile_diagnostics import band_mean_samples
from pub_style import apply_style, PAL, label_panel

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
STRAIN_DIR = os.path.join(RESULTS_DIR, "strain_mode")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 250

SCENARIOS = [
    ("uniform_transport", "Uniform transport", "Ashkenazy & Tziperman (2021)"),
    ("soderlund2014_equator", "Equator-enhanced", "Soderlund et al. (2014)"),
    ("lemasquerier2023_polar", "Polar-enhanced", "Lemasquerier et al. (2023)"),
]

EQ_BAND = (0.0, 30.0)
MID_BAND = (30.0, 60.0)
POLAR_BAND = (60.0, 90.0)

JUNO_D_COND = 29.0
JUNO_ERR = 10.0

# Cross-section colours
C_LID = "#4A90D9"
C_CONV = "#E8645A"
C_LID_DARK = "#2C5F99"
C_CONV_DARK = "#B83B31"
C_TOTAL = "#333333"

# Strain overlay colours (darker/shifted versions)
C_LID_STRAIN = "#1B4F72"
C_CONV_STRAIN = "#922B21"
C_TOTAL_STRAIN = "#111111"


def load(name, mode):
    if mode == "global":
        path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    else:
        path = os.path.join(STRAIN_DIR, f"mc_2d_{name}_{N_ITER}_strain.npz")
    return dict(np.load(path, allow_pickle=True))


def pct(arr, lo=16, hi=84):
    return np.median(arr, axis=0), np.percentile(arr, lo, axis=0), np.percentile(arr, hi, axis=0)


def plot_cross_section(ax, name, title, citation, letter):
    """Stacked cross-section: global as filled bands, strain as dashed overlay."""

    # --- Global (filled bands + solid lines) ---
    d = load(name, "global")
    lat = d["latitudes_deg"]
    H = d["H_profiles"]
    Dc = d["D_cond_profiles"]
    n_valid_g = len(H)

    H_med, H_lo, H_hi = pct(H)
    Dc_med, Dc_lo, Dc_hi = pct(Dc)

    # Conductive lid band
    ax.fill_between(lat, Dc_lo, Dc_hi, color=C_LID, alpha=0.25, zorder=2)
    ax.plot(lat, Dc_med, color=C_LID_DARK, lw=1.8, zorder=4,
            label=r"D$_{\rm cond}$ base (lid)")

    # Convective sublayer fill
    ax.fill_between(lat, Dc_med, H_med, color=C_CONV, alpha=0.30, zorder=2)

    # Total shell band + line
    ax.fill_between(lat, H_lo, H_hi, color=C_TOTAL, alpha=0.10, zorder=1)
    ax.plot(lat, H_med, color=C_TOTAL, lw=1.8, zorder=4,
            label=r"H$_{\rm total}$ (ice-ocean)")

    # Surface
    ax.axhline(0, color="k", lw=0.8, zorder=5)

    # --- Strain (dashed overlay, no fill) ---
    d_s = load(name, "strain")
    H_s = d_s["H_profiles"]
    Dc_s = d_s["D_cond_profiles"]
    n_valid_s = len(H_s)

    H_s_med = np.median(H_s, axis=0)
    Dc_s_med = np.median(Dc_s, axis=0)

    ax.plot(lat, Dc_s_med, color=C_LID_STRAIN, lw=1.5, ls="--", zorder=6,
            label=r"D$_{\rm cond}$ (strain)")
    ax.plot(lat, H_s_med, color=C_TOTAL_STRAIN, lw=1.5, ls="--", zorder=6,
            label=r"H$_{\rm total}$ (strain)")

    # Stats box
    eq_H_g = H_med[0]
    pole_H_g = H_med[-1]
    dH_g = pole_H_g - eq_H_g

    eq_H_s = H_s_med[0]
    pole_H_s = H_s_med[-1]
    dH_s = pole_H_s - eq_H_s

    stats = (
        f"N = {n_valid_g} / {n_valid_s}\n"
        f"dH global: {dH_g:+.1f} km\n"
        f"dH strain: {dH_s:+.1f} km\n"
        f"diff: {dH_s - dH_g:+.1f} km"
    )
    ax.text(
        0.97, 0.03, stats,
        transform=ax.transAxes, fontsize=5.5,
        va="bottom", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"),
    )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("Latitude (deg)", fontsize=8)
    ax.invert_yaxis()
    label_panel(ax, letter)
    ax.set_title(f"{title}\n{citation}", fontsize=7.5, loc="left")


def plot_violins(ax):
    """D_cond violins: global (blue) vs strain (red), three bands per scenario."""

    c_global = PAL.BLUE
    c_strain = PAL.RED

    n_scen = len(SCENARIOS)
    group_centres = np.arange(n_scen) * 1.0

    for mode, color, x_offset in [
        ("global", c_global, -0.15),
        ("strain", c_strain, +0.15),
    ]:
        for i, (name, _, _) in enumerate(SCENARIOS):
            d = load(name, mode)
            lats = d["latitudes_deg"]
            Dc = d["D_cond_profiles"]

            eq = band_mean_samples(lats, Dc, EQ_BAND)
            mid = band_mean_samples(lats, Dc, MID_BAND)
            polar = band_mean_samples(lats, Dc, POLAR_BAND)

            for band_data, band_offset, band_alpha in [
                (eq, -0.08, 0.55),
                (mid, 0.0, 0.35),
                (polar, 0.08, 0.55),
            ]:
                pos = group_centres[i] + x_offset + band_offset
                parts = ax.violinplot(
                    [band_data], positions=[pos], widths=0.07,
                    showmeans=False, showmedians=False, showextrema=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor("k")
                    body.set_linewidth(0.3)
                    body.set_alpha(band_alpha)

                med = np.median(band_data)
                hw = 0.025
                ax.hlines(med, pos - hw, pos + hw, color="k", lw=1.0, zorder=5)

                # Annotate median
                ax.text(pos, med - 1.0, f"{med:.0f}",
                        ha="center", va="top", fontsize=4.5, fontweight="bold", zorder=7)

    # Juno
    ax.axhspan(JUNO_D_COND - JUNO_ERR, JUNO_D_COND + JUNO_ERR,
               color=PAL.GREEN, alpha=0.08, zorder=0)
    ax.axhline(JUNO_D_COND, color=PAL.GREEN, lw=0.7, ls="--", zorder=0)
    ax.text(
        n_scen - 0.55, JUNO_D_COND + 0.8, "Juno MWR: 29 +/- 10 km",
        fontsize=6, color=PAL.GREEN, ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"),
    )

    ax.set_xticks(group_centres)
    ax.set_xticklabels([t for _, t, _ in SCENARIOS], fontsize=8)
    ax.set_ylabel(r"D$_{\rm cond}$ (km)", fontsize=10)
    ax.set_ylim(0, None)
    ax.set_title(
        r"D$_{\rm cond}$ by latitude band: global grain (blue) vs strain-scaled (red)"
        "\nEach triplet: equatorial / mid-latitude / polar",
        fontsize=8, fontweight="bold",
    )
    label_panel(ax, "d")

    global_patch = mpatches.Patch(facecolor=c_global, alpha=0.5, edgecolor="k", lw=0.5,
                                   label="Global grain (benchmark)")
    strain_patch = mpatches.Patch(facecolor=c_strain, alpha=0.5, edgecolor="k", lw=0.5,
                                   label="Strain-scaled grain")
    ax.legend(handles=[global_patch, strain_patch], loc="upper left", fontsize=7)


def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.22, height_ratios=[1, 1.1])

    # Top row: cross-section panels
    letters = "abc"
    for idx, (name, title, citation) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[0, idx])
        plot_cross_section(ax, name, title, citation, letters[idx])
        if idx == 0:
            ax.set_ylabel("Depth from surface (km)", fontsize=8)

    # Cross-section legend
    cs_handles = [
        mpatches.Patch(facecolor=C_LID, alpha=0.5, edgecolor=C_LID_DARK,
                       label=r"D$_{\rm cond}$ (lid)"),
        mpatches.Patch(facecolor=C_CONV, alpha=0.5, edgecolor=C_CONV_DARK,
                       label=r"D$_{\rm conv}$ (sublayer)"),
        mpatches.Patch(facecolor=C_TOTAL, alpha=0.3, edgecolor=C_TOTAL,
                       label=r"H$_{\rm total}$ (16-84th pctl)"),
        mlines.Line2D([], [], color="0.2", lw=1.5, ls="--", label="Strain-scaled"),
    ]
    fig.legend(
        handles=cs_handles, loc="upper center",
        ncol=4, fontsize=7, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.52),
    )

    # Bottom row: violin panel
    ax_v = fig.add_subplot(gs[1, :])
    plot_violins(ax_v)

    fig.suptitle(
        "Global vs strain-scaled grain size  |  250 MC samples, 19 columns\n"
        "Solid/filled = global grain (benchmark), dashed/red = strain-scaled (sensitivity)",
        fontsize=10, fontweight="bold", y=0.98,
    )

    save_path = os.path.join(FIGURES_DIR, "global_vs_strain_250mc.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
