"""
Quick diagnostic plot for the 20% tidal-uplift verification run.

Shows stacked shell structure (D_cond + D_conv) with uncertainty bands
for three scenarios, 20 MC iterations each. No Ra filtering — all valid
samples included to show the full prior-predictive structure.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from pub_style import apply_style, PAL, label_panel, save_fig, figsize_double_tall

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")

SCENARIOS = [
    ("uniform_transport",       "Uniform transport",       "Ashkenazy & Tziperman (2021)"),
    ("soderlund2014_equator",   "Equatorial-enhanced",     "Soderlund et al. (2014)"),
    ("lemasquerier2023_polar",  "Polar-enhanced",          "Lemasquerier et al. (2023)"),
]

N_ITER = 20

C_LID      = "#4A90D9"
C_CONV     = "#E8645A"
C_LID_DARK = "#2C5F99"
C_CONV_DARK= "#B83B31"
C_TOTAL    = "#333333"


def pct_band(arr, lo=16, hi=84):
    return np.median(arr, axis=0), np.percentile(arr, lo, axis=0), np.percentile(arr, hi, axis=0)


def plot_panel(ax, name, title, citation, letter, show_ylabel=True):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{N_ITER}.npz")
    d = dict(np.load(path, allow_pickle=True))

    lat = d["latitudes_deg"]
    H  = d["H_profiles"]
    Dc = d["D_cond_profiles"]
    Dv = d["D_conv_profiles"]
    Ra = d["Ra_profiles"]
    Nu = d["Nu_profiles"]

    n_valid = len(H)

    H_med,  H_lo,  H_hi  = pct_band(H)
    Dc_med, Dc_lo, Dc_hi = pct_band(Dc)

    # -- Stacked structure (depth from surface) --
    # Conductive lid band
    ax.fill_between(lat, Dc_lo, Dc_hi, color=C_LID, alpha=0.25, zorder=2)
    ax.plot(lat, Dc_med, color=C_LID_DARK, lw=1.8, zorder=4,
            label=r"$D_{\mathrm{cond}}$ (lid base)")

    # Convective sublayer fill between lid base and total
    ax.fill_between(lat, Dc_med, H_med, color=C_CONV, alpha=0.30, zorder=2)

    # Total shell band + line
    ax.fill_between(lat, H_lo, H_hi, color=C_TOTAL, alpha=0.10, zorder=1)
    ax.plot(lat, H_med, color=C_TOTAL, lw=1.8, zorder=4,
            label=r"$H_{\mathrm{total}}$ (ice-ocean)")

    # Surface line
    ax.axhline(0, color="k", lw=0.8, zorder=5)

    # Stats annotation
    eq_H    = H_med[0]
    pole_H  = H_med[-1]
    eq_Dc   = Dc_med[0]
    pole_Dc = Dc_med[-1]
    eq_Dv   = eq_H - eq_Dc
    pole_Dv = pole_H - pole_Dc
    dH      = pole_H - eq_H
    eq_lid  = eq_Dc / eq_H * 100
    pole_lid= pole_Dc / pole_H * 100

    # Median Ra and Nu at equator and pole
    med_Ra_eq   = np.median(Ra[:, 0])
    med_Ra_pole = np.median(Ra[:, -1])
    med_Nu_eq   = np.median(Nu[:, 0])
    med_Nu_pole = np.median(Nu[:, -1])
    frac_nu_gt2 = np.mean(Nu > 2.0)

    stats = (
        f"N = {n_valid}\n"
        f"Eq:   H={eq_H:.1f}, Dc={eq_Dc:.1f}, Dv={eq_Dv:.1f}\n"
        f"Pole: H={pole_H:.1f}, Dc={pole_Dc:.1f}, Dv={pole_Dv:.1f}\n"
        f"dH = {dH:+.1f} km\n"
        f"Ra: {med_Ra_eq:.0f} / {med_Ra_pole:.0f}\n"
        f"Nu>2: {frac_nu_gt2:.0%}"
    )
    ax.text(
        0.97, 0.03, stats,
        transform=ax.transAxes, fontsize=6.5,
        va="bottom", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"),
    )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_xlabel("Latitude (deg)")
    if show_ylabel:
        ax.set_ylabel("Depth from surface (km)")
    ax.invert_yaxis()

    label_panel(ax, letter)
    ax.set_title(f"{title}\n{citation}", fontsize=8, loc="left")


def main():
    apply_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, (name, title, citation) in enumerate(SCENARIOS):
        plot_panel(axes[idx], name, title, citation,
                   letter=chr(97 + idx),
                   show_ylabel=(idx == 0))

    # Shared legend
    legend_elements = [
        Patch(facecolor=C_LID, alpha=0.5, edgecolor=C_LID_DARK,
              label=r"Conductive lid ($D_{\mathrm{cond}}$)"),
        Patch(facecolor=C_CONV, alpha=0.5, edgecolor=C_CONV_DARK,
              label=r"Convective sublayer ($D_{\mathrm{conv}}$)"),
        Patch(facecolor=C_TOTAL, alpha=0.3, edgecolor=C_TOTAL,
              label=r"Total shell $H$ (16-84th pctl)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=7.5, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.suptitle(
        "Shell structure with 20% tidal uplift  |  20 MC samples, 19 columns\n"
        "All valid samples (no Ra filter)  |  q_tidal_scale = 1.20",
        fontsize=9, fontweight="bold", y=1.04,
    )

    save_path = os.path.join(FIGURES_DIR, "diagnostic_20pct_uplift.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
