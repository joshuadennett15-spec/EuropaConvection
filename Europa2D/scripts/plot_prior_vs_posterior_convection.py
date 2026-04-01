"""
4-scenario three-way comparison: 0.6mm prior vs Juno-reweighted vs 1.5mm prior.

Layout: 4 rows (one per scenario) x 4 columns (H_total, D_cond, Conv%, Nu|conv).
Every result is shown — no view is privileged over another.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import apply_style, PAL, label_panel, save_fig, add_minor_gridlines, DOUBLE_COL

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")

JUNO = 29.0
SIGMA_EFF = np.sqrt(10.0**2 + 3.0**2)

C_PRIOR = "0.55"
C_JUNO = PAL.BLUE
C_GRAIN = PAL.RED

SCENARIOS = [
    ("uniform_transport",             "Uniform transport"),
    ("soderlund2014_equator",         "Equator-enhanced"),
    ("lemasquerier2023_polar",        "Polar-enhanced"),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced"),
]


def _interp_at(lat, profiles, target):
    return np.array([np.interp(target, lat, profiles[i])
                     for i in range(profiles.shape[0])])


def _gaussian_lk(dc):
    return np.exp(-0.5 * ((dc - JUNO) / SIGMA_EFF)**2)


def _weighted_profile(arr, weights):
    n_lat = arr.shape[1]
    med = np.zeros(n_lat)
    for j in range(n_lat):
        col = arr[:, j]
        idx = np.argsort(col)
        cw = np.cumsum(weights[idx])
        cw /= cw[-1]
        med[j] = col[idx[np.searchsorted(cw, 0.50)]]
    return med


def _weighted_conv_frac(Nu, weights):
    conv = (Nu > 1.1).astype(float)
    return np.sum(weights[:, None] * conv, axis=0)


def _conditional_median(arr, Nu):
    n_lat = arr.shape[1]
    med = np.full(n_lat, np.nan)
    for j in range(n_lat):
        mask = Nu[:, j] > 1.1
        if mask.sum() > 5:
            med[j] = np.median(arr[mask, j])
    return med


def _conditional_weighted_median(arr, Nu, weights):
    n_lat = arr.shape[1]
    med = np.full(n_lat, np.nan)
    for j in range(n_lat):
        mask = Nu[:, j] > 1.1
        if mask.sum() < 5:
            continue
        vals = arr[mask, j]
        w = weights[mask]
        w = w / w.sum()
        idx = np.argsort(vals)
        cw = np.cumsum(w[idx])
        med[j] = vals[idx[np.searchsorted(cw, 0.50)]]
    return med


def main():
    apply_style()

    fig, axes = plt.subplots(
        4, 4,
        figsize=(DOUBLE_COL, DOUBLE_COL * 1.10),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.38)

    col_titles = [
        r"$H_{\rm total}$ (km)",
        r"$D_{\rm cond}$ (km)",
        "Convecting (%)",
        "Nu | convecting",
    ]
    panel_idx = 0

    for row, (key, title) in enumerate(SCENARIOS):
        # Load 0.6mm and 1.5mm
        f06 = os.path.join(RESULTS_DIR, f"mc_2d_{key}_250_grain06mm.npz")
        f15 = os.path.join(RESULTS_DIR, f"mc_2d_{key}_250.npz")

        if not os.path.exists(f06):
            print(f"WARNING: {f06} not found, skipping {key}")
            continue

        old = dict(np.load(f06, allow_pickle=True))
        new = dict(np.load(f15, allow_pickle=True))

        lat = old["latitudes_deg"]
        lat_n = new["latitudes_deg"]

        # Juno importance weights for 0.6mm
        dc35 = _interp_at(lat, old["D_cond_profiles"], 35.0)
        lk = _gaussian_lk(dc35)
        w = lk / lk.sum()

        # --- Column 0: H_total ---
        ax = axes[row, 0]
        ax.plot(lat, np.median(old["H_profiles"], axis=0), color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_profile(old["H_profiles"], w), color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.median(new["H_profiles"], axis=0), color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(15, 65)
        if row == 0:
            ax.set_title(col_titles[0], fontsize=7, fontweight="bold")
        ax.text(-0.35, 0.5, title, transform=ax.transAxes, fontsize=6.5,
                fontweight="bold", va="center", ha="center", rotation=90)
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

        # --- Column 1: D_cond ---
        ax = axes[row, 1]
        ax.plot(lat, np.median(old["D_cond_profiles"], axis=0), color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_profile(old["D_cond_profiles"], w), color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.median(new["D_cond_profiles"], axis=0), color=C_GRAIN, lw=1.0, ls="-.")
        ax.errorbar(35.0, JUNO, yerr=10.0, fmt="D", ms=3, color=PAL.ORANGE,
                    ecolor=PAL.ORANGE, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=5)
        ax.set_ylim(10, 55)
        if row == 0:
            ax.set_title(col_titles[1], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

        # --- Column 2: Convecting fraction ---
        ax = axes[row, 2]
        ax.plot(lat, np.mean(old["Nu_profiles"] > 1.1, axis=0) * 100,
                color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_conv_frac(old["Nu_profiles"], w) * 100,
                color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.mean(new["Nu_profiles"] > 1.1, axis=0) * 100,
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(0, 70)
        ax.axhline(50, color="0.7", lw=0.4, ls=":", zorder=0)
        if row == 0:
            ax.set_title(col_titles[2], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

        # --- Column 3: Conditional Nu ---
        ax = axes[row, 3]
        ax.plot(lat, _conditional_median(old["Nu_profiles"], old["Nu_profiles"]),
                color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _conditional_weighted_median(old["Nu_profiles"], old["Nu_profiles"], w),
                color=C_JUNO, lw=1.5)
        ax.plot(lat_n, _conditional_median(new["Nu_profiles"], new["Nu_profiles"]),
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(1, 15)
        if row == 0:
            ax.set_title(col_titles[3], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + panel_idx))
        panel_idx += 1

    # X-axis labels on bottom row
    for ax in axes[3, :]:
        ax.set_xlabel(r"Latitude ($\degree$)", fontsize=7)
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))

    # Legend
    legend_elements = [
        Line2D([0], [0], color=C_PRIOR, lw=1.0, ls="--", label="0.6 mm prior"),
        Line2D([0], [0], color=C_JUNO, lw=1.5, label="0.6 mm + Juno reweighted"),
        Line2D([0], [0], color=C_GRAIN, lw=1.0, ls="-.", label="1.5 mm prior"),
    ]
    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01),
               columnspacing=2.0, handletextpad=0.5)

    save_fig(fig, "prior_vs_juno_vs_grain_4scenario", FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
