"""
Latitude-resolved ice shell structure: D_cond and D_conv vs latitude.

Produces a 2x2 figure (one panel per ocean heat transport scenario) showing
the median stacked shell structure (depth from surface) with uncertainty
bands, from equator (0°) to pole (90°).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

RA_CRIT = 1000.0

SCENARIOS = [
    ("uniform_transport", "Uniform", "Ashkenazy & Tziperman (2021)"),
    ("soderlund2014_equator", "Equatorial-enhanced", "Soderlund et al. (2014)"),
    ("lemasquerier2023_polar", "Polar-enhanced", "Lemasquerier et al. (2023)"),
    ("lemasquerier2023_polar_strong", "Polar-enhanced (strong)", "Lemasquerier et al. (2023)"),
]

C_LID = "#4A90D9"
C_CONV = "#E8645A"
C_LID_DARK = "#2C5F99"
C_CONV_DARK = "#B83B31"
C_TOTAL = "#333333"


def load_scenario(name, n_iter=500):
    """Load NPZ for a scenario, return dict of arrays."""
    path = os.path.join(RESULTS_DIR, f"mc_2d_{name}_{n_iter}.npz")
    return dict(np.load(path, allow_pickle=True))


def percentile_band(arr, lo=16, hi=84, axis=0):
    """Return (median, lo_pct, hi_pct) along axis."""
    return (
        np.median(arr, axis=axis),
        np.percentile(arr, lo, axis=axis),
        np.percentile(arr, hi, axis=axis),
    )


def _plot_panel(ax, scenario_name, label, citation, panel_letter, show_xlabel=True, show_ylabel=True):
    """Plot a single latitude-structure panel on the given axes."""
    data = load_scenario(scenario_name)

    lat = data["latitudes_deg"]
    H = data["H_profiles"]
    Dc = data["D_cond_profiles"]
    Dv = data["D_conv_profiles"]
    Ra = data["Ra_profiles"]

    conv_frac = np.mean(Ra >= RA_CRIT, axis=1)
    m_conv = conv_frac >= 0.5
    n_conv = m_conv.sum()
    n_total = len(H)

    H_c = H[m_conv]
    Dc_c = Dc[m_conv]
    Dv_c = Dv[m_conv]

    H_med, H_lo, H_hi = percentile_band(H_c)
    Dc_med, Dc_lo, Dc_hi = percentile_band(Dc_c)

    Dbase_c = Dc_c + Dv_c
    Db_med, Db_lo, Db_hi = percentile_band(Dbase_c)

    ax.fill_between(lat, Dc_lo, Dc_hi, color=C_LID, alpha=0.25, zorder=2)
    ax.plot(lat, Dc_med, color=C_LID_DARK, lw=2, zorder=4,
            label=r"$D_{\rm cond}$ base (lid)")

    ax.fill_between(lat, Dc_med, H_med, color=C_CONV, alpha=0.30, zorder=2)

    ax.fill_between(lat, H_lo, H_hi, color=C_TOTAL, alpha=0.10, zorder=1)
    ax.plot(lat, H_med, color=C_TOTAL, lw=2, zorder=4,
            label=r"$H_{\rm total}$ (ice-ocean)")

    ax.fill_between(lat, Dc_lo, Dc_hi, color=C_LID, alpha=0.20, zorder=3)

    ax.axhline(0, color="k", lw=0.8, zorder=5)

    eq_H = H_med[0]
    pole_H = H_med[-1]
    eq_lid = Dc_med[0] / H_med[0] * 100
    pole_lid = Dc_med[-1] / H_med[-1] * 100
    delta_H = pole_H - eq_H
    stats = (
        f"N = {n_conv}/{n_total}\n"
        f"Eq: H={eq_H:.1f} km, lid={eq_lid:.0f}%\n"
        f"Pole: H={pole_H:.1f} km, lid={pole_lid:.0f}%\n"
        f"\u0394H = {delta_H:+.1f} km"
    )
    ax.text(
        0.97, 0.03, stats,
        transform=ax.transAxes, fontsize=8,
        va="bottom", ha="right", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#ccc"),
    )

    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    if show_xlabel:
        ax.set_xlabel("Latitude (\u00b0)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Depth from surface (km)", fontsize=10)

    ax.set_title(
        f"({panel_letter})  {label}\n{citation}",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax.invert_yaxis()


def _shared_legend(fig):
    """Add the shared legend below the figure."""
    legend_elements = [
        Patch(facecolor=C_LID, alpha=0.5, edgecolor=C_LID_DARK,
              label=r"Conductive lid ($D_{\rm cond}$)"),
        Patch(facecolor=C_CONV, alpha=0.5, edgecolor=C_CONV_DARK,
              label=r"Convective sublayer ($D_{\rm conv}$)"),
        Patch(facecolor=C_TOTAL, alpha=0.3, edgecolor=C_TOTAL,
              label=r"Total shell $H$ (16\u201384th pctl)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=10, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )


SUPTITLE_BASE = (
    "Ice shell internal structure vs. latitude \u2014 Andrade rheology, audited priors\n"
    r"Convective branch (Ra $\geq$ Ra$_{\rm crit}$), 500 MC samples, 37 columns"
)

SCENARIOS_UNIFORM = SCENARIOS[:2]   # Uniform + Soderlund
SCENARIOS_POLAR = SCENARIOS[2:]     # Lemasquerier conservative + strong


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Figure A: Uniform + Soderlund (1x2) ---
    fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig_a.subplots_adjust(wspace=0.12)
    for idx, (scenario_name, label, citation) in enumerate(SCENARIOS_UNIFORM):
        _plot_panel(axes_a[idx], scenario_name, label, citation,
                    panel_letter=chr(97 + idx),
                    show_ylabel=(idx == 0))
    _shared_legend(fig_a)
    fig_a.suptitle(
        "Uniform and equatorial-enhanced ocean transport\n" + SUPTITLE_BASE,
        fontsize=12, fontweight="bold", y=1.06,
    )
    path_a = os.path.join(FIGURES_DIR, "latitude_structure_uniform.png")
    fig_a.savefig(path_a, dpi=200, bbox_inches="tight")
    print(f"Saved: {path_a}")
    plt.close(fig_a)

    # --- Figure B: Lemasquerier polar cases (1x2) ---
    fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig_b.subplots_adjust(wspace=0.12)
    for idx, (scenario_name, label, citation) in enumerate(SCENARIOS_POLAR):
        _plot_panel(axes_b[idx], scenario_name, label, citation,
                    panel_letter=chr(97 + idx),
                    show_ylabel=(idx == 0))
    path_b = os.path.join(FIGURES_DIR, "latitude_structure_polar.png")
    fig_b.savefig(path_b, dpi=200, bbox_inches="tight")
    print(f"Saved: {path_b}")
    plt.close(fig_b)

    # --- Also keep the combined 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.12)
    panels = axes.ravel()
    for idx, (scenario_name, label, citation) in enumerate(SCENARIOS):
        _plot_panel(panels[idx], scenario_name, label, citation,
                    panel_letter=chr(97 + idx),
                    show_xlabel=(idx >= 2),
                    show_ylabel=(idx % 2 == 0))
    _shared_legend(fig)
    fig.suptitle(SUPTITLE_BASE, fontsize=13, fontweight="bold", y=1.01)
    save_path = os.path.join(FIGURES_DIR, "latitude_structure.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
