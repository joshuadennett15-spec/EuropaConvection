"""
Publication figure: 4-scenario comparison from 250 MC samples each.

Figure 1 (main): Continuous latitude profiles with percentile envelopes,
                 band-mean summaries, and Juno D_cond constraint.

Figure 2 (diagnostics): Convecting fraction + conditional Ra/Nu.

Uses corrected dt=1e12 results.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import (apply_style, PAL, label_panel, save_fig,
                        add_minor_gridlines, DOUBLE_COL)
from profile_diagnostics import band_mean_samples

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 250

# Juno induction constraint (Khurana et al. 2024)
JUNO_DCOND_KM = 29.0
JUNO_DCOND_ERR = 10.0
JUNO_LAT_DEG = 35.0

RA_CRIT = 1000.0

# Latitude bands for area-weighted summaries
BAND_EQ = (0.0, 10.0)
BAND_MID = (30.0, 50.0)
BAND_POLAR = (80.0, 90.0)

SCENARIOS = [
    ("uniform_transport",             "Uniform transport",      PAL.BLACK),
    ("soderlund2014_equator",         "Equator-enhanced",       "#B8860B"),
    ("lemasquerier2023_polar",        "Polar-enhanced",         PAL.BLUE),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced",  PAL.RED),
]

CITATIONS = {
    "uniform_transport": "Ashkenazy & Tziperman (2021)",
    "soderlund2014_equator": "Soderlund et al. (2014)",
    "lemasquerier2023_polar": "Lemasquerier et al. (2023)",
    "lemasquerier2023_polar_strong": "Lemasquerier et al. (2023)",
}


def _load(scenario_key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{scenario_key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _interp_at(lat, arr, target_lat):
    """Interpolate a per-latitude array at a single target latitude for each sample."""
    # arr shape: (n_samples, n_lat)
    return np.array([np.interp(target_lat, lat, arr[i]) for i in range(arr.shape[0])])


# ── Figure 1: Shell structure (main figure) ──────────────────────────────

def plot_thickness_figure():
    """
    2x2 panels, one per scenario. Each panel shows:
    - H_total: median + 1-sigma + 2-sigma envelopes
    - D_cond: median + 1-sigma envelope
    - D_conv: stacked fill between D_cond and H_total medians
    - Band-mean markers for equatorial, mid-latitude, polar
    - Juno D_cond constraint at 35 deg
    - Compact band-mean table
    """
    apply_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.78),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.12)

    for idx, (key, title, sc_color) in enumerate(SCENARIOS):
        ax = axes.flat[idx]
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Dv = d["D_conv_profiles"]
        n_valid = int(d["n_valid"])

        # ── Percentile envelopes ──
        H_med = np.median(H, axis=0)
        H_p16, H_p84 = np.percentile(H, 16, axis=0), np.percentile(H, 84, axis=0)
        H_p05, H_p95 = np.percentile(H, 5, axis=0), np.percentile(H, 95, axis=0)

        Dc_med = np.median(Dc, axis=0)
        Dc_p16, Dc_p84 = np.percentile(Dc, 16, axis=0), np.percentile(Dc, 84, axis=0)

        # ── 2-sigma envelope (H_total, lightest) ──
        ax.fill_between(lat, H_p05, H_p95, color="0.82", alpha=0.35, lw=0)
        # ── 1-sigma envelope (H_total) ──
        ax.fill_between(lat, H_p16, H_p84, color="0.68", alpha=0.30, lw=0)

        # ── Stacked cross-section fill ──
        ax.fill_between(lat, 0, Dc_med, color=PAL.CYAN, alpha=0.22, lw=0)
        ax.fill_between(lat, Dc_med, H_med, color=PAL.ORANGE, alpha=0.22, lw=0)

        # ── D_cond 1-sigma envelope ──
        ax.fill_between(lat, Dc_p16, Dc_p84, color=PAL.CYAN, alpha=0.12, lw=0)

        # ── Median lines ──
        ax.plot(lat, H_med, color=PAL.BLACK, lw=1.5, zorder=3)
        ax.plot(lat, Dc_med, color=PAL.BLUE, lw=1.2, zorder=3)

        # ── Band-mean markers ──
        bands = [
            (BAND_EQ,    r"0--10$\degree$",  "o", 4.0),
            (BAND_MID,   r"30--50$\degree$", "s", 3.5),
            (BAND_POLAR, r"80--90$\degree$", "^", 4.0),
        ]
        for band, _blabel, marker, ms in bands:
            band_center = (band[0] + band[1]) / 2
            h_band = band_mean_samples(lat, H, band)
            dc_band = band_mean_samples(lat, Dc, band)
            # H_total band marker
            ax.plot(band_center, np.median(h_band), marker=marker, ms=ms,
                    color=PAL.BLACK, mec=PAL.BLACK, mfc="white", mew=0.8, zorder=4)
            # D_cond band marker
            ax.plot(band_center, np.median(dc_band), marker=marker, ms=ms,
                    color=PAL.BLUE, mec=PAL.BLUE, mfc="white", mew=0.8, zorder=4)

        # ── Juno D_cond constraint at 35 deg ──
        dc_at_juno = _interp_at(lat, Dc, JUNO_LAT_DEG)
        dc_juno_med = np.median(dc_at_juno)
        ax.errorbar(
            JUNO_LAT_DEG, JUNO_DCOND_KM,
            yerr=JUNO_DCOND_ERR, fmt="D", ms=4,
            color=PAL.RED, ecolor=PAL.RED, elinewidth=0.8, capsize=2.5, capthick=0.8,
            zorder=6,
        )

        # ── Panel formatting ──
        add_minor_gridlines(ax)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 70)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

        # Title
        cite = CITATIONS[key]
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        ax.text(0.01, 0.91, cite, transform=ax.transAxes, fontsize=5.5,
                fontstyle="italic", color="0.45", va="top")
        label_panel(ax, chr(97 + idx))

        # ── Band-mean summary table ──
        h_eq = band_mean_samples(lat, H, BAND_EQ)
        h_mid = band_mean_samples(lat, H, BAND_MID)
        h_po = band_mean_samples(lat, H, BAND_POLAR)
        dc_eq = band_mean_samples(lat, Dc, BAND_EQ)
        dc_mid = band_mean_samples(lat, Dc, BAND_MID)
        dc_po = band_mean_samples(lat, Dc, BAND_POLAR)

        table = (
            f"         eq    mid   pole\n"
            f"H    {np.median(h_eq):5.1f} {np.median(h_mid):5.1f} {np.median(h_po):5.1f}\n"
            f"Dc   {np.median(dc_eq):5.1f} {np.median(dc_mid):5.1f} {np.median(dc_po):5.1f}\n"
            f"Juno Dc(35)={dc_juno_med:.1f} km\n"
            f"N={n_valid}"
        )
        ax.text(
            0.98, 0.97, table,
            transform=ax.transAxes, fontsize=5, va="top", ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                      ec="0.75", lw=0.3),
        )

    # ── Axis labels ──
    for ax in axes[1, :]:
        ax.set_xlabel(r"Latitude ($\degree$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Thickness (km)")

    # ── Shared legend ──
    legend_elements = [
        Line2D([0], [0], color=PAL.BLACK, lw=1.5, label=r"$H_{\rm total}$ median"),
        Line2D([0], [0], color=PAL.BLUE, lw=1.2, label=r"$D_{\rm cond}$ median"),
        Patch(fc=PAL.CYAN, alpha=0.30, ec="none", label=r"$D_{\rm cond}$ (lid)"),
        Patch(fc=PAL.ORANGE, alpha=0.30, ec="none", label=r"$D_{\rm conv}$ (sublayer)"),
        Patch(fc="0.75", alpha=0.35, ec="none", label=r"$H_{\rm total}$ 1$\sigma$/2$\sigma$"),
        Line2D([0], [0], marker="D", ms=4, ls="none",
               color=PAL.RED, label=f"Juno $D_{{\\rm cond}}$ ({JUNO_LAT_DEG:.0f}$\\degree$)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", ncol=3, fontsize=6,
        bbox_to_anchor=(0.5, -0.03),
        columnspacing=1.5, handletextpad=0.5,
    )

    save_fig(fig, f"4scenario_thickness_{N_ITER}mc", FIGURES_DIR)


# ── Figure 2: Convection diagnostics ────────────────────────────────────

def plot_diagnostics_figure():
    """3-panel: convecting fraction, conditional Nu, conditional Ra."""
    apply_style()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.36),
        sharex=True,
    )
    fig.subplots_adjust(wspace=0.40)

    ax_frac, ax_nu, ax_ra = axes

    for key, title, color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        Nu = d["Nu_profiles"]
        Ra = d["Ra_profiles"]
        n_lat = len(lat)

        # (a) Convecting fraction vs latitude
        conv_frac = np.mean(Nu > 1.1, axis=0)
        ax_frac.plot(lat, conv_frac * 100, color=color, lw=1.3, label=title)

        # (b) Conditional Nu (convecting samples only) — median + IQR
        nu_med = np.full(n_lat, np.nan)
        nu_lo = np.full(n_lat, np.nan)
        nu_hi = np.full(n_lat, np.nan)
        for j in range(n_lat):
            mask = Nu[:, j] > 1.1
            if np.sum(mask) > 5:
                nu_med[j] = np.median(Nu[mask, j])
                nu_lo[j] = np.percentile(Nu[mask, j], 25)
                nu_hi[j] = np.percentile(Nu[mask, j], 75)
        ax_nu.fill_between(lat, nu_lo, nu_hi, color=color, alpha=0.10, lw=0)
        ax_nu.plot(lat, nu_med, color=color, lw=1.3)

        # (c) Conditional Ra (convecting samples only) — median + IQR
        ra_med = np.full(n_lat, np.nan)
        ra_lo = np.full(n_lat, np.nan)
        ra_hi = np.full(n_lat, np.nan)
        for j in range(n_lat):
            mask = Ra[:, j] > RA_CRIT
            if np.sum(mask) > 5:
                ra_med[j] = np.median(Ra[mask, j])
                ra_lo[j] = np.percentile(Ra[mask, j], 25)
                ra_hi[j] = np.percentile(Ra[mask, j], 75)
        ax_ra.fill_between(lat, ra_lo, ra_hi, color=color, alpha=0.10, lw=0)
        ax_ra.plot(lat, ra_med, color=color, lw=1.3)

    # ── (a) formatting ──
    ax_frac.set_ylabel("Convecting samples (%)")
    ax_frac.set_ylim(0, 100)
    ax_frac.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax_frac.axhline(50, color="0.6", lw=0.5, ls=":", zorder=0)
    label_panel(ax_frac, "a")
    add_minor_gridlines(ax_frac)
    ax_frac.legend(fontsize=5.5, loc="lower left")

    # ── (b) formatting ──
    ax_nu.set_ylabel("Nu | convecting")
    ax_nu.set_ylim(1, None)
    label_panel(ax_nu, "b")
    add_minor_gridlines(ax_nu)

    # ── (c) formatting ──
    ax_ra.set_ylabel("Ra | convecting")
    ax_ra.set_yscale("log")
    ax_ra.axhline(RA_CRIT, color="0.6", lw=0.5, ls=":", zorder=0,
                  label=r"$Ra_{\rm crit}$")
    label_panel(ax_ra, "c")
    add_minor_gridlines(ax_ra)

    for ax in axes:
        ax.set_xlim(0, 90)
        ax.set_xlabel(r"Latitude ($\degree$)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))

    save_fig(fig, f"4scenario_diagnostics_{N_ITER}mc", FIGURES_DIR)


# ── Summary table (printed to console) ──────────────────────────────────

def print_summary_table():
    """Print a formatted band-mean summary table for all scenarios."""
    print(f"\n{"-"*90}")
    print(f"  Band-mean summary (N={N_ITER} MC, area-weighted cos(phi))")
    print(f"{"-"*90}")
    header = (
        f"  {'Scenario':<26s}"
        f"  {'H_eq':>6s} {'H_mid':>6s} {'H_pole':>6s}"
        f"  {'Dc_eq':>6s} {'Dc_mid':>6s} {'Dc_pole':>6s}"
        f"  {'Dc(35)':>7s}"
        f"  {'Conv%eq':>7s} {'Conv%po':>7s}"
    )
    print(header)
    print(f"  {"":-<26s}  {"":->6s} {"":->6s} {"":->6s}  {"":->6s} {"":->6s} {"":->6s}  {"":->7s}  {"":->7s} {"":->7s}")

    for key, title, _color in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Nu = d["Nu_profiles"]

        h_eq = np.median(band_mean_samples(lat, H, BAND_EQ))
        h_mid = np.median(band_mean_samples(lat, H, BAND_MID))
        h_po = np.median(band_mean_samples(lat, H, BAND_POLAR))
        dc_eq = np.median(band_mean_samples(lat, Dc, BAND_EQ))
        dc_mid = np.median(band_mean_samples(lat, Dc, BAND_MID))
        dc_po = np.median(band_mean_samples(lat, Dc, BAND_POLAR))
        dc_35 = np.median(_interp_at(lat, Dc, JUNO_LAT_DEG))
        conv_eq = np.mean(Nu[:, 0] > 1.1) * 100
        conv_po = np.mean(Nu[:, -1] > 1.1) * 100

        print(
            f"  {title:<26s}"
            f"  {h_eq:6.1f} {h_mid:6.1f} {h_po:6.1f}"
            f"  {dc_eq:6.1f} {dc_mid:6.1f} {dc_po:6.1f}"
            f"  {dc_35:7.1f}"
            f"  {conv_eq:6.0f}% {conv_po:6.0f}%"
        )

    print(f"\n  Juno constraint: D_cond({JUNO_LAT_DEG:.0f} deg) = {JUNO_DCOND_KM} +/- {JUNO_DCOND_ERR} km")
    print(f"  Bands: eq={BAND_EQ}, mid={BAND_MID}, polar={BAND_POLAR}")
    print(f"{"-"*90}\n")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating publication figures from 250 MC results...")
    print_summary_table()
    plot_thickness_figure()
    plot_diagnostics_figure()
    print("Done.")


if __name__ == "__main__":
    main()
