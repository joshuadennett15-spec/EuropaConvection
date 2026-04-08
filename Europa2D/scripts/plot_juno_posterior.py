"""
Publication figure: Prior vs Juno-reweighted posterior for D_cond at 35 deg.

Shows how the Juno MWR constraint shifts the D_cond distribution for each
ocean transport scenario.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import apply_style, PAL, label_panel, save_fig, add_minor_gridlines, DOUBLE_COL

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")
N_ITER = 250

JUNO_DCOND_KM = 29.0
JUNO_DCOND_SIGMA_OBS = 10.0
MODEL_DISCREPANCY = 3.0
SIGMA_EFF = np.sqrt(JUNO_DCOND_SIGMA_OBS**2 + MODEL_DISCREPANCY**2)
JUNO_LAT_DEG = 35.0

SCENARIOS = [
    ("uniform_transport",             "Uniform transport",      PAL.BLACK),
    ("soderlund2014_equator",         "Equator-enhanced",       "#B8860B"),
    ("lemasquerier2023_polar",        "Polar-enhanced",         PAL.BLUE),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced",  PAL.RED),
]


def _load(key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _interp_at_lat(lat, profiles, target):
    return np.array([np.interp(target, lat, profiles[i])
                     for i in range(profiles.shape[0])])


def _gaussian_likelihood(d_cond):
    return np.exp(-0.5 * ((d_cond - JUNO_DCOND_KM) / SIGMA_EFF) ** 2)


def _weighted_kde(samples, weights, grid):
    """KDE with importance weights via resampling."""
    # Resample according to weights for KDE
    n_resample = 5000
    rng = np.random.default_rng(42)
    indices = rng.choice(len(samples), size=n_resample, p=weights / weights.sum())
    resampled = samples[indices]
    kde = gaussian_kde(resampled, bw_method=0.25)
    return kde(grid)


def main():
    apply_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(DOUBLE_COL, DOUBLE_COL * 0.62),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.12)

    grid = np.linspace(0, 70, 300)

    for idx, (key, title, color) in enumerate(SCENARIOS):
        ax = axes.flat[idx]
        d = _load(key)
        lat = d["latitudes_deg"]
        Dc = d["D_cond_profiles"]

        dc_35 = _interp_at_lat(lat, Dc, JUNO_LAT_DEG)

        # Prior KDE
        prior_kde = gaussian_kde(dc_35, bw_method=0.25)
        prior_density = prior_kde(grid)

        # Importance weights
        lk = _gaussian_likelihood(dc_35)
        weights = lk / lk.sum()

        # Posterior KDE
        post_density = _weighted_kde(dc_35, weights, grid)

        # Normalize both to same peak height for visual comparison
        prior_density = prior_density / prior_density.max()
        post_density = post_density / post_density.max()

        # Plot prior
        ax.fill_between(grid, 0, prior_density, color=color, alpha=0.12, lw=0)
        ax.plot(grid, prior_density, color=color, lw=1.0, ls="--", alpha=0.6,
                label="Prior" if idx == 0 else None)

        # Plot posterior
        ax.fill_between(grid, 0, post_density, color=color, alpha=0.25, lw=0)
        ax.plot(grid, post_density, color=color, lw=1.5,
                label="Posterior" if idx == 0 else None)

        # Juno constraint band
        ax.axvspan(JUNO_DCOND_KM - JUNO_DCOND_SIGMA_OBS,
                   JUNO_DCOND_KM + JUNO_DCOND_SIGMA_OBS,
                   color=PAL.RED, alpha=0.06, zorder=0)
        ax.axvline(JUNO_DCOND_KM, color=PAL.RED, lw=0.7, ls=":", alpha=0.7,
                   label="Juno MWR" if idx == 0 else None)

        # Prior and posterior medians
        prior_med = float(np.median(dc_35))
        post_med = float(np.sum(weights * dc_35))
        ax.axvline(prior_med, color=color, lw=0.6, ls="--", alpha=0.5)
        ax.axvline(post_med, color=color, lw=0.9, ls="-", alpha=0.8)

        # Annotation: shift
        shift = post_med - prior_med
        ax.text(
            0.97, 0.95,
            f"Prior med: {prior_med:.1f} km\n"
            f"Post mean: {post_med:.1f} km\n"
            f"Shift: +{shift:.1f} km",
            transform=ax.transAxes, fontsize=5.5, va="top", ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.92,
                      ec="0.75", lw=0.3),
        )

        # Panel formatting
        add_minor_gridlines(ax)
        ax.set_xlim(0, 65)
        ax.set_ylim(0, 1.15)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax.set_title(title, fontsize=8, fontweight="bold", loc="left")
        label_panel(ax, chr(97 + idx))

    for ax in axes[1, :]:
        ax.set_xlabel(r"$D_{\rm cond}$ at 35$\degree$ (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized density")

    # Shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=2.0, handletextpad=0.5,
    )

    save_fig(fig, "juno_prior_posterior_dcond35", FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
