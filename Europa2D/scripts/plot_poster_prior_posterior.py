#!/usr/bin/env python3
"""
Poster figure: Prior vs. posterior density for Juno-constrained parameters.

Three-panel KDE plot showing how the Juno MWR D_cond constraint (29 +/- 10 km)
shifts the distributions of (a) D_cond, (b) q_basal, and (c) d_grain.

Uses uniform-transport round 1 data (all three ocean models are
indistinguishable at 35 deg latitude).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import gaussian_kde
from pub_style import apply_style, PAL, figsize_double, label_panel, save_fig

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS = os.path.join(
    os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ',
    'results', 'midlat_juno',
)
NPZ_PATH = os.path.join(RESULTS, 'midlat35_uniform_round1.npz')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# ── Juno observation ─────────────────────────────────────────────────────────
JUNO_D_OBS = 29.0       # km
JUNO_SIGMA_OBS = 10.0   # km
SIGMA_MODEL = 3.0        # km
SIGMA_TOTAL = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)

# ── Physical constants for q_basal reconstruction ────────────────────────────
R_EUROPA = 1_560_800.0   # m
AREA_EUROPA = 4.0 * np.pi * R_EUROPA**2


def compute_weights(D_cond):
    """Gaussian importance weights for Juno likelihood."""
    log_w = -0.5 * ((D_cond - JUNO_D_OBS) / SIGMA_TOTAL) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def extract_q_basal(data):
    """Reconstruct q_basal (mW/m^2) from NPZ arrays."""
    D_H2O = data['param_D_H2O']
    H_rad = data['param_H_rad']
    R_rock = R_EUROPA - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / AREA_EUROPA
    q_tidal = data['param_P_tidal'] / AREA_EUROPA
    return (q_rad + q_tidal) * 1e3  # mW/m^2


def weighted_percentile(values, weights, pct):
    """Weighted percentile via CDF interpolation."""
    idx = np.argsort(values)
    cum_w = np.cumsum(weights[idx])
    return float(np.interp(pct / 100.0, cum_w, values[idx]))


def main():
    apply_style()

    # ── Poster overrides (larger text for readability at distance) ───────
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })

    data = np.load(NPZ_PATH)
    D_cond = data['D_cond_km']
    q_basal = extract_q_basal(data)
    d_grain = data['param_d_grain'] * 1e3  # mm

    # Compute importance weights
    w = compute_weights(D_cond)

    # Resample from posterior for KDE
    rng = np.random.default_rng(42)
    n_resample = 20_000
    idx_post = rng.choice(len(D_cond), size=n_resample, p=w, replace=True)

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    col_prior = PAL.CYAN
    col_post = PAL.ORANGE
    col_juno = PAL.BLACK

    # ── Panel (a): D_cond ────────────────────────────────────────────────
    ax = axes[0]
    x_dc = np.linspace(0, 70, 400)
    kde_prior = gaussian_kde(D_cond, bw_method=0.15)
    kde_post = gaussian_kde(D_cond[idx_post], bw_method=0.15)

    ax.fill_between(x_dc, kde_prior(x_dc), alpha=0.25, color=col_prior,
                    linewidth=0)
    ax.plot(x_dc, kde_prior(x_dc), color=col_prior, lw=1.5, label='Prior')
    ax.fill_between(x_dc, kde_post(x_dc), alpha=0.35, color=col_post,
                    linewidth=0)
    ax.plot(x_dc, kde_post(x_dc), color=col_post, lw=1.8, label='Posterior')

    # Juno constraint Gaussian
    juno_pdf = np.exp(-0.5 * ((x_dc - JUNO_D_OBS) / SIGMA_TOTAL) ** 2)
    juno_pdf /= SIGMA_TOTAL * np.sqrt(2 * np.pi)
    # Scale to roughly match KDE peak for visual clarity
    juno_scale = kde_post(x_dc).max() / juno_pdf.max() * 0.7
    ax.plot(x_dc, juno_pdf * juno_scale, '--', color=col_juno, lw=1.2,
            label=f'Juno ({JUNO_D_OBS:.0f} $\\pm$ {JUNO_SIGMA_OBS:.0f} km)')

    # Posterior median + 68% CI
    dc_med = weighted_percentile(D_cond, w, 50)
    dc_16 = weighted_percentile(D_cond, w, 15.87)
    dc_84 = weighted_percentile(D_cond, w, 84.13)
    ymax = ax.get_ylim()[1]
    ax.axvline(dc_med, color=col_post, lw=0.8, ls='-', alpha=0.6)
    ax.axvspan(dc_16, dc_84, alpha=0.08, color=col_post)

    ax.set_xlabel(r'$D_\mathrm{cond}$ (km)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.8)
    ax.set_title(r'Conductive lid thickness')
    label_panel(ax, 'a')

    # Annotation
    ax.annotate(
        f'{dc_med:.0f}$^{{+{dc_84 - dc_med:.0f}}}_{{-{dc_med - dc_16:.0f}}}$ km',
        xy=(dc_med, kde_post(np.array([dc_med]))[0]),
        xytext=(dc_med + 12, kde_post(np.array([dc_med]))[0] * 0.85),
        fontsize=9, color=col_post,
        arrowprops=dict(arrowstyle='->', color=col_post, lw=0.8),
    )

    # ── Panel (b): q_basal ───────────────────────────────────────────────
    ax = axes[1]
    x_q = np.linspace(2, 28, 400)
    kde_prior_q = gaussian_kde(q_basal, bw_method=0.15)
    kde_post_q = gaussian_kde(q_basal[idx_post], bw_method=0.15)

    ax.fill_between(x_q, kde_prior_q(x_q), alpha=0.25, color=col_prior,
                    linewidth=0)
    ax.plot(x_q, kde_prior_q(x_q), color=col_prior, lw=1.5, label='Prior')
    ax.fill_between(x_q, kde_post_q(x_q), alpha=0.35, color=col_post,
                    linewidth=0)
    ax.plot(x_q, kde_post_q(x_q), color=col_post, lw=1.8, label='Posterior')

    q_med = weighted_percentile(q_basal, w, 50)
    q_16 = weighted_percentile(q_basal, w, 15.87)
    q_84 = weighted_percentile(q_basal, w, 84.13)
    ax.axvline(q_med, color=col_post, lw=0.8, ls='-', alpha=0.6)
    ax.axvspan(q_16, q_84, alpha=0.08, color=col_post)

    ax.set_xlabel(r'$q_\mathrm{basal}$ (mW m$^{-2}$)')
    ax.set_ylabel('')
    ax.set_xlim(2, 28)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.8)
    ax.set_title('Basal heat flux')
    label_panel(ax, 'b')

    ax.annotate(
        f'{q_med:.1f}$^{{+{q_84 - q_med:.1f}}}_{{-{q_med - q_16:.1f}}}$ mW m$^{{-2}}$',
        xy=(q_med, kde_post_q(np.array([q_med]))[0]),
        xytext=(q_med + 4, kde_post_q(np.array([q_med]))[0] * 0.85),
        fontsize=9, color=col_post,
        arrowprops=dict(arrowstyle='->', color=col_post, lw=0.8),
    )

    # ── Panel (c): d_grain ───────────────────────────────────────────────
    ax = axes[2]
    x_dg = np.linspace(0, 3.5, 400)
    kde_prior_dg = gaussian_kde(d_grain, bw_method=0.15)
    kde_post_dg = gaussian_kde(d_grain[idx_post], bw_method=0.15)

    ax.fill_between(x_dg, kde_prior_dg(x_dg), alpha=0.25, color=col_prior,
                    linewidth=0)
    ax.plot(x_dg, kde_prior_dg(x_dg), color=col_prior, lw=1.5, label='Prior')
    ax.fill_between(x_dg, kde_post_dg(x_dg), alpha=0.35, color=col_post,
                    linewidth=0)
    ax.plot(x_dg, kde_post_dg(x_dg), color=col_post, lw=1.8, label='Posterior')

    dg_med = weighted_percentile(d_grain, w, 50)
    dg_16 = weighted_percentile(d_grain, w, 15.87)
    dg_84 = weighted_percentile(d_grain, w, 84.13)
    ax.axvline(dg_med, color=col_post, lw=0.8, ls='-', alpha=0.6)
    ax.axvspan(dg_16, dg_84, alpha=0.08, color=col_post)

    ax.set_xlabel(r'$d_\mathrm{grain}$ (mm)')
    ax.set_ylabel('')
    ax.set_xlim(0, 3.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', framealpha=0.8)
    ax.set_title('Ice grain size')
    label_panel(ax, 'c')

    ax.annotate(
        f'{dg_med:.2f}$^{{+{dg_84 - dg_med:.2f}}}_{{-{dg_med - dg_16:.2f}}}$ mm',
        xy=(dg_med, kde_post_dg(np.array([dg_med]))[0]),
        xytext=(dg_med + 0.8, kde_post_dg(np.array([dg_med]))[0] * 0.85),
        fontsize=9, color=col_post,
        arrowprops=dict(arrowstyle='->', color=col_post, lw=0.8),
    )

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, 'poster_prior_posterior', FIGURES_DIR, formats=('png', 'pdf'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()
