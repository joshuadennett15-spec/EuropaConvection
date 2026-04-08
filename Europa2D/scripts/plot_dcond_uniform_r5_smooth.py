#!/usr/bin/env python3
"""
Smooth KDE replot of D_cond prior vs posterior for Uniform round 5,
with LaTeX-rendered labels.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

import matplotlib.pyplot as plt
from pub_style import PAL, save_fig

# ── Constants ────────────────────────────────────────────────────────────────
JUNO_D_OBS = 24.0       # pure-water Juno MWR interpretation
JUNO_SIGMA_OBS = 10.0
SIGMA_MODEL = 3.0
SIGMA_TOTAL = np.sqrt(JUNO_SIGMA_OBS**2 + SIGMA_MODEL**2)

RESULTS = os.path.join(
    os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ',
    'results', 'midlat_juno',
)
NPZ = os.path.join(RESULTS, 'midlat35_uniform_round5.npz')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def compute_weights(D_cond):
    log_w = -0.5 * ((D_cond - JUNO_D_OBS) / SIGMA_TOTAL) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def weighted_percentile(values, weights, pct):
    idx = np.argsort(values)
    cum_w = np.cumsum(weights[idx])
    return float(np.interp(pct / 100.0, cum_w, values[idx]))


def main():
    # ── Style: poster-friendly sizes with LaTeX ──────────────────────────
    mpl.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    data = np.load(NPZ)
    D_cond = data['D_cond_km']
    w = compute_weights(D_cond)

    # Resample for posterior KDE
    rng = np.random.default_rng(42)
    idx_post = rng.choice(len(D_cond), size=25_000, p=w, replace=True)

    # ── KDEs ─────────────────────────────────────────────────────────────
    x = np.linspace(0, 70, 500)
    kde_prior = gaussian_kde(D_cond, bw_method=0.20)
    kde_post = gaussian_kde(D_cond[idx_post], bw_method=0.20)

    y_prior = kde_prior(x)
    y_post = kde_post(x)

    # Juno Gaussian
    juno_pdf = np.exp(-0.5 * ((x - JUNO_D_OBS) / SIGMA_TOTAL) ** 2)
    juno_pdf /= SIGMA_TOTAL * np.sqrt(2 * np.pi)

    # Posterior stats
    dc_med = weighted_percentile(D_cond, w, 50)
    dc_16 = weighted_percentile(D_cond, w, 15.87)
    dc_84 = weighted_percentile(D_cond, w, 84.13)

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Prior
    ax.fill_between(x, y_prior, alpha=0.20, color=PAL.CYAN, linewidth=0)
    ax.plot(x, y_prior, color=PAL.CYAN, lw=1.8, label='Prior')

    # Posterior
    ax.fill_between(x, y_post, alpha=0.30, color=PAL.ORANGE, linewidth=0)
    ax.plot(x, y_post, color=PAL.ORANGE, lw=2.0, label='Posterior')

    # Juno constraint — scale to visual comparison
    juno_scale = y_post.max() / juno_pdf.max() * 0.75
    ax.plot(x, juno_pdf * juno_scale, '--', color=PAL.BLACK, lw=1.3,
            label=r'Juno $\mathcal{N}(24,\,10)$ km')
    ax.axvline(JUNO_D_OBS, color=PAL.BLACK, lw=0.5, ls=':', alpha=0.4)

    # 68% credible interval shading
    mask_ci = (x >= dc_16) & (x <= dc_84)
    ax.fill_between(x[mask_ci], y_post[mask_ci], alpha=0.12,
                    color=PAL.ORANGE, linewidth=0)
    ax.axvline(dc_med, color=PAL.RED, lw=1.0, ls='-', alpha=0.6)

    # Annotation
    y_at_med = float(kde_post(np.array([dc_med]))[0])
    ax.annotate(
        rf'${dc_med:.0f}^{{+{dc_84 - dc_med:.0f}}}_{{{-(dc_med - dc_16):.0f}}}$ km',
        xy=(dc_med, y_at_med),
        xytext=(dc_med + 14, y_at_med * 0.92),
        fontsize=11, color=PAL.RED,
        arrowprops=dict(arrowstyle='->', color=PAL.RED, lw=0.9),
    )

    ax.set_xlabel(r'Conductive lid thickness $D_{\mathrm{cond}}$ (km)')
    ax.set_ylabel(r'Probability density')
    ax.set_xlim(0, 65)
    ax.set_ylim(bottom=0)
    ax.legend(loc='center right', frameon=True, framealpha=0.85,
              edgecolor='0.8', fancybox=False,
              bbox_to_anchor=(1.0, 0.5))
    ax.set_title(r'Uniform transport @ $35^{\circ}$ --- Pure water')

    fig.tight_layout()
    save_fig(fig, 'dcond_uniform_round5_smooth', FIGURES_DIR,
             formats=('png', 'pdf'))


if __name__ == '__main__':
    main()
