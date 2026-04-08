#!/usr/bin/env python3
"""
Bayesian shrinkage figure: prior vs posterior for parameters constrained
by the pure-water Juno MWR observation (D_cond = 24 +/- 10 km at 35 deg).

Shows only parameters with positive shrinkage > 5%.
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
from pub_style import PAL, save_fig, label_panel

# ── Constants ────────────────────────────────────────────────────────────────
D_OBS = 24.0
SIGMA_OBS = 10.0
SIGMA_MODEL = 3.0
SIGMA_TOTAL = np.sqrt(SIGMA_OBS**2 + SIGMA_MODEL**2)

NPZ = os.path.join(
    os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ',
    'results', 'midlat_juno', 'midlat35_uniform_round1.npz',
)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

R_EUROPA = 1_560_800.0
AREA_EUROPA = 4.0 * np.pi * R_EUROPA**2


def compute_weights(D_cond):
    log_w = -0.5 * ((D_cond - D_OBS) / SIGMA_TOTAL) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / w.sum()


def weighted_pct(values, weights, pct):
    idx = np.argsort(values)
    cum_w = np.cumsum(weights[idx])
    return float(np.interp(pct / 100.0, cum_w, values[idx]))


def shrinkage(values, weights):
    p16_prior = np.percentile(values, 15.87)
    p84_prior = np.percentile(values, 84.13)
    prior_w = p84_prior - p16_prior
    if prior_w < 1e-15:
        return 0.0
    p16_post = weighted_pct(values, weights, 15.87)
    p84_post = weighted_pct(values, weights, 84.13)
    return 1.0 - (p84_post - p16_post) / prior_w


def main():
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

    data = np.load(NPZ)
    D_cond = data['D_cond_km']
    w = compute_weights(D_cond)

    rng = np.random.default_rng(42)
    n_resample = 20_000
    idx_post = rng.choice(len(D_cond), size=n_resample, p=w, replace=True)

    # ── Parameters with shrinkage > 5% ───────────────────────────────────
    panels = [
        {
            'values': D_cond,
            'label': r'$D_{\mathrm{cond}}$ (km)',
            'title': r'Conductive lid thickness',
            'xlim': (0, 65),
            'bw': 0.18,
        },
        {
            'values': data['param_Q_v'] / 1e3,
            'label': r'$Q_v$ (kJ mol$^{-1}$)',
            'title': r'Volume diffusion activation energy',
            'xlim': (50, 68),
            'bw': 0.20,
        },
        {
            'values': data['lid_fractions'],
            'label': r'$D_{\mathrm{cond}} / H_{\mathrm{total}}$',
            'title': r'Lid fraction',
            'xlim': (0, 1.05),
            'bw': 0.18,
        },
        {
            'values': data['thicknesses_km'],
            'label': r'$H_{\mathrm{total}}$ (km)',
            'title': r'Total shell thickness',
            'xlim': (0, 120),
            'bw': 0.18,
        },
    ]

    col_prior = PAL.CYAN
    col_post = PAL.ORANGE

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    axes = axes.ravel()

    for i, (ax, p) in enumerate(zip(axes, panels)):
        vals = p['values']
        s = shrinkage(vals, w)
        xlo, xhi = p['xlim']
        x = np.linspace(xlo, xhi, 500)

        # Prior KDE
        kde_prior = gaussian_kde(vals, bw_method=p['bw'])
        y_prior = kde_prior(x)
        ax.fill_between(x, y_prior, alpha=0.20, color=col_prior, linewidth=0)
        ax.plot(x, y_prior, color=col_prior, lw=1.5, label='Prior')

        # Posterior KDE
        kde_post = gaussian_kde(vals[idx_post], bw_method=p['bw'])
        y_post = kde_post(x)
        ax.fill_between(x, y_post, alpha=0.30, color=col_post, linewidth=0)
        ax.plot(x, y_post, color=col_post, lw=1.8, label='Posterior')

        # Posterior median + 68% CI
        med = weighted_pct(vals, w, 50)
        p16 = weighted_pct(vals, w, 15.87)
        p84 = weighted_pct(vals, w, 84.13)
        ax.axvline(med, color=PAL.RED, lw=0.9, alpha=0.6)
        ax.axvspan(p16, p84, alpha=0.08, color=col_post)

        # Format credible interval string
        if med >= 10:
            ci_str = rf'${med:.0f}^{{+{p84 - med:.0f}}}_{{{-(med - p16):.0f}}}$'
        elif med >= 1:
            ci_str = rf'${med:.1f}^{{+{p84 - med:.1f}}}_{{{-(med - p16):.1f}}}$'
        else:
            ci_str = rf'${med:.2f}^{{+{p84 - med:.2f}}}_{{{-(med - p16):.2f}}}$'

        # Shrinkage badge
        ax.text(
            0.97, 0.95,
            rf'Shrinkage: {100 * s:.0f}\%',
            transform=ax.transAxes, fontsize=8.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', alpha=0.9),
        )

        # Median annotation
        y_at_med = float(kde_post(np.array([med]))[0])
        # Pick annotation position that avoids clutter
        x_text = med + (xhi - xlo) * 0.12
        if x_text > xhi * 0.85:
            x_text = med - (xhi - xlo) * 0.15
            ha = 'right'
        else:
            ha = 'left'
        ax.annotate(
            ci_str, xy=(med, y_at_med),
            xytext=(x_text, y_at_med * 0.80),
            fontsize=9, color=PAL.RED,
            arrowprops=dict(arrowstyle='->', color=PAL.RED, lw=0.7),
            ha=ha,
        )

        ax.set_xlabel(p['label'])
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.set_title(p['title'])
        label_panel(ax, chr(ord('a') + i))

        if i == 0:
            ax.set_ylabel('Probability density')
        elif i == 2:
            ax.set_ylabel('Probability density')

        if i == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.85,
                      edgecolor='0.8', fancybox=False)

    fig.suptitle(
        r'Bayesian parameter constraints from Juno MWR'
        r' ($D_{\mathrm{cond}} = 24 \pm 10$ km, $\phi = 35^{\circ}$)',
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, 'bayesian_shrinkage', FIGURES_DIR, formats=('png', 'pdf'))


if __name__ == '__main__':
    main()
