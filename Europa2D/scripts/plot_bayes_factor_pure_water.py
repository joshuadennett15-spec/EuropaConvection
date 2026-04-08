#!/usr/bin/env python3
"""
Bayes factor bar chart: ocean transport models vs uniform reference,
pure-water Juno constraint (D_cond = 24 +/- 10 km, 30-40 deg band).

Y-axis: Bayes factor B relative to Uniform.
X-axis: Ocean heat transport scenario.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

import matplotlib.pyplot as plt
from pub_style import PAL, save_fig
from profile_diagnostics import band_mean_samples

# ── Constants ────────────────────────────────────────────────────────────────
D_OBS = 24.0
SIGMA_OBS = 10.0
SIGMA_MODEL = 3.0
SIGMA_TOTAL = np.sqrt(SIGMA_OBS**2 + SIGMA_MODEL**2)
JUNO_BAND = (30.0, 40.0)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

SCENARIOS = [
    ('uniform_transport',       'Uniform',            PAL.BLUE),
    ('soderlund2014_equator',   'Equator-\nenhanced',  PAL.ORANGE),
    ('lemasquerier2023_polar',  'Polar-\nenhanced',    PAL.GREEN),
]


def compute_log_evidence(D_band):
    """Log marginal likelihood via importance sampling."""
    log_L = -0.5 * ((D_band - D_OBS) / SIGMA_TOTAL) ** 2
    N = len(log_L)
    return np.log(np.sum(np.exp(log_L))) - np.log(N)


def bootstrap_log_evidence(D_band, n_boot=2000, rng=None):
    """Bootstrap confidence interval on log-evidence."""
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(D_band)
    boot_logZ = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        boot_logZ[b] = compute_log_evidence(D_band[idx])
    return boot_logZ


def main():
    mpl.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.6,
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

    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # ── Compute evidences ────────────────────────────────────────────────
    labels = []
    colors = []
    log_Zs = []
    boot_logZs = []

    for key, label, color in SCENARIOS:
        path = os.path.join(RESULTS_DIR, f'mc_2d_{key}_500.npz')
        d = np.load(path, allow_pickle=True)
        lats = d['latitudes_deg']
        D_band = band_mean_samples(lats, d['D_cond_profiles'], JUNO_BAND)

        log_Zs.append(compute_log_evidence(D_band))
        boot_logZs.append(bootstrap_log_evidence(D_band, rng=rng))
        labels.append(label)
        colors.append(color)

    # Bayes factors relative to Uniform
    ref_logZ = log_Zs[0]
    ref_boot = boot_logZs[0]

    BFs = [np.exp(lz - ref_logZ) for lz in log_Zs]

    # Bootstrap BF CIs
    BF_lo = []
    BF_hi = []
    for i in range(len(SCENARIOS)):
        boot_BF = np.exp(boot_logZs[i] - ref_boot)
        BF_lo.append(np.percentile(boot_BF, 16))
        BF_hi.append(np.percentile(boot_BF, 84))

    err_lo = [BFs[i] - BF_lo[i] for i in range(len(BFs))]
    err_hi = [BF_hi[i] - BFs[i] for i in range(len(BFs))]

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    x_pos = np.arange(len(SCENARIOS))
    bar_width = 0.55

    bars = ax.bar(
        x_pos, BFs, width=bar_width, color=colors, edgecolor='k',
        linewidth=0.6, alpha=0.75, zorder=3,
        yerr=[err_lo, err_hi], capsize=4,
        error_kw=dict(lw=1.0, capthick=0.8, color='0.3', zorder=4),
    )

    # Reference line at B = 1
    ax.axhline(1.0, color='0.4', lw=0.8, ls='--', zorder=2)

    # Jeffreys thresholds (shaded bands)
    ymax = 4.0
    ax.axhspan(1/3, 3, alpha=0.04, color='0.5', zorder=1)
    ax.text(
        0.97, 0.72,
        r'Inconclusive ($\frac{1}{3} < B < 3$)',
        fontsize=8, color='0.5', ha='right', va='bottom',
        transform=ax.transAxes,
    )

    # BF value labels on bars
    for i, bf in enumerate(BFs):
        ax.text(
            x_pos[i], bf + err_hi[i] + 0.06,
            rf'$B = {bf:.2f}$',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold',
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Bayes factor $B$ (vs.\ Uniform)')
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, len(SCENARIOS) - 0.5)

    ax.set_title(
        r'Model comparison: pure water Juno constraint'
        '\n'
        r'{\small $D_{\mathrm{cond}} = 24 \pm 10$ km at $30$--$40^{\circ}$ latitude}'
    )

    fig.tight_layout()
    save_fig(fig, 'bayes_factor_pure_water', FIGURES_DIR, formats=('png', 'pdf'))


if __name__ == '__main__':
    main()
