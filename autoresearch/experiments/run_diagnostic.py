"""Diagnostic pass: JS(D_cond) vs latitude, clamp fraction vs latitude,
active-only D_conv, and convecting fraction.

Runs baseline + heatbal_ocean + heatbal_total at 30 samples.
Answers: where is the real signal, and how much is clamp artifact?
"""
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))
sys.path.insert(0, str(_REPO / 'autoresearch'))

import numpy as np
from convection_2d import ConvectionHypothesis
from monte_carlo_2d import MonteCarloRunner2D
from objectives import _js_divergence, _JS_BIN_EDGES


def run_scenario(pattern, q_star, n_samples, hypothesis=None):
    runner = MonteCarloRunner2D(
        n_iterations=n_samples, seed=42, n_workers=12,
        n_lat=19, nx=31,
        ocean_pattern=pattern, q_star=q_star,
        max_steps=1500, eq_threshold=1e-12,
        verbose=False, hypothesis=hypothesis,
    )
    return runner.run()


def js_vs_latitude(mc_a, mc_b):
    """Compute JS(D_cond) at each latitude index."""
    n_lat = mc_a.D_cond_profiles.shape[1]
    js_vals = np.zeros(n_lat)
    for i in range(n_lat):
        js_vals[i] = _js_divergence(
            mc_a.D_cond_profiles[:, i],
            mc_b.D_cond_profiles[:, i],
        )
    return js_vals


def clamp_fraction_vs_latitude(mc):
    """Fraction of samples hitting 0.95H clamp at each latitude."""
    D_cond = mc.D_cond_profiles  # (n_valid, n_lat)
    H = mc.H_profiles            # (n_valid, n_lat)
    return np.mean(D_cond >= 0.94 * H, axis=0)


def convecting_fraction_vs_latitude(mc):
    """Fraction of samples with D_conv > 0.01 km at each latitude."""
    D_conv = mc.D_conv_profiles
    return np.mean(D_conv > 0.01, axis=0)


def active_dconv_stats(mc):
    """D_conv stats only for columns that are actually convecting (D_conv > 0.5 km)."""
    D_conv = mc.D_conv_profiles  # (n_valid, n_lat)
    n_lat = D_conv.shape[1]
    medians = np.zeros(n_lat)
    means = np.zeros(n_lat)
    for i in range(n_lat):
        col = D_conv[:, i]
        active = col[col > 0.5]
        if len(active) > 0:
            medians[i] = np.median(active)
            means[i] = np.mean(active)
    return medians, means


def main():
    n_samples = 30
    experiments = [
        ('baseline', None),
        ('heatbal_ocean', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
        ('heatbal_total', ConvectionHypothesis('heat_balance', {'include_tidal': True})),
    ]

    scenario_cfgs = [
        ('uniform', 'uniform', None),
        ('polar', 'polar_enhanced', 0.455),
        ('equator', 'equator_enhanced', 0.4),
    ]

    for exp_name, hyp in experiments:
        t0 = time.time()
        print(f'\n{"="*70}')
        print(f'  {exp_name}')
        print(f'{"="*70}')

        mc_by_scenario = {}
        for sname, pattern, qs in scenario_cfgs:
            mc_by_scenario[sname] = run_scenario(pattern, qs, n_samples, hyp)

        lats = mc_by_scenario['uniform'].latitudes_deg
        dt = time.time() - t0
        print(f'  ({dt:.0f}s)')

        # 1. JS(D_cond) vs latitude for each scenario pair
        print(f'\n  JS(D_cond) vs latitude:')
        print(f'  {"Lat":>5s}  {"U-P":>8s}  {"U-E":>8s}  {"P-E":>8s}')
        js_up = js_vs_latitude(mc_by_scenario['uniform'], mc_by_scenario['polar'])
        js_ue = js_vs_latitude(mc_by_scenario['uniform'], mc_by_scenario['equator'])
        js_pe = js_vs_latitude(mc_by_scenario['polar'], mc_by_scenario['equator'])
        for i in range(len(lats)):
            print(f'  {lats[i]:5.1f}  {js_up[i]:8.4f}  {js_ue[i]:8.4f}  {js_pe[i]:8.4f}')

        peak_lat_up = lats[np.argmax(js_up)]
        peak_lat_ue = lats[np.argmax(js_ue)]
        peak_lat_pe = lats[np.argmax(js_pe)]
        print(f'  Peak JS latitude: U-P={peak_lat_up:.0f}deg, U-E={peak_lat_ue:.0f}deg, P-E={peak_lat_pe:.0f}deg')
        print(f'  Min pairwise JS (scorer metric): {min(js_up.min(), js_ue.min(), js_pe.min()):.6f}')

        # 2. Clamp fraction vs latitude
        print(f'\n  0.95H clamp fraction vs latitude:')
        print(f'  {"Lat":>5s}  {"uniform":>8s}  {"polar":>8s}  {"equator":>8s}')
        for sname in ['uniform', 'polar', 'equator']:
            cf = clamp_fraction_vs_latitude(mc_by_scenario[sname])
            if sname == 'uniform':
                cf_u = cf
            elif sname == 'polar':
                cf_p = cf
            else:
                cf_e = cf
        for i in range(len(lats)):
            print(f'  {lats[i]:5.1f}  {cf_u[i]:8.1%}  {cf_p[i]:8.1%}  {cf_e[i]:8.1%}')

        # 3. Convecting fraction vs latitude
        print(f'\n  Convecting fraction (D_conv > 0.01 km) vs latitude:')
        print(f'  {"Lat":>5s}  {"uniform":>8s}  {"polar":>8s}  {"equator":>8s}')
        for sname in ['uniform', 'polar', 'equator']:
            cvf = convecting_fraction_vs_latitude(mc_by_scenario[sname])
            if sname == 'uniform':
                cvf_u = cvf
            elif sname == 'polar':
                cvf_p = cvf
            else:
                cvf_e = cvf
        for i in range(len(lats)):
            print(f'  {lats[i]:5.1f}  {cvf_u[i]:8.1%}  {cvf_p[i]:8.1%}  {cvf_e[i]:8.1%}')

        # 4. Active-only D_conv
        print(f'\n  Active-only D_conv median (km) vs latitude:')
        print(f'  {"Lat":>5s}  {"uniform":>8s}  {"polar":>8s}  {"equator":>8s}')
        for sname in ['uniform', 'polar', 'equator']:
            med, _ = active_dconv_stats(mc_by_scenario[sname])
            if sname == 'uniform':
                dv_u = med
            elif sname == 'polar':
                dv_p = med
            else:
                dv_e = med
        for i in range(len(lats)):
            print(f'  {lats[i]:5.1f}  {dv_u[i]:8.2f}  {dv_p[i]:8.2f}  {dv_e[i]:8.2f}')

    print(f'\n{"="*70}')
    print('  DIAGNOSTIC COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
