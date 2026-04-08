"""Onset gate diagnostic: is the Ra_crit=1000 check killing convection?

3-run comparison, same seed, same parameters:
  A. baseline (onset gate ON, Ra_crit=1000)
  B. ra_onset with ra_crit_override=1 (gate effectively disabled via adjuster)
  C. heatbal_ocean + ra_crit=1 combined (heat-balance with gate disabled)

Reports: convecting fraction, D_conv/H, D_cond@35, Ra, Nu, profile JS, Juno fit.
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
from objectives import compute_latitude_score


def run_scenario(pattern, q_star, n_samples, seed, hypothesis=None):
    runner = MonteCarloRunner2D(
        n_iterations=n_samples, seed=seed, n_workers=12,
        n_lat=19, nx=31,
        ocean_pattern=pattern, q_star=q_star,
        max_steps=1500, eq_threshold=1e-12,
        verbose=False, hypothesis=hypothesis,
    )
    return runner.run()


def convecting_frac(mc):
    """Fraction of (sample, latitude) pairs with D_conv > 0.5 km."""
    return float(np.mean(mc.D_conv_profiles > 0.5))


def convecting_frac_by_lat(mc):
    return np.mean(mc.D_conv_profiles > 0.5, axis=0)


def dconv_over_H(mc):
    """Median D_conv/H at each latitude."""
    ratio = mc.D_conv_profiles / np.maximum(mc.H_profiles, 1.0)
    return np.median(ratio, axis=0)


def clamp_frac(mc):
    return float(np.mean(mc.D_cond_profiles >= 0.94 * mc.H_profiles))


def run_experiment(name, hypothesis, n_samples, seed):
    scenarios_cfg = [
        ('uniform', 'uniform', None),
        ('polar', 'polar_enhanced', 0.455),
        ('equator', 'equator_enhanced', 0.4),
    ]
    scenarios = {}
    mc_objs = {}
    for sname, pattern, qs in scenarios_cfg:
        mc = run_scenario(pattern, qs, n_samples, seed, hypothesis)
        mc_objs[sname] = mc
        scenarios[sname] = {
            'latitudes_deg': mc.latitudes_deg,
            'D_cond_profiles': mc.D_cond_profiles,
            'D_conv_profiles': mc.D_conv_profiles,
            'H_profiles': mc.H_profiles,
            'Ra_profiles': mc.Ra_profiles,
        }
    score, metrics = compute_latitude_score(scenarios, consistency_error=0.0)
    return score, metrics, mc_objs


def print_diagnostics(name, score, metrics, mc_objs):
    print(f'\n{"="*70}')
    print(f'  {name}')
    print(f'{"="*70}')
    print(f'  Score:           {score:.2f}')
    print(f'  profile_JS_min:  {metrics["profile_JS_min"]:.4f}')
    print(f'  JS@35:           {metrics["JS_35"]:.4f}')
    print(f'  JS_peak:         {metrics["JS_peak"]:.4f} at {metrics["phi_peak_js"]:.0f} deg')
    print(f'  D_conv contrast: {metrics["D_conv_contrast"]:.2f} km')
    print(f'  D_cond@35:       {metrics["D_cond_35_median"]:.1f} km')
    print(f'  Juno excess:     {metrics["juno_excess"]:.1f} km')

    lats = mc_objs['uniform'].latitudes_deg

    # Per-scenario summary
    for sname in ['uniform', 'polar', 'equator']:
        mc = mc_objs[sname]
        cf = convecting_frac(mc)
        cl = clamp_frac(mc)
        D_cond_med = np.median(mc.D_cond_profiles, axis=0)
        D_conv_med = np.median(mc.D_conv_profiles, axis=0)
        H_med = np.median(mc.H_profiles, axis=0)
        Ra_med = np.median(mc.Ra_profiles, axis=0)
        Nu_med = np.median(mc.Nu_profiles, axis=0) if hasattr(mc, 'Nu_profiles') and mc.Nu_profiles is not None else np.ones(len(lats))

        print(f'\n  --- {sname} ---')
        print(f'  Convecting fraction: {cf:.1%}')
        print(f'  Clamp 0.95H:         {cl:.1%}')
        print(f'  {"Lat":>5s}  {"Dcond":>7s}  {"Dconv":>7s}  {"H":>7s}  {"Dconv/H":>7s}  {"Ra":>8s}  {"Nu":>6s}  {"conv%":>6s}')

        conv_by_lat = convecting_frac_by_lat(mc)
        dconv_h = dconv_over_H(mc)

        for i in range(0, len(lats), 2):  # every other latitude for readability
            print(f'  {lats[i]:5.1f}  {D_cond_med[i]:7.1f}  {D_conv_med[i]:7.1f}  '
                  f'{H_med[i]:7.1f}  {dconv_h[i]:7.3f}  {Ra_med[i]:8.0f}  '
                  f'{Nu_med[i]:6.2f}  {conv_by_lat[i]:6.1%}')


def main():
    seed = 42
    n_samples = 30

    experiments = [
        ('A: baseline (onset gate ON)', None),
        ('B: ra_crit=1 (onset gate OFF)', ConvectionHypothesis('ra_onset', {'ra_crit_override': 1})),
        ('C: heatbal + ra_crit=1', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
    ]

    print(f'Seed: {seed}, Samples: {n_samples}')
    print(f'Question: Is the Ra_crit=1000 onset gate killing convection?')

    all_results = []
    for name, hyp in experiments:
        t0 = time.time()
        score, metrics, mc_objs = run_experiment(name, hyp, n_samples, seed)
        dt = time.time() - t0
        print_diagnostics(f'{name}  ({dt:.0f}s)', score, metrics, mc_objs)
        all_results.append((name, score, metrics))

    # Comparison table
    print(f'\n{"="*70}')
    print('  COMPARISON')
    print(f'{"="*70}')
    print(f'  {"Experiment":40s}  {"Score":>7s}  {"pJS":>7s}  {"Dc@35":>7s}  {"Dconv_c":>7s}  {"Juno":>5s}')
    for name, score, m in all_results:
        print(f'  {name:40s}  {score:+7.2f}  {m["profile_JS_min"]:7.4f}  '
              f'{m["D_cond_35_median"]:7.1f}  {m["D_conv_contrast"]:7.2f}  {m["juno_excess"]:5.1f}')

    print(f'\n{"="*70}')
    print('  ONSET DIAGNOSTIC COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
