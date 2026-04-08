"""Focused validation: baseline + heatbal_ocean + heatbal_total, 30 samples x 2 seeds.

Checks before 150-sample campaign:
1. D_cond separation at 35 deg stable across seeds (not a fluke)
2. Heat-balance not dominated by 0.05H / 0.95H clamps
3. Latitude profiles make physical sense (higher q_ocean -> thinner D_cond)
4. heatbal_total does not blow up or erase heatbal_ocean signal
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
    mc = runner.run()
    return mc


def run_experiment(name, hypothesis, n_samples, seed):
    scenario_cfgs = [
        ('uniform', 'uniform', None),
        ('polar', 'polar_enhanced', 0.455),
        ('equator', 'equator_enhanced', 0.4),
    ]
    scenarios = {}
    mc_objects = {}
    for sname, pattern, qs in scenario_cfgs:
        mc = run_scenario(pattern, qs, n_samples, seed, hypothesis)
        mc_objects[sname] = mc
        scenarios[sname] = {
            'latitudes_deg': mc.latitudes_deg,
            'D_cond_profiles': mc.D_cond_profiles,
            'D_conv_profiles': mc.D_conv_profiles,
            'H_profiles': mc.H_profiles,
            'Ra_profiles': mc.Ra_profiles,
        }

    score, metrics = compute_latitude_score(scenarios, consistency_error=0.0)
    return score, metrics, mc_objects


def check_clamp_fraction(mc, H_total_profiles):
    """Check what fraction of columns hit the 0.05H or 0.95H clamp."""
    D_cond = mc.D_cond_profiles  # (n_valid, n_lat)
    H = H_total_profiles          # (n_valid, n_lat)
    low_clamp = np.mean(D_cond <= 0.06 * H)   # near 0.05H
    high_clamp = np.mean(D_cond >= 0.94 * H)  # near 0.95H
    return low_clamp, high_clamp


def print_latitude_profile(mc, label):
    """Print D_cond and D_conv median profiles for physical sense check."""
    lats = mc.latitudes_deg
    D_cond_med = np.median(mc.D_cond_profiles, axis=0)
    D_conv_med = np.median(mc.D_conv_profiles, axis=0)
    H_med = np.median(mc.H_profiles, axis=0)
    Ra_med = np.median(mc.Ra_profiles, axis=0)

    print(f'\n  Latitude profile ({label}):')
    print(f'  {"Lat":>5s}  {"D_cond":>8s}  {"D_conv":>8s}  {"H_total":>8s}  {"Ra":>8s}  {"frac_conv":>10s}')
    for i in range(0, len(lats), max(1, len(lats) // 10)):
        frac = D_conv_med[i] / max(H_med[i], 0.01)
        print(f'  {lats[i]:5.1f}  {D_cond_med[i]:8.2f}  {D_conv_med[i]:8.2f}  {H_med[i]:8.2f}  {Ra_med[i]:8.0f}  {frac:10.3f}')


def main():
    seeds = [42, 137]
    n_samples = 30
    experiments = [
        ('baseline', None),
        ('heatbal_ocean', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
        ('heatbal_total', ConvectionHypothesis('heat_balance', {'include_tidal': True})),
    ]

    all_results = {}

    for name, hyp in experiments:
        print(f'\n{"="*70}')
        print(f'  EXPERIMENT: {name}')
        print(f'{"="*70}')

        seed_scores = []
        for seed in seeds:
            t0 = time.time()
            score, metrics, mc_objs = run_experiment(name, hyp, n_samples, seed)
            dt = time.time() - t0

            print(f'\n  Seed {seed}: score={score:.2f} ({dt:.0f}s)')
            print(f'    JS(D_cond):      {metrics.get("JS_discriminability", 0):.6f}')
            print(f'    JS(D_conv):      {metrics.get("JS_discriminability_Dconv", 0):.6f}')
            print(f'    D_conv contrast: {metrics.get("D_conv_contrast", 0):.2f} km')
            print(f'    D_cond@35:       {metrics.get("D_cond_35_median", 0):.1f} km')
            print(f'    Ra eq/pole:      {metrics.get("Ra_eq_median", 0):.0f} / {metrics.get("Ra_pole_median", 0):.0f}')

            # Check 2: Clamp fraction
            for sname, mc in mc_objs.items():
                lo, hi = check_clamp_fraction(mc, mc.H_profiles)
                if lo > 0.05 or hi > 0.05:
                    print(f'    CLAMP WARNING ({sname}): {lo:.1%} at 0.05H, {hi:.1%} at 0.95H')

            # Check 3: Print latitude profiles for physical sense
            if seed == seeds[0]:
                for sname in ['uniform', 'polar', 'equator']:
                    print_latitude_profile(mc_objs[sname], f'{name}/{sname}')

            seed_scores.append((score, metrics))

        all_results[name] = seed_scores

    # Summary: Check 1 — seed stability
    print(f'\n{"="*70}')
    print('  SEED STABILITY CHECK')
    print(f'{"="*70}')
    for name, seed_results in all_results.items():
        scores = [s for s, _ in seed_results]
        js_dcond = [m.get('JS_discriminability', 0) for _, m in seed_results]
        js_dconv = [m.get('JS_discriminability_Dconv', 0) for _, m in seed_results]
        print(f'\n  {name}:')
        print(f'    Score:     {scores[0]:.2f} / {scores[1]:.2f}  (diff={abs(scores[0]-scores[1]):.2f})')
        print(f'    JS(Dcond): {js_dcond[0]:.6f} / {js_dcond[1]:.6f}')
        print(f'    JS(Dconv): {js_dconv[0]:.6f} / {js_dconv[1]:.6f}')

    # Check 4: heatbal_total vs heatbal_ocean
    print(f'\n{"="*70}')
    print('  HEATBAL_TOTAL vs HEATBAL_OCEAN')
    print(f'{"="*70}')
    if 'heatbal_ocean' in all_results and 'heatbal_total' in all_results:
        ocean_js = np.mean([m.get('JS_discriminability', 0) for _, m in all_results['heatbal_ocean']])
        total_js = np.mean([m.get('JS_discriminability', 0) for _, m in all_results['heatbal_total']])
        ocean_dcond = np.mean([m.get('D_cond_35_median', 0) for _, m in all_results['heatbal_ocean']])
        total_dcond = np.mean([m.get('D_cond_35_median', 0) for _, m in all_results['heatbal_total']])
        print(f'  Ocean: JS={ocean_js:.6f}, D_cond@35={ocean_dcond:.1f} km')
        print(f'  Total: JS={total_js:.6f}, D_cond@35={total_dcond:.1f} km')
        if total_js < ocean_js * 0.5:
            print('  WARNING: tidal inclusion erases >50% of ocean signal')
        elif total_js > ocean_js * 1.5:
            print('  NOTE: tidal inclusion amplifies signal')
        else:
            print('  OK: tidal inclusion preserves signal')

    print(f'\n{"="*70}')
    print('  VALIDATION COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
