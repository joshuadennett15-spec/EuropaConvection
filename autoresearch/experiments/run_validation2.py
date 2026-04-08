"""Validation pass 2: baseline + heatbal_ocean + heatbal_total, 30 samples x 2 seeds.
Uses updated profile-level scorer. Confirms ranking before 150-sample campaign."""
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


def run_experiment(name, hypothesis, n_samples, seed):
    scenario_cfgs = [
        ('uniform', 'uniform', None),
        ('polar', 'polar_enhanced', 0.455),
        ('equator', 'equator_enhanced', 0.4),
    ]
    scenarios = {}
    for sname, pattern, qs in scenario_cfgs:
        runner = MonteCarloRunner2D(
            n_iterations=n_samples, seed=seed, n_workers=12,
            n_lat=19, nx=31,
            ocean_pattern=pattern, q_star=qs,
            max_steps=1500, eq_threshold=1e-12,
            verbose=False, hypothesis=hypothesis,
        )
        mc = runner.run()
        scenarios[sname] = {
            'latitudes_deg': mc.latitudes_deg,
            'D_cond_profiles': mc.D_cond_profiles,
            'D_conv_profiles': mc.D_conv_profiles,
            'H_profiles': mc.H_profiles,
            'Ra_profiles': mc.Ra_profiles,
        }
    score, metrics = compute_latitude_score(scenarios, consistency_error=0.0)
    return score, metrics


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
        print(f'  {name}')
        print(f'{"="*70}')
        seed_results = []
        for seed in seeds:
            t0 = time.time()
            score, metrics = run_experiment(name, hyp, n_samples, seed)
            dt = time.time() - t0
            print(f'\n  Seed {seed}: score={score:.2f} ({dt:.0f}s)')
            print(f'    profile_JS_min:  {metrics["profile_JS_min"]:.4f}')
            print(f'    profile_JS_mean: {metrics["profile_JS_mean"]:.4f}')
            print(f'    JS@35 (report):  {metrics["JS_35"]:.4f}')
            print(f'    JS_peak:         {metrics["JS_peak"]:.4f} at {metrics["phi_peak_js"]:.0f} deg')
            print(f'    D_conv contrast: {metrics["D_conv_contrast"]:.2f} km')
            print(f'    D_cond@35:       {metrics["D_cond_35_median"]:.1f} km')
            print(f'    Juno excess:     {metrics["juno_excess"]:.1f} km')
            seed_results.append((score, metrics))
        all_results[name] = seed_results

    # Seed stability
    print(f'\n{"="*70}')
    print('  SEED STABILITY')
    print(f'{"="*70}')
    for name, results in all_results.items():
        s = [r[0] for r in results]
        pj = [r[1]['profile_JS_min'] for r in results]
        dc = [r[1]['D_cond_35_median'] for r in results]
        print(f'\n  {name}:')
        print(f'    Score:          {s[0]:+.2f} / {s[1]:+.2f}  (diff={abs(s[0]-s[1]):.2f})')
        print(f'    profile_JS_min: {pj[0]:.4f} / {pj[1]:.4f}  (diff={abs(pj[0]-pj[1]):.4f})')
        print(f'    D_cond@35:      {dc[0]:.1f} / {dc[1]:.1f} km')

    # Ranking
    print(f'\n{"="*70}')
    print('  RANKING (mean across seeds)')
    print(f'{"="*70}')
    ranked = sorted(all_results.items(), key=lambda x: np.mean([r[0] for r in x[1]]))
    for i, (name, results) in enumerate(ranked):
        mean_score = np.mean([r[0] for r in results])
        mean_pj = np.mean([r[1]['profile_JS_min'] for r in results])
        mean_dc = np.mean([r[1]['D_cond_35_median'] for r in results])
        print(f'  {i+1}. {name:20s}  score={mean_score:+.2f}  pJS={mean_pj:.4f}  D_cond@35={mean_dc:.1f}km')

    print(f'\n{"="*70}')
    print('  VALIDATION 2 COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
