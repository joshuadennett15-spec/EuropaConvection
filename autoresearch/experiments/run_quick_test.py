"""Quick test: baseline + heatbal_ocean + heatbal_total, 15 samples, new profile scorer."""
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


def run_scenario(pattern, q_star, n_samples, hypothesis=None):
    runner = MonteCarloRunner2D(
        n_iterations=n_samples, seed=42, n_workers=12,
        n_lat=9, nx=21,
        ocean_pattern=pattern, q_star=q_star,
        max_steps=300, eq_threshold=1e-10,
        verbose=False, hypothesis=hypothesis,
    )
    mc = runner.run()
    return {
        'latitudes_deg': mc.latitudes_deg,
        'D_cond_profiles': mc.D_cond_profiles,
        'D_conv_profiles': mc.D_conv_profiles,
        'H_profiles': mc.H_profiles,
        'Ra_profiles': mc.Ra_profiles,
    }


def main():
    experiments = [
        ('baseline', None),
        ('heatbal_ocean', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
        ('heatbal_total', ConvectionHypothesis('heat_balance', {'include_tidal': True})),
    ]

    n_samples = 15

    for name, hyp in experiments:
        t0 = time.time()
        print(f'\n{"="*60}')
        print(f'  EXPERIMENT: {name}')
        print(f'{"="*60}')

        scenarios = {}
        for sname, pattern, qs in [
            ('uniform', 'uniform', None),
            ('polar', 'polar_enhanced', 0.455),
            ('equator', 'equator_enhanced', 0.4),
        ]:
            scenarios[sname] = run_scenario(pattern, qs, n_samples, hypothesis=hyp)

        score, metrics = compute_latitude_score(scenarios, consistency_error=0.0)
        dt = time.time() - t0
        print(f'  Score:           {score:.2f}  ({dt:.0f}s)')
        print(f'  profile_JS_min:  {metrics["profile_JS_min"]:.4f}')
        print(f'  profile_JS_mean: {metrics["profile_JS_mean"]:.4f}')
        print(f'  JS@35 (report):  {metrics["JS_35"]:.4f}')
        print(f'  JS_peak:         {metrics["JS_peak"]:.4f}  at {metrics["phi_peak_js"]:.0f} deg')
        print(f'  D_conv contrast: {metrics["D_conv_contrast"]:.2f} km')
        print(f'  D_cond@35:       {metrics["D_cond_35_median"]:.1f} km')
        print(f'  Juno excess:     {metrics["juno_excess"]:.1f} km')

    print(f'\n{"="*60}')
    print('  QUICK TEST COMPLETE')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
