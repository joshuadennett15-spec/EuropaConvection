"""150-sample campaign: baseline + heatbal_ocean + heatbal_total.

Canonical seed=42, repeat seed=137 if runtime allows.
Saves full diagnostics per experiment for thesis figures.
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))
sys.path.insert(0, str(_REPO / 'autoresearch'))

import numpy as np
from convection_2d import ConvectionHypothesis
from monte_carlo_2d import MonteCarloRunner2D
from objectives import compute_latitude_score, _js_divergence, _JS_BIN_EDGES

OUT_DIR = Path(__file__).parent / 'campaign_150'


def run_scenario(pattern, q_star, n_samples, seed, hypothesis=None):
    runner = MonteCarloRunner2D(
        n_iterations=n_samples, seed=seed, n_workers=12,
        n_lat=19, nx=31,
        ocean_pattern=pattern, q_star=q_star,
        max_steps=1500, eq_threshold=1e-12,
        verbose=False, hypothesis=hypothesis,
    )
    return runner.run()


def clamp_fraction_vs_latitude(mc):
    D_cond = mc.D_cond_profiles
    H = mc.H_profiles
    return np.mean(D_cond >= 0.94 * H, axis=0)


def convecting_fraction_vs_latitude(mc):
    D_conv = mc.D_conv_profiles
    return np.mean(D_conv > 0.01, axis=0)


def profile_stats(arr):
    """Return median, 16th, 84th percentiles along axis=0."""
    return {
        'median': np.median(arr, axis=0),
        'p16': np.percentile(arr, 15.87, axis=0),
        'p84': np.percentile(arr, 84.13, axis=0),
    }


def run_experiment(name, hypothesis, n_samples, seed):
    scenario_cfgs = [
        ('uniform', 'uniform', None),
        ('polar', 'polar_enhanced', 0.455),
        ('equator', 'equator_enhanced', 0.4),
    ]

    scenarios_for_scorer = {}
    diagnostics = {}

    for sname, pattern, qs in scenario_cfgs:
        mc = run_scenario(pattern, qs, n_samples, seed, hypothesis)

        scenarios_for_scorer[sname] = {
            'latitudes_deg': mc.latitudes_deg,
            'D_cond_profiles': mc.D_cond_profiles,
            'D_conv_profiles': mc.D_conv_profiles,
            'H_profiles': mc.H_profiles,
            'Ra_profiles': mc.Ra_profiles,
        }

        diagnostics[sname] = {
            'latitudes_deg': mc.latitudes_deg.tolist(),
            'n_valid': int(mc.n_valid),
            'n_iterations': int(mc.n_iterations),
            'clamp_fraction': clamp_fraction_vs_latitude(mc).tolist(),
            'convecting_fraction': convecting_fraction_vs_latitude(mc).tolist(),
            'D_cond': {k: v.tolist() for k, v in profile_stats(mc.D_cond_profiles).items()},
            'D_conv': {k: v.tolist() for k, v in profile_stats(mc.D_conv_profiles).items()},
            'H_total': {k: v.tolist() for k, v in profile_stats(mc.H_profiles).items()},
            'Ra': {k: v.tolist() for k, v in profile_stats(mc.Ra_profiles).items()},
        }

    score, metrics = compute_latitude_score(scenarios_for_scorer, consistency_error=0.0)
    return score, metrics, diagnostics


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ('baseline', None),
        ('heatbal_ocean', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
        ('heatbal_total', ConvectionHypothesis('heat_balance', {'include_tidal': True})),
    ]

    seeds = [42, 137]
    n_samples = 150

    summary = []

    for name, hyp in experiments:
        for seed in seeds:
            tag = f'{name}_s{seed}'
            print(f'\n{"="*70}')
            print(f'  {tag}  (n={n_samples})')
            print(f'{"="*70}')

            t0 = time.time()
            score, metrics, diagnostics = run_experiment(name, hyp, n_samples, seed)
            dt = time.time() - t0

            print(f'  Score:           {score:.2f}  ({dt:.0f}s)')
            print(f'  profile_JS_min:  {metrics["profile_JS_min"]:.4f}')
            print(f'  profile_JS_mean: {metrics["profile_JS_mean"]:.4f}')
            print(f'  JS@35 (report):  {metrics["JS_35"]:.4f}')
            print(f'  JS_peak:         {metrics["JS_peak"]:.4f} at {metrics["phi_peak_js"]:.0f} deg')
            print(f'  D_conv contrast: {metrics["D_conv_contrast"]:.2f} km')
            print(f'  D_cond@35:       {metrics["D_cond_35_median"]:.1f} km')
            print(f'  Juno excess:     {metrics["juno_excess"]:.1f} km')

            # Clamp summary
            for sname in ['uniform', 'polar', 'equator']:
                cf = diagnostics[sname]['clamp_fraction']
                mean_cf = np.mean(cf)
                print(f'  Clamp 0.95H ({sname:8s}): {mean_cf:.1%} mean')

            entry = {
                'experiment': name,
                'seed': seed,
                'n_samples': n_samples,
                'score': float(score),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in metrics.items()},
                'diagnostics': diagnostics,
                'runtime_seconds': dt,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

            # Save per-experiment JSON
            out_path = OUT_DIR / f'{tag}.json'
            with open(out_path, 'w') as f:
                json.dump(entry, f, cls=_Enc, indent=2)
            print(f'  Saved: {out_path.name}')

            summary.append({
                'tag': tag,
                'experiment': name,
                'seed': seed,
                'score': float(score),
                'profile_JS_min': float(metrics['profile_JS_min']),
                'profile_JS_mean': float(metrics['profile_JS_mean']),
                'JS_35': float(metrics['JS_35']),
                'JS_peak': float(metrics['JS_peak']),
                'phi_peak_js': float(metrics['phi_peak_js']),
                'D_conv_contrast': float(metrics['D_conv_contrast']),
                'D_cond_35_median': float(metrics['D_cond_35_median']),
                'juno_excess': float(metrics['juno_excess']),
                'runtime_seconds': dt,
            })

    # Final ranking
    print(f'\n{"="*70}')
    print('  CAMPAIGN RESULTS')
    print(f'{"="*70}')

    # Group by experiment, average across seeds
    by_exp = {}
    for s in summary:
        by_exp.setdefault(s['experiment'], []).append(s)

    print(f'\n  {"Experiment":20s}  {"Score":>8s}  {"pJS_min":>8s}  {"pJS_mean":>8s}  {"JS@35":>8s}  {"Dcond35":>8s}  {"Dconv_c":>8s}')
    print(f'  {"-"*20}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}')

    ranked = sorted(by_exp.items(), key=lambda x: np.mean([r['score'] for r in x[1]]))
    for name, runs in ranked:
        ms = np.mean([r['score'] for r in runs])
        mpj = np.mean([r['profile_JS_min'] for r in runs])
        mpjm = np.mean([r['profile_JS_mean'] for r in runs])
        mj35 = np.mean([r['JS_35'] for r in runs])
        mdc = np.mean([r['D_cond_35_median'] for r in runs])
        mdvc = np.mean([r['D_conv_contrast'] for r in runs])
        print(f'  {name:20s}  {ms:+8.2f}  {mpj:8.4f}  {mpjm:8.4f}  {mj35:8.4f}  {mdc:8.1f}  {mdvc:8.2f}')

    # Seed stability
    print(f'\n  Seed stability (score diff):')
    for name, runs in by_exp.items():
        if len(runs) >= 2:
            diff = abs(runs[0]['score'] - runs[1]['score'])
            pj_diff = abs(runs[0]['profile_JS_min'] - runs[1]['profile_JS_min'])
            print(f'    {name:20s}  score_diff={diff:.2f}  pJS_diff={pj_diff:.4f}')

    # Save summary
    summary_path = OUT_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, cls=_Enc, indent=2)
    print(f'\n  Summary saved: {summary_path}')

    print(f'\n{"="*70}')
    print('  CAMPAIGN COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
