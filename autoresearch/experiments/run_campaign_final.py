"""Final 150-sample campaign: onset sensitivity + heat-balance interaction.

4 experiments, seed=42 canonical, seed=137 repeat:
  1. baseline_ra1000          — legacy production behavior
  2. onset_relaxed_ra100      — onset-sensitive alternative
  3. onset_ra100_heatbal      — onset relaxation + heat-balance coupling
  4. heatbal_ra1000           — heat-balance on legacy baseline

Cleanly separates onset sensitivity from heat-balance coupling.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))
sys.path.insert(0, str(_REPO / 'autoresearch'))

import numpy as np
from convection_2d import ConvectionHypothesis
from monte_carlo_2d import MonteCarloRunner2D
from objectives import compute_latitude_score

OUT_DIR = Path(__file__).parent / 'campaign_final'


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def run_scenario(pattern, q_star, n_samples, seed, hypothesis=None):
    runner = MonteCarloRunner2D(
        n_iterations=n_samples, seed=seed, n_workers=12,
        n_lat=19, nx=31,
        ocean_pattern=pattern, q_star=q_star,
        max_steps=1500, eq_threshold=1e-12,
        verbose=False, hypothesis=hypothesis,
    )
    return runner.run()


def clamp_frac_by_lat(mc):
    return np.mean(mc.D_cond_profiles >= 0.94 * mc.H_profiles, axis=0)


def conv_frac_by_lat(mc):
    return np.mean(mc.D_conv_profiles > 0.5, axis=0)


def profile_stats(arr):
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
            'clamp_fraction': clamp_frac_by_lat(mc).tolist(),
            'convecting_fraction': conv_frac_by_lat(mc).tolist(),
            'D_cond': {k: v.tolist() for k, v in profile_stats(mc.D_cond_profiles).items()},
            'D_conv': {k: v.tolist() for k, v in profile_stats(mc.D_conv_profiles).items()},
            'H_total': {k: v.tolist() for k, v in profile_stats(mc.H_profiles).items()},
        }

    score, metrics = compute_latitude_score(scenarios_for_scorer, consistency_error=0.0)
    return score, metrics, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=150)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_samples = args.samples
    seeds = args.seeds

    experiments = [
        ('baseline_ra1000', None),
        ('onset_relaxed_ra100', ConvectionHypothesis('ra_onset', {'ra_crit_override': 100})),
        ('onset_ra100_heatbal', ConvectionHypothesis('ra_onset_heatbal', {
            'ra_crit_override': 100, 'include_tidal': False,
        })),
        ('heatbal_ra1000', ConvectionHypothesis('heat_balance', {'include_tidal': False})),
    ]

    summary = []

    for name, hyp in experiments:
        for seed in seeds:
            tag = f'{name}_s{seed}'

            # Check if already done
            out_path = OUT_DIR / f'{tag}.json'
            if out_path.exists():
                print(f'[SKIP] {tag} (already exists)')
                with open(out_path) as f:
                    entry = json.load(f)
                summary.append({
                    'tag': tag, 'experiment': name, 'seed': seed,
                    'score': entry['score'],
                    **{k: v for k, v in entry['metrics'].items()},
                })
                continue

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

            for sname in ['uniform', 'polar', 'equator']:
                cf_mean = np.mean(diagnostics[sname]['clamp_fraction'])
                cv_mean = np.mean(diagnostics[sname]['convecting_fraction'])
                print(f'  {sname:8s}: clamp={cf_mean:.1%}  convecting={cv_mean:.1%}')

            entry = {
                'experiment': name, 'seed': seed, 'n_samples': n_samples,
                'score': float(score),
                'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in metrics.items()},
                'diagnostics': diagnostics,
                'runtime_seconds': dt,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

            with open(out_path, 'w') as f:
                json.dump(entry, f, cls=_Enc, indent=2)
            print(f'  Saved: {out_path.name}')

            summary.append({
                'tag': tag, 'experiment': name, 'seed': seed,
                'score': float(score),
                **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                   for k, v in metrics.items()},
            })

    # Ranking
    print(f'\n{"="*70}')
    print('  FINAL CAMPAIGN RESULTS')
    print(f'{"="*70}')

    by_exp = {}
    for s in summary:
        by_exp.setdefault(s['experiment'], []).append(s)

    print(f'\n  {"Experiment":25s}  {"Score":>8s}  {"pJS_min":>8s}  {"JS@35":>8s}  {"Dcond35":>8s}  {"Dconv_c":>8s}  {"Juno":>5s}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*5}')

    ranked = sorted(by_exp.items(), key=lambda x: np.mean([r['score'] for r in x[1]]))
    for name, runs in ranked:
        ms = np.mean([r['score'] for r in runs])
        mpj = np.mean([r.get('profile_JS_min', 0) for r in runs])
        mj35 = np.mean([r.get('JS_35', 0) for r in runs])
        mdc = np.mean([r.get('D_cond_35_median', 0) for r in runs])
        mdvc = np.mean([r.get('D_conv_contrast', 0) for r in runs])
        mje = np.mean([r.get('juno_excess', 0) for r in runs])
        print(f'  {name:25s}  {ms:+8.2f}  {mpj:8.4f}  {mj35:8.4f}  {mdc:8.1f}  {mdvc:8.2f}  {mje:5.1f}')

    # Seed stability
    print(f'\n  Seed stability:')
    for name, runs in by_exp.items():
        if len(runs) >= 2:
            scores = [r['score'] for r in runs]
            pjs = [r.get('profile_JS_min', 0) for r in runs]
            print(f'    {name:25s}  score_diff={abs(scores[0]-scores[1]):.2f}  pJS_diff={abs(pjs[0]-pjs[1]):.4f}')

    # Save summary
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, cls=_Enc, indent=2)

    print(f'\n{"="*70}')
    print('  CAMPAIGN COMPLETE')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
