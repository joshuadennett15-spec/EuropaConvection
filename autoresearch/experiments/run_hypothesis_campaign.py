"""Run the hypothesis campaign and print ranked results."""
import json
import os
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
from harness import ExperimentHarness


def _load_config():
    config_path = Path(__file__).parent / 'hypothesis_config.json'
    with open(config_path) as f:
        return json.load(f)


def _load_log(log_path):
    completed = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                completed.add(entry['experiment'])
    return completed


def _append_log(log_path, entry):
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(log_path, 'a') as f:
        f.write(json.dumps(entry, cls=_Enc) + '\n')


def run_campaign():
    config = _load_config()
    n_samples = config['n_samples']
    n_workers = config['n_workers']
    experiments = config['experiments']

    log_path = Path(__file__).parent / 'hypothesis_results.jsonl'
    completed = _load_log(log_path)

    harness = ExperimentHarness()

    results = []
    for exp in experiments:
        name = exp['name']
        if name in completed:
            print(f"[SKIP] {name} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'='*60}")

        hyp_def = exp.get('hypothesis')
        hypothesis = None
        if hyp_def is not None:
            hypothesis = ConvectionHypothesis(
                mechanism=hyp_def['mechanism'],
                params=hyp_def['params'],
            )

        config_overrides = exp.get('config_overrides', {})
        grain_mode = config_overrides.get('grain_latitude_mode', 'global')
        grain_exp = config_overrides.get('grain_strain_exponent', 0.5)

        t0 = time.time()

        score, metrics = harness._run_latitude_experiment(
            n_samples=n_samples,
            n_workers=n_workers,
            hypothesis=hypothesis,
            grain_latitude_mode=grain_mode,
            grain_strain_exponent=grain_exp,
        )

        runtime = time.time() - t0

        entry = {
            'experiment': name,
            'hypothesis': exp.get('hypothesis'),
            'config_overrides': config_overrides,
            'latitude_score': float(score),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in metrics.items()},
            'runtime_seconds': runtime,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        _append_log(log_path, entry)
        results.append(entry)

        print(f"  Score: {score:.2f}")
        print(f"  JS(D_cond): {metrics.get('JS_discriminability', 0):.4f}")
        print(f"  JS(D_conv): {metrics.get('JS_discriminability_Dconv', 0):.4f}")
        print(f"  D_conv contrast: {metrics.get('D_conv_contrast', 0):.2f} km")
        print(f"  Runtime: {runtime:.1f}s")

    # Load all results for leaderboard
    if log_path.exists():
        all_results = []
        with open(log_path) as f:
            for line in f:
                all_results.append(json.loads(line))
    else:
        all_results = results

    if all_results:
        print(f"\n{'='*60}")
        print("  LEADERBOARD (lower score = better)")
        print(f"{'='*60}")
        ranked = sorted(all_results, key=lambda r: r['latitude_score'])
        for i, r in enumerate(ranked):
            m = r['metrics']
            print(f"  {i+1}. {r['experiment']:20s}  score={r['latitude_score']:+8.2f}"
                  f"  JS_Dcond={m.get('JS_discriminability', 0):.4f}"
                  f"  JS_Dconv={m.get('JS_discriminability_Dconv', 0):.4f}"
                  f"  D_conv_contrast={m.get('D_conv_contrast', 0):.1f}km")


if __name__ == '__main__':
    run_campaign()
