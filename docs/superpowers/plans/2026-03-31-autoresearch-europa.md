# Autoresearch for EuropaConvection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous experiment harness that lets an AI agent run solver optimization, parameter search, and latitude-realism experiments on the Europa 2D ice shell model.

**Architecture:** Six files in `autoresearch/`: scoring functions (`objectives.py`), experiment runner (`harness.py`), agent instructions (`program.md`), autonomous loop entry point (`run.py`), plus `reference/` and `best.json` for state. The harness imports existing Europa2D code directly — no new dependencies.

**Tech Stack:** Python 3.10+, numpy, scipy (existing deps). Uses `Europa2D/src/` APIs (AxialSolver2D, MonteCarloRunner2D, LatitudeProfile, LatitudeParameterSampler).

---

### Task 1: Create `objectives.py` — Scoring Functions

**Files:**
- Create: `autoresearch/objectives.py`
- Test: `autoresearch/tests/test_objectives.py`

- [ ] **Step 1: Write failing test for solver score**

```python
# autoresearch/tests/test_objectives.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from objectives import compute_solver_score


def test_solver_score_baseline_equals_one():
    """Baseline metrics should produce score ≈ 1.0 (normalized)."""
    ref = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    result = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    score, metrics = compute_solver_score(result, ref)
    assert score == pytest.approx(1.0, abs=0.01)
    assert metrics['max_T_err'] == pytest.approx(0.0, abs=1e-10)


def test_solver_score_penalty_on_large_error():
    """Temperature error > 0.1 K triggers hard penalty."""
    ref = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    result = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.2}
    score, metrics = compute_solver_score(result, ref)
    assert score > 1000.0
    assert metrics['max_T_err'] == pytest.approx(0.2, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/tests/test_objectives.py::test_solver_score_baseline_equals_one -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'objectives'`

- [ ] **Step 3: Implement `compute_solver_score`**

```python
# autoresearch/objectives.py
"""Composite scoring functions for autoresearch experiment modes."""
import math
import numpy as np
from typing import Any, Dict, Tuple


# --- Solver mode ---

_SOLVER_WEIGHTS = {'time': 0.5, 'err': 0.35, 'iter': 0.15}
_SOLVER_ERR_THRESHOLD = 0.1  # K


def compute_solver_score(
    result: Dict[str, Any],
    reference: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Score a solver experiment against the frozen reference.

    Args:
        result: Dict with 'time', 'steps', 'T_2d' from the experiment.
        reference: Dict with 'time', 'steps', 'T_2d' from reference/solver_ref.json.

    Returns:
        (score, metrics) where lower score is better.
    """
    max_T_err = float(np.max(np.abs(
        np.asarray(result['T_2d']) - np.asarray(reference['T_2d'])
    )))
    t_ratio = result['time'] / reference['time']
    iter_ratio = result['steps'] / reference['steps']

    w = _SOLVER_WEIGHTS
    score = w['time'] * t_ratio + w['err'] * (max_T_err / _SOLVER_ERR_THRESHOLD) + w['iter'] * iter_ratio

    if max_T_err > _SOLVER_ERR_THRESHOLD:
        score += 1000.0

    metrics = {
        'time': result['time'],
        'time_ref': reference['time'],
        'steps': result['steps'],
        'steps_ref': reference['steps'],
        'max_T_err': max_T_err,
    }
    return score, metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest autoresearch/tests/test_objectives.py -k solver -v`
Expected: 2 PASSED

- [ ] **Step 5: Write failing test for physics score**

```python
# append to autoresearch/tests/test_objectives.py
from objectives import compute_physics_score


def test_physics_score_perfect_match():
    """D_cond at 35° = 29 km should give low score."""
    mc_results = {
        'D_cond_profiles': np.full((100, 19), 29.0),
        'H_profiles': np.full((100, 19), 30.0),
        'n_valid': 100,
        'n_iterations': 100,
        'latitudes_deg': np.linspace(0, 90, 19),
    }
    score, metrics = compute_physics_score(mc_results)
    assert score < 2.0  # good fit
    assert metrics['D_cond_35_median'] == pytest.approx(29.0, abs=0.1)


def test_physics_score_thin_shell_penalty():
    """Shells < 15 km should trigger penalty."""
    mc_results = {
        'D_cond_profiles': np.full((100, 19), 10.0),
        'H_profiles': np.full((100, 19), 10.0),
        'n_valid': 100,
        'n_iterations': 100,
        'latitudes_deg': np.linspace(0, 90, 19),
    }
    score, _ = compute_physics_score(mc_results)
    assert score > 100.0  # heavy penalty
```

- [ ] **Step 6: Implement `compute_physics_score`**

```python
# append to autoresearch/objectives.py

# --- Physics mode ---

_JUNO_D_COND_MU = 29.0       # km
_JUNO_D_COND_SIGMA_OBS = 10.0 # km
_MODEL_DISCREPANCY = 3.0      # km
_JUNO_SIGMA_EFF = math.sqrt(_JUNO_D_COND_SIGMA_OBS**2 + _MODEL_DISCREPANCY**2)
_JUNO_LATITUDE_DEG = 35.0
_H_TOTAL_MIN = 15.0           # km, Wakita et al. 2024
_YIELD_WEIGHT = 5.0


def _find_lat_index(latitudes_deg: np.ndarray, target_deg: float) -> int:
    """Return index of the latitude bin closest to target_deg."""
    return int(np.argmin(np.abs(latitudes_deg - target_deg)))


def compute_physics_score(
    mc_results: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Score a physics-mode MC ensemble against Juno constraints.

    Args:
        mc_results: Dict with D_cond_profiles (n_valid, n_lat),
            H_profiles (n_valid, n_lat), n_valid, n_iterations,
            latitudes_deg (n_lat,).

    Returns:
        (score, metrics) where lower is better.
    """
    lats = np.asarray(mc_results['latitudes_deg'])
    idx_35 = _find_lat_index(lats, _JUNO_LATITUDE_DEG)

    D_cond_35 = np.asarray(mc_results['D_cond_profiles'])[:, idx_35]
    D_cond_35_median = float(np.median(D_cond_35))

    # Gaussian negative log-likelihood at the median
    nll = 0.5 * ((D_cond_35_median - _JUNO_D_COND_MU) / _JUNO_SIGMA_EFF) ** 2

    # H_total < 15 km penalty
    H_all = np.asarray(mc_results['H_profiles'])
    H_min_per_sample = H_all.min(axis=1)
    thin_frac = float(np.mean(H_min_per_sample < _H_TOTAL_MIN))
    thin_penalty = 100.0 * thin_frac

    # Valid yield
    valid_frac = mc_results['n_valid'] / max(mc_results['n_iterations'], 1)
    yield_penalty = _YIELD_WEIGHT * (1.0 - valid_frac)

    score = nll + thin_penalty + yield_penalty

    metrics = {
        'D_cond_35_median': D_cond_35_median,
        'D_cond_35_mean': float(np.mean(D_cond_35)),
        'D_cond_35_std': float(np.std(D_cond_35)),
        'thin_fraction': thin_frac,
        'valid_fraction': valid_frac,
        'nll': nll,
    }
    return score, metrics
```

- [ ] **Step 7: Run physics score tests**

Run: `python -m pytest autoresearch/tests/test_objectives.py -k physics -v`
Expected: 2 PASSED

- [ ] **Step 8: Write failing test for latitude score**

```python
# append to autoresearch/tests/test_objectives.py
from objectives import compute_latitude_score


def test_latitude_score_rewards_contrast():
    """Higher D_conv contrast should give lower (better) score."""
    lats = np.linspace(0, 90, 19)
    # Low contrast scenario
    low = _make_scenario_results(lats, d_conv_eq=2.0, d_conv_pole=2.1)
    # High contrast scenario
    high = _make_scenario_results(lats, d_conv_eq=4.0, d_conv_pole=1.0)

    score_low, _ = compute_latitude_score(
        scenarios={'uniform': low, 'polar': low, 'equator': low},
        consistency_error=0.01,
    )
    score_high, _ = compute_latitude_score(
        scenarios={'uniform': high, 'polar': high, 'equator': high},
        consistency_error=0.01,
    )
    assert score_high < score_low


def _make_scenario_results(lats, d_conv_eq, d_conv_pole):
    n_samples = 100
    n_lat = len(lats)
    # Linear D_conv from equator to pole
    d_conv_profile = np.linspace(d_conv_eq, d_conv_pole, n_lat)
    D_conv = np.tile(d_conv_profile, (n_samples, 1)) + np.random.default_rng(42).normal(0, 0.1, (n_samples, n_lat))
    D_cond = np.full((n_samples, n_lat), 25.0) + np.random.default_rng(43).normal(0, 2.0, (n_samples, n_lat))
    Ra = np.tile(np.linspace(50, 10, n_lat), (n_samples, 1))
    H = D_cond + D_conv
    return {
        'D_conv_profiles': D_conv,
        'D_cond_profiles': D_cond,
        'Ra_profiles': Ra,
        'H_profiles': H,
        'n_valid': n_samples,
        'n_iterations': n_samples,
        'latitudes_deg': lats,
    }
```

- [ ] **Step 9: Implement `compute_latitude_score`**

```python
# append to autoresearch/objectives.py

# --- Latitude mode ---

_LAT_WEIGHTS = {
    'dconv': 1.0,
    'disc': 2.0,
    'ra': 0.5,
    '1d2d': 5.0,
    'juno': 3.0,
    'sanity': 10.0,
}

# JS divergence histogram settings
_JS_BIN_EDGES = np.arange(5.0, 61.0, 1.0)  # 1 km bins, 5-60 km


def _js_divergence(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two sample sets using fixed bins."""
    hist_a, _ = np.histogram(samples_a, bins=_JS_BIN_EDGES, density=True)
    hist_b, _ = np.histogram(samples_b, bins=_JS_BIN_EDGES, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = hist_a + eps
    q = hist_b + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_latitude_score(
    scenarios: Dict[str, Dict[str, Any]],
    consistency_error: float,
) -> Tuple[float, Dict[str, float]]:
    """Score a latitude-mode triple-scenario experiment.

    Args:
        scenarios: Dict mapping scenario name ('uniform', 'polar', 'equator')
            to MC result dicts, each with D_conv_profiles, D_cond_profiles,
            Ra_profiles, H_profiles, latitudes_deg (all same shape).
        consistency_error: |H_2d - H_1d| / H_1d from the fixed calibration check.

    Returns:
        (score, metrics) where lower is better.
    """
    # Use first scenario's latitudes (all must match)
    first = next(iter(scenarios.values()))
    lats = np.asarray(first['latitudes_deg'])
    idx_35 = _find_lat_index(lats, _JUNO_LATITUDE_DEG)

    # D_conv contrast: max across scenarios
    d_conv_contrasts = []
    for name, res in scenarios.items():
        D_conv_median = np.median(np.asarray(res['D_conv_profiles']), axis=0)
        d_conv_contrasts.append(float(np.max(D_conv_median) - np.min(D_conv_median)))
    d_conv_contrast = max(d_conv_contrasts)

    # Ra gradient: use uniform scenario median
    uniform = scenarios.get('uniform', first)
    Ra_median = np.median(np.asarray(uniform['Ra_profiles']), axis=0)
    # Avoid log(0) — if Ra is zero at pole, use small value
    ra_eq = max(float(Ra_median[0]), 1e-6)
    ra_pole = max(float(Ra_median[-1]), 1e-6)
    ra_log_ratio = math.log(ra_eq / ra_pole) if ra_eq > ra_pole else 0.0

    # JS divergence on D_cond at 35° — minimum pairwise
    scenario_names = list(scenarios.keys())
    js_values = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            d_a = np.asarray(scenarios[scenario_names[i]]['D_cond_profiles'])[:, idx_35]
            d_b = np.asarray(scenarios[scenario_names[j]]['D_cond_profiles'])[:, idx_35]
            js_values.append(_js_divergence(d_a, d_b))
    min_js = min(js_values) if js_values else 0.0

    # Juno soft check at 35°
    D_cond_35_all = []
    for res in scenarios.values():
        D_cond_35_all.append(np.median(np.asarray(res['D_cond_profiles'])[:, idx_35]))
    D_cond_35_median = float(np.mean(D_cond_35_all))
    juno_excess = max(0.0, abs(D_cond_35_median - _JUNO_D_COND_MU) - _JUNO_D_COND_SIGMA_OBS)

    # Sanity penalty
    sanity = 0.0
    for res in scenarios.values():
        H = np.asarray(res['H_profiles'])
        if np.any(H <= 0):
            sanity += 1.0
        Ra = np.asarray(res['Ra_profiles'])
        if np.any(Ra < 0):
            sanity += 1.0

    w = _LAT_WEIGHTS
    score = (
        -w['dconv'] * d_conv_contrast
        - w['disc'] * min_js
        - w['ra'] * ra_log_ratio
        + w['1d2d'] * max(0.0, consistency_error - 0.05) * 100.0
        + w['juno'] * juno_excess * 10.0
        + w['sanity'] * sanity
    )

    metrics = {
        'D_conv_contrast': d_conv_contrast,
        'JS_discriminability': min_js,
        'Ra_eq_median': ra_eq,
        'Ra_pole_median': ra_pole,
        'Ra_log_ratio': ra_log_ratio,
        'consistency_error': consistency_error,
        'D_cond_35_median': D_cond_35_median,
        'juno_excess': juno_excess,
        'sanity_penalty': sanity,
    }
    return score, metrics


# --- Dispatcher ---

def compute_score(mode: str, **kwargs) -> Tuple[float, Dict[str, float]]:
    """Dispatch to the appropriate scoring function by mode name."""
    if mode == 'solver':
        return compute_solver_score(kwargs['result'], kwargs['reference'])
    elif mode == 'physics':
        return compute_physics_score(kwargs['mc_results'])
    elif mode == 'latitude':
        return compute_latitude_score(kwargs['scenarios'], kwargs['consistency_error'])
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

- [ ] **Step 10: Run all objective tests**

Run: `python -m pytest autoresearch/tests/test_objectives.py -v`
Expected: 5 PASSED

- [ ] **Step 11: Commit**

```bash
git add autoresearch/objectives.py autoresearch/tests/test_objectives.py
git commit -m "feat(autoresearch): add composite scoring functions for three experiment modes"
```

---

### Task 2: Create `harness.py` — Experiment Runner

**Files:**
- Create: `autoresearch/harness.py`
- Test: `autoresearch/tests/test_harness.py`

- [ ] **Step 1: Write failing test for harness init mode**

```python
# autoresearch/tests/test_harness.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from harness import ExperimentHarness


def test_init_creates_reference_and_best(tmp_path):
    """--init should create reference/ dir and best.json."""
    harness = ExperimentHarness(base_dir=str(tmp_path))
    # Mock the actual model runs
    with patch.object(harness, '_run_solver_baseline') as mock_solver, \
         patch.object(harness, '_run_physics_baseline') as mock_physics, \
         patch.object(harness, '_run_latitude_baseline') as mock_latitude:
        mock_solver.return_value = ({'time': 2.0, 'steps': 200, 'T_2d': [[150.0]]}, 1.0, {})
        mock_physics.return_value = ({'D_cond_35_median': 29.0}, 1.5, {})
        mock_latitude.return_value = ({'D_conv_contrast': 0.8}, 2.0, {})
        harness.init()

    assert (tmp_path / 'reference' / 'solver_ref.json').exists()
    assert (tmp_path / 'reference' / 'physics_ref.json').exists()
    assert (tmp_path / 'reference' / 'latitude_ref.json').exists()
    assert (tmp_path / 'best.json').exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest autoresearch/tests/test_harness.py::test_init_creates_reference_and_best -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ExperimentHarness` — init, solver mode, failure handling**

```python
# autoresearch/harness.py
"""Experiment harness for autoresearch — run, score, log."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add Europa2D/src and EuropaProjectDJ/src to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'EuropaProjectDJ' / 'src'))

from objectives import compute_solver_score, compute_physics_score, compute_latitude_score


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class ExperimentHarness:
    """Run-score-log harness for autoresearch experiments."""

    # Reference solver parameters (equatorial, Howell defaults)
    _REF_PARAMS = {
        'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
        'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
        'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
        'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
    }
    _REF_T_SURF = 104.0
    _REF_Q_OCEAN = 0.025
    _REF_NX = 31
    _REF_DT = 5e12
    _REF_THICKNESS = 20e3

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or Path(__file__).parent)
        self.ref_dir = self.base_dir / 'reference'
        self.best_path = self.base_dir / 'best.json'
        self.log_path = self.base_dir / 'experiments.jsonl'

    def init(self):
        """Initialize reference artifacts and best.json."""
        self.ref_dir.mkdir(parents=True, exist_ok=True)

        solver_result, solver_score, solver_metrics = self._run_solver_baseline()
        physics_result, physics_score, physics_metrics = self._run_physics_baseline()
        latitude_result, latitude_score, latitude_metrics = self._run_latitude_baseline()

        self._write_json(self.ref_dir / 'solver_ref.json', solver_result)
        self._write_json(self.ref_dir / 'physics_ref.json', physics_result)
        self._write_json(self.ref_dir / 'latitude_ref.json', latitude_result)

        best = {
            'solver': {'score': solver_score, 'metrics': solver_metrics},
            'physics': {'score': physics_score, 'metrics': physics_metrics},
            'latitude': {'score': latitude_score, 'metrics': latitude_metrics},
        }
        self._write_json(self.best_path, best)
        print("=== BASELINE INITIALIZED ===")
        print(f"  Solver score:   {solver_score:.4f}")
        print(f"  Physics score:  {physics_score:.4f}")
        print(f"  Latitude score: {latitude_score:.4f}")

    def run(self, mode: str, tag: str, n_samples: int = 250, n_workers: int = 8):
        """Run an experiment, score it, log the result."""
        try:
            if mode == 'solver':
                score, metrics = self._run_solver_experiment()
            elif mode == 'physics':
                score, metrics = self._run_physics_experiment(n_samples, n_workers)
            elif mode == 'latitude':
                score, metrics = self._run_latitude_experiment(n_samples, n_workers)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Compare to best
            best = self._load_best()
            prev_score = best.get(mode, {}).get('score', float('inf'))
            delta = score - prev_score
            improved = delta < 0

            # Log
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': mode,
                'tag': tag,
                'status': 'ok',
                'score': score,
                'delta': delta,
                'improved': improved,
                'metrics': metrics,
                'git_sha': self._get_git_sha(),
            }
            self._append_log(entry)

            # Update best if improved
            if improved:
                best[mode] = {'score': score, 'metrics': metrics}
                self._write_json(self.best_path, best)

            # Print structured result
            self._print_result(mode, tag, score, prev_score, delta, improved, metrics)

        except Exception:
            tb = traceback.format_exc()
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': mode,
                'tag': tag,
                'status': 'failed',
                'score': float('inf'),
                'error': tb[-500:],  # last 500 chars of traceback
            }
            self._append_log(entry)
            print(f"=== EXPERIMENT FAILED ===")
            print(f"Mode: {mode}")
            print(f"Tag: {tag}")
            print(f"Error:\n{tb}")
            # Exit 0 so agent loop continues
            return

    # --- Solver mode ---

    def _run_solver_baseline(self):
        """Run solver reference and return (result_dict, score, metrics)."""
        result = self._run_single_solver()
        ref = result  # baseline IS the reference
        score, metrics = compute_solver_score(result, ref)
        return result, score, metrics

    def _run_solver_experiment(self):
        """Run solver 5 times, score against frozen reference."""
        ref_data = self._load_json(self.ref_dir / 'solver_ref.json')
        times = []
        last_result = None
        for _ in range(5):
            result = self._run_single_solver()
            times.append(result['time'])
            last_result = result
        last_result['time'] = float(np.median(times))
        return compute_solver_score(last_result, ref_data)

    def _run_single_solver(self):
        """Run one solver evaluation at reference params."""
        from latitude_profile import LatitudeProfile
        from axial_solver import AxialSolver2D

        profile = LatitudeProfile(
            T_eq=self._REF_T_SURF,
            epsilon_eq=self._REF_PARAMS['epsilon_0'],
            epsilon_pole=self._REF_PARAMS['epsilon_0'],
            q_ocean_mean=self._REF_Q_OCEAN,
            ocean_pattern="uniform",
        )
        solver = AxialSolver2D(
            n_lat=1, nx=self._REF_NX, dt=self._REF_DT,
            latitude_profile=profile, physics_params=dict(self._REF_PARAMS),
            use_convection=True, initial_thickness=self._REF_THICKNESS,
        )
        t0 = time.perf_counter()
        result = solver.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False)
        elapsed = time.perf_counter() - t0

        return {
            'time': elapsed,
            'steps': result['steps'],
            'T_2d': result['T_2d'],
            'H_profile_km': result['H_profile_km'],
        }

    # --- Physics mode ---

    def _run_physics_baseline(self):
        """Run physics baseline MC ensemble."""
        mc_results = self._run_mc_ensemble("uniform", n_samples=250, n_workers=8)
        result_dict = self._mc_to_dict(mc_results)
        score, metrics = compute_physics_score(result_dict)
        return result_dict, score, metrics

    def _run_physics_experiment(self, n_samples: int, n_workers: int):
        """Run physics-mode MC ensemble and score."""
        mc_results = self._run_mc_ensemble("uniform", n_samples=n_samples, n_workers=n_workers)
        result_dict = self._mc_to_dict(mc_results)
        return compute_physics_score(result_dict)

    # --- Latitude mode ---

    def _run_latitude_baseline(self):
        """Run latitude baseline (3 scenarios + calibration check)."""
        score, metrics = self._run_latitude_experiment(n_samples=250, n_workers=8)
        return metrics, score, metrics  # metrics IS the ref data for latitude mode

    def _run_latitude_experiment(self, n_samples: int, n_workers: int):
        """Run 3 scenarios + 1D/2D calibration, then score."""
        scenario_configs = [
            ('uniform', 'uniform', None),
            ('polar', 'polar_enhanced', 0.455),
            ('equator', 'equator_enhanced', 0.4),
        ]
        scenarios = {}
        for name, pattern, q_star in scenario_configs:
            mc = self._run_mc_ensemble(pattern, n_samples=n_samples, n_workers=n_workers, q_star=q_star)
            scenarios[name] = self._mc_to_dict(mc)

        consistency_error = self._run_calibration_check()
        return compute_latitude_score(scenarios, consistency_error)

    def _run_calibration_check(self) -> float:
        """Fixed 1D/2D consistency check at equatorial reference params."""
        from latitude_profile import LatitudeProfile
        from axial_solver import AxialSolver2D
        from Solver import Thermal_Solver
        from Boundary_Conditions import FixedTemperature

        params = dict(self._REF_PARAMS)
        params['T_surf'] = self._REF_T_SURF

        # 1D solve
        bc_1d = FixedTemperature(temperature=self._REF_T_SURF)
        solver_1d = Thermal_Solver(
            nx=self._REF_NX, thickness=self._REF_THICKNESS, dt=1e11,
            surface_bc=bc_1d, use_convection=True, physics_params=params,
        )
        for _ in range(500):
            v = solver_1d.solve_step(self._REF_Q_OCEAN)
            if abs(v) < 1e-12:
                break
        H_1d = solver_1d.H / 1000.0

        # 2D single-column solve
        profile = LatitudeProfile(
            T_eq=self._REF_T_SURF,
            epsilon_eq=self._REF_PARAMS['epsilon_0'],
            epsilon_pole=self._REF_PARAMS['epsilon_0'],
            q_ocean_mean=self._REF_Q_OCEAN,
            ocean_pattern="uniform",
        )
        solver_2d = AxialSolver2D(
            n_lat=1, nx=self._REF_NX, dt=self._REF_DT,
            latitude_profile=profile, physics_params=dict(self._REF_PARAMS),
            use_convection=True, initial_thickness=self._REF_THICKNESS,
        )
        result_2d = solver_2d.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False)
        H_2d = float(result_2d['H_profile_km'][0])

        if H_1d <= 0:
            return 1.0  # degenerate — flag as inconsistent
        return abs(H_2d - H_1d) / H_1d

    def _run_mc_ensemble(self, ocean_pattern: str, n_samples: int, n_workers: int,
                         q_star: Optional[float] = None):
        """Run one MC ensemble via MonteCarloRunner2D."""
        from monte_carlo_2d import MonteCarloRunner2D

        runner = MonteCarloRunner2D(
            n_iterations=n_samples,
            seed=42,
            n_workers=n_workers,
            ocean_pattern=ocean_pattern,
            q_star=q_star,
            verbose=False,
        )
        return runner.run()

    def _mc_to_dict(self, mc) -> Dict[str, Any]:
        """Extract scoring-relevant fields from MonteCarloResults2D."""
        return {
            'D_cond_profiles': np.asarray(mc.D_cond_profiles),
            'D_conv_profiles': np.asarray(mc.D_conv_profiles),
            'Ra_profiles': np.asarray(mc.Ra_profiles),
            'H_profiles': np.asarray(mc.H_profiles),
            'n_valid': mc.n_valid,
            'n_iterations': mc.n_iterations,
            'latitudes_deg': np.asarray(mc.latitudes_deg),
        }

    # --- Utilities ---

    def _load_best(self) -> Dict:
        if self.best_path.exists():
            return self._load_json(self.best_path)
        return {}

    def _get_git_sha(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            return result.stdout.strip()
        except Exception:
            return 'unknown'

    def _append_log(self, entry: Dict):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry, cls=_NumpyEncoder) + '\n')

    def _write_json(self, path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, cls=_NumpyEncoder, indent=2)

    def _load_json(self, path: Path) -> Dict:
        with open(path) as f:
            return json.load(f)

    def _print_result(self, mode, tag, score, prev, delta, improved, metrics):
        status = "IMPROVED" if improved else "no improvement"
        print(f"=== EXPERIMENT RESULT ===")
        print(f"Mode: {mode}")
        print(f"Tag: {tag}")
        print(f"Score: {score:.4f} (prev: {prev:.4f}, delta: {delta:.4f}, {status})")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='Autoresearch experiment harness')
    parser.add_argument('--init', action='store_true', help='Initialize reference artifacts')
    parser.add_argument('--mode', choices=['solver', 'physics', 'latitude'], help='Experiment mode')
    parser.add_argument('--tag', default='unnamed', help='Experiment tag/description')
    parser.add_argument('--n-samples', type=int, default=250, help='MC samples per ensemble')
    parser.add_argument('--n-workers', type=int, default=8, help='Parallel workers')
    args = parser.parse_args()

    harness = ExperimentHarness()

    if args.init:
        harness.init()
    elif args.mode:
        harness.run(args.mode, args.tag, args.n_samples, args.n_workers)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run harness init test**

Run: `python -m pytest autoresearch/tests/test_harness.py -v`
Expected: 1 PASSED

- [ ] **Step 5: Write failing test for harness failure handling**

```python
# append to autoresearch/tests/test_harness.py

def test_failed_experiment_logs_infinity(tmp_path):
    """A crashing experiment should log score=Infinity and not raise."""
    harness = ExperimentHarness(base_dir=str(tmp_path))
    # Create minimal best.json
    (tmp_path / 'best.json').write_text('{}')

    with patch.object(harness, '_run_solver_experiment', side_effect=RuntimeError("boom")):
        harness.run('solver', 'bad_change')

    log_path = tmp_path / 'experiments.jsonl'
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip())
    assert entry['status'] == 'failed'
    assert entry['score'] == float('inf')
    assert 'boom' in entry['error']
```

- [ ] **Step 6: Run failure handling test**

Run: `python -m pytest autoresearch/tests/test_harness.py::test_failed_experiment_logs_infinity -v`
Expected: PASS (already implemented in step 3)

- [ ] **Step 7: Commit**

```bash
git add autoresearch/harness.py autoresearch/tests/test_harness.py
git commit -m "feat(autoresearch): add experiment harness with init, run, and failure handling"
```

---

### Task 3: Create `autoresearch/tests/__init__.py` and Package Init

**Files:**
- Create: `autoresearch/__init__.py`
- Create: `autoresearch/tests/__init__.py`

- [ ] **Step 1: Create package init files**

```python
# autoresearch/__init__.py
```

```python
# autoresearch/tests/__init__.py
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "import autoresearch"`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add autoresearch/__init__.py autoresearch/tests/__init__.py
git commit -m "chore(autoresearch): add package init files"
```

---

### Task 4: Create `run.py` — Autonomous Loop Entry Point

**Files:**
- Create: `autoresearch/run.py`

- [ ] **Step 1: Implement `run.py`**

```python
# autoresearch/run.py
"""Entry point for autonomous autoresearch experiment loop.

Usage:
    python autoresearch/run.py --max-experiments 20 --mode latitude
    python autoresearch/run.py --max-experiments 10 --mode solver
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), check=check)


def _git_sha() -> str:
    result = _run_cmd(['git', 'rev-parse', '--short', 'HEAD'], check=False)
    return result.stdout.strip()


def _create_branch() -> str:
    ts = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    branch = f"autoresearch/run-{ts}"
    _run_cmd(['git', 'checkout', '-b', branch])
    print(f"Created branch: {branch}")
    return branch


def _ensure_baseline():
    ref_dir = Path(__file__).parent / 'reference'
    if not ref_dir.exists():
        print("No reference/ found. Initializing baseline...")
        _run_cmd([sys.executable, str(Path(__file__).parent / 'harness.py'), '--init'])
        _run_cmd(['git', 'add', 'autoresearch/reference/', 'autoresearch/best.json'])
        _run_cmd(['git', 'commit', '-m', 'autoresearch: initialize baseline reference artifacts'])


def main():
    parser = argparse.ArgumentParser(description='Launch autonomous autoresearch loop')
    parser.add_argument('--max-experiments', type=int, default=20, help='Max experiments to run')
    parser.add_argument('--mode', required=True, choices=['solver', 'physics', 'latitude'])
    parser.add_argument('--n-samples', type=int, default=250)
    parser.add_argument('--n-workers', type=int, default=8)
    args = parser.parse_args()

    branch = _create_branch()
    _ensure_baseline()

    print(f"\n{'=' * 60}")
    print(f"AUTORESEARCH — Mode: {args.mode}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"Branch: {branch}")
    print(f"{'=' * 60}")
    print()
    print("Ready for AI agent to begin experiment loop.")
    print("The agent should:")
    print("  1. Read autoresearch/program.md")
    print("  2. Read autoresearch/best.json")
    print("  3. Formulate hypothesis, modify code, run harness, evaluate")
    print()
    print(f"Harness command:")
    print(f"  python autoresearch/harness.py --mode {args.mode} --tag \"<description>\" "
          f"--n-samples {args.n_samples} --n-workers {args.n_workers}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify run.py parses arguments**

Run: `python autoresearch/run.py --help`
Expected: Usage text with --max-experiments, --mode, --n-samples, --n-workers

- [ ] **Step 3: Commit**

```bash
git add autoresearch/run.py
git commit -m "feat(autoresearch): add autonomous loop entry point"
```

---

### Task 5: Create `program.md` — Agent Instructions

**Files:**
- Create: `autoresearch/program.md`

- [ ] **Step 1: Write `program.md`**

This is a large markdown file. The content should be synthesized from spec sections 6.1-6.6. Here is the full document:

```markdown
# Autoresearch: Europa Ice Shell Optimization

You are an AI research agent running autonomous experiments on a 2D axisymmetric Europa ice shell model. Your goal is to improve the model's physical realism, observational agreement, and computational performance through systematic experimentation.

## Project Overview

Europa (Jupiter's moon) has a global ocean beneath an ice shell. This project models the ice shell's thermal structure using a 2D latitude-dependent solver. Two codebases:

- **Europa2D/** — Active 2D axisymmetric model (your primary target)
- **EuropaProjectDJ/** — 1D reference solver (trusted baseline, modify cautiously)

### Current State

The 2D model's shell thickness is dominated by surface temperature variation (96 K equator, 46 K pole). The conductive lid at poles is ~14 km thicker than at the equator. The convective sublayer is nearly uniform (~2 km) regardless of latitude, ocean pattern, or tidal strain. This means ocean transport scenarios are barely distinguishable — a known limitation you should try to improve.

## Observational Constraints

| Observable | Value | Uncertainty | Source | Notes |
|---|---|---|---|---|
| D_cond at 35° lat | 29 km | ± 10 km | Levin et al. 2025, Juno MWR | Single flyby at ~35° latitude |
| H_total minimum | > 15 km | hard bound | Wakita et al. 2024 | Impact basin formation |
| H_total preferred | > 20 km | soft bound | Wakita et al. 2024 | Multiring formation |
| Surface heat flux (equil.) | 20-50 mW/m² | order of magnitude | Hussmann & Spohn 2002 | For equilibrium shells |
| T_surf equatorial | 96 K | ± 5 K | Ashkenazy 2019 | Annual mean |
| T_surf polar | 46 K | ± 5 K | Ashkenazy 2019 | Annual mean |

**Critical:** The Juno D_cond is measured at ~35° latitude, not globally averaged. Apply it at the correct latitude bin.

## Physics Primer

- **D_cond scaling:** D_cond ~ ∫[T_surf to T_interior] k(T)/q dT. k(T) = 567/T W/m·K.
- **Why T_surf dominates:** 50 K equator-to-pole contrast → ~14 km D_cond variation. Typical q_basal heterogeneity → ~5 km. k(T) amplifies: k at 46 K is ~2x k at 96 K.
- **Ocean transport regimes:**
  - Soderlund (2014): equatorial enhancement via Hadley cells (q* = 0.4)
  - Lemasquerier (2023): polar enhancement, ocean transposes mantle tidal pattern
  - Ashkenazy & Tziperman (2021): uniform (efficient meridional mixing)
- **q* parameter:** q* = 0.91 × mantle_tidal_fraction. Pure tidal → q* = 0.91.
- **Tidal strain:** Mantle-core pattern: ε(φ) ~ √(1 + 3sin²φ), 4:1 pole-to-equator dissipation.
- **Convection:** Nu = 0.3446 × Ra^(1/3). Stagnant-lid, ~10 K sublayer ΔT. Ra_crit = 1000.
- **Grain recrystallization:** Higher tidal strain → smaller grains → lower viscosity.

## Experiment Protocol

1. Read `autoresearch/best.json` to understand current best scores.
2. Formulate a hypothesis: "If I change X, I expect Y because Z."
3. Make ONE focused code change per experiment.
4. Commit the change: `git add <files> && git commit -m "autoresearch: <hypothesis>"`
5. Run: `python autoresearch/harness.py --mode <mode> --tag "<description>"`
6. Evaluate: did the score improve? Does the physics make sense?
7. If improved: run smoke test `python -m pytest Europa2D/tests/test_validation.py -x`
8. If smoke passes AND score improved: run full suite `python -m pytest Europa2D/tests/`
9. Commit the result: `git commit -m "autoresearch(<mode>): <tag> — score X.XX (ΔY.YY, IMPROVED/no improvement)"`
10. If failed or tests broken: `git checkout -- Europa2D/src/` to revert, then try next idea.

## Experiment Modes

### `--mode solver`
Optimize wall-clock time and convergence without breaking accuracy.
- Modify: `Europa2D/src/axial_solver.py`
- Budget: ~30 seconds per experiment
- Key metric: wall-clock time (median of 5 runs)
- Guardrail: max temperature error < 0.1 K vs frozen reference

### `--mode physics`
Improve agreement with Juno and other observational constraints.
- Modify: `Europa2D/src/latitude_sampler.py`, `latitude_profile.py`, `literature_scenarios.py`, `axial_solver.py`
- Budget: ~5-10 minutes per experiment (250-sample MC)
- Key metric: negative log-likelihood of D_cond at 35° against Juno MWR

### `--mode latitude`
Make the 2D model produce realistic latitude-dependent shell structure.
- Modify: same as physics mode
- Budget: ~15-20 minutes per experiment (3 × 250-sample MC)
- Key metrics: D_conv contrast, scenario discriminability (JS divergence), Ra gradient
- This is the hardest and most valuable mode.

## Research Questions (Prioritized)

### Priority 1 — Latitude realism
1. Can q_basal patterns be made strong enough to compete with T_surf without violating Juno?
2. Does the convection closure need latitude-dependent parameters?
3. What q_tidal_scale value produces the most realistic latitude structure?
4. Should mantle_tidal_fraction have a narrower or shifted prior?

### Priority 2 — Physics optimization
5. Are sampling distributions well-centered on observational constraints?
6. Does the 1.20x tidal flux scale have the right magnitude?
7. Can additional constraints tighten the posterior?

### Priority 3 — Solver performance
8. Can the radial grid be coarsened without losing accuracy?
9. Is the convergence criterion too tight or too loose?
10. Can lateral diffusion be skipped (τ ~ 200 Gyr)?

## Files You May Modify

| Scope | File | Purpose |
|---|---|---|
| Always | `Europa2D/src/axial_solver.py` | Solver numerics |
| Always | `Europa2D/src/latitude_sampler.py` | Sampling distributions |
| Always | `Europa2D/src/latitude_profile.py` | Latitude parameterizations |
| Always | `Europa2D/src/literature_scenarios.py` | Scenario configs |
| Cautious | `Europa2D/src/monte_carlo_2d.py` | MC runner (needs rationale) |
| Cautious | `Europa2D/src/profile_diagnostics.py` | Diagnostics (needs rationale) |
| Cautious | `EuropaProjectDJ/src/` | 1D solver (needs strong justification + test both suites) |

## Files You Must NOT Modify

- `autoresearch/harness.py`
- `autoresearch/objectives.py`
- `autoresearch/program.md` (this file)
- Test files (unless adding NEW tests for NEW physics)

## Stopping Criteria

Stop experimenting when:
- You have reached the maximum experiment count
- 3 consecutive experiments show no improvement
- A test failure you cannot fix after 2 attempts
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/program.md
git commit -m "docs(autoresearch): add agent instructions (program.md)"
```

---

### Task 6: Add `.gitignore` for Autoresearch Runtime Artifacts

**Files:**
- Create: `autoresearch/.gitignore`

- [ ] **Step 1: Create .gitignore**

```gitignore
# autoresearch/.gitignore
# Runtime artifacts — tracked on experiment branches, not master
experiments.jsonl
reference/
best.json
```

- [ ] **Step 2: Commit**

```bash
git add autoresearch/.gitignore
git commit -m "chore(autoresearch): add gitignore for runtime artifacts"
```

---

### Task 7: Integration Test — Full Solver Mode Round-Trip

**Files:**
- Create: `autoresearch/tests/test_integration.py`

- [ ] **Step 1: Write integration test for solver mode**

```python
# autoresearch/tests/test_integration.py
"""Integration test: full solver-mode experiment round-trip."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pytest
from pathlib import Path
from harness import ExperimentHarness


@pytest.mark.slow
def test_solver_round_trip(tmp_path):
    """Init baseline, run solver experiment, verify log and scoring."""
    harness = ExperimentHarness(base_dir=str(tmp_path))

    # Init baseline (runs real solver — takes ~10s)
    harness.init()

    ref_path = tmp_path / 'reference' / 'solver_ref.json'
    best_path = tmp_path / 'best.json'
    assert ref_path.exists()
    assert best_path.exists()

    # Run an experiment (no code change — should match baseline)
    harness.run('solver', 'no_change_control')

    log_path = tmp_path / 'experiments.jsonl'
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().split('\n')[-1])
    assert entry['status'] == 'ok'
    assert entry['mode'] == 'solver'
    # Score should be close to baseline (small timing variance)
    assert entry['score'] < 2.0
    assert entry['metrics']['max_T_err'] < 0.01  # same code → near-zero error
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest autoresearch/tests/test_integration.py -v -m slow --timeout=120`
Expected: 1 PASSED (takes ~15 seconds for init + experiment)

- [ ] **Step 3: Commit**

```bash
git add autoresearch/tests/test_integration.py
git commit -m "test(autoresearch): add solver-mode integration round-trip test"
```

---

### Task 8: Final Verification — All Tests Pass

**Files:** No new files

- [ ] **Step 1: Run all autoresearch tests**

Run: `python -m pytest autoresearch/tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run existing Europa2D tests (ensure no regressions)**

Run: `python -m pytest Europa2D/tests/ -v`
Expected: All existing tests PASS (autoresearch doesn't touch model code)

- [ ] **Step 3: Verify harness CLI works end-to-end**

Run: `python autoresearch/harness.py --help`
Expected: Usage text with --init, --mode, --tag, --n-samples, --n-workers

Run: `python autoresearch/run.py --help`
Expected: Usage text with --max-experiments, --mode

- [ ] **Step 4: Final commit**

```bash
git add -A autoresearch/
git commit -m "feat(autoresearch): complete autoresearch harness for Europa ice shell experiments"
```
