import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from objectives import compute_solver_score, compute_physics_score, compute_latitude_score


def test_solver_score_baseline_equals_one():
    """Baseline metrics should produce score ~ 1.0 (normalized)."""
    ref = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    result = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    score, metrics = compute_solver_score(result, ref)
    assert score == pytest.approx(0.65, abs=0.01)
    assert metrics['max_T_err'] == pytest.approx(0.0, abs=1e-10)


def test_solver_score_penalty_on_large_error():
    """Temperature error > 0.1 K triggers hard penalty."""
    ref = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.0}
    result = {'time': 2.0, 'steps': 200, 'T_2d': np.ones((1, 31)) * 150.2}
    score, metrics = compute_solver_score(result, ref)
    assert score > 1000.0
    assert metrics['max_T_err'] == pytest.approx(0.2, abs=0.01)


def test_physics_score_perfect_match():
    """D_cond at 35 deg = 29 km should give low score."""
    mc_results = {
        'D_cond_profiles': np.full((100, 19), 29.0),
        'H_profiles': np.full((100, 19), 30.0),
        'n_valid': 100,
        'n_iterations': 100,
        'latitudes_deg': np.linspace(0, 90, 19),
    }
    score, metrics = compute_physics_score(mc_results)
    assert score < 2.0
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
    assert score > 100.0


def test_latitude_score_rewards_contrast():
    """Higher D_conv contrast should give lower (better) score."""
    lats = np.linspace(0, 90, 19)
    rng = np.random.default_rng(42)

    def _make(d_conv_eq, d_conv_pole):
        n_samples, n_lat = 100, len(lats)
        d_conv_profile = np.linspace(d_conv_eq, d_conv_pole, n_lat)
        D_conv = np.tile(d_conv_profile, (n_samples, 1)) + rng.normal(0, 0.1, (n_samples, n_lat))
        D_cond = np.full((n_samples, n_lat), 25.0) + rng.normal(0, 2.0, (n_samples, n_lat))
        Ra = np.tile(np.linspace(50, 10, n_lat), (n_samples, 1))
        return {
            'D_conv_profiles': D_conv,
            'D_cond_profiles': D_cond,
            'Ra_profiles': Ra,
            'H_profiles': D_cond + D_conv,
            'n_valid': n_samples,
            'n_iterations': n_samples,
            'latitudes_deg': lats,
        }

    low = _make(2.0, 2.1)
    high = _make(4.0, 1.0)

    score_low, _ = compute_latitude_score(
        scenarios={'uniform': low, 'polar': low, 'equator': low},
        consistency_error=0.01,
    )
    score_high, _ = compute_latitude_score(
        scenarios={'uniform': high, 'polar': high, 'equator': high},
        consistency_error=0.01,
    )
    assert score_high < score_low
