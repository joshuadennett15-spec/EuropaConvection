import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from monte_carlo_2d import MonteCarloRunner2D, MonteCarloResults2D


class TestMonteCarloRunner2D:

    def test_runs_and_returns_results(self):
        """Smoke test: 10 iterations should complete."""
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert isinstance(results, MonteCarloResults2D)
        assert results.n_valid > 0
        assert results.H_profiles.shape[1] == 5  # n_lat
        assert results.ocean_pattern == "polar_enhanced"
        # Default: mantle_tidal_fraction=0.5 -> q_star=0.455 -> a=0.536
        expected_a = 3.0 * 0.455 / (3.0 - 0.455)
        assert results.ocean_amplitude == pytest.approx(expected_a, rel=1e-3)
        assert results.T_floor == pytest.approx(52.0)
        assert results.q_star == pytest.approx(0.455, rel=1e-3)
        assert results.mantle_tidal_fraction == pytest.approx(0.5)

    def test_H_profiles_are_reasonable(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert np.all(results.H_profiles > 0)
        assert np.all(results.H_profiles < 200)

    def test_statistics_computed(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert results.H_median.shape == (5,)
        assert results.H_mean.shape == (5,)
        assert results.H_sigma_low.shape == (5,)
        assert results.H_sigma_high.shape == (5,)
