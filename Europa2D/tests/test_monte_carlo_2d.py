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
        assert results.T_floor == pytest.approx(46.0)
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


def test_results_default_t_floor_matches_ashkenazy():
    """MonteCarloResults2D default T_floor must be 46.0 K (Ashkenazy 2019)."""
    import numpy as np
    from monte_carlo_2d import MonteCarloResults2D

    results = MonteCarloResults2D(
        H_profiles=np.zeros((1, 5)),
        latitudes_deg=np.linspace(0, 90, 5),
        n_iterations=1,
        n_valid=1,
        H_median=np.zeros(5),
        H_mean=np.zeros(5),
        H_sigma_low=np.zeros(5),
        H_sigma_high=np.zeros(5),
        runtime_seconds=0.0,
        ocean_pattern="uniform",
        ocean_amplitude=0.0,
    )
    assert results.T_floor == 46.0, (
        f"Default T_floor={results.T_floor}, expected 46.0 K (Ashkenazy 2019)"
    )
