"""Tests for convection_2d hypothesis adjusters."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import pytest
from convection_2d import ConvectionHypothesis, make_adjuster
from latitude_profile import LatitudeProfile
from Convection import ConvectionState


def _make_test_state(**overrides):
    """Create a ConvectionState with sensible defaults for testing."""
    defaults = dict(
        idx_c=20, z_c=14000.0, D_cond=14000.0, D_conv=6000.0,
        T_c=230.0, Ti=250.0, Ra=1500.0, Nu=3.5, is_convecting=True,
    )
    defaults.update(overrides)
    return ConvectionState(**defaults)


class TestConvectionHypothesis:
    """Tests for hypothesis dataclass and make_adjuster dispatch."""

    def test_make_adjuster_returns_none_for_none_hypothesis(self):
        profile = LatitudeProfile()
        result = make_adjuster(None, 0.0, profile)
        assert result is None

    def test_make_adjuster_returns_callable_for_heat_balance(self):
        hypothesis = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False, "max_iterations": 5, "tolerance": 1e-4},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adjuster = make_adjuster(hypothesis, 0.5, profile)
        assert callable(adjuster)

    def test_make_adjuster_returns_callable_for_ra_onset(self):
        hypothesis = ConvectionHypothesis(
            mechanism="ra_onset",
            params={"ra_crit_override": 1200},
        )
        profile = LatitudeProfile()
        adjuster = make_adjuster(hypothesis, 0.0, profile)
        assert callable(adjuster)

    def test_make_adjuster_returns_callable_for_tidal_viscosity(self):
        hypothesis = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        profile = LatitudeProfile()
        adjuster = make_adjuster(hypothesis, 0.5, profile)
        assert callable(adjuster)

    def test_make_adjuster_raises_for_unknown_mechanism(self):
        hypothesis = ConvectionHypothesis(mechanism="nonsense", params={})
        profile = LatitudeProfile()
        with pytest.raises(ValueError, match="Unknown mechanism"):
            make_adjuster(hypothesis, 0.0, profile)
