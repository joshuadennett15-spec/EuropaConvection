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


class TestHeatBalanceAdjuster:
    """Tests for the heat-balance D_cond correction."""

    def test_higher_q_ocean_thins_conductive_lid(self):
        """Higher q_ocean should reduce D_cond (thinner lid to conduct more heat)."""
        state_lo = _make_test_state()
        state_hi = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)

        profile = LatitudeProfile(q_ocean_mean=0.02)
        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False, "max_iterations": 5, "tolerance": 1e-4},
        )
        adj_lo = make_adjuster(hyp, 0.0, profile)
        adj_hi = make_adjuster(hyp, 0.0, profile)

        adj_lo(state_lo, T, z, 20e3, 0.01)
        adj_hi(state_hi, T, z, 20e3, 0.05)

        assert state_hi.D_cond < state_lo.D_cond, (
            f"D_cond should decrease with higher q_ocean: "
            f"lo={state_lo.D_cond:.0f}, hi={state_hi.D_cond:.0f}"
        )

    def test_d_conv_increases_when_q_ocean_increases(self):
        """Thinner lid means thicker convective layer."""
        state_lo = _make_test_state()
        state_hi = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)

        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj_lo = make_adjuster(hyp, 0.0, profile)
        adj_hi = make_adjuster(hyp, 0.0, profile)

        adj_lo(state_lo, T, z, 20e3, 0.01)
        adj_hi(state_hi, T, z, 20e3, 0.05)

        assert state_hi.D_conv > state_lo.D_conv

    def test_non_convecting_state_unchanged(self):
        """If is_convecting=False, adjuster should not modify state."""
        state = _make_test_state(is_convecting=False, Ra=500.0, Nu=1.0)
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)
        D_cond_before = state.D_cond

        hyp = ConvectionHypothesis(mechanism="heat_balance", params={"include_tidal": False})
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj = make_adjuster(hyp, 0.0, profile)
        adj(state, T, z, 20e3, 0.02)

        assert state.D_cond == D_cond_before

    def test_d_cond_clamped_to_physical_range(self):
        """D_cond must stay within [0.05*H, 0.95*H]."""
        state = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)
        H = 20e3

        hyp = ConvectionHypothesis(mechanism="heat_balance", params={"include_tidal": False})
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj = make_adjuster(hyp, 0.0, profile)

        adj(state, T, z, H, 10.0)  # extreme q_ocean
        assert state.D_cond >= 0.05 * H
        assert state.D_conv <= 0.95 * H


class TestRaOnsetAdjuster:

    def test_high_ra_crit_shuts_off_weak_convection(self):
        state = _make_test_state(Ra=1500.0, Nu=3.5, is_convecting=True)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 2000})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.is_convecting is False
        assert state.Nu == 1.0

    def test_low_ra_crit_keeps_convection(self):
        state = _make_test_state(Ra=1500.0, Nu=3.5, is_convecting=True)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 800})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.is_convecting is True
        assert state.Nu == 3.5

    def test_ra_onset_can_enable_subcritical_column(self):
        state = _make_test_state(Ra=500.0, Nu=1.0, is_convecting=False)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 400})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.is_convecting is True
        assert state.Nu > 1.0


class TestTidalViscosityAdjuster:

    def test_higher_strain_increases_nu(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        state_eq = _make_test_state(Ra=1500.0, Nu=3.5)
        state_pole = _make_test_state(Ra=1500.0, Nu=3.5)

        adj_eq = make_adjuster(hyp, 0.0, profile)
        adj_pole = make_adjuster(hyp, np.pi / 2, profile)

        adj_eq(state_eq, np.zeros(31), np.zeros(31), 20e3, 0.02)
        adj_pole(state_pole, np.zeros(31), np.zeros(31), 20e3, 0.02)

        assert state_pole.Nu > state_eq.Nu

    def test_equatorial_column_still_softened(self):
        """At equator, epsilon_0/epsilon_ref=1, so softening factor = 1/(1+1) = 0.5 for eta."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        state = _make_test_state(Ra=1500.0, Nu=3.5)
        adj = make_adjuster(hyp, 0.0, profile)
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.Ra > 1500.0

    def test_non_convecting_unchanged(self):
        state = _make_test_state(is_convecting=False, Ra=100.0, Nu=1.0)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        adj = make_adjuster(hyp, np.pi / 4, LatitudeProfile())
        Nu_before = state.Nu
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.Nu == Nu_before
