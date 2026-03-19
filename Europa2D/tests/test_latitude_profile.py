import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src  # triggers import path setup

import numpy as np
import pytest
from latitude_profile import LatitudeProfile


class TestSurfaceTemperature:
    """Tests for T_s(phi) = T_eq * max(cos(phi), cos(85deg))^(1/4)"""

    def test_equator_returns_T_eq(self):
        profile = LatitudeProfile(T_eq=110.0)
        assert profile.surface_temperature(0.0) == pytest.approx(110.0, abs=0.01)

    def test_pole_is_colder_than_equator(self):
        profile = LatitudeProfile(T_eq=110.0)
        T_pole = profile.surface_temperature(np.radians(90.0))
        assert T_pole < 60.0
        assert T_pole > 30.0

    def test_monotonically_decreasing(self):
        profile = LatitudeProfile(T_eq=110.0)
        lats = np.linspace(0, np.pi / 2, 20)
        temps = np.array([profile.surface_temperature(phi) for phi in lats])
        assert np.all(np.diff(temps) <= 0)

    def test_floor_prevents_zero(self):
        profile = LatitudeProfile(T_eq=110.0)
        T_pole = profile.surface_temperature(np.radians(90.0))
        assert T_pole > 0

    def test_array_input(self):
        profile = LatitudeProfile(T_eq=110.0)
        lats = np.array([0.0, np.radians(45), np.radians(90)])
        temps = profile.surface_temperature(lats)
        assert temps.shape == (3,)
        assert temps[0] > temps[1] > temps[2]


class TestTidalStrain:
    """Tests for eps_0(phi) = eps_eq + (eps_pole - eps_eq) * sin^2(phi)"""

    def test_equator_returns_epsilon_eq(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(0.0) == pytest.approx(6e-6, rel=1e-6)

    def test_pole_returns_epsilon_pole(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(np.pi / 2) == pytest.approx(1.2e-5, rel=1e-6)

    def test_midlatitude_is_between(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        eps_45 = profile.tidal_strain(np.radians(45))
        assert 6e-6 < eps_45 < 1.2e-5

    def test_monotonically_increasing(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        lats = np.linspace(0, np.pi / 2, 20)
        strains = np.array([profile.tidal_strain(phi) for phi in lats])
        assert np.all(np.diff(strains) >= 0)


class TestOceanHeatFlux:
    """Tests for q_ocean(phi) with normalization."""

    def test_uniform_is_constant(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="uniform")
        lats = np.linspace(0, np.pi / 2, 10)
        fluxes = profile.ocean_heat_flux(lats)
        assert np.allclose(fluxes, 0.02, rtol=1e-10)

    def test_polar_enhanced_higher_at_pole(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="polar_enhanced")
        q_eq = profile.ocean_heat_flux(0.0)
        q_pole = profile.ocean_heat_flux(np.pi / 2)
        assert q_pole > q_eq

    def test_equator_enhanced_higher_at_equator(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="equator_enhanced")
        q_eq = profile.ocean_heat_flux(0.0)
        q_pole = profile.ocean_heat_flux(np.pi / 2)
        assert q_eq > q_pole

    def test_normalization_preserves_global_mean(self):
        """Integral of q(phi)*cos(phi) dphi over [0, pi/2] should equal q_mean * 1."""
        from scipy.integrate import quad
        for pattern in ["uniform", "polar_enhanced", "equator_enhanced"]:
            profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern=pattern)
            numerator, _ = quad(
                lambda phi: profile.ocean_heat_flux(phi) * np.cos(phi),
                0, np.pi / 2
            )
            denominator, _ = quad(np.cos, 0, np.pi / 2)  # = 1.0
            mean = numerator / denominator
            assert mean == pytest.approx(0.02, rel=0.01), f"Failed for {pattern}: mean={mean}"


class TestEvaluateAt:
    """Tests for the convenience method that returns all params at a latitude."""

    def test_returns_dict_with_required_keys(self):
        profile = LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        result = profile.evaluate_at(np.radians(45))
        assert 'T_surf' in result
        assert 'epsilon_0' in result
        assert 'q_ocean' in result

    def test_values_match_individual_methods(self):
        profile = LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02)
        phi = np.radians(30)
        result = profile.evaluate_at(phi)
        assert result['T_surf'] == pytest.approx(profile.surface_temperature(phi))
        assert result['epsilon_0'] == pytest.approx(profile.tidal_strain(phi))
        assert result['q_ocean'] == pytest.approx(profile.ocean_heat_flux(phi))


class TestTidalStrainBeuthe:
    """Tests for eps_0(phi) = eps_eq * sqrt(1 + c * sin^2(phi)), c = (eps_pole/eps_eq)^2 - 1."""

    def test_equator_returns_epsilon_eq(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(0.0) == pytest.approx(6e-6, rel=1e-10)

    def test_pole_returns_epsilon_pole(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(np.pi / 2) == pytest.approx(1.2e-5, rel=1e-10)

    def test_midlatitude_heating_ratio_matches_beuthe(self):
        """At 45 deg, eps^2/eps_eq^2 should be 1 + c*0.5, not 2.25 (old linear)."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        c = (1.2e-5 / 6e-6) ** 2 - 1  # = 3.0
        eps_45 = profile.tidal_strain(np.radians(45))
        ratio = (eps_45 / 6e-6) ** 2
        expected = 1.0 + c * 0.5  # = 2.5
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_default_c_equals_3_for_beuthe_pattern(self):
        """With defaults (6e-6, 1.2e-5), c = (2)^2 - 1 = 3, giving the exact
        Beuthe (2013) whole-shell eccentricity-tide pattern."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        c = (profile.epsilon_pole / profile.epsilon_eq) ** 2 - 1
        assert c == pytest.approx(3.0, rel=1e-10)

    def test_monotonically_increasing(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        lats = np.linspace(0, np.pi / 2, 50)
        strains = profile.tidal_strain(lats)
        assert np.all(np.diff(strains) >= 0)

    def test_heating_ratio_pole_to_equator_is_4(self):
        """eps_pole^2 / eps_eq^2 = (1.2e-5)^2 / (6e-6)^2 = 4."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        ratio = (profile.tidal_strain(np.pi / 2) / profile.tidal_strain(0.0)) ** 2
        assert ratio == pytest.approx(4.0, rel=1e-10)
