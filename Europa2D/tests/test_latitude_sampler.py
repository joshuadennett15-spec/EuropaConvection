import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_sampler import LatitudeParameterSampler
from latitude_profile import LatitudeProfile


class TestLatitudeParameterSampler:

    def test_returns_dict_and_profile(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, profile = sampler.sample()
        assert isinstance(params, dict)
        assert isinstance(profile, LatitudeProfile)

    def test_shared_params_present(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        required = ['d_grain', 'Q_v', 'Q_b', 'mu_ice', 'D0v', 'D0b', 'd_del',
                     'f_porosity', 'f_salt', 'B_k', 'T_phi', 'H_rad',
                     'q_basal', 'q_tidal']
        for key in required:
            assert key in params, f"Missing key: {key}"

    def test_reproducible_with_seed(self):
        s1 = LatitudeParameterSampler(seed=42)
        s2 = LatitudeParameterSampler(seed=42)
        p1, prof1 = s1.sample()
        p2, prof2 = s2.sample()
        assert p1['d_grain'] == p2['d_grain']
        assert prof1.T_eq == prof2.T_eq

    def test_profile_has_valid_values(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert 80 < profile.T_eq < 140
        assert 2e-6 <= profile.epsilon_eq <= 2e-5
        assert 2e-6 <= profile.epsilon_pole <= 3.4e-5
        assert profile.q_ocean_mean > 0

    def test_audited_fixed_params(self):
        """Audited priors: f_salt=0, B_k=1, T_phi=150."""
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        assert params['f_salt'] == 0.0
        assert params['B_k'] == 1.0
        assert params['T_phi'] == 150.0

    def test_audited_q_basal_range(self):
        """q_basal should be uniform in [10, 30] mW/m²."""
        sampler = LatitudeParameterSampler(seed=0)
        q_vals = [sampler.sample()[0]['q_basal'] for _ in range(200)]
        assert all(10e-3 <= q <= 30e-3 for q in q_vals)

    def test_audited_H_rad_positive(self):
        """H_rad must be truncated > 0."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            params, _ = sampler.sample()
            assert params['H_rad'] > 0

    def test_audited_f_porosity_range(self):
        """f_porosity tightened to [0, 0.10]."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            params, _ = sampler.sample()
            assert 0.0 <= params['f_porosity'] <= 0.10


class TestSamplerNewFields:

    def test_profile_has_T_floor(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'T_floor')
        assert 40 <= profile.T_floor <= 70

    def test_profile_has_mantle_tidal_fraction(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'mantle_tidal_fraction')
        assert 0.0 < profile.mantle_tidal_fraction < 1.0

    def test_T_floor_independent_of_q_ocean(self):
        """T_floor should NOT be derived from q_ocean_mean."""
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        # T_floor should be sampled from Normal(52, 5), not 52 + 240*q
        assert profile.T_floor != pytest.approx(52.0 + 240.0 * profile.q_ocean_mean, rel=0.01)

    def test_T_floor_less_than_T_eq(self):
        """Guard: T_floor must always be < T_eq."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            _, profile = sampler.sample()
            assert profile.T_floor < profile.T_eq

    def test_T_floor_varies_across_samples(self):
        """T_floor should be sampled, not a constant default."""
        sampler = LatitudeParameterSampler(seed=0)
        floors = [sampler.sample()[1].T_floor for _ in range(50)]
        assert len(set(floors)) > 1, "T_floor should vary across samples"

    def test_mantle_tidal_fraction_varies_across_samples(self):
        """mantle_tidal_fraction should be sampled, not a constant default."""
        sampler = LatitudeParameterSampler(seed=0)
        fracs = [sampler.sample()[1].mantle_tidal_fraction for _ in range(50)]
        assert len(set(fracs)) > 1, "mantle_tidal_fraction should vary across samples"

    def test_mantle_tidal_fraction_range(self):
        """mantle_tidal_fraction ~ Uniform(0.1, 0.9)."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            _, profile = sampler.sample()
            assert 0.1 <= profile.mantle_tidal_fraction <= 0.9

    def test_equator_enhanced_has_explicit_q_star(self):
        """For equator_enhanced pattern, q_star should be sampled directly."""
        sampler = LatitudeParameterSampler(seed=42, ocean_pattern="equator_enhanced")
        _, profile = sampler.sample()
        assert profile.q_star is not None
        assert 0.1 <= profile.q_star <= 0.8

    def test_polar_enhanced_has_no_explicit_q_star(self):
        """For polar_enhanced, q_star should be None (derived from mantle_tidal_fraction)."""
        sampler = LatitudeParameterSampler(seed=42, ocean_pattern="polar_enhanced")
        _, profile = sampler.sample()
        assert profile.q_star is None
