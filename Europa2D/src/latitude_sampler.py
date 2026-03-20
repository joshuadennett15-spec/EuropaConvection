"""
Parameter sampler for 2D Monte Carlo runs.

Shared shell physics is drawn from the audited 1D baseline so the 2D model is
directly comparable to the audited `EuropaProjectDJ` workflow. Latitude
structure is then imposed separately through `LatitudeProfile`.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
from typing import Dict, Optional, Tuple

from audited_sampler import AuditedShellSampler
from constants import Planetary
from latitude_profile import LatitudeProfile, OceanPattern


class LatitudeParameterSampler:
    """
    Sample shared shell physics plus latitude-structure controls for Europa2D.

    Shared shell properties are held constant across latitude columns within one
    realization:
        d_grain, mu_ice, Q_v, Q_b, H_rad, D_H2O, f_porosity,
        f_salt, B_k, T_phi, D0v, D0b, d_del.

    Latitude dependence enters only through:
        T_eq/T_floor, epsilon_eq/epsilon_pole, q_ocean_mean via ocean pattern,
        q_star, and mantle_tidal_fraction.
    """

    SHARED_AUDITED_KEYS = (
        'd_grain', 'd_del', 'D0v', 'D0b',
        'mu_ice', 'D_H2O', 'Q_v', 'Q_b',
        'H_rad', 'f_porosity', 'f_salt', 'T_phi', 'B_k',
    )

    LATITUDE_STRUCTURE_KEYS = (
        'T_eq', 'T_floor', 'epsilon_eq', 'epsilon_pole',
        'q_ocean_mean', 'ocean_pattern', 'ocean_amplitude',
        'q_star', 'mantle_tidal_fraction', 'tidal_pattern',
    )

    def __init__(
        self,
        seed: Optional[int] = None,
        ocean_pattern: OceanPattern = "uniform",
        ocean_amplitude: Optional[float] = None,
        q_star: Optional[float] = None,
        tidal_pattern: str = "mantle_core",
    ):
        seed_sequence = np.random.SeedSequence(seed)
        shared_seq, latitude_seq = seed_sequence.spawn(2)

        self._shared_sampler = AuditedShellSampler()
        self._shared_sampler.rng = np.random.default_rng(shared_seq)
        self.rng = np.random.default_rng(latitude_seq)

        self.ocean_pattern = ocean_pattern
        self.ocean_amplitude = ocean_amplitude
        self.q_star_override = q_star
        self._tidal_pattern = tidal_pattern

    @classmethod
    def shared_parameter_names(cls) -> Tuple[str, ...]:
        """Shell properties that do not vary by latitude in this 2D proxy."""
        return cls.SHARED_AUDITED_KEYS

    @classmethod
    def latitude_structure_names(cls) -> Tuple[str, ...]:
        """Controls that define the latitude dependence of one realization."""
        return cls.LATITUDE_STRUCTURE_KEYS

    def sample(self) -> Tuple[Dict[str, float], LatitudeProfile]:
        """
        Sample all parameters for one 2D MC iteration.

        Returns:
            (shared_params, latitude_profile) tuple
        """
        audited_params = self._shared_sampler.sample()
        D_H2O = audited_params['D_H2O']
        H_rad = audited_params['H_rad']

        # Latitude-dependent surface forcing:
        # keep an equatorial anchor rather than reusing the audited global 1D
        # T_surf value, because T_surf is an explicitly latitude-varying field
        # in the 2D model.
        T_eq = self.rng.normal(96.0, 5.0)
        T_eq = np.clip(T_eq, 80.0, 115.0)

        # Ashkenazy (2019): annual-mean polar floor 46 K at Q=0.05 W/m^2,
        # rising to ~53 K at Q=0.2 W/m^2.  Sample independently of
        # q_ocean_mean to avoid double-counting polar thermal effects.
        T_floor = self.rng.normal(46.0, 4.0)
        T_floor = float(np.clip(T_floor, 38.0, 58.0))
        T_floor = min(T_floor, T_eq - 1.0)

        # Latitude-varying tidal strain uses equatorial and polar anchors.
        epsilon_eq = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_eq = np.clip(epsilon_eq, 2e-6, 2e-5)

        epsilon_pole = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_pole = np.clip(epsilon_pole, 2e-6, 3.4e-5)

        # Mean basal heat flux inherits the audited 1D shell prior. In the
        # current 2D proxy that global-mean basal flux is redistributed by the
        # latitude-only ocean pattern.
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_radiogenic = (H_rad * M_rock) / Planetary.AREA
        q_tidal_global = audited_params['P_tidal'] / Planetary.AREA
        q_basal_global = q_radiogenic + q_tidal_global
        q_ocean_mean = q_basal_global

        mantle_tidal_fraction = float(self.rng.uniform(0.1, 0.9))

        # For equator-enhanced cases, keep a direct q* prior matched to the
        # Soderlund-style benchmark. Polar/uniform cases derive q* from the
        # tidal fraction unless a scenario override is provided.
        q_star_explicit = self.q_star_override
        if q_star_explicit is None and self.ocean_pattern == "equator_enhanced":
            q_star_explicit = self.rng.normal(0.4, 0.1)
            q_star_explicit = float(np.clip(q_star_explicit, 0.1, 0.8))

        # tidal_pattern is held fixed per MC campaign, not sampled
        tidal_pattern = self._tidal_pattern  # set in constructor, default "mantle_core"

        profile = LatitudeProfile(
            T_eq=T_eq,
            T_floor=T_floor,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=self.ocean_amplitude,
            q_star=q_star_explicit,
            mantle_tidal_fraction=mantle_tidal_fraction,
            tidal_pattern=tidal_pattern,
        )

        shared_params = {
            key: audited_params[key]
            for key in self.SHARED_AUDITED_KEYS
        }
        shared_params['q_basal'] = q_basal_global
        shared_params['q_tidal'] = q_tidal_global

        return shared_params, profile
