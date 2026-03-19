"""
Parameter sampler for 2D Monte Carlo runs.

Samples shared physics parameters (identical across columns) and
latitude-dependent amplitudes that define a LatitudeProfile.

Based on Howell (2021) distributions extended for 2D.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
from typing import Dict, Optional, Tuple

from constants import Thermal, Planetary
from latitude_profile import LatitudeProfile, OceanPattern


class LatitudeParameterSampler:
    """
    Samples shared + latitude-dependent parameters for 2D MC runs.

    Returns a tuple of (shared_params dict, LatitudeProfile instance).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        ocean_pattern: OceanPattern = "polar_enhanced",
        ocean_amplitude: Optional[float] = None,
        q_star: Optional[float] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.ocean_pattern = ocean_pattern
        self.ocean_amplitude = ocean_amplitude
        self.q_star_override = q_star

    def _sample_truncated_normal(self, mean: float, sigma: float,
                                 low: float = -np.inf, high: float = np.inf) -> float:
        while True:
            sample = self.rng.normal(mean, sigma)
            if low <= sample <= high:
                return sample

    def sample(self) -> Tuple[Dict[str, float], LatitudeProfile]:
        """
        Sample all parameters for one 2D MC iteration.

        Uses audited priors: direct q_basal sampling (no P_tidal),
        fixed f_salt=0, fixed B_k=1, tightened f_porosity and epsilon clips.

        Returns:
            (shared_params, latitude_profile) tuple
        """
        # === Shared parameters (audited priors) ===
        d_grain = 10 ** self.rng.normal(np.log10(7e-4), 0.5)
        d_grain = np.clip(d_grain, 1e-5, 5e-3)

        mu_ice = self._sample_truncated_normal(3.5e9, 0.5e9, low=2.0e9, high=5.0e9)

        Q_v = self.rng.normal(59.4e3, 0.05 * 59.4e3)
        Q_b = self.rng.normal(49.0e3, 0.05 * 49.0e3)

        H_rad = self._sample_truncated_normal(4.5e-12, 1.0e-12, low=0.0)
        T_phi = 150.0  # Fixed (audited)

        D_H2O = self.rng.normal(127e3, 21e3)
        D_H2O = np.clip(D_H2O, 80e3, 200e3)

        f_porosity = self.rng.uniform(0.0, 0.10)  # Tightened from [0, 0.30]
        f_salt = 0.0   # Fixed (audited)
        B_k = 1.0      # Fixed (audited)

        D0v = max(self.rng.normal(9.1e-4, 0.033 * 9.1e-4), 1e-8)
        D0b = max(self.rng.normal(8.4e-4, 0.033 * 8.4e-4), 1e-8)
        d_del_mean = np.mean([9.04e-10, 5.22e-10])
        d_del_std = np.std([9.04e-10, 5.22e-10])
        d_del = max(self.rng.normal(d_del_mean, d_del_std), 1e-12)

        # === Latitude-dependent amplitudes ===
        T_eq = self.rng.normal(110.0, 5.0)
        T_eq = np.clip(T_eq, 90.0, 130.0)

        # T_floor: independent of q_ocean_mean (avoids double-counting)
        # Ashkenazy (2019): T_pole = 51-52 K at zero internal heating
        T_floor = self.rng.normal(52.0, 5.0)
        T_floor = float(np.clip(T_floor, 40.0, 70.0))
        # Ensure T_floor < T_eq
        T_floor = min(T_floor, T_eq - 1.0)

        epsilon_eq = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_eq = np.clip(epsilon_eq, 2e-6, 2e-5)  # Tightened lower bound

        epsilon_pole = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_pole = np.clip(epsilon_pole, 2e-6, 3.4e-5)  # Audited bounds

        # === Ocean heat flux: direct q_basal sampling (audited) ===
        q_basal_global = self.rng.uniform(10e-3, 30e-3)  # W/m², U(10, 30) mW/m²

        R_europa = Planetary.RADIUS
        R_rock = R_europa - D_H2O
        A_surface = Planetary.AREA
        rho_rock = 3500.0
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * rho_rock

        q_radiogenic = (H_rad * M_rock) / A_surface
        q_tidal_global = max(0.0, q_basal_global - q_radiogenic)
        q_ocean_mean = q_basal_global  # Redistributed via pattern

        # mantle_tidal_fraction: radiogenic vs tidal partition
        # Lemasquerier (2023): q* = 0.91 * mantle_tidal_fraction
        mantle_tidal_fraction = float(self.rng.uniform(0.1, 0.9))

        # q_star resolution:
        # 1. If scenario specifies q_star, use it (no sampling — deterministic contrast)
        # 2. For equator_enhanced without override: sample Normal(0.4, 0.1) clipped [0.1, 0.8]
        # 3. For polar_enhanced/uniform without override: leave None (derived from mantle_tidal_fraction)
        q_star_explicit = self.q_star_override
        if q_star_explicit is None and self.ocean_pattern == "equator_enhanced":
            q_star_explicit = self.rng.normal(0.4, 0.1)
            q_star_explicit = float(np.clip(q_star_explicit, 0.1, 0.8))

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
        )

        shared_params = {
            'd_grain': d_grain, 'd_del': d_del,
            'D0v': D0v, 'D0b': D0b,
            'mu_ice': mu_ice,
            'D_H2O': D_H2O,
            'Q_v': Q_v, 'Q_b': Q_b,
            'H_rad': H_rad,
            'q_basal': q_basal_global,
            'q_tidal': q_tidal_global,
            'f_porosity': f_porosity,
            'f_salt': f_salt,
            'T_phi': T_phi,
            'B_k': B_k,
        }

        return shared_params, profile
