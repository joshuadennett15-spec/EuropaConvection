"""
Latitude-dependent parameter profiles for Europa's ice shell.

Provides continuous functions for surface temperature, tidal strain,
and ocean heat flux as functions of geographic latitude phi.

Convention: phi = 0 at equator, phi = pi/2 at pole.

References:
    - Ojakangas & Stevenson (1989): Surface temperature distribution
    - Tobie et al. (2003): Tidal strain patterns
    - Soderlund et al. (2014): Ocean heat transport
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
from typing import Literal, Union
from dataclasses import dataclass

from constants import Planetary

OceanPattern = Literal["uniform", "polar_enhanced", "equator_enhanced"]

FloatOrArray = Union[float, npt.NDArray[np.float64]]


@dataclass(frozen=True)
class LatitudeProfile:
    """
    Latitude-dependent physical parameters for Europa's ice shell.

    All angles are geographic latitude in radians:
        phi = 0 at equator, phi = pi/2 at pole.

    Attributes:
        T_eq: Equatorial surface temperature (K)
        epsilon_eq: Tidal strain at equator
        epsilon_pole: Tidal strain at pole
        q_ocean_mean: Global mean ocean heat flux (W/m^2)
        ocean_pattern: Heat flux distribution pattern
    """
    T_eq: float = 110.0
    epsilon_eq: float = 6.0e-6
    epsilon_pole: float = 1.2e-5
    q_ocean_mean: float = 0.02
    ocean_pattern: OceanPattern = "polar_enhanced"
    T_floor: float = 52.0

    def __post_init__(self):
        if self.T_floor <= 0:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be positive."
            )
        if self.T_floor >= self.T_eq:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be less than T_eq ({self.T_eq} K). "
                "A polar floor >= equatorial temperature is non-physical for Europa."
            )

    def surface_temperature(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Surface temperature as a function of latitude.

        T_s(phi) = ((T_eq^4 - T_floor^4) * cos(phi) + T_floor^4)^(1/4)

        Reparameterized energy balance: T_s(0) = T_eq exactly, T_s(pi/2) = T_floor.
        The T_floor default (52 K) is from Ashkenazy (2019), absorbing obliquity,
        seasonal insolation, thermal inertia, and Jupiter longwave radiation.

        References:
            Ojakangas & Stevenson (1989): radiative equilibrium framework
            Ashkenazy (2019): full seasonal energy balance, T_pole = 51-52 K

        Args:
            phi: Geographic latitude in radians (0=equator, pi/2=pole)

        Returns:
            Surface temperature (K)
        """
        phi_arr = np.asarray(phi)
        T_eq4 = self.T_eq ** 4
        T_fl4 = self.T_floor ** 4
        result = ((T_eq4 - T_fl4) * np.cos(phi_arr) + T_fl4) ** 0.25
        return float(result) if np.ndim(phi) == 0 else result

    def tidal_strain(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Tidal strain amplitude as a function of latitude.

        eps_0(phi) = eps_eq * sqrt(1 + c * sin^2(phi))
        where c = (eps_pole / eps_eq)^2 - 1

        This ensures eps_0^2(phi) = eps_eq^2 * (1 + c*sin^2(phi)), which
        reproduces the Beuthe (2013) zonally-averaged whole-shell eccentricity-tide
        dissipation pattern: q_tidal ~ 1 + 3*sin^2(phi) when c = 3.

        References:
            Beuthe (2013): spatial patterns of tidal heating, Icarus 223, 308-329
            Tobie et al. (2003): ~4:1 pole-to-equator dissipation ratio

        Args:
            phi: Geographic latitude in radians (0=equator, pi/2=pole)

        Returns:
            Tidal strain amplitude (dimensionless)
        """
        phi_arr = np.asarray(phi)
        c = (self.epsilon_pole / self.epsilon_eq) ** 2 - 1.0
        sin2 = np.sin(phi_arr) ** 2
        result = self.epsilon_eq * np.sqrt(1.0 + c * sin2)
        return float(result) if np.ndim(phi) == 0 else result

    def ocean_heat_flux(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Ocean heat flux as a function of latitude.

        Supports three patterns, all normalized to preserve the global mean:
        - uniform: q(phi) = q_mean
        - polar_enhanced: q proportional to 1 + 2*sin^2(phi), Soderlund et al. (2014)
        - equator_enhanced: q proportional to 1 + 2*cos^2(phi)

        Normalization: integral_0^{pi/2} q(phi)cos(phi) dphi / integral_0^{pi/2} cos(phi) dphi = q_mean

        Args:
            phi: Geographic latitude in radians

        Returns:
            Ocean heat flux (W/m^2)
        """
        phi_arr = np.asarray(phi)

        if self.ocean_pattern == "uniform":
            result = np.full_like(phi_arr, self.q_ocean_mean, dtype=float)
        elif self.ocean_pattern == "polar_enhanced":
            # Shape: 1 + 2*sin^2(phi)
            # Analytical: integral_0^{pi/2} (1+2sin^2(phi))cos(phi) dphi = 5/3
            # integral_0^{pi/2} cos(phi) dphi = 1
            # So normalization factor = 5/3
            norm = 5.0 / 3.0
            shape = 1.0 + 2.0 * np.sin(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        elif self.ocean_pattern == "equator_enhanced":
            # Shape: 1 + 2*cos^2(phi)
            # Analytical: integral_0^{pi/2} (1+2cos^2(phi))cos(phi) dphi = 7/3
            norm = 7.0 / 3.0
            shape = 1.0 + 2.0 * np.cos(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        else:
            raise ValueError(f"Unknown ocean pattern: {self.ocean_pattern}")

        return float(result) if np.ndim(phi) == 0 else result

    def evaluate_at(self, phi: float) -> dict:
        """
        Evaluate all latitude-dependent parameters at a single latitude.

        Args:
            phi: Geographic latitude in radians

        Returns:
            Dict with keys: T_surf, epsilon_0, q_ocean
        """
        return {
            'T_surf': self.surface_temperature(phi),
            'epsilon_0': self.tidal_strain(phi),
            'q_ocean': self.ocean_heat_flux(phi),
        }
