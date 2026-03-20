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
from typing import Literal, Optional, Union
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
    T_eq: float = 96.0
    epsilon_eq: float = 6.0e-6
    epsilon_pole: float = 1.2e-5
    q_ocean_mean: float = 0.02
    ocean_pattern: OceanPattern = "uniform"
    ocean_amplitude: Optional[float] = None
    T_floor: float = 46.0
    q_star: Optional[float] = None
    mantle_tidal_fraction: float = 0.5
    strict_q_star: bool = True

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
        if not (0.0 <= self.mantle_tidal_fraction <= 1.0):
            raise ValueError(
                f"mantle_tidal_fraction ({self.mantle_tidal_fraction}) must be in [0, 1]. "
                "It represents q_tidal / (q_tidal + q_radiogenic)."
            )
        if self.q_star is not None:
            if self.strict_q_star and self.q_star > 0.91:
                raise ValueError(
                    f"q_star ({self.q_star}) exceeds Lemasquerier (2023) physical range "
                    f"(max ~0.91 for pure tidal mantle heating). "
                    f"Set strict_q_star=False for exploratory runs."
                )
            if self.ocean_pattern == "polar_enhanced" and self.q_star >= 3.0:
                raise ValueError(
                    f"q_star ({self.q_star}) >= 3.0 causes singularity in polar_enhanced "
                    f"amplitude inversion: a = 3*q_star/(3-q_star)."
                )
            if self.ocean_pattern == "equator_enhanced" and self.q_star >= 1.5:
                raise ValueError(
                    f"q_star ({self.q_star}) >= 1.5 causes singularity in equator_enhanced "
                    f"amplitude inversion: a = 3*q_star/(3-2*q_star)."
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
        if self.epsilon_eq == 0.0:
            result = np.zeros_like(phi_arr, dtype=float)
            return float(result) if np.ndim(phi) == 0 else result
        c = (self.epsilon_pole / self.epsilon_eq) ** 2 - 1.0
        sin2 = np.sin(phi_arr) ** 2
        result = self.epsilon_eq * np.sqrt(1.0 + c * sin2)
        return float(result) if np.ndim(phi) == 0 else result

    def resolved_q_star(self) -> float:
        """
        Return the effective q_star (Lemasquerier 2023 contrast parameter).

        Resolution order:
        1. Explicit q_star field (if not None)
        2. Derived from mantle_tidal_fraction: q_star = 0.91 * mantle_tidal_fraction

        Returns 0.0 for uniform pattern.

        References:
            Lemasquerier et al. (2023): q* = 0.91 for pure tidal mantle heating
        """
        if self.ocean_pattern == "uniform":
            return 0.0
        if self.q_star is not None:
            return float(self.q_star)
        return 0.91 * self.mantle_tidal_fraction

    def _q_star_to_amplitude(self, q_star: float) -> float:
        """
        Convert q_star to shape function amplitude a.

        polar_enhanced:   a = 3*q_star / (3 - q_star)
        equator_enhanced: a = 3*q_star / (3 - 2*q_star)
        uniform:          a = 0
        """
        if self.ocean_pattern == "uniform":
            return 0.0
        if self.ocean_pattern == "polar_enhanced":
            return 3.0 * q_star / (3.0 - q_star)
        if self.ocean_pattern == "equator_enhanced":
            return 3.0 * q_star / (3.0 - 2.0 * q_star)
        raise ValueError(f"Unknown ocean pattern: {self.ocean_pattern}")

    def resolved_ocean_amplitude(self) -> float:
        """
        Return the contrast amplitude used by the ocean heat-flux pattern.

        Resolution order:
        1. Explicit ocean_amplitude (if not None) -- backward compat
        2. Derived from q_star via pattern-specific inversion
        3. Derived from mantle_tidal_fraction -> q_star -> amplitude

        References:
            Lemasquerier et al. (2023): q* contrast parameter
            Soderlund et al. (2014): ~40% zonal-mean variation
        """
        if self.ocean_amplitude is not None:
            return float(self.ocean_amplitude)
        if self.ocean_pattern == "uniform":
            return 0.0
        q = self.resolved_q_star()
        return self._q_star_to_amplitude(q)

    def ocean_heat_flux(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Ocean heat flux as a function of latitude.

        Supports three patterns, all normalized to preserve the global mean:
        - uniform: q(phi) = q_mean
        - polar_enhanced: q proportional to 1 + a*sin^2(phi)
        - equator_enhanced: q proportional to 1 + a*cos^2(phi)

        The amplitude a is resolved via resolved_ocean_amplitude():
            ocean_amplitude > q_star > mantle_tidal_fraction

        Normalization: integral_0^{pi/2} q(phi)cos(phi) dphi
                     / integral_0^{pi/2} cos(phi) dphi = q_mean

        Args:
            phi: Geographic latitude in radians

        Returns:
            Ocean heat flux (W/m^2)
        """
        phi_arr = np.asarray(phi)
        a = self.resolved_ocean_amplitude()

        if self.ocean_pattern == "uniform":
            result = np.full_like(phi_arr, self.q_ocean_mean, dtype=float)
        elif self.ocean_pattern == "polar_enhanced":
            # Shape: 1 + a*sin^2(phi)
            # Analytical norm: 1 + a/3
            norm = 1.0 + a / 3.0
            shape = 1.0 + a * np.sin(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        elif self.ocean_pattern == "equator_enhanced":
            # Shape: 1 + a*cos^2(phi)
            # Analytical norm: 1 + 2*a/3
            norm = 1.0 + 2.0 * a / 3.0
            shape = 1.0 + a * np.cos(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        else:
            raise ValueError(f"Unknown ocean pattern: {self.ocean_pattern}")

        return float(result) if np.ndim(phi) == 0 else result

    def ocean_endpoint_fluxes(self) -> tuple:
        """
        Return (q_equator, q_pole) ocean heat fluxes.

        Convenience for endpoint ratio calculations and diagnostics.

        Returns:
            Tuple of (q_equator, q_pole) in W/m^2
        """
        return (self.ocean_heat_flux(0.0), self.ocean_heat_flux(np.pi / 2))

    def ocean_endpoint_ratio(self) -> float:
        """
        Return the larger/smaller endpoint ratio for ocean heat flux.

        For polar_enhanced: q_pole / q_eq (>= 1)
        For equator_enhanced: q_eq / q_pole (>= 1)
        For uniform: 1.0

        Returns:
            Endpoint ratio (always >= 1)
        """
        q_eq, q_pole = self.ocean_endpoint_fluxes()
        if q_eq == 0 and q_pole == 0:
            return 1.0
        return max(q_eq, q_pole) / min(q_eq, q_pole)

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
