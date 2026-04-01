"""
Experimental convection hypothesis adjusters for Europa2D.

Each hypothesis is a factory that returns a convection_adjuster callback.
The callback mutates a ConvectionState in place during
build_conductivity_profile(), before Nu is applied to k_profile.

This is the only coupling point with the 1D solver.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from Convection import ConvectionState, IceConvection
from Physics import IcePhysics
from constants import Thermal, Planetary, Rheology, Convection as ConvectionConstants
from latitude_profile import LatitudeProfile


@dataclass(frozen=True)
class ConvectionHypothesis:
    """Defines which convection adjustment to apply and its parameters."""
    mechanism: str   # "heat_balance" | "ra_onset" | "tidal_viscosity"
    params: dict     # mechanism-specific parameters


def make_adjuster(
    hypothesis: Optional[ConvectionHypothesis],
    phi: float,
    profile: LatitudeProfile,
) -> Optional[Callable]:
    """Create a convection_adjuster closure for one latitude column.

    Args:
        hypothesis: ConvectionHypothesis or None (no adjustment)
        phi: Geographic latitude in radians for this column
        profile: LatitudeProfile instance

    Returns:
        Callable matching the convection_adjuster signature, or None.
    """
    if hypothesis is None:
        return None

    mechanism = hypothesis.mechanism
    params = hypothesis.params

    if mechanism == "heat_balance":
        include_tidal = params.get("include_tidal", False)
        max_iterations = params.get("max_iterations", 5)
        tolerance = params.get("tolerance", 1e-4)
        epsilon_0 = float(profile.tidal_strain(phi))
        mu_ice = 3.3e9

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _heat_balance_adjuster(
                state, T_profile, z_grid, total_thickness, q_ocean,
                include_tidal, max_iterations, tolerance, epsilon_0, mu_ice,
            )
        return adjuster

    if mechanism == "ra_onset":
        ra_crit = params["ra_crit_override"]

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _ra_onset_adjuster(state, ra_crit)
        return adjuster

    if mechanism == "tidal_viscosity":
        epsilon_0_local = float(profile.tidal_strain(phi))
        epsilon_ref = params.get("epsilon_ref", 6e-6)
        n = params.get("softening_exponent", 1.0)

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n)
        return adjuster

    raise ValueError(f"Unknown mechanism: {mechanism!r}")


# --- Hypothesis implementations (stubs, filled in Tasks 5-7) ---

def _heat_balance_adjuster(state, T_profile, z_grid, H, q_ocean,
                           include_tidal, max_iterations, tolerance,
                           epsilon_0, mu_ice):
    """Adjust D_cond so conductive lid flux matches local q_ocean."""
    pass


def _ra_onset_adjuster(state, ra_crit_override):
    """Override is_convecting with custom Ra_crit."""
    pass


def _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n):
    """Rescale Ra and Nu using tidal-softened viscosity."""
    pass
