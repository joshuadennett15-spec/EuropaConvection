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
    """Adjust D_cond so conductive lid flux matches local q_ocean.

    At equilibrium the conductive lid must transport q_ocean:
        q_lid = k_lid * (T_c - T_surface) / D_cond = q_ocean
    This gives: D_cond_eq = k_lid * (T_c - T_surface) / q_ocean
    Then D_conv = H - D_cond_eq, and Ra/Nu are recomputed.
    """
    if not state.is_convecting or q_ocean <= 0:
        return

    T_surface = float(T_profile[0])
    T_c = state.T_c
    T_melt = float(T_profile[-1])

    T_mean_lid = 0.5 * (T_surface + T_c)
    k_lid = float(Thermal.conductivity(T_mean_lid))

    q_tidal_contrib = 0.0
    if include_tidal and epsilon_0 > 0 and state.D_conv > 0:
        T_mean_conv = 0.5 * (T_c + T_melt)
        try:
            q_vol = IcePhysics.tidal_heating(
                np.array([T_mean_conv]),
                epsilon_0=epsilon_0, mu_ice=mu_ice,
                use_composite_viscosity=True,
            )
            q_tidal_contrib = float(q_vol[0]) * state.D_conv
        except Exception:
            q_tidal_contrib = 0.0

    q_total = q_ocean + q_tidal_contrib
    if q_total <= 0:
        return

    dT_lid = T_c - T_surface
    if dT_lid <= 0:
        return

    D_cond_eq = k_lid * dT_lid / q_total
    D_cond_eq = max(0.05 * H, min(0.95 * H, D_cond_eq))
    D_conv_new = H - D_cond_eq

    idx_new = int(np.searchsorted(z_grid, D_cond_eq))
    idx_new = max(1, min(len(z_grid) - 2, idx_new))

    if state.D_conv > 0:
        Ra_new = state.Ra * (D_conv_new / state.D_conv) ** 3
    else:
        Ra_new = 0.0

    if Ra_new > 0:
        Nu_new = max(1.0, ConvectionConstants.NU_PREFACTOR * Ra_new ** (1.0 / 3.0))
    else:
        Nu_new = 1.0

    state.D_cond = D_cond_eq
    state.D_conv = D_conv_new
    state.z_c = D_cond_eq
    state.idx_c = idx_new
    state.Ra = Ra_new
    state.Nu = Nu_new
    state.is_convecting = Ra_new >= ConvectionConstants.RA_CRIT


def _ra_onset_adjuster(state, ra_crit_override):
    """Override is_convecting with custom Ra_crit."""
    should_convect = state.Ra >= ra_crit_override

    if should_convect and not state.is_convecting:
        state.is_convecting = True
        if state.Ra > 0:
            state.Nu = max(1.0, ConvectionConstants.NU_PREFACTOR * state.Ra ** (1.0 / 3.0))
        else:
            state.Nu = 1.0
    elif not should_convect and state.is_convecting:
        state.is_convecting = False
        state.Nu = 1.0


def _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n):
    """Rescale Ra and Nu using tidal-softened viscosity.

    eta_eff = eta_default / (1 + (epsilon_0 / epsilon_ref)^n)
    Ra_adj = Ra_default * (1 + (epsilon_0 / epsilon_ref)^n)
    Nu_adj = C * Ra_adj^(1/3)
    """
    if not state.is_convecting or epsilon_ref <= 0:
        return

    softening = 1.0 + (epsilon_0_local / epsilon_ref) ** n
    Ra_adj = state.Ra * softening

    Nu_adj = max(1.0, ConvectionConstants.NU_PREFACTOR * Ra_adj ** (1.0 / 3.0))

    state.Ra = Ra_adj
    state.Nu = Nu_adj
