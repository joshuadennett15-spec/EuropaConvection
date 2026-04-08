"""
Deterministic 2D attribution suite investigating D_conv responsiveness.

Runs four single-realization test scenarios to isolate the effects of:
- Surface temperature gradent
- Ocean heat flux redistribution
- Tidal strain latitude dependence

The scenarios share identical baseline parameters.
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src  # triggers import path setup

import numpy as np

from constants import Thermal
from axial_solver import AxialSolver2D
from attribution_cases import build_paired_attribution_profiles
from latitude_profile import LatitudeProfile
from monte_carlo_2d import save_results_2d
from profile_diagnostics import (
    HIGH_LAT_BAND,
    LOW_LAT_BAND,
    compute_profile_diagnostics,
)
from run_2d_single import FIXED_PARAMS


def build_attribution_scenarios(
    q_ocean_mean: float = 0.02,
    T_eq: float = Thermal.SURFACE_TEMP_MEAN,
    T_floor: float = 50.0,
    epsilon_eq: float = 6e-6,
    epsilon_pole: float = 1.2e-5,
) -> dict[str, LatitudeProfile]:
    """Create four single-variable-varying profiles."""
    source_profile = LatitudeProfile(
        T_eq=T_eq,
        T_floor=T_floor,
        epsilon_eq=epsilon_eq,
        epsilon_pole=epsilon_pole,
        q_ocean_mean=q_ocean_mean,
        ocean_pattern="polar_enhanced",
        q_star=0.455,
        surface_pattern="latitude",
        strict_q_star=False,
    )
    return build_paired_attribution_profiles(source_profile)


def run_attribution_suite(n_lat: int = 19, nx: int = 31, max_steps: int = 500) -> None:
    """Run four scenarios and print comparative diagnostics."""
    scenarios = build_attribution_scenarios()
    results = {}

    print(f"\nRunning 4-case Attribution Suite (n_lat={n_lat}, nx={nx})")
    print("-" * 65)

    for name, profile in scenarios.items():
        solver = AxialSolver2D(
            n_lat=n_lat, nx=nx, dt=1e12,
            latitude_profile=profile, physics_params=dict(FIXED_PARAMS),
            use_convection=True, initial_thickness=25e3, rannacher_steps=4
        )
        result = solver.run_to_equilibrium(threshold=1e-12, max_steps=max_steps, verbose=False)
        results[name] = result

        # Compute band means
        lats = result["latitudes_deg"]
        diag = result["diagnostics"]
        
        low_idx = (lats >= LOW_LAT_BAND[0]) & (lats <= LOW_LAT_BAND[1])
        high_idx = (lats >= HIGH_LAT_BAND[0]) & (lats <= HIGH_LAT_BAND[1])
        
        def calc_band_mean(arr, idx, phi_deg):
            weights = np.cos(np.radians(phi_deg[idx]))
            if weights.sum() == 0:
                return 0.0
            return np.average(arr[idx], weights=weights)

        H = result["H_profile_km"]
        D_cond = np.array([d['D_cond_km'] for d in diag])
        D_conv = np.array([d['D_conv_km'] for d in diag])
        T_c = np.array([d['T_c'] for d in diag])
        Ti = np.array([d.get('Ti', 0.0) for d in diag])
        Ra = np.array([d['Ra'] for d in diag])

        print(f"[{name.upper()}]")
        print(f"  0-10 deg:  H={calc_band_mean(H, low_idx, lats):.1f} km, "
              f"D_cond={calc_band_mean(D_cond, low_idx, lats):.1f} km, "
              f"D_conv={calc_band_mean(D_conv, low_idx, lats):.1f} km, "
              f"T_c={calc_band_mean(T_c, low_idx, lats):.1f} K, "
              f"Ti={calc_band_mean(Ti, low_idx, lats):.1f} K, "
              f"Ra={calc_band_mean(Ra, low_idx, lats):.0f}")
        
        print(f"  80-90 deg: H={calc_band_mean(H, high_idx, lats):.1f} km, "
              f"D_cond={calc_band_mean(D_cond, high_idx, lats):.1f} km, "
              f"D_conv={calc_band_mean(D_conv, high_idx, lats):.1f} km, "
              f"T_c={calc_band_mean(T_c, high_idx, lats):.1f} K, "
              f"Ti={calc_band_mean(Ti, high_idx, lats):.1f} K, "
              f"Ra={calc_band_mean(Ra, high_idx, lats):.0f}\n")
              
        # Save output
        out_dir = os.path.join(_PROJECT_DIR, "results", "attribution")
        os.makedirs(out_dir, exist_ok=True)
        # minimal save dictionary format to bypass MonteCarloResults2D requirement
        save_dict = {
            'H_km': H, 'D_cond_km': D_cond, 'D_conv_km': D_conv,
            'T_c': T_c, 'Ti': Ti, 'Ra': Ra,
            'latitudes_deg': lats
        }
        np.savez(os.path.join(out_dir, f"attrib_{name}.npz"), **save_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2D single-variable attribution scenarios.")
    parser.add_argument("--n-lat", type=int, default=19, help="Number of latitude columns.")
    parser.add_argument("--nx", type=int, default=31, help="Radial nodes per column.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per case.")
    args = parser.parse_args()
    
    run_attribution_suite(n_lat=args.n_lat, nx=args.nx, max_steps=args.max_steps)
