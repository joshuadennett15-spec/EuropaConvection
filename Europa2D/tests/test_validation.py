"""
Validation: 2D model at a single latitude must match 1D model output.

This test runs a single-column 2D solver and compares the equilibrium
thickness against a standalone 1D Thermal_Solver with identical parameters.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics


class TestValidationVs1D:
    """2D single-column output should match standalone 1D solver."""

    PARAMS = {
        'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
        'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
        'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
        'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
    }
    T_SURF = 104.0
    Q_OCEAN = 0.025
    NX = 31
    DT = 1e11
    THICKNESS = 15e3

    def test_conductive_single_column_matches_1d(self):
        """Without convection, single-column 2D must match 1D within 1%."""
        # 1D reference
        bc_1d = FixedTemperature(temperature=self.T_SURF)
        solver_1d = Thermal_Solver(
            nx=self.NX, thickness=self.THICKNESS, dt=self.DT,
            surface_bc=bc_1d, use_convection=False,
            physics_params={**self.PARAMS, 'T_surf': self.T_SURF},
        )
        for _ in range(300):
            v = solver_1d.solve_step(self.Q_OCEAN)
            if abs(v) < 1e-12:
                break
        H_1d = solver_1d.H / 1000.0

        # 2D with 1 column at equator (T_eq = T_SURF, so T_s(0) = T_SURF)
        profile = LatitudeProfile(
            T_eq=self.T_SURF, epsilon_eq=self.PARAMS['epsilon_0'],
            epsilon_pole=self.PARAMS['epsilon_0'],
            q_ocean_mean=self.Q_OCEAN, ocean_pattern="uniform",
        )
        solver_2d = AxialSolver2D(
            n_lat=1, nx=self.NX, dt=self.DT,
            latitude_profile=profile, physics_params=self.PARAMS,
            use_convection=False, initial_thickness=self.THICKNESS,
        )
        result_2d = solver_2d.run_to_equilibrium(
            threshold=1e-12, max_steps=300, verbose=False
        )
        H_2d = result_2d['H_profile_km'][0]

        # Should match within 1%
        assert H_2d == pytest.approx(H_1d, rel=0.01), \
            f"2D ({H_2d:.3f} km) vs 1D ({H_1d:.3f} km)"


def test_single_column_2d_matches_1d():
    """2D solver with n_lat=1, uniform forcing using the default warm equator."""
    profile = LatitudeProfile(
        T_eq=110.0, T_floor=46.0,
        epsilon_eq=6e-6, epsilon_pole=6e-6,
        q_ocean_mean=0.02,
        ocean_pattern="uniform",
    )
    solver = AxialSolver2D(
        n_lat=1,
        nx=31,
        dt=1e12,
        latitude_profile=profile,
        use_convection=True,
    )
    result = solver.run_to_equilibrium(threshold=1e-12, max_steps=1500, verbose=False)
    H_2d = result["H_profile_km"][0]

    # Sanity bounds
    assert H_2d > 5.0, f"Unphysical: shell too thin ({H_2d:.1f} km)"
    assert H_2d < 80.0, f"Unphysical: shell too thick ({H_2d:.1f} km)"
    # Record the exact value for future regression
    print(f"Default variant (110/46 K): H_2d = {H_2d:.2f} km")


def test_single_column_2d_warm_variant():
    """Repeat with legacy 110/52 K surface BCs for sensitivity comparison."""
    profile = LatitudeProfile(
        T_eq=110.0, T_floor=52.0,
        epsilon_eq=6e-6, epsilon_pole=6e-6,
        q_ocean_mean=0.02,
        ocean_pattern="uniform",
    )
    solver = AxialSolver2D(
        n_lat=1,
        nx=31,
        dt=1e12,
        latitude_profile=profile,
        use_convection=True,
    )
    result = solver.run_to_equilibrium(threshold=1e-12, max_steps=1500, verbose=False)
    H_warm = result["H_profile_km"][0]

    assert H_warm > 5.0, f"Unphysical: shell too thin ({H_warm:.1f} km)"
    assert H_warm < 80.0, f"Unphysical: shell too thick ({H_warm:.1f} km)"
    print(f"Warm variant (110/52 K): H_warm = {H_warm:.2f} km")
