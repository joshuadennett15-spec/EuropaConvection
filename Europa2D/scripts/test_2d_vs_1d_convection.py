"""
Diagnostic: isolate what suppresses convection in 2D vs 1D.

Tests run 50 MC samples under 5 configurations to identify the culprit:

1. 1D_baseline       — standard 1D MC (the reference that convects well)
2. 2D_uniform_full   — full 2D solver, uniform ocean, lateral coupling ON
3. 2D_uniform_nolat  — 2D solver, uniform ocean, lateral coupling OFF
4. 1D_with_2d_flags  — 1D solver but with use_composite_transition_closure=True
                        and use_onset_consistent_partition=True (the 2D defaults)
5. 1D_no_2d_flags    — 1D solver with these flags explicitly False (same as baseline)

If #3 matches #1, lateral diffusion is the culprit.
If #4 diverges from #1 and matches #2/#3, the convection flags are the culprit.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import time

from audited_sampler import AuditedShellSampler
from constants import Planetary, Thermal, Convection as ConvConst
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics
from axial_solver import AxialSolver2D
from latitude_profile import LatitudeProfile


N_SAMPLES = 50
BASE_SEED = 42


def run_1d_sample(sample_id, extra_phys=None):
    """Run a single 1D sample, optionally with extra physics flags."""
    sampler = AuditedShellSampler(seed=BASE_SEED + sample_id)
    params = sampler.sample()

    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_rad = params['H_rad']
    P_tidal = params['P_tidal']

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_radiogenic = (H_rad * M_rock) / Planetary.AREA
    q_tidal = P_tidal / Planetary.AREA
    q_basal = q_radiogenic + q_tidal

    # Build physics params dict, injecting any extra flags
    phys = dict(params)
    if extra_phys:
        phys.update(extra_phys)

    # Warm start
    T_melt = Thermal.MELT_TEMP
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = (k_mean * (T_melt - T_surf)) / q_basal
    H_guess *= 8.0  # convective multiplier
    H_guess = np.clip(H_guess, 5e3, 100e3)

    surface_bc = FixedTemperature(temperature=T_surf)
    solver = Thermal_Solver(
        nx=31,
        thickness=H_guess,
        dt=1e12,
        total_time=5e14,
        coordinate_system='auto',
        surface_bc=surface_bc,
        rannacher_steps=4,
        use_convection=True,
        physics_params=phys,
    )

    for step in range(1500):
        velocity = solver.solve_step(q_basal)
        if abs(velocity) < 1e-12:
            break

    H_km = solver.H / 1000.0
    D_H2O_km = D_H2O / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O_km * 0.99 or H_km > 200:
        return None

    is_conv = False
    Ra = 0.0
    Nu = 1.0
    D_conv = 0.0
    if solver.convection_state is not None:
        state = solver.convection_state
        Ra = state.Ra
        Nu = state.Nu
        D_conv = state.D_conv / 1000.0
        if state.D_conv > 0 and state.Ra >= ConvConst.RA_CRIT:
            is_conv = True

    return {
        'H_km': H_km, 'is_convective': is_conv, 'Ra': Ra, 'Nu': Nu,
        'D_conv_km': D_conv, 'q_basal': q_basal * 1000, 'T_surf': T_surf,
    }


def run_2d_sample(sample_id, lateral_on=True):
    """Run a single 2D uniform sample, optionally disabling lateral coupling."""
    sampler = AuditedShellSampler(seed=BASE_SEED + sample_id)
    params = sampler.sample()

    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_rad = params['H_rad']
    P_tidal = params['P_tidal']

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_radiogenic = (H_rad * M_rock) / Planetary.AREA
    q_tidal = P_tidal / Planetary.AREA
    q_basal = q_radiogenic + q_tidal

    # Use uniform profile with equatorial T everywhere (no lat variation)
    profile = LatitudeProfile(
        T_eq=T_surf,
        T_floor=T_surf - 1.0,  # must be < T_eq
        epsilon_eq=params.get('epsilon_0', 6e-6),
        epsilon_pole=params.get('epsilon_0', 6e-6),  # same everywhere
        q_ocean_mean=q_basal,
        ocean_pattern="uniform",
        surface_pattern="uniform",  # T_eq everywhere, no lat gradient
    )

    # Build physics params
    phys = {k: params[k] for k in (
        'd_grain', 'd_del', 'D0v', 'D0b',
        'mu_ice', 'D_H2O', 'Q_v', 'Q_b',
        'H_rad', 'f_porosity', 'f_salt', 'T_phi', 'B_k',
    ) if k in params}

    # Warm start
    T_melt = Thermal.MELT_TEMP
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = (k_mean * (T_melt - T_surf)) / q_basal
    H_guess *= 8.0
    H_guess = np.clip(H_guess, 5e3, 100e3)

    n_lat = 5  # small for speed; uniform so all columns identical
    solver = AxialSolver2D(
        n_lat=n_lat,
        nx=31,
        dt=5e12,
        latitude_profile=profile,
        physics_params=phys,
        use_convection=True,
        initial_thickness=H_guess,
        rannacher_steps=4,
    )

    q_ocean_profile = np.full(n_lat, q_basal)

    for step in range(500):
        if lateral_on:
            velocities = solver.solve_step(q_ocean_profile)
        else:
            # Only radial solve, skip lateral diffusion
            velocities = np.zeros(n_lat)
            for j, col in enumerate(solver.columns):
                velocities[j] = col.solve_step(q_ocean_profile[j])

        if np.max(np.abs(velocities)) < 1e-12:
            break

    # Report equatorial column (index 0)
    col = solver.columns[0]
    H_km = col.H / 1000.0
    D_H2O_km = D_H2O / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O_km * 0.99 or H_km > 200:
        return None

    is_conv = False
    Ra = 0.0
    Nu = 1.0
    D_conv = 0.0
    if col.convection_state is not None:
        state = col.convection_state
        Ra = state.Ra
        Nu = state.Nu
        D_conv = state.D_conv / 1000.0
        if state.D_conv > 0 and state.Ra >= ConvConst.RA_CRIT:
            is_conv = True

    return {
        'H_km': H_km, 'is_convective': is_conv, 'Ra': Ra, 'Nu': Nu,
        'D_conv_km': D_conv, 'q_basal': q_basal * 1000, 'T_surf': T_surf,
    }


def summarize(name, results):
    n_valid = len(results)
    if n_valid == 0:
        print(f"  {name}: 0 valid samples")
        return
    n_conv = sum(1 for r in results if r['is_convective'])
    H_all = [r['H_km'] for r in results]
    H_conv = [r['H_km'] for r in results if r['is_convective']]
    H_cond = [r['H_km'] for r in results if not r['is_convective']]
    Ra_conv = [r['Ra'] for r in results if r['is_convective']]
    Nu_conv = [r['Nu'] for r in results if r['is_convective']]

    print(f"  {name}:")
    print(f"    Valid: {n_valid}/{N_SAMPLES}  |  Conv: {n_conv} ({100*n_conv/n_valid:.0f}%)  |  Cond: {n_valid-n_conv}")
    if H_conv:
        print(f"    Conv H: {np.mean(H_conv):.1f} +/- {np.std(H_conv):.1f} km  (median {np.median(H_conv):.1f})")
        print(f"    Conv Ra: {np.median(Ra_conv):.1e}  Nu: {np.median(Nu_conv):.1f}")
    if H_cond:
        print(f"    Cond H: {np.mean(H_cond):.1f} +/- {np.std(H_cond):.1f} km  (median {np.median(H_cond):.1f})")
    print(f"    All H: {np.mean(H_all):.1f} +/- {np.std(H_all):.1f} km  (median {np.median(H_all):.1f})")


def main():
    configs = {
        "1D_baseline (no 2D flags)": lambda i: run_1d_sample(i, extra_phys=None),
        "1D + composite_transition_closure=True": lambda i: run_1d_sample(i, extra_phys={
            'use_composite_transition_closure': True,
            'use_onset_consistent_partition': False,
        }),
        "1D + onset_consistent_partition=True": lambda i: run_1d_sample(i, extra_phys={
            'use_composite_transition_closure': False,
            'use_onset_consistent_partition': True,
        }),
        "1D + BOTH 2D flags": lambda i: run_1d_sample(i, extra_phys={
            'use_composite_transition_closure': True,
            'use_onset_consistent_partition': True,
        }),
        "2D uniform (lateral ON)": lambda i: run_2d_sample(i, lateral_on=True),
        "2D uniform (lateral OFF)": lambda i: run_2d_sample(i, lateral_on=False),
    }

    for name, runner in configs.items():
        t0 = time.time()
        results = []
        for i in range(N_SAMPLES):
            r = runner(i)
            if r is not None:
                results.append(r)
        elapsed = time.time() - t0
        summarize(name, results)
        print(f"    Time: {elapsed:.1f}s")
        print()


if __name__ == "__main__":
    main()
