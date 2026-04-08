"""
Quick test: run 50 1D MC iterations at equatorial T_surf with q_basal
scaled to 0.85x (the equatorial fraction from polar-enhanced scenario).

Purpose: verify whether 0.85 * q_basal still produces convective interiors,
i.e. the ~15% reduction alone is NOT what suppresses convection in 2D runs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
from Monte_Carlo import SolverConfig
from audited_sampler import AuditedShellSampler
from constants import Planetary
import multiprocessing as mp
from functools import partial
import time


# Scale factor to apply to q_basal (0.85 = equatorial value in polar-enhanced)
Q_SCALE_FACTORS = [1.0, 0.85, 0.73, 0.50]


def run_scaled_sample(sample_id, base_seed, q_scale, sampler_class, config):
    """Run one 1D sample with q_basal multiplied by q_scale."""
    sampler = sampler_class(seed=base_seed + sample_id)
    params = sampler.sample()

    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_rad = params['H_rad']
    P_tidal = params['P_tidal']

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    A_surface = Planetary.AREA

    q_radiogenic = (H_rad * M_rock) / A_surface
    q_silicate_tidal = P_tidal / A_surface
    q_basal_full = q_radiogenic + q_silicate_tidal

    # Apply the scale factor
    q_basal = q_basal_full * q_scale

    from Solver import Thermal_Solver
    from Boundary_Conditions import FixedTemperature
    from Physics import IcePhysics
    from constants import Thermal, Convection as ConvConst

    # Warm start
    if config.use_warm_start and q_basal > 0:
        T_melt = Thermal.MELT_TEMP
        delta_T = T_melt - T_surf
        k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
        H_guess = (k_mean * delta_T) / q_basal
        if config.use_convection:
            H_guess *= 8.0
        H_guess = np.clip(H_guess, 5e3, 100e3)
    else:
        H_guess = config.initial_thickness

    surface_bc = FixedTemperature(temperature=T_surf)
    solver = Thermal_Solver(
        nx=config.nx,
        thickness=H_guess,
        dt=config.dt,
        total_time=config.total_time,
        coordinate_system=config.coordinate_system,
        surface_bc=surface_bc,
        rannacher_steps=config.rannacher_steps,
        use_convection=config.use_convection,
        physics_params=params,
    )

    for step in range(config.max_steps):
        velocity = solver.solve_step(q_basal)
        if abs(velocity) < config.eq_threshold:
            break

    H_km = solver.H / 1000.0
    D_H2O_km = D_H2O / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O_km * 0.99 or H_km > 200:
        return None

    is_convective = False
    Ra = 0.0
    Nu = 1.0
    D_conv = 0.0
    D_cond = H_km
    if solver.convection_state is not None:
        state = solver.convection_state
        Ra = state.Ra
        Nu = state.Nu
        D_conv = state.D_conv / 1000.0
        D_cond = state.D_cond / 1000.0
        if state.D_conv > 0 and state.Ra >= ConvConst.RA_CRIT:
            is_convective = True

    return {
        'H_km': H_km,
        'q_basal_full': q_basal_full * 1000,  # mW/m2
        'q_basal_used': q_basal * 1000,
        'is_convective': is_convective,
        'Ra': Ra,
        'Nu': Nu,
        'D_conv_km': D_conv,
        'D_cond_km': D_cond,
        'T_surf': T_surf,
    }


def main():
    N = 50
    seed = 42
    config = SolverConfig()

    print(f"Running {N} 1D samples at each q_basal scale factor...")
    print(f"Scale factors: {Q_SCALE_FACTORS}")
    print()

    for scale in Q_SCALE_FACTORS:
        t0 = time.time()
        results = []
        for i in range(N):
            r = run_scaled_sample(i, seed, scale, AuditedShellSampler, config)
            if r is not None:
                results.append(r)

        elapsed = time.time() - t0
        n_valid = len(results)
        n_conv = sum(1 for r in results if r['is_convective'])
        n_cond = n_valid - n_conv

        thicknesses = [r['H_km'] for r in results]
        conv_thicknesses = [r['H_km'] for r in results if r['is_convective']]
        cond_thicknesses = [r['H_km'] for r in results if not r['is_convective']]
        q_used = [r['q_basal_used'] for r in results]

        print(f"=== q_basal x {scale:.2f} ===")
        print(f"  Valid: {n_valid}/{N},  Convective: {n_conv} ({100*n_conv/max(n_valid,1):.0f}%),  Conductive: {n_cond}")
        print(f"  q_basal used: {np.mean(q_used):.1f} +/- {np.std(q_used):.1f} mW/m2")
        if conv_thicknesses:
            print(f"  Convective H: {np.mean(conv_thicknesses):.1f} +/- {np.std(conv_thicknesses):.1f} km  (median {np.median(conv_thicknesses):.1f})")
            conv_Ra = [r['Ra'] for r in results if r['is_convective']]
            conv_Nu = [r['Nu'] for r in results if r['is_convective']]
            print(f"  Convective Ra: {np.median(conv_Ra):.1e},  Nu: {np.median(conv_Nu):.1f}")
        if cond_thicknesses:
            print(f"  Conductive H: {np.mean(cond_thicknesses):.1f} +/- {np.std(cond_thicknesses):.1f} km  (median {np.median(cond_thicknesses):.1f})")
        print(f"  All H: {np.mean(thicknesses):.1f} +/- {np.std(thicknesses):.1f} km  (median {np.median(thicknesses):.1f})")
        print(f"  Time: {elapsed:.1f}s")
        print()


if __name__ == "__main__":
    main()
