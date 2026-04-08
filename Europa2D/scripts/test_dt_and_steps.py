"""
Diagnostic round 2: isolate dt and max_steps effects.

The first round showed that lateral diffusion and convection flags
don't matter. The remaining structural differences between 1D and 2D are:
  - dt: 1D uses 1e12, 2D uses 5e12
  - max_steps: 1D uses 1500, 2D uses 500

This test runs:
  1. 1D with dt=1e12, max_steps=1500  (baseline, expect 64% conv)
  2. 1D with dt=5e12, max_steps=500   (match 2D time-stepping)
  3. 1D with dt=5e12, max_steps=1500  (large dt, many steps)
  4. 1D with dt=1e12, max_steps=500   (small dt, fewer steps)
  5. 2D uniform (lateral OFF) with dt=1e12, max_steps=1500  (match 1D stepping)
  6. 2D uniform (lateral OFF) with dt=5e12, max_steps=500   (original 2D)
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


def get_params_and_qbasal(sample_id):
    """Sample params and compute q_basal."""
    sampler = AuditedShellSampler(seed=BASE_SEED + sample_id)
    params = sampler.sample()
    D_H2O = params['D_H2O']
    H_rad = params['H_rad']
    P_tidal = params['P_tidal']
    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_rad = (H_rad * M_rock) / Planetary.AREA
    q_tidal = P_tidal / Planetary.AREA
    q_basal = q_rad + q_tidal
    return params, q_basal


def warm_start(T_surf, q_basal):
    T_melt = Thermal.MELT_TEMP
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = (k_mean * (T_melt - T_surf)) / q_basal * 8.0
    return float(np.clip(H_guess, 5e3, 100e3))


def run_1d(sample_id, dt, max_steps):
    params, q_basal = get_params_and_qbasal(sample_id)
    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_guess = warm_start(T_surf, q_basal)

    solver = Thermal_Solver(
        nx=31, thickness=H_guess, dt=dt, total_time=5e14,
        coordinate_system='auto',
        surface_bc=FixedTemperature(temperature=T_surf),
        rannacher_steps=4, use_convection=True,
        physics_params=dict(params),
    )

    for step in range(max_steps):
        velocity = solver.solve_step(q_basal)
        if abs(velocity) < 1e-12:
            break

    H_km = solver.H / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O / 1000.0 * 0.99 or H_km > 200:
        return None

    is_conv = False
    Ra, Nu, D_conv = 0.0, 1.0, 0.0
    if solver.convection_state is not None:
        s = solver.convection_state
        Ra, Nu, D_conv = s.Ra, s.Nu, s.D_conv / 1000.0
        if s.D_conv > 0 and s.Ra >= ConvConst.RA_CRIT:
            is_conv = True

    return {'H_km': H_km, 'is_convective': is_conv, 'Ra': Ra, 'Nu': Nu,
            'D_conv_km': D_conv, 'steps': step+1}


def run_2d_column(sample_id, dt, max_steps):
    """Run 2D solver (lateral OFF) — just independent columns with 2D flags."""
    params, q_basal = get_params_and_qbasal(sample_id)
    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_guess = warm_start(T_surf, q_basal)

    profile = LatitudeProfile(
        T_eq=T_surf, T_floor=T_surf - 1.0,
        epsilon_eq=params.get('epsilon_0', 6e-6),
        epsilon_pole=params.get('epsilon_0', 6e-6),
        q_ocean_mean=q_basal, ocean_pattern="uniform",
        surface_pattern="uniform",
    )

    phys = {k: params[k] for k in (
        'd_grain', 'd_del', 'D0v', 'D0b', 'mu_ice', 'D_H2O', 'Q_v', 'Q_b',
        'H_rad', 'f_porosity', 'f_salt', 'T_phi', 'B_k',
    ) if k in params}

    n_lat = 3
    solver = AxialSolver2D(
        n_lat=n_lat, nx=31, dt=dt,
        total_time=5e14,
        latitude_profile=profile,
        physics_params=phys,
        use_convection=True,
        initial_thickness=H_guess,
        rannacher_steps=4,
    )

    q_profile = np.full(n_lat, q_basal)

    for step in range(max_steps):
        velocities = np.zeros(n_lat)
        for j, col in enumerate(solver.columns):
            velocities[j] = col.solve_step(q_profile[j])
        if np.max(np.abs(velocities)) < 1e-12:
            break

    col = solver.columns[0]
    H_km = col.H / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O / 1000.0 * 0.99 or H_km > 200:
        return None

    is_conv = False
    Ra, Nu, D_conv = 0.0, 1.0, 0.0
    if col.convection_state is not None:
        s = col.convection_state
        Ra, Nu, D_conv = s.Ra, s.Nu, s.D_conv / 1000.0
        if s.D_conv > 0 and s.Ra >= ConvConst.RA_CRIT:
            is_conv = True

    return {'H_km': H_km, 'is_convective': is_conv, 'Ra': Ra, 'Nu': Nu,
            'D_conv_km': D_conv, 'steps': step+1}


def summarize(name, results):
    n_valid = len(results)
    if n_valid == 0:
        print(f"  {name}: 0 valid")
        return
    n_conv = sum(1 for r in results if r['is_convective'])
    H_conv = [r['H_km'] for r in results if r['is_convective']]
    H_cond = [r['H_km'] for r in results if not r['is_convective']]
    H_all = [r['H_km'] for r in results]
    steps = [r['steps'] for r in results]

    print(f"  {name}:")
    print(f"    Valid: {n_valid}/{N_SAMPLES}  |  Conv: {n_conv} ({100*n_conv/n_valid:.0f}%)")
    if H_conv:
        Ra_conv = [r['Ra'] for r in results if r['is_convective']]
        Nu_conv = [r['Nu'] for r in results if r['is_convective']]
        print(f"    Conv H: {np.mean(H_conv):.1f} +/- {np.std(H_conv):.1f} km  Ra={np.median(Ra_conv):.1e}  Nu={np.median(Nu_conv):.1f}")
    if H_cond:
        print(f"    Cond H: {np.mean(H_cond):.1f} +/- {np.std(H_cond):.1f} km")
    print(f"    All H: {np.mean(H_all):.1f} +/- {np.std(H_all):.1f} km  median={np.median(H_all):.1f}")
    print(f"    Steps: mean={np.mean(steps):.0f}  max={np.max(steps)}")


def main():
    configs = [
        ("1D  dt=1e12  steps=1500", lambda i: run_1d(i, dt=1e12, max_steps=1500)),
        ("1D  dt=5e12  steps=500",  lambda i: run_1d(i, dt=5e12, max_steps=500)),
        ("1D  dt=5e12  steps=1500", lambda i: run_1d(i, dt=5e12, max_steps=1500)),
        ("1D  dt=1e12  steps=500",  lambda i: run_1d(i, dt=1e12, max_steps=500)),
        ("2D  dt=1e12  steps=1500", lambda i: run_2d_column(i, dt=1e12, max_steps=1500)),
        ("2D  dt=5e12  steps=500",  lambda i: run_2d_column(i, dt=5e12, max_steps=500)),
    ]

    for name, runner in configs:
        t0 = time.time()
        results = [r for i in range(N_SAMPLES) if (r := runner(i)) is not None]
        elapsed = time.time() - t0
        summarize(name, results)
        print(f"    Time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
