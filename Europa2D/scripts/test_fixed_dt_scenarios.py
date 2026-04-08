"""
Verification: run 25 MC samples per scenario with the corrected dt=1e12.

Compares all four literature ocean scenarios plus a 1D baseline to confirm
the fix restores proper convective behavior.
"""
import sys, os

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

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
from latitude_sampler import LatitudeParameterSampler
from literature_scenarios import SCENARIOS


N_SAMPLES = 25
BASE_SEED = 42
DT = 1e12
MAX_STEPS = 1500
N_LAT = 19
NX = 31


def run_1d_baseline(sample_id):
    """1D global MC sample (reference)."""
    sampler = AuditedShellSampler(seed=BASE_SEED + sample_id)
    params = sampler.sample()
    T_surf = params['T_surf']
    D_H2O = params['D_H2O']
    H_rad = params['H_rad']
    P_tidal = params['P_tidal']

    R_rock = Planetary.RADIUS - D_H2O
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
    q_basal = (H_rad * M_rock) / Planetary.AREA + P_tidal / Planetary.AREA

    T_melt = Thermal.MELT_TEMP
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = float(np.clip((k_mean * (T_melt - T_surf)) / q_basal * 8.0, 5e3, 100e3))

    solver = Thermal_Solver(
        nx=NX, thickness=H_guess, dt=DT, total_time=5e14,
        coordinate_system='auto',
        surface_bc=FixedTemperature(temperature=T_surf),
        rannacher_steps=4, use_convection=True,
        physics_params=dict(params),
    )

    for step in range(MAX_STEPS):
        velocity = solver.solve_step(q_basal)
        if abs(velocity) < 1e-12:
            break

    H_km = solver.H / 1000.0
    if H_km <= 0.5 or H_km >= D_H2O / 1000.0 * 0.99 or H_km > 200:
        return None

    is_conv = False
    Ra, Nu = 0.0, 1.0
    if solver.convection_state is not None:
        s = solver.convection_state
        Ra, Nu = s.Ra, s.Nu
        if s.D_conv > 0 and s.Ra >= ConvConst.RA_CRIT:
            is_conv = True

    return {'H_km': H_km, 'is_convective': is_conv, 'Ra': Ra, 'Nu': Nu}


def run_2d_scenario(sample_id, scenario_name):
    """2D MC sample for a given ocean scenario."""
    sc = SCENARIOS[scenario_name]
    sampler = LatitudeParameterSampler(
        seed=BASE_SEED + sample_id,
        ocean_pattern=sc.ocean_pattern,
        q_star=sc.q_star if sc.q_star > 0 else None,
    )
    shared_params, profile = sampler.sample()
    D_H2O = shared_params['D_H2O']

    q_mean = profile.q_ocean_mean
    if q_mean > 0:
        k_mean = float(Thermal.conductivity(190.0))
        H_guess = k_mean * 170.0 / q_mean
        H_guess = float(np.clip(H_guess * 1.5, 5e3, 80e3))
    else:
        H_guess = 20e3

    solver = AxialSolver2D(
        n_lat=N_LAT, nx=NX, dt=DT,
        latitude_profile=profile,
        physics_params=shared_params,
        use_convection=True,
        initial_thickness=H_guess,
        rannacher_steps=4,
    )

    result = solver.run_to_equilibrium(
        threshold=1e-12, max_steps=MAX_STEPS, verbose=False,
    )

    H_km = result['H_profile_km']
    diag = result['diagnostics']
    lats_deg = result['latitudes_deg']

    eq_mask = lats_deg <= 10
    po_mask = lats_deg >= 80
    H_eq = np.mean(H_km[eq_mask])
    H_po = np.mean(H_km[po_mask])

    Ra_arr = np.array([d['Ra'] for d in diag])
    Nu_arr = np.array([d['Nu'] for d in diag])
    Ra_eq = np.mean(Ra_arr[eq_mask])
    Nu_eq = np.mean(Nu_arr[eq_mask])
    Ra_po = np.mean(Ra_arr[po_mask])
    Nu_po = np.mean(Nu_arr[po_mask])

    conv_eq = Ra_eq >= ConvConst.RA_CRIT
    conv_po = Ra_po >= ConvConst.RA_CRIT

    return {
        'H_eq': H_eq, 'H_po': H_po,
        'H_mean': np.mean(H_km),
        'Ra_eq': Ra_eq, 'Ra_po': Ra_po,
        'Nu_eq': Nu_eq, 'Nu_po': Nu_po,
        'conv_eq': conv_eq, 'conv_po': conv_po,
        'converged': result['converged'],
    }


def summarize_1d(name, results):
    n = len(results)
    n_conv = sum(1 for r in results if r['is_convective'])
    H_all = [r['H_km'] for r in results]
    H_conv = [r['H_km'] for r in results if r['is_convective']]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Valid: {n}/{N_SAMPLES}  Conv: {n_conv} ({100*n_conv/n:.0f}%)")
    print(f"  All H:  {np.mean(H_all):.1f} +/- {np.std(H_all):.1f} km  (median {np.median(H_all):.1f})")
    if H_conv:
        Ra_c = [r['Ra'] for r in results if r['is_convective']]
        Nu_c = [r['Nu'] for r in results if r['is_convective']]
        print(f"  Conv H: {np.mean(H_conv):.1f} +/- {np.std(H_conv):.1f} km  Ra={np.median(Ra_c):.1e}  Nu={np.median(Nu_c):.1f}")


def summarize_2d(name, results):
    n = len(results)
    n_conv_eq = sum(1 for r in results if r['conv_eq'])
    n_conv_po = sum(1 for r in results if r['conv_po'])
    H_eq = [r['H_eq'] for r in results]
    H_po = [r['H_po'] for r in results]
    H_mean = [r['H_mean'] for r in results]
    Ra_eq = [r['Ra_eq'] for r in results if r['conv_eq']]
    Ra_po = [r['Ra_po'] for r in results if r['conv_po']]
    Nu_eq = [r['Nu_eq'] for r in results if r['conv_eq']]
    Nu_po = [r['Nu_po'] for r in results if r['conv_po']]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Valid: {n}/{N_SAMPLES}")
    print(f"  Equator (0-10 deg):")
    print(f"    H: {np.mean(H_eq):.1f} +/- {np.std(H_eq):.1f} km  Conv: {n_conv_eq} ({100*n_conv_eq/n:.0f}%)")
    if Ra_eq:
        print(f"    Ra: {np.median(Ra_eq):.1e}  Nu: {np.median(Nu_eq):.1f}")
    print(f"  Pole (80-90 deg):")
    print(f"    H: {np.mean(H_po):.1f} +/- {np.std(H_po):.1f} km  Conv: {n_conv_po} ({100*n_conv_po/n:.0f}%)")
    if Ra_po:
        print(f"    Ra: {np.median(Ra_po):.1e}  Nu: {np.median(Nu_po):.1f}")
    print(f"  Global mean H: {np.mean(H_mean):.1f} +/- {np.std(H_mean):.1f} km")


def main():
    print(f"dt={DT:.0e}, max_steps={MAX_STEPS}, N={N_SAMPLES}, n_lat={N_LAT}")

    # 1D baseline
    t0 = time.time()
    results_1d = [r for i in range(N_SAMPLES) if (r := run_1d_baseline(i)) is not None]
    summarize_1d("1D Global Baseline", results_1d)
    print(f"  Time: {time.time()-t0:.0f}s")
    sys.stdout.flush()

    # 2D scenarios
    for scenario_name in SCENARIOS:
        t0 = time.time()
        results_2d = []
        for i in range(N_SAMPLES):
            r = run_2d_scenario(i, scenario_name)
            if r is not None:
                results_2d.append(r)
            if (i + 1) % 5 == 0:
                print(f"  ... {scenario_name}: {i+1}/{N_SAMPLES} done")
                sys.stdout.flush()
        sc = SCENARIOS[scenario_name]
        summarize_2d(f"2D {sc.title} ({sc.citation})", results_2d)
        print(f"  Time: {time.time()-t0:.0f}s")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
