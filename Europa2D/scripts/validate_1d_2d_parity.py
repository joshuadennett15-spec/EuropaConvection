"""
Step 4 parity validation: 2D single-column must reproduce 1D solver.

Strict single-column check using actual draws from the equatorial-baseline
MC archive. For each draw the script:
  - Reconstructs q_basal from sampled H_rad, P_tidal, D_H2O (Howell 2021)
  - Applies the production warm-start (conductive guess * 8, clipped)
  - Runs both the 1D Thermal_Solver and a 2D AxialSolver2D(n_lat=1, uniform)
  - Checks convergence before comparing QoIs
  - Compares H, D_cond, D_conv, Ra, Nu

Usage:
    python Europa2D/scripts/validate_1d_2d_parity.py
    python Europa2D/scripts/validate_1d_2d_parity.py --archive path/to/archive.npz --n-batch 50
"""
import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

import numpy as np

from axial_solver import AxialSolver2D
from latitude_profile import LatitudeProfile
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from constants import Planetary, Thermal, Convection as ConvConst

# ── Solver settings (match production MC, but with extra headroom) ───────────
NX = 31
DT = 1e12
MAX_STEPS = 5000  # production uses 1500, but warm-start clamp at 100 km needs ~3000
THRESHOLD = 1e-12

# QoI tolerances (relative)
TOLERANCES = {
    "H_km": 0.02,        # 2% on total thickness
    "D_cond_km": 0.02,   # 2% on conductive lid
    "D_conv_km": 0.05,   # 5% on convective layer (smaller, noisier)
    "Ra": 0.02,          # 2% on Rayleigh number
    "Nu": 0.02,          # 2% on Nusselt number
}

# Keys that the 2D physics_params dict needs (everything except T_surf,
# epsilon_0 which go through LatitudeProfile, and D_H2O / H_rad / P_tidal
# which are only used for q_basal reconstruction).
_PHYSICS_KEYS = (
    "d_grain", "d_del", "D0v", "D0b", "mu_ice", "Q_v", "Q_b",
    "f_porosity", "f_salt", "T_phi", "B_k", "D_H2O",
)


def _reconstruct_q_basal(params: dict) -> float:
    """Reconstruct basal heat flux from sampled parameters (Howell 2021)."""
    D_H2O = params["D_H2O"]
    H_rad = params["H_rad"]
    P_tidal = params["P_tidal"]

    R_rock = Planetary.RADIUS - D_H2O
    rho_rock = 3500.0
    M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * rho_rock

    q_radiogenic = (H_rad * M_rock) / Planetary.AREA
    q_silicate_tidal = P_tidal / Planetary.AREA
    return q_radiogenic + q_silicate_tidal


def _warm_start_thickness(q_basal: float, T_surf: float) -> float:
    """Production warm-start: conductive guess * 8, clipped to [5, 100] km."""
    T_melt = Thermal.MELT_TEMP
    k_mean = Thermal.conductivity((T_surf + T_melt) / 2)
    H_guess = (k_mean * (T_melt - T_surf)) / q_basal
    H_guess *= 8.0  # empirical convective multiplier
    return float(np.clip(H_guess, 5e3, 100e3))


def _extract_qois(solver) -> dict:
    """Extract QoIs from a converged 1D Thermal_Solver."""
    H_km = solver.H / 1000.0
    if solver.convection_state is not None:
        diag = solver.get_convection_diagnostics()
        return {
            "H_km": H_km,
            "D_cond_km": diag["D_cond_km"],
            "D_conv_km": diag["D_conv_km"],
            "Ra": diag["Ra"],
            "Nu": diag["Nu"],
        }
    return {
        "H_km": H_km,
        "D_cond_km": H_km,
        "D_conv_km": 0.0,
        "Ra": 0.0,
        "Nu": 1.0,
    }


def _load_archive_draw(archive, idx: int) -> dict:
    """Load a single parameter draw from an archive by index."""
    params = {}
    for k in sorted(archive.files):
        if k.startswith("param_"):
            name = k[len("param_"):]
            params[name] = float(archive[k][idx])
    return params


def _run_1d(params: dict, q_basal: float, H_init: float) -> tuple:
    """Run 1D solver. Returns (qois_dict, converged_bool)."""
    T_surf = params["T_surf"]
    bc = FixedTemperature(temperature=T_surf)

    physics = dict(params)
    physics["use_composite_transition_closure"] = True
    physics["use_onset_consistent_partition"] = True

    solver = Thermal_Solver(
        nx=NX,
        thickness=H_init,
        dt=DT,
        surface_bc=bc,
        use_convection=True,
        physics_params=physics,
    )

    converged = False
    for step in range(MAX_STEPS):
        v = solver.solve_step(q_basal)
        if abs(v) < THRESHOLD:
            converged = True
            break

    return _extract_qois(solver), converged


def _run_2d_single_column(params: dict, q_basal: float, H_init: float) -> tuple:
    """Run 2D solver with n_lat=1, uniform. Returns (qois_dict, converged_bool)."""
    T_surf = params["T_surf"]
    epsilon_0 = params.get("epsilon_0", 1e-5)

    profile = LatitudeProfile(
        T_eq=T_surf,
        epsilon_eq=epsilon_0,
        epsilon_pole=epsilon_0,
        q_ocean_mean=q_basal,
        ocean_pattern="uniform",
        surface_pattern="uniform",
    )

    physics = {k: params[k] for k in _PHYSICS_KEYS if k in params}

    solver = AxialSolver2D(
        n_lat=1,
        nx=NX,
        dt=DT,
        latitude_profile=profile,
        physics_params=physics,
        use_convection=True,
        initial_thickness=H_init,
        rannacher_steps=4,
    )

    result = solver.run_to_equilibrium(
        threshold=THRESHOLD,
        max_steps=MAX_STEPS,
        verbose=False,
    )

    H_km = result["H_profile_km"][0]
    diag = result["diagnostics"][0]
    converged = result["converged"]

    qois = {
        "H_km": H_km,
        "D_cond_km": diag.get("D_cond_km", H_km),
        "D_conv_km": diag.get("D_conv_km", 0.0),
        "Ra": diag.get("Ra", 0.0),
        "Nu": diag.get("Nu", 1.0),
    }
    return qois, converged


def _compare(qoi_1d: dict, qoi_2d: dict) -> list:
    """Compare QoIs. Returns list of (name, val_1d, val_2d, rel_err, tol, pass)."""
    rows = []
    for name, tol in TOLERANCES.items():
        v1 = qoi_1d[name]
        v2 = qoi_2d[name]
        if abs(v1) < 1e-12:
            rel_err = abs(v2 - v1)
            passed = rel_err < tol
        else:
            rel_err = abs(v2 - v1) / abs(v1)
            passed = rel_err < tol
        rows.append((name, v1, v2, rel_err, tol, passed))
    return rows


def _print_comparison(rows: list, label: str) -> bool:
    """Pretty-print comparison table. Returns True if all pass."""
    all_pass = all(r[5] for r in rows)
    status = "PASS" if all_pass else "FAIL"
    print(f"\n{'=' * 72}")
    print(f"  {label}  [{status}]")
    print(f"{'=' * 72}")
    print(f"  {'QoI':<12s}  {'1D':>12s}  {'2D':>12s}  {'rel_err':>10s}  {'tol':>8s}  {'ok':>4s}")
    print(f"  {'-' * 60}")
    for name, v1, v2, rel_err, tol, passed in rows:
        mark = "OK" if passed else "FAIL"
        print(f"  {name:<12s}  {v1:12.4f}  {v2:12.4f}  {rel_err:10.6f}  {tol:8.4f}  {mark:>4s}")
    return all_pass


def run_batch_check(archive_path: str, n_batch: int, seed: int) -> dict:
    """Batch parity check using actual archive draws with per-sample q_basal."""
    archive = np.load(archive_path)
    n_total = int(archive["n_valid"])
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_batch, n_total), replace=False)

    n_pass = 0
    n_skip_convergence = 0
    max_errs = {name: 0.0 for name in TOLERANCES}
    sample_details = []

    print(f"\n--- Batch parity: {len(indices)} actual archive draws ---")
    t0 = time.time()

    for i, idx in enumerate(indices):
        params = _load_archive_draw(archive, idx)

        q_basal = _reconstruct_q_basal(params)
        H_init = _warm_start_thickness(q_basal, params["T_surf"])

        qoi_1d, conv_1d = _run_1d(params, q_basal, H_init)
        qoi_2d, conv_2d = _run_2d_single_column(params, q_basal, H_init)

        # Skip comparison if either solver didn't converge
        if not (conv_1d and conv_2d):
            n_skip_convergence += 1
            sample_details.append({
                "index": int(idx),
                "skipped": True,
                "reason": f"convergence: 1D={conv_1d}, 2D={conv_2d}",
                "q_basal_mW": q_basal * 1000,
                "H_init_km": H_init / 1000,
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(indices):
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(indices)}] {n_pass} pass, "
                      f"{n_skip_convergence} skipped, {elapsed:.1f}s")
            continue

        rows = _compare(qoi_1d, qoi_2d)
        sample_pass = all(r[5] for r in rows)
        if sample_pass:
            n_pass += 1

        for name, _, _, rel_err, _, _ in rows:
            max_errs[name] = max(max_errs[name], rel_err)

        sample_details.append({
            "index": int(idx),
            "skipped": False,
            "passed": sample_pass,
            "q_basal_mW": q_basal * 1000,
            "H_init_km": H_init / 1000,
            "qoi_1d": {k: float(v) for k, v in qoi_1d.items()},
            "qoi_2d": {k: float(v) for k, v in qoi_2d.items()},
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(indices):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{len(indices)}] {n_pass} pass, "
                  f"{n_skip_convergence} skipped, {elapsed:.1f}s "
                  f"({rate:.1f} samples/s)")

    n_compared = len(indices) - n_skip_convergence
    all_pass = (n_pass == n_compared) and n_compared > 0
    status = "PASS" if all_pass else "FAIL"

    print(f"\n{'=' * 72}")
    print(f"  Batch parity: {n_pass}/{n_compared} compared passed, "
          f"{n_skip_convergence} skipped (non-converged)  [{status}]")
    print(f"{'=' * 72}")

    if n_compared > 0:
        print(f"  {'QoI':<12s}  {'max_rel_err':>12s}  {'tol':>8s}  {'ok':>4s}")
        print(f"  {'-' * 40}")
        for name, tol in TOLERANCES.items():
            mark = "OK" if max_errs[name] < tol else "FAIL"
            print(f"  {name:<12s}  {max_errs[name]:12.6f}  {tol:8.4f}  {mark:>4s}")

    # Print one example for visual inspection
    for detail in sample_details:
        if not detail.get("skipped", True):
            print(f"\n  Example (archive idx {detail['index']}, "
                  f"q_basal={detail['q_basal_mW']:.2f} mW/m²):")
            for qoi_name in TOLERANCES:
                v1 = detail["qoi_1d"][qoi_name]
                v2 = detail["qoi_2d"][qoi_name]
                print(f"    {qoi_name:<12s}  1D={v1:10.4f}  2D={v2:10.4f}")
            break

    return {
        "test": "batch_parity",
        "n_samples": len(indices),
        "n_compared": n_compared,
        "n_pass": n_pass,
        "n_skip_convergence": n_skip_convergence,
        "passed": all_pass,
        "max_relative_errors": max_errs,
        "seed": seed,
        "samples": sample_details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: 1D-2D parity validation using production physics."
    )
    parser.add_argument(
        "--archive",
        default=os.path.join(
            _SCRIPT_DIR, "..", "..", "EuropaProjectDJ", "results",
            "eq_baseline_andrade.npz"
        ),
        help="Path to equatorial baseline MC archive (.npz)",
    )
    parser.add_argument(
        "--n-batch", type=int, default=30,
        help="Number of random archive draws for batch check",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--output", default=None,
        help="JSON output path (default: Europa2D/results/parity_validation.json)",
    )
    args = parser.parse_args()

    archive_path = os.path.abspath(args.archive)
    if not os.path.exists(archive_path):
        print(f"ERROR: Archive not found: {archive_path}")
        sys.exit(1)

    output_path = args.output or os.path.join(
        _SCRIPT_DIR, "..", "results", "parity_validation.json"
    )

    print(f"Archive: {archive_path}")
    print(f"Physics config (from config.json):")
    print(f"  NU_SCALING         = {ConvConst.NU_SCALING}")

    from ConfigManager import ConfigManager
    print(f"  CONDUCTIVITY_MODEL = {ConfigManager.get('thermal', 'CONDUCTIVITY_MODEL', 'Carnahan')}")
    print(f"  CREEP_MODEL        = {ConfigManager.get('rheology', 'CREEP_MODEL', 'diffusion')}")

    report = {"archive": archive_path, "results": []}

    # Batch check with actual archive draws
    batch_result = run_batch_check(archive_path, args.n_batch, args.seed)
    report["results"].append(batch_result)

    # Overall verdict
    all_pass = all(r["passed"] for r in report["results"])
    report["verdict"] = "PASS" if all_pass else "FAIL"

    print(f"\n{'#' * 72}")
    print(f"  OVERALL VERDICT: {report['verdict']}")
    print(f"{'#' * 72}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(report, default=_convert))
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults written to: {output_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
