"""
Paired 4-case attribution ensemble for the 2D Europa shell model.

Each realization samples one shared shell state, then runs:
    baseline, surface_only, ocean_only, strain_only
with that exact same draw so case-to-case differences are attributable to the
forcing toggle rather than Monte Carlo noise.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from collections import Counter
from typing import Any, Final

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")

# Match the main 2D MC runner: keep one BLAS thread per worker process so
# 12-process Windows runs do not degenerate into N x M thread oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src  # triggers import path setup

import numpy as np

from attribution_cases import ATTRIBUTION_CASES, build_paired_attribution_profiles
from axial_solver import AxialSolver2D
from constants import Thermal, Convection as ConvectionConstants
from latitude_sampler import LatitudeParameterSampler
from literature_scenarios import get_scenario, list_scenarios
from profile_diagnostics import HIGH_LAT_BAND, LOW_LAT_BAND, band_mean_samples


PROFILE_METRICS: Final[tuple[str, ...]] = (
    "H_km",
    "D_cond_km",
    "D_conv_km",
    "Ra",
    "Nu",
    "T_c",
    "Ti",
)

NU_ACTIVE_THRESHOLD: Final[float] = 1.01

SAMPLED_SHARED_KEYS: Final[tuple[str, ...]] = (
    "d_grain",
    "d_del",
    "D0v",
    "D0b",
    "mu_ice",
    "D_H2O",
    "Q_v",
    "Q_b",
    "H_rad",
    "f_porosity",
    "f_salt",
    "T_phi",
    "B_k",
    "q_basal",
    "q_tidal",
    "q_basal_inherited",
    "q_tidal_inherited",
    "q_tidal_scale",
)

SAMPLED_PROFILE_KEYS: Final[tuple[str, ...]] = (
    "T_eq",
    "T_floor",
    "epsilon_eq",
    "epsilon_pole",
    "q_ocean_mean",
    "mantle_tidal_fraction",
    "resolved_q_star",
)


def _initial_thickness_guess(q_mean: float, use_convection: bool, initial_thickness: float) -> float:
    """Warm-start thickness matching the 2D MC runner heuristics."""
    if q_mean <= 0.0:
        return initial_thickness
    k_mean = float(Thermal.conductivity(190.0))
    H_guess = k_mean * 170.0 / q_mean
    if use_convection:
        H_guess *= 1.5
    return float(np.clip(H_guess, 5e3, 80e3))


def _interpolate_invalid_columns(
    metric_arrays: dict[str, np.ndarray],
    latitudes_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate invalid latitude columns for every extracted diagnostic profile."""
    if np.all(valid_mask):
        return metric_arrays

    fixed: dict[str, np.ndarray] = {}
    for name, values in metric_arrays.items():
        fixed[name] = np.interp(latitudes_deg, latitudes_deg[valid_mask], values[valid_mask])
    return fixed


def _extract_metric_arrays(result: dict[str, Any], latitudes_deg: np.ndarray) -> dict[str, np.ndarray]:
    """Extract the profile metrics used by the paired ensemble."""
    diag = result["diagnostics"]
    arrays = {
        "H_km": np.asarray(result["H_profile_km"], dtype=float),
        "D_cond_km": np.array([d["D_cond_km"] for d in diag], dtype=float),
        "D_conv_km": np.array([d["D_conv_km"] for d in diag], dtype=float),
        "Ra": np.array([d["Ra"] for d in diag], dtype=float),
        "Nu": np.array([d["Nu"] for d in diag], dtype=float),
        "T_c": np.array([d.get("T_c", 0.0) for d in diag], dtype=float),
        "Ti": np.array([d.get("Ti", 0.0) for d in diag], dtype=float),
    }

    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Non-finite {name} profile in attribution sample.")

    valid_mask = (
        (arrays["H_km"] > 0.5)
        & (arrays["H_km"] < 200.0)
        & np.isfinite(arrays["H_km"])
    )
    if np.sum(valid_mask) < len(latitudes_deg) * 0.5:
        raise RuntimeError("Too many invalid columns in attribution sample.")

    return _interpolate_invalid_columns(arrays, latitudes_deg, valid_mask)


def _run_paired_sample(
    sample_id: int,
    base_seed: int,
    scenario_name: str,
    n_lat: int,
    nx: int,
    dt: float,
    use_convection: bool,
    max_steps: int,
    eq_threshold: float,
    initial_thickness: float,
    rannacher_steps: int,
    coordinate_system: str,
    grain_latitude_mode: str,
    q_tidal_scale: float,
    T_floor: float,
) -> dict[str, Any]:
    """Run one paired attribution realization."""
    try:
        scenario = get_scenario(scenario_name)
        sampler = LatitudeParameterSampler(
            seed=base_seed + sample_id,
            ocean_pattern=scenario.ocean_pattern,
            q_star=scenario.q_star if scenario.q_star > 0 else None,
            grain_latitude_mode=grain_latitude_mode,
            q_tidal_scale=q_tidal_scale,
            T_floor_mean=T_floor,
        )
        shared_params, sampled_profile = sampler.sample()
        cases = build_paired_attribution_profiles(sampled_profile)
        H_guess = _initial_thickness_guess(
            q_mean=sampled_profile.q_ocean_mean,
            use_convection=use_convection,
            initial_thickness=initial_thickness,
        )

        latitudes_deg: np.ndarray | None = None
        case_arrays: dict[str, dict[str, np.ndarray]] = {}
        case_converged: dict[str, bool] = {}
        case_steps: dict[str, int] = {}
        for case_name in ATTRIBUTION_CASES:
            solver = AxialSolver2D(
                n_lat=n_lat,
                nx=nx,
                dt=dt,
                latitude_profile=cases[case_name],
                physics_params=dict(shared_params),
                use_convection=use_convection,
                initial_thickness=H_guess,
                rannacher_steps=rannacher_steps,
                coordinate_system=coordinate_system,
            )
            result = solver.run_to_equilibrium(
                threshold=eq_threshold,
                max_steps=max_steps,
                verbose=False,
            )

            latitudes_deg = np.asarray(result["latitudes_deg"], dtype=float)
            case_arrays[case_name] = _extract_metric_arrays(result, latitudes_deg)
            case_converged[case_name] = bool(result["converged"])
            case_steps[case_name] = int(result["steps"])

        sample_meta = {
            **{key: float(shared_params[key]) for key in SAMPLED_SHARED_KEYS},
            "T_eq": float(sampled_profile.T_eq),
            "T_floor": float(sampled_profile.T_floor),
            "epsilon_eq": float(sampled_profile.epsilon_eq),
            "epsilon_pole": float(sampled_profile.epsilon_pole),
            "q_ocean_mean": float(sampled_profile.q_ocean_mean),
            "mantle_tidal_fraction": float(sampled_profile.mantle_tidal_fraction),
            "resolved_q_star": float(sampled_profile.resolved_q_star()),
        }

        return {
            "ok": True,
            "latitudes_deg": latitudes_deg,
            "case_arrays": case_arrays,
            "case_converged": case_converged,
            "case_steps": case_steps,
            "sample_meta": sample_meta,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _band_samples(latitudes_deg: np.ndarray, profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return low/high-band mean arrays for an (n_samples, n_lat) profile matrix."""
    low = band_mean_samples(latitudes_deg, profiles, LOW_LAT_BAND)
    high = band_mean_samples(latitudes_deg, profiles, HIGH_LAT_BAND)
    return low, high


def _band_medians(latitudes_deg: np.ndarray, profiles: np.ndarray) -> tuple[float, float]:
    """Return median low/high-band means for an (n_samples, n_lat) profile matrix."""
    low, high = _band_samples(latitudes_deg, profiles)
    return float(np.median(low)), float(np.median(high))


def _active_dconv_summary(dconv_samples: np.ndarray, ra_samples: np.ndarray) -> tuple[float, float]:
    """Return active-only D_conv median and all-sample p75 for one latitude band."""
    active_mask = ra_samples >= ConvectionConstants.RA_CRIT
    active_median = float(np.median(dconv_samples[active_mask])) if np.any(active_mask) else float("nan")
    p75_all = float(np.percentile(dconv_samples, 75.0))
    return active_median, p75_all


def _format_optional_km(value: float) -> str:
    """Format a kilometer value, preserving empty active-only summaries."""
    if np.isfinite(value):
        return f"{value:.1f} km"
    return "n/a"


def _print_case_summary(
    latitudes_deg: np.ndarray,
    metric_stacks: dict[str, np.ndarray],
    converged_stack: np.ndarray,
    step_stack: np.ndarray,
) -> None:
    """Print median band summaries for each attribution case and paired deltas."""
    print("=" * 76)
    print("PAIRED ATTRIBUTION ENSEMBLE (median band means)")
    print("=" * 76)
    print(
        f"Onset diagnostics use band-mean thresholds: "
        f"Ra_crit={ConvectionConstants.RA_CRIT:.0f}, Nu_active>{NU_ACTIVE_THRESHOLD:.2f}"
    )
    print("Active D_conv is the median over supercritical samples only; p75 uses all samples.")
    print(
        "Convergence by case: "
        + ", ".join(
            f"{case_name}={int(np.sum(converged_stack[:, case_index]))}/{converged_stack.shape[0]}"
            for case_index, case_name in enumerate(ATTRIBUTION_CASES)
        )
    )
    print(
        "Median steps by case: "
        + ", ".join(
            f"{case_name}={int(np.median(step_stack[:, case_index]))}"
            for case_index, case_name in enumerate(ATTRIBUTION_CASES)
        )
    )
    print("-" * 76)

    for case_index, case_name in enumerate(ATTRIBUTION_CASES):
        H_low, H_high = _band_medians(latitudes_deg, metric_stacks["H_km"][:, case_index, :])
        Dcond_low, Dcond_high = _band_medians(latitudes_deg, metric_stacks["D_cond_km"][:, case_index, :])
        dconv_low_samples, dconv_high_samples = _band_samples(
            latitudes_deg,
            metric_stacks["D_conv_km"][:, case_index, :],
        )
        Dconv_low = float(np.median(dconv_low_samples))
        Dconv_high = float(np.median(dconv_high_samples))
        Tc_low, Tc_high = _band_medians(latitudes_deg, metric_stacks["T_c"][:, case_index, :])
        Ti_low, Ti_high = _band_medians(latitudes_deg, metric_stacks["Ti"][:, case_index, :])
        ra_low_samples, ra_high_samples = _band_samples(latitudes_deg, metric_stacks["Ra"][:, case_index, :])
        nu_low_samples, nu_high_samples = _band_samples(latitudes_deg, metric_stacks["Nu"][:, case_index, :])
        Ra_low = float(np.median(ra_low_samples))
        Ra_high = float(np.median(ra_high_samples))
        Nu_low = float(np.median(nu_low_samples))
        Nu_high = float(np.median(nu_high_samples))
        Dconv_active_low, Dconv_p75_low = _active_dconv_summary(dconv_low_samples, ra_low_samples)
        Dconv_active_high, Dconv_p75_high = _active_dconv_summary(dconv_high_samples, ra_high_samples)
        supercrit_low = float(np.mean(ra_low_samples >= ConvectionConstants.RA_CRIT))
        supercrit_high = float(np.mean(ra_high_samples >= ConvectionConstants.RA_CRIT))
        nu_active_low = float(np.mean(nu_low_samples > NU_ACTIVE_THRESHOLD))
        nu_active_high = float(np.mean(nu_high_samples > NU_ACTIVE_THRESHOLD))

        print(f"[{case_name.upper()}]")
        print(
            f"  0-10 deg:  H={H_low:.1f} km, D_cond={Dcond_low:.1f} km, "
            f"D_conv={Dconv_low:.1f} km, T_c={Tc_low:.1f} K, Ti={Ti_low:.1f} K, "
            f"Ra={Ra_low:.0f}, Nu={Nu_low:.2f}, f_Ra_crit={supercrit_low:.2f}, f_Nu_active={nu_active_low:.2f}"
        )
        print(
            f"             active D_conv={_format_optional_km(Dconv_active_low)}, "
            f"D_conv p75(all)={Dconv_p75_low:.1f} km"
        )
        print(
            f"  80-90 deg: H={H_high:.1f} km, D_cond={Dcond_high:.1f} km, "
            f"D_conv={Dconv_high:.1f} km, T_c={Tc_high:.1f} K, Ti={Ti_high:.1f} K, "
            f"Ra={Ra_high:.0f}, Nu={Nu_high:.2f}, f_Ra_crit={supercrit_high:.2f}, f_Nu_active={nu_active_high:.2f}"
        )
        print(
            f"             active D_conv={_format_optional_km(Dconv_active_high)}, "
            f"D_conv p75(all)={Dconv_p75_high:.1f} km"
        )
        print(
            f"             converged={int(np.sum(converged_stack[:, case_index]))}/{converged_stack.shape[0]}, "
            f"median steps={int(np.median(step_stack[:, case_index]))}"
        )
        print()

    print("-" * 76)
    print("PAIRED DELTA VS BASELINE (median band means)")
    print("-" * 76)
    baseline = {name: metric_stacks[name][:, 0, :] for name in ("H_km", "D_cond_km", "D_conv_km", "Ra")}
    for case_index, case_name in enumerate(ATTRIBUTION_CASES[1:], start=1):
        delta_H = metric_stacks["H_km"][:, case_index, :] - baseline["H_km"]
        delta_Dcond = metric_stacks["D_cond_km"][:, case_index, :] - baseline["D_cond_km"]
        delta_Dconv = metric_stacks["D_conv_km"][:, case_index, :] - baseline["D_conv_km"]
        delta_Ra = metric_stacks["Ra"][:, case_index, :] - baseline["Ra"]
        dH_low, dH_high = _band_medians(latitudes_deg, delta_H)
        dDc_low, dDc_high = _band_medians(latitudes_deg, delta_Dcond)
        dDv_low, dDv_high = _band_medians(latitudes_deg, delta_Dconv)
        dRa_low, dRa_high = _band_medians(latitudes_deg, delta_Ra)
        print(f"[{case_name.upper()} - BASELINE]")
        print(
            f"  0-10 deg:  dH={dH_low:+.1f} km, dD_cond={dDc_low:+.1f} km, "
            f"dD_conv={dDv_low:+.1f} km, dRa={dRa_low:+.0f}"
        )
        print(
            f"  80-90 deg: dH={dH_high:+.1f} km, dD_cond={dDc_high:+.1f} km, "
            f"dD_conv={dDv_high:+.1f} km, dRa={dRa_high:+.0f}"
        )
        print()


def _print_failure_summary(n_iterations: int, valid_results: list[dict[str, Any]], failed_results: list[dict[str, Any]]) -> None:
    """Print valid/failed realization counts and the top failure reasons."""
    print(f"Completed: {len(valid_results)}/{n_iterations} valid realizations")
    if not failed_results:
        return
    print(f"Failed: {len(failed_results)}/{n_iterations} realizations")
    counts = Counter(result["error"] for result in failed_results)
    print("Top failure reasons:")
    for reason, count in counts.most_common(5):
        print(f"  {count}x {reason}")
    print("-" * 76)


def run_paired_attribution_ensemble(
    scenario_name: str = "lemasquerier2023_polar",
    n_iterations: int = 50,
    seed: int = 42,
    n_workers: int = 1,
    n_lat: int = 19,
    nx: int = 31,
    dt: float = 5e12,
    max_steps: int = 500,
    eq_threshold: float = 1e-12,
    initial_thickness: float = 20e3,
    use_convection: bool = True,
    rannacher_steps: int = 4,
    coordinate_system: str = "auto",
    grain_latitude_mode: str = "global",
    q_tidal_scale: float = 1.0,
    T_floor: float = 46.0,
) -> str:
    """Run the paired 4-case attribution ensemble and save an NPZ archive."""
    scenario = get_scenario(scenario_name)
    print("=" * 76)
    print("PAIRED 4-CASE ATTRIBUTION ENSEMBLE")
    print("=" * 76)
    print(f"Scenario: {scenario.name} ({scenario.citation})")
    print(f"Iterations: {n_iterations}, Workers: {n_workers}")
    print(f"Grid: n_lat={n_lat}, nx={nx}, max_steps={max_steps}")
    print("-" * 76)
    if scenario.ocean_pattern == "uniform":
        print("Warning: uniform_transport makes ocean_only identical to baseline.")

    start_time = time.time()
    worker = partial(
        _run_paired_sample,
        base_seed=seed,
        scenario_name=scenario_name,
        n_lat=n_lat,
        nx=nx,
        dt=dt,
        use_convection=use_convection,
        max_steps=max_steps,
        eq_threshold=eq_threshold,
        initial_thickness=initial_thickness,
        rannacher_steps=rannacher_steps,
        coordinate_system=coordinate_system,
        grain_latitude_mode=grain_latitude_mode,
        q_tidal_scale=q_tidal_scale,
        T_floor=T_floor,
    )

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = list(pool.imap_unordered(worker, range(n_iterations)))
    else:
        results = [worker(i) for i in range(n_iterations)]

    runtime = time.time() - start_time
    valid_results = [r for r in results if r.get("ok")]
    failed_results = [r for r in results if not r.get("ok")]
    if not valid_results:
        raise RuntimeError("No valid paired attribution realizations completed.")

    latitudes_deg = np.asarray(valid_results[0]["latitudes_deg"], dtype=float)
    case_names = np.array(ATTRIBUTION_CASES, dtype="<U16")

    metric_stacks = {
        metric: np.array(
            [
                [sample["case_arrays"][case_name][metric] for case_name in ATTRIBUTION_CASES]
                for sample in valid_results
            ],
            dtype=float,
        )
        for metric in PROFILE_METRICS
    }
    sample_meta_arrays = {
        key: np.array([sample["sample_meta"][key] for sample in valid_results], dtype=float)
        for key in SAMPLED_SHARED_KEYS + SAMPLED_PROFILE_KEYS
    }
    converged_stack = np.array(
        [
            [sample["case_converged"][case_name] for case_name in ATTRIBUTION_CASES]
            for sample in valid_results
        ],
        dtype=bool,
    )
    step_stack = np.array(
        [
            [sample["case_steps"][case_name] for case_name in ATTRIBUTION_CASES]
            for sample in valid_results
        ],
        dtype=int,
    )

    _print_failure_summary(n_iterations, valid_results, failed_results)
    _print_case_summary(latitudes_deg, metric_stacks, converged_stack, step_stack)

    save_dict: dict[str, Any] = {
        "scenario": np.array(scenario.name),
        "scenario_citation": np.array(scenario.citation),
        "case_names": case_names,
        "latitudes_deg": latitudes_deg,
        "n_iterations": n_iterations,
        "n_valid": len(valid_results),
        "n_failed": len(failed_results),
        "runtime_seconds": runtime,
        "q_tidal_scale": q_tidal_scale,
        "T_floor_mean": T_floor,
        "grain_latitude_mode": np.array(grain_latitude_mode),
        "Ra_crit": ConvectionConstants.RA_CRIT,
        "Nu_active_threshold": NU_ACTIVE_THRESHOLD,
        "case_converged": converged_stack,
        "case_steps": step_stack,
    }

    for metric, stack in metric_stacks.items():
        save_dict[f"{metric}_profiles"] = stack
        save_dict[f"{metric}_median"] = np.median(stack, axis=0)
        save_dict[f"{metric}_mean"] = np.mean(stack, axis=0)
        save_dict[f"{metric}_sigma_low"] = np.percentile(stack, 15.87, axis=0)
        save_dict[f"{metric}_sigma_high"] = np.percentile(stack, 84.13, axis=0)
        for case_index, case_name in enumerate(ATTRIBUTION_CASES):
            low, high = _band_samples(latitudes_deg, stack[:, case_index, :])
            save_dict[f"{metric}_{case_name}_low_band"] = low
            save_dict[f"{metric}_{case_name}_high_band"] = high

    for case_index, case_name in enumerate(ATTRIBUTION_CASES):
        ra_low, ra_high = _band_samples(latitudes_deg, metric_stacks["Ra"][:, case_index, :])
        nu_low, nu_high = _band_samples(latitudes_deg, metric_stacks["Nu"][:, case_index, :])
        dconv_low, dconv_high = _band_samples(latitudes_deg, metric_stacks["D_conv_km"][:, case_index, :])
        dconv_active_low, dconv_p75_low = _active_dconv_summary(dconv_low, ra_low)
        dconv_active_high, dconv_p75_high = _active_dconv_summary(dconv_high, ra_high)
        save_dict[f"Ra_supercritical_{case_name}_low_fraction"] = float(
            np.mean(ra_low >= ConvectionConstants.RA_CRIT)
        )
        save_dict[f"Ra_supercritical_{case_name}_high_fraction"] = float(
            np.mean(ra_high >= ConvectionConstants.RA_CRIT)
        )
        save_dict[f"Nu_active_{case_name}_low_fraction"] = float(
            np.mean(nu_low > NU_ACTIVE_THRESHOLD)
        )
        save_dict[f"Nu_active_{case_name}_high_fraction"] = float(
            np.mean(nu_high > NU_ACTIVE_THRESHOLD)
        )
        save_dict[f"D_conv_active_{case_name}_low_median"] = dconv_active_low
        save_dict[f"D_conv_active_{case_name}_high_median"] = dconv_active_high
        save_dict[f"D_conv_{case_name}_low_p75"] = dconv_p75_low
        save_dict[f"D_conv_{case_name}_high_p75"] = dconv_p75_high

    for key, values in sample_meta_arrays.items():
        save_dict[f"sample_{key}"] = values

    if failed_results:
        failure_counts = Counter(result["error"] for result in failed_results)
        save_dict["failure_reasons"] = np.array(list(failure_counts.keys()), dtype="<U256")
        save_dict["failure_counts"] = np.array(list(failure_counts.values()), dtype=int)

    out_dir = os.path.join(_PROJECT_DIR, "results", "attribution")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir,
        f"paired_attribution_{scenario.name}_{n_iterations}.npz",
    )
    np.savez(output_path, **save_dict)
    print(f"Saved paired attribution archive to: {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired 4-case 2D attribution ensemble with shared sampled physics per realization."
    )
    parser.add_argument(
        "--scenario",
        choices=list_scenarios(),
        default="lemasquerier2023_polar",
        help="Ocean-transport scenario used for the ocean_only case.",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Number of paired realizations.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-workers", type=int, default=max(1, mp.cpu_count() - 1), help="Worker processes.")
    parser.add_argument("--n-lat", type=int, default=19, help="Latitude columns.")
    parser.add_argument("--nx", type=int, default=31, help="Radial nodes per column.")
    parser.add_argument("--dt", type=float, default=5e12, help="Time step in seconds.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum solver steps per case.")
    parser.add_argument("--q-tidal-scale", type=float, default=1.20, help="Scale factor applied to inherited tidal/ocean flux.")
    parser.add_argument("--t-floor", type=float, default=50.0, help="Mean polar floor used by the 2D sampler.")
    parser.add_argument(
        "--grain-mode",
        choices=["global", "strain", "strain_temperature"],
        default="global",
        help="Latitude grain-size coupling mode for the shared sampled profile.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = _parse_args()
    run_paired_attribution_ensemble(
        scenario_name=args.scenario,
        n_iterations=args.iterations,
        seed=args.seed,
        n_workers=args.n_workers,
        n_lat=args.n_lat,
        nx=args.nx,
        dt=args.dt,
        max_steps=args.max_steps,
        grain_latitude_mode=args.grain_mode,
        q_tidal_scale=args.q_tidal_scale,
        T_floor=args.t_floor,
    )
