"""Targeted diagnostic for the 2D onset-definition mismatch hypothesis.

This runner is intentionally more explicit than the generic quick/validation
passes. It asks one narrow question:

    "Is the hard local-Ra onset gate collapsing warm sublayers that already
    satisfy the Green/Deschamps rheological transition criterion?"

Unlike the MC aggregate interface, this script runs explicit solver samples so
it can measure the latent transition depth directly from each equilibrium
temperature profile. That lets it distinguish:

- a genuinely fully conductive column
- a column with a rheological transition that the solver collapsed
- a column that is active once the onset gate is relaxed

The output is designed to be readable by humans and downstream models alike.
It prints a structured report and can optionally save a JSON payload with the
full aggregated metrics.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "Europa2D" / "src"))
sys.path.insert(0, str(_REPO / "EuropaProjectDJ" / "src"))
sys.path.insert(0, str(_REPO / "autoresearch"))

from axial_solver import AxialSolver2D
from constants import Thermal
from convection_2d import ConvectionHypothesis
from latitude_sampler import LatitudeParameterSampler
from objectives import compute_latitude_score


DEFAULT_SCENARIOS: Tuple[Tuple[str, str, Optional[float]], ...] = (
    ("uniform", "uniform", None),
    ("polar", "polar_enhanced", 0.455),
    ("equator", "equator_enhanced", 0.4),
)


def _build_experiments() -> List[Tuple[str, Optional[ConvectionHypothesis]]]:
    return [
        ("baseline", None),
        ("ra_onset_1000", ConvectionHypothesis("ra_onset", {"ra_crit_override": 1000.0})),
        ("ra_onset_100", ConvectionHypothesis("ra_onset", {"ra_crit_override": 100.0})),
        ("ra_onset_10", ConvectionHypothesis("ra_onset", {"ra_crit_override": 10.0})),
        ("ra_onset_1", ConvectionHypothesis("ra_onset", {"ra_crit_override": 1.0})),
    ]


def _find_lat_index(latitudes_deg: np.ndarray, target_deg: float) -> int:
    return int(np.argmin(np.abs(np.asarray(latitudes_deg) - target_deg)))


def _safe_float(value: Any) -> float:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return float(value)


def _active_only_median(values: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    n_lat = values.shape[1]
    out = np.zeros(n_lat)
    for idx in range(n_lat):
        active_vals = values[active_mask[:, idx], idx]
        out[idx] = float(np.median(active_vals)) if active_vals.size else 0.0
    return out


def _find_transition_depth_m(
    T_profile: np.ndarray,
    z_grid: np.ndarray,
    T_c: float,
    total_thickness: float,
) -> float:
    """Locate the depth where the equilibrium profile first reaches T_c."""
    if not np.isfinite(T_c):
        return total_thickness

    warm_indices = np.where(T_profile >= T_c)[0]
    if warm_indices.size == 0:
        return total_thickness

    idx_c = int(warm_indices[0])
    if idx_c <= 0:
        return float(z_grid[0])
    if idx_c >= len(z_grid):
        return total_thickness

    T_above = float(T_profile[idx_c - 1])
    T_below = float(T_profile[idx_c])
    z_above = float(z_grid[idx_c - 1])
    z_below = float(z_grid[idx_c])
    if T_below <= T_above:
        return float(z_grid[idx_c])

    frac = (T_c - T_above) / (T_below - T_above)
    frac = float(np.clip(frac, 0.0, 1.0))
    return z_above + frac * (z_below - z_above)


def _run_single_diagnostic_sample(
    sample_id: int,
    base_seed: int,
    n_lat: int,
    nx: int,
    dt: float,
    max_steps: int,
    eq_threshold: float,
    initial_thickness: float,
    ocean_pattern: str,
    q_star: Optional[float],
    hypothesis_mechanism: Optional[str],
    hypothesis_params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Run one explicit 2D sample and return rich per-column diagnostics."""
    try:
        sampler = LatitudeParameterSampler(
            seed=base_seed + sample_id,
            ocean_pattern=ocean_pattern,
            q_star=q_star,
        )
        shared_params, profile = sampler.sample()
        D_H2O = float(shared_params["D_H2O"])

        q_mean = float(profile.q_ocean_mean)
        if q_mean > 0.0:
            k_mean = float(Thermal.conductivity(190.0))
            H_guess = k_mean * 170.0 / q_mean
            H_guess *= 1.5
            H_guess = float(np.clip(H_guess, 5e3, 80e3))
        else:
            H_guess = float(initial_thickness)

        hypothesis = None
        if hypothesis_mechanism is not None:
            hypothesis = ConvectionHypothesis(
                mechanism=hypothesis_mechanism,
                params=hypothesis_params or {},
            )

        solver = AxialSolver2D(
            n_lat=n_lat,
            nx=nx,
            dt=dt,
            latitude_profile=profile,
            physics_params=shared_params,
            use_convection=True,
            initial_thickness=H_guess,
            rannacher_steps=4,
            coordinate_system="auto",
            hypothesis=hypothesis,
        )
        result = solver.run_to_equilibrium(
            threshold=eq_threshold,
            max_steps=max_steps,
            verbose=False,
        )

        H_km = np.asarray(result["H_profile_km"])
        D_H2O_km = D_H2O / 1000.0
        valid_mask = (H_km > 0.5) & (H_km < D_H2O_km * 0.99) & (H_km < 200.0)
        if np.sum(valid_mask) < len(H_km) * 0.5:
            return None

        latitudes_deg = np.asarray(result["latitudes_deg"])
        diagnostics = result["diagnostics"]

        D_cond_km = np.zeros(n_lat)
        D_conv_km = np.zeros(n_lat)
        latent_D_conv_km = np.zeros(n_lat)
        H_total_km = np.zeros(n_lat)
        Ra = np.zeros(n_lat)
        Nu = np.zeros(n_lat)
        T_c = np.zeros(n_lat)
        transition_exists = np.zeros(n_lat, dtype=bool)
        active = np.zeros(n_lat, dtype=bool)
        collapsed = np.zeros(n_lat, dtype=bool)
        upper_clamp = np.zeros(n_lat, dtype=bool)
        lower_clamp = np.zeros(n_lat, dtype=bool)

        for j, col in enumerate(solver.columns):
            state = col.convection_state
            if state is None:
                return None

            T_profile = np.asarray(col.T)
            z_grid = np.asarray(col._get_depths())
            H = float(col.H)
            T_surface = float(T_profile[0])
            T_c_j = float(state.T_c)
            z_transition = _find_transition_depth_m(T_profile, z_grid, T_c_j, H)
            latent_D_conv_m = max(0.0, H - z_transition)

            D_cond_km[j] = float(diagnostics[j]["D_cond_km"])
            D_conv_km[j] = float(diagnostics[j]["D_conv_km"])
            latent_D_conv_km[j] = latent_D_conv_m / 1000.0
            H_total_km[j] = H / 1000.0
            Ra[j] = float(diagnostics[j]["Ra"])
            Nu[j] = float(diagnostics[j]["Nu"])
            T_c[j] = T_c_j

            has_transition = T_c_j > T_surface + 1.0 and latent_D_conv_m > 50.0
            is_active = bool(state.is_convecting and state.D_conv > 50.0)
            is_collapsed = bool(has_transition and not is_active and state.D_conv <= 50.0)

            transition_exists[j] = has_transition
            active[j] = is_active
            collapsed[j] = is_collapsed
            upper_clamp[j] = bool(state.D_cond >= 0.94 * H)
            lower_clamp[j] = bool(state.D_cond <= 0.06 * H)

        return {
            "latitudes_deg": latitudes_deg,
            "H_km": H_total_km,
            "D_cond_km": D_cond_km,
            "D_conv_km": D_conv_km,
            "latent_D_conv_km": latent_D_conv_km,
            "Ra": Ra,
            "Nu": Nu,
            "T_c": T_c,
            "transition_exists": transition_exists,
            "active": active,
            "collapsed": collapsed,
            "upper_clamp": upper_clamp,
            "lower_clamp": lower_clamp,
        }
    except Exception:
        return None


def _stack_samples(samples: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([sample[key] for sample in samples])


def _scenario_summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise RuntimeError("Scenario produced no valid diagnostic samples.")

    latitudes = np.asarray(samples[0]["latitudes_deg"])
    D_cond = _stack_samples(samples, "D_cond_km")
    D_conv = _stack_samples(samples, "D_conv_km")
    latent_D_conv = _stack_samples(samples, "latent_D_conv_km")
    H = _stack_samples(samples, "H_km")
    Ra = _stack_samples(samples, "Ra")
    Nu = _stack_samples(samples, "Nu")
    transition_mask = _stack_samples(samples, "transition_exists").astype(bool)
    active_mask = _stack_samples(samples, "active").astype(bool)
    collapsed_mask = _stack_samples(samples, "collapsed").astype(bool)
    upper_clamp_mask = _stack_samples(samples, "upper_clamp").astype(bool)
    lower_clamp_mask = _stack_samples(samples, "lower_clamp").astype(bool)

    idx_35 = _find_lat_index(latitudes, 35.0)
    active_only_dconv = _active_only_median(D_conv, active_mask)

    per_lat = []
    for i, lat in enumerate(latitudes):
        per_lat.append(
            {
                "lat_deg": _safe_float(lat),
                "D_cond_median_km": _safe_float(np.median(D_cond[:, i])),
                "D_cond_p16_km": _safe_float(np.percentile(D_cond[:, i], 15.87)),
                "D_cond_p84_km": _safe_float(np.percentile(D_cond[:, i], 84.13)),
                "D_conv_median_km": _safe_float(np.median(D_conv[:, i])),
                "latent_D_conv_median_km": _safe_float(np.median(latent_D_conv[:, i])),
                "active_only_D_conv_median_km": _safe_float(active_only_dconv[i]),
                "transition_fraction": _safe_float(np.mean(transition_mask[:, i])),
                "active_fraction": _safe_float(np.mean(active_mask[:, i])),
                "collapsed_fraction": _safe_float(np.mean(collapsed_mask[:, i])),
                "upper_clamp_fraction": _safe_float(np.mean(upper_clamp_mask[:, i])),
                "lower_clamp_fraction": _safe_float(np.mean(lower_clamp_mask[:, i])),
                "Ra_median": _safe_float(np.median(Ra[:, i])),
                "Nu_median": _safe_float(np.median(Nu[:, i])),
                "H_median_km": _safe_float(np.median(H[:, i])),
            }
        )

    overall = {
        "n_valid": len(samples),
        "transition_fraction_all": _safe_float(np.mean(transition_mask)),
        "active_fraction_all": _safe_float(np.mean(active_mask)),
        "collapsed_fraction_all": _safe_float(np.mean(collapsed_mask)),
        "upper_clamp_fraction_all": _safe_float(np.mean(upper_clamp_mask)),
        "lower_clamp_fraction_all": _safe_float(np.mean(lower_clamp_mask)),
        "D_cond_35_median_km": _safe_float(np.median(D_cond[:, idx_35])),
        "D_conv_35_median_km": _safe_float(np.median(D_conv[:, idx_35])),
        "latent_D_conv_35_median_km": _safe_float(np.median(latent_D_conv[:, idx_35])),
        "Ra_35_median": _safe_float(np.median(Ra[:, idx_35])),
        "Nu_35_median": _safe_float(np.median(Nu[:, idx_35])),
    }

    scoring_payload = {
        "latitudes_deg": latitudes,
        "D_cond_profiles": D_cond,
        "D_conv_profiles": D_conv,
        "H_profiles": H,
        "Ra_profiles": Ra,
    }

    return {
        "latitudes_deg": latitudes,
        "per_latitude": per_lat,
        "overall": overall,
        "scoring_payload": scoring_payload,
    }


def _run_scenario(
    pattern: str,
    q_star: Optional[float],
    n_samples: int,
    seed: int,
    n_workers: int,
    n_lat: int,
    nx: int,
    max_steps: int,
    eq_threshold: float,
    hypothesis: Optional[ConvectionHypothesis],
) -> Dict[str, Any]:
    worker = partial(
        _run_single_diagnostic_sample,
        base_seed=seed,
        n_lat=n_lat,
        nx=nx,
        dt=1e12,
        max_steps=max_steps,
        eq_threshold=eq_threshold,
        initial_thickness=20e3,
        ocean_pattern=pattern,
        q_star=q_star,
        hypothesis_mechanism=hypothesis.mechanism if hypothesis else None,
        hypothesis_params=hypothesis.params if hypothesis else None,
    )

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            chunksize = max(1, n_samples // max(1, n_workers * 2))
            samples = list(
                sample
                for sample in pool.imap_unordered(worker, range(n_samples), chunksize=chunksize)
                if sample is not None
            )
    else:
        samples = [worker(i) for i in range(n_samples)]
        samples = [sample for sample in samples if sample is not None]

    return _scenario_summary(samples)


def _run_experiment(
    name: str,
    hypothesis: Optional[ConvectionHypothesis],
    n_samples: int,
    seeds: Iterable[int],
    n_workers: int,
    n_lat: int,
    nx: int,
    max_steps: int,
    eq_threshold: float,
) -> Dict[str, Any]:
    seed_reports = []

    for seed in seeds:
        t0 = time.time()
        scenarios = {}
        for scenario_name, pattern, q_star in DEFAULT_SCENARIOS:
            scenarios[scenario_name] = _run_scenario(
                pattern=pattern,
                q_star=q_star,
                n_samples=n_samples,
                seed=seed,
                n_workers=n_workers,
                n_lat=n_lat,
                nx=nx,
                max_steps=max_steps,
                eq_threshold=eq_threshold,
                hypothesis=hypothesis,
            )

        score, metrics = compute_latitude_score(
            {key: payload["scoring_payload"] for key, payload in scenarios.items()},
            consistency_error=0.0,
        )
        seed_reports.append(
            {
                "seed": int(seed),
                "runtime_seconds": _safe_float(time.time() - t0),
                "score": _safe_float(score),
                "metrics": {k: _safe_float(v) for k, v in metrics.items()},
                "scenarios": scenarios,
            }
        )

    return {"name": name, "hypothesis": hypothesis, "seed_reports": seed_reports}


def _mean_metric(seed_reports: List[Dict[str, Any]], key: str) -> float:
    return float(np.mean([report["metrics"][key] for report in seed_reports]))


def _mean_overall(seed_reports: List[Dict[str, Any]], scenario_name: str, key: str) -> float:
    return float(
        np.mean(
            [report["scenarios"][scenario_name]["overall"][key] for report in seed_reports]
        )
    )


def _experiment_comparison(
    baseline: Dict[str, Any],
    experiment: Dict[str, Any],
) -> Dict[str, float]:
    deltas = {}
    deltas["profile_JS_min_delta"] = _mean_metric(
        experiment["seed_reports"], "profile_JS_min"
    ) - _mean_metric(
        baseline["seed_reports"], "profile_JS_min"
    )
    deltas["D_cond_35_delta_km"] = _mean_metric(
        experiment["seed_reports"], "D_cond_35_median"
    ) - _mean_metric(
        baseline["seed_reports"], "D_cond_35_median"
    )

    for scenario_name, _, _ in DEFAULT_SCENARIOS:
        prefix = scenario_name
        deltas[f"{prefix}_active_fraction_delta"] = _mean_overall(
            experiment["seed_reports"], scenario_name, "active_fraction_all"
        ) - _mean_overall(
            baseline["seed_reports"], scenario_name, "active_fraction_all"
        )
        deltas[f"{prefix}_collapsed_fraction_delta"] = _mean_overall(
            experiment["seed_reports"], scenario_name, "collapsed_fraction_all"
        ) - _mean_overall(
            baseline["seed_reports"], scenario_name, "collapsed_fraction_all"
        )
        deltas[f"{prefix}_upper_clamp_delta"] = _mean_overall(
            experiment["seed_reports"], scenario_name, "upper_clamp_fraction_all"
        ) - _mean_overall(
            baseline["seed_reports"], scenario_name, "upper_clamp_fraction_all"
        )
    return {k: _safe_float(v) for k, v in deltas.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, ConvectionHypothesis):
        return {"mechanism": value.mechanism, "params": value.params}
    return value


def _print_seed_header(seed_report: Dict[str, Any]) -> None:
    metrics = seed_report["metrics"]
    print(
        f"  Seed {seed_report['seed']}: score={seed_report['score']:+.2f}"
        f" ({seed_report['runtime_seconds']:.0f}s)"
    )
    print(f"    profile_JS_min:  {metrics['profile_JS_min']:.4f}")
    print(f"    profile_JS_mean: {metrics['profile_JS_mean']:.4f}")
    print(f"    JS@35:           {metrics['JS_35']:.4f}")
    print(f"    JS_peak:         {metrics['JS_peak']:.4f} at {metrics['phi_peak_js']:.0f} deg")
    print(f"    D_cond@35:       {metrics['D_cond_35_median']:.1f} km")
    print(f"    D_conv contrast: {metrics['D_conv_contrast']:.2f} km")
    print(f"    Ra eq/pole:      {metrics['Ra_eq_median']:.0f} / {metrics['Ra_pole_median']:.0f}")


def _print_scenario_overview(scenarios: Dict[str, Any]) -> None:
    print("    Scenario overview:")
    print("      name       transition   active   collapsed   clamp(0.95H)   D_cond@35   D_conv@35   latent@35")
    for scenario_name, _, _ in DEFAULT_SCENARIOS:
        overall = scenarios[scenario_name]["overall"]
        print(
            f"      {scenario_name:8s}  "
            f"{100*overall['transition_fraction_all']:8.1f}%  "
            f"{100*overall['active_fraction_all']:6.1f}%  "
            f"{100*overall['collapsed_fraction_all']:9.1f}%  "
            f"{100*overall['upper_clamp_fraction_all']:11.1f}%  "
            f"{overall['D_cond_35_median_km']:8.2f}  "
            f"{overall['D_conv_35_median_km']:8.2f}  "
            f"{overall['latent_D_conv_35_median_km']:9.2f}"
        )


def _print_latitude_table(
    scenario_summary: Dict[str, Any],
    scenario_name: str,
    baseline_summary: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"    Latitude profile ({scenario_name}):")
    print(
        "      Lat    D_cond   D_conv   latent   trans_f   active_f   collapse_f   clamp_f"
        "        Ra       Nu"
    )
    baseline_by_lat = {}
    if baseline_summary is not None:
        baseline_by_lat = {
            round(row["lat_deg"], 6): row for row in baseline_summary["per_latitude"]
        }

    for row in scenario_summary["per_latitude"]:
        key = round(row["lat_deg"], 6)
        if baseline_summary is None:
            line = (
                f"      {row['lat_deg']:4.1f}  {row['D_cond_median_km']:8.2f}"
                f"  {row['D_conv_median_km']:7.2f}  {row['latent_D_conv_median_km']:7.2f}"
                f"  {row['transition_fraction']:8.3f}  {row['active_fraction']:8.3f}"
                f"  {row['collapsed_fraction']:10.3f}  {row['upper_clamp_fraction']:8.3f}"
                f"  {row['Ra_median']:8.1f}  {row['Nu_median']:7.2f}"
            )
        else:
            base = baseline_by_lat[key]
            line = (
                f"      {row['lat_deg']:4.1f}  {row['D_cond_median_km']:8.2f}"
                f"  {row['D_conv_median_km']:7.2f}  {row['latent_D_conv_median_km']:7.2f}"
                f"  {row['transition_fraction']:8.3f}  {row['active_fraction']:8.3f}"
                f"  {row['collapsed_fraction']:10.3f}  {row['upper_clamp_fraction']:8.3f}"
                f"  {row['Ra_median']:8.1f}  {row['Nu_median']:7.2f}"
                f"   | dActive={row['active_fraction'] - base['active_fraction']:+.3f}"
                f" dCollapse={row['collapsed_fraction'] - base['collapsed_fraction']:+.3f}"
            )
        print(line)


def _print_baseline_comparison(name: str, comparison: Dict[str, float]) -> None:
    print(f"  Delta vs baseline ({name}):")
    print(f"    profile_JS_min delta: {comparison['profile_JS_min_delta']:+.4f}")
    print(f"    D_cond@35 delta:      {comparison['D_cond_35_delta_km']:+.2f} km")
    for scenario_name, _, _ in DEFAULT_SCENARIOS:
        print(
            f"    {scenario_name:8s} active {comparison[f'{scenario_name}_active_fraction_delta']:+.3f}"
            f" | collapsed {comparison[f'{scenario_name}_collapsed_fraction_delta']:+.3f}"
            f" | clamp {comparison[f'{scenario_name}_upper_clamp_delta']:+.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a targeted onset-definition diagnostic for Europa2D."
    )
    parser.add_argument("--samples", type=int, default=12, help="Samples per scenario per seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 137],
        help="One or more MC seeds.",
    )
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--n-lat", type=int, default=19)
    parser.add_argument("--nx", type=int, default=31)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--eq-threshold", type=float, default=1e-11)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["baseline", "ra_onset_1000", "ra_onset_100", "ra_onset_10", "ra_onset_1"],
        default=None,
        help="Subset of experiments to run.",
    )
    parser.add_argument(
        "--full-latitude-tables",
        action="store_true",
        help="Print the full per-latitude table for every scenario/seed. Default prints only the first seed.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiments = _build_experiments()
    if args.experiments is not None:
        wanted = set(args.experiments)
        experiments = [item for item in experiments if item[0] in wanted]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "samples": args.samples,
            "seeds": args.seeds,
            "workers": args.workers,
            "n_lat": args.n_lat,
            "nx": args.nx,
            "max_steps": args.max_steps,
            "eq_threshold": args.eq_threshold,
            "experiments": [name for name, _ in experiments],
        },
        "experiments": [],
    }

    baseline_report = None

    print("=" * 78)
    print("  ONSET-DEFINITION DIAGNOSTIC")
    print("=" * 78)
    print("  Question: is the hard local-Ra onset gate collapsing warm sublayers that")
    print("  already satisfy the Green/Deschamps rheological transition criterion?")

    for name, hypothesis in experiments:
        print(f"\n{'=' * 78}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'=' * 78}")
        experiment_report = _run_experiment(
            name=name,
            hypothesis=hypothesis,
            n_samples=args.samples,
            seeds=args.seeds,
            n_workers=args.workers,
            n_lat=args.n_lat,
            nx=args.nx,
            max_steps=args.max_steps,
            eq_threshold=args.eq_threshold,
        )

        for idx, seed_report in enumerate(experiment_report["seed_reports"]):
            _print_seed_header(seed_report)
            _print_scenario_overview(seed_report["scenarios"])

            if idx == 0 or args.full_latitude_tables:
                for scenario_name, _, _ in DEFAULT_SCENARIOS:
                    baseline_summary = None
                    if baseline_report is not None and name != "baseline":
                        baseline_summary = baseline_report["seed_reports"][idx]["scenarios"][scenario_name]
                    _print_latitude_table(
                        seed_report["scenarios"][scenario_name],
                        scenario_name,
                        baseline_summary=baseline_summary,
                    )

        if baseline_report is None:
            baseline_report = experiment_report
        else:
            comparison = _experiment_comparison(baseline_report, experiment_report)
            experiment_report["vs_baseline"] = comparison
            _print_baseline_comparison(name, comparison)

        report["experiments"].append(experiment_report)

    print(f"\n{'=' * 78}")
    print("  RANKING (mean across seeds)")
    print(f"{'=' * 78}")
    ranked = sorted(
        report["experiments"],
        key=lambda exp: np.mean([seed["score"] for seed in exp["seed_reports"]]),
    )
    for idx, exp in enumerate(ranked, start=1):
        mean_score = float(np.mean([seed["score"] for seed in exp["seed_reports"]]))
        mean_pjs = _mean_metric(exp["seed_reports"], "profile_JS_min")
        mean_d35 = _mean_metric(exp["seed_reports"], "D_cond_35_median")
        mean_active = np.mean(
            [
                _mean_overall(exp["seed_reports"], scenario_name, "active_fraction_all")
                for scenario_name, _, _ in DEFAULT_SCENARIOS
            ]
        )
        mean_collapsed = np.mean(
            [
                _mean_overall(exp["seed_reports"], scenario_name, "collapsed_fraction_all")
                for scenario_name, _, _ in DEFAULT_SCENARIOS
            ]
        )
        print(
            f"  {idx}. {exp['name']:14s}  score={mean_score:+.2f}  pJS={mean_pjs:.4f}"
            f"  D_cond@35={mean_d35:.1f} km  active={100*mean_active:.1f}%"
            f"  collapsed={100*mean_collapsed:.1f}%"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(_to_jsonable(report), indent=2))
        print(f"\nSaved JSON report to: {args.output_json}")

    print(f"\n{'=' * 78}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
