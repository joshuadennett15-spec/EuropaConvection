"""
Production Sobol sensitivity analysis for the 1D Europa thermal model.

Wraps the existing EuropaProjectDJ sobol_workflow infrastructure with three
physics configurations that map to the experiment matrix:

  baseline  — Howell conductivity / diffusion creep / Green Nu scaling
  improved  — Carnahan conductivity / GBS composite creep / DV2021 scaling
  wattmeter — same as improved + wattmeter grain-size model

Each config produces Saltelli-sampled parameter sets over 8 grouped priors,
runs the 1D solver in parallel, and computes S1/ST indices with 95% CIs.

Usage:
    python run_sobol_analysis.py --N 128 --config baseline --workers 6
    python run_sobol_analysis.py --N 256 --config all --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

# Wire EuropaProjectDJ source and scripts onto the path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DJ_SRC = _PROJECT_ROOT / "EuropaProjectDJ" / "src"
_DJ_SCRIPTS = _PROJECT_ROOT / "EuropaProjectDJ" / "scripts"
sys.path.insert(0, str(_DJ_SRC))
sys.path.insert(0, str(_DJ_SCRIPTS))

from Monte_Carlo import SolverConfig  # noqa: E402
from sobol_workflow import (  # noqa: E402
    DEFAULT_ANALYSIS_OUTPUTS,
    build_salib_problem,
    compute_sobol_indices,
    default_convergence_schedule,
    effective_dimension,
    evaluate_sobol_design,
    expected_sobol_rows,
    generate_sobol_design,
    is_power_of_two,
    ordered_unique,
    sobol_results_to_rows,
    summarize_top_total_indices,
)
from run_sobol_suite import write_csv  # noqa: E402

# ── Output directory ────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "sobol"

# ── Physics configurations ──────────────────────────────────────────────────
# Maps to the three thesis experiment columns:
#   baseline  = Howell / diffusion / Green
#   improved  = Carnahan / GBS / DV2021
#   wattmeter = improved + wattmeter grain

CONFIG_NAMES = ("baseline", "improved", "wattmeter")

# QoIs to analyse (subset that converges well at moderate N)
ANALYSIS_QOIS: Sequence[str] = (
    "thickness_km",
    "D_cond_km",
    "D_conv_km",
    "lid_fraction",
    "convective_flag",
)


@dataclass(frozen=True)
class PhysicsConfig:
    """Immutable descriptor for one physics configuration."""
    name: str
    label: str
    conductivity_model: str
    creep_model: str
    nu_scaling: str
    grain_mode: str


PHYSICS_CONFIGS: Dict[str, PhysicsConfig] = {
    "baseline": PhysicsConfig(
        name="baseline",
        label="Baseline (Howell/diffusion/Green)",
        conductivity_model="Howell",
        creep_model="diffusion",
        nu_scaling="green",
        grain_mode="sampled",
    ),
    "improved": PhysicsConfig(
        name="improved",
        label="Improved (Carnahan/GBS/DV2021)",
        conductivity_model="Carnahan",
        creep_model="composite_gbs",
        nu_scaling="dv2021",
        grain_mode="sampled",
    ),
    "wattmeter": PhysicsConfig(
        name="wattmeter",
        label="Wattmeter (Carnahan/GBS/DV2021/WM)",
        conductivity_model="Carnahan",
        creep_model="composite_gbs",
        nu_scaling="dv2021",
        grain_mode="wattmeter",
    ),
}


def _build_solver_config(physics: PhysicsConfig, nx: int, max_steps: int) -> SolverConfig:
    """Create a SolverConfig with physics overrides baked in."""
    config = SolverConfig()
    config.nx = nx
    config.max_steps = max_steps
    # SolverConfig doesn't carry physics-selector fields natively;
    # the sobol_workflow._evaluate_fixed_params reads config attributes
    # if present.  We attach them as dynamic attributes.
    config.conductivity_model = physics.conductivity_model  # type: ignore[attr-defined]
    config.creep_model = physics.creep_model  # type: ignore[attr-defined]
    config.nu_scaling = physics.nu_scaling  # type: ignore[attr-defined]
    config.grain_mode = physics.grain_mode  # type: ignore[attr-defined]
    return config


def _run_single_config(
    physics: PhysicsConfig,
    *,
    base_n: int,
    scenario: str,
    grouped: bool,
    seed: int,
    n_workers: int,
    nx: int,
    max_steps: int,
    num_resamples: int,
    conf_level: float,
    output_qois: Sequence[str],
    output_dir: Path,
    quiet: bool,
) -> Dict[str, Any]:
    """Run the full Sobol pipeline for one physics configuration."""
    run_label = f"{physics.name}_N{base_n}"
    run_dir = output_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    problem = build_salib_problem(scenario, grouped=grouped)
    factor_labels = (
        ordered_unique(problem["groups"]) if "groups" in problem
        else list(problem["names"])
    )
    n_evals = expected_sobol_rows(problem, base_n, calc_second_order=False)
    schedule = default_convergence_schedule(base_n)

    config = _build_solver_config(physics, nx, max_steps)

    print(f"\n{'=' * 68}")
    print(f"CONFIG: {physics.label}")
    print(f"  Run label      : {run_label}")
    print(f"  Scenario       : {scenario}")
    print(f"  Base N         : {base_n}")
    print(f"  Eff dimension  : {effective_dimension(problem)}")
    print(f"  Total evals    : {n_evals:,}")
    print(f"  Schedule       : {schedule}")
    print(f"  Workers        : {n_workers}")
    print(f"{'=' * 68}")

    t0 = time.time()

    # 1. Generate Sobol design in unit hypercube
    unit_design = generate_sobol_design(
        problem,
        base_n,
        calc_second_order=False,
        seed=seed,
    )

    # 2. Evaluate 1D solver on every design row
    evaluation = evaluate_sobol_design(
        unit_design,
        scenario,
        config,
        n_workers=n_workers,
        physical_output_policy="keep",
        verbose=not quiet,
    )

    # 3. Compute Sobol indices with convergence checkpoints
    sobol_results = compute_sobol_indices(
        problem,
        evaluation["outputs"],
        output_names=list(output_qois),
        base_sample_sizes=schedule,
        calc_second_order=False,
        num_resamples=num_resamples,
        conf_level=conf_level,
        seed=seed,
    )

    elapsed = time.time() - t0

    # 4. Summarise top total-order factors
    summary = summarize_top_total_indices(problem, sobol_results)

    # 5. Persist raw design + outputs as compressed NPZ
    np.savez_compressed(
        run_dir / f"{run_label}_design.npz",
        X_unit=unit_design,
        **{f"input_{k}": v for k, v in evaluation["prior_inputs"].items()},
        **{f"diag_{k}": v for k, v in evaluation["diagnostics"].items()},
        **{f"output_{k}": v for k, v in evaluation["outputs"].items()},
        error_type=evaluation["errors"]["error_type"],
        error_message=evaluation["errors"]["error_message"],
    )

    # 6. CSV of Sobol indices
    csv_rows = sobol_results_to_rows(problem, sobol_results)
    write_csv(csv_rows, run_dir / f"{run_label}_indices.csv")

    # 7. JSON manifest with full provenance
    manifest = {
        "run_label": run_label,
        "physics_config": {
            "name": physics.name,
            "label": physics.label,
            "conductivity_model": physics.conductivity_model,
            "creep_model": physics.creep_model,
            "nu_scaling": physics.nu_scaling,
            "grain_mode": physics.grain_mode,
        },
        "scenario": scenario,
        "grouped": grouped,
        "factor_labels": factor_labels,
        "parameter_names": list(problem["names"]),
        "base_samples": base_n,
        "schedule": schedule,
        "seed": seed,
        "n_workers": n_workers,
        "nx": nx,
        "max_steps": max_steps,
        "num_resamples": num_resamples,
        "conf_level": conf_level,
        "expected_evaluations": n_evals,
        "elapsed_seconds": round(elapsed, 1),
        "numerical_success_rate": float(
            np.nanmean(evaluation["outputs"]["numerical_success"])
        ),
        "physical_valid_rate": float(
            np.nanmean(evaluation["outputs"]["physical_flag"])
        ),
        "requested_outputs": list(output_qois),
        "top_total_order": summary,
    }
    manifest_path = run_dir / f"{run_label}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Print summary
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Numerical success: {manifest['numerical_success_rate']:.1%}")
    print(f"  Physical valid:    {manifest['physical_valid_rate']:.1%}")
    for qoi_name, top_rows in summary.items():
        print(f"\n  {qoi_name}:")
        for row in top_rows:
            print(
                f"    {row['factor']:25s}  "
                f"ST={row['ST']:.3f} +/- {row['ST_conf']:.3f}  "
                f"S1={row['S1']:.3f} +/- {row['S1_conf']:.3f}"
            )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production Sobol sensitivity analysis for the 1D Europa thermal model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--N", type=int, default=128,
        help="Base Sobol sample size (power of 2). Default: 128.",
    )
    parser.add_argument(
        "--config", default="baseline",
        help=(
            "Physics config to run. One of: baseline, improved, wattmeter, all. "
            "Default: baseline."
        ),
    )
    parser.add_argument(
        "--scenario", default="global_audited",
        help="Sobol scenario name from sobol_workflow. Default: global_audited.",
    )
    parser.add_argument(
        "--grouped", action="store_true",
        help="Use grouped Sobol indices (by physics block) instead of per-parameter.",
    )
    parser.add_argument(
        "--workers", type=int, default=6,
        help="Number of parallel worker processes. Default: 6.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Sobol scrambling / bootstrap seed. Default: 42.",
    )
    parser.add_argument(
        "--nx", type=int, default=31,
        help="Finite-difference grid size. Default: 31.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1500,
        help="Maximum solver steps per evaluation. Default: 1500.",
    )
    parser.add_argument(
        "--num-resamples", type=int, default=1000,
        help="Bootstrap resamples for SALib CIs. Default: 1000.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory. Default: autoresearch/experiments/results/sobol/.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-evaluation progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not is_power_of_two(args.N):
        raise ValueError(f"--N must be a power of two, received {args.N}")

    # Resolve configs to run
    if args.config == "all":
        configs_to_run = list(CONFIG_NAMES)
    elif args.config in PHYSICS_CONFIGS:
        configs_to_run = [args.config]
    else:
        valid = ", ".join(CONFIG_NAMES)
        raise ValueError(f"Unknown --config '{args.config}'. Valid: {valid}, all")

    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_manifests = {}
    for config_name in configs_to_run:
        physics = PHYSICS_CONFIGS[config_name]
        manifest = _run_single_config(
            physics,
            base_n=args.N,
            scenario=args.scenario,
            grouped=args.grouped,
            seed=args.seed,
            n_workers=args.workers,
            nx=args.nx,
            max_steps=args.max_steps,
            num_resamples=args.num_resamples,
            conf_level=0.95,
            output_qois=ANALYSIS_QOIS,
            output_dir=output_dir,
            quiet=args.quiet,
        )
        all_manifests[config_name] = manifest

    # Write combined manifest when running all configs
    if len(configs_to_run) > 1:
        combined_path = output_dir / "combined_manifest.json"
        with combined_path.open("w", encoding="utf-8") as fh:
            json.dump(all_manifests, fh, indent=2)
        print(f"\nCombined manifest: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
