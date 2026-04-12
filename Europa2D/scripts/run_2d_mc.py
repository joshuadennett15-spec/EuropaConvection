"""Full 2D Monte Carlo runs for literature-backed scenarios."""
import argparse
import multiprocessing as mp
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src

from literature_scenarios import DEFAULT_SCENARIO, get_scenario, list_scenarios
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run 2D Monte Carlo for literature scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["all", *list_scenarios()],
        default=DEFAULT_SCENARIO,
        help="Scenario preset to run. Use 'all' to loop over all literature presets.",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Number of Monte Carlo iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--n-lat", type=int, default=37, help="Number of latitude columns.")
    parser.add_argument("--nx", type=int, default=31, help="Radial nodes per column.")
    parser.add_argument("--dt", type=float, default=1e12, help="Time step in seconds.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Maximum solver steps per sample.")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of worker processes.",
    )
    parser.add_argument("--q-tidal-scale", type=float, default=1.20, help="Scale factor applied to ocean heat flux.")
    parser.add_argument(
        "--grain-mode",
        choices=["global", "strain"],
        default="global",
        help="Grain latitude mode: 'global' (benchmark) or 'strain' (recrystallization).",
    )
    return parser.parse_args()


def run_mc_scenario(
    scenario_name: str,
    results_dir: str,
    iterations: int,
    seed: int,
    n_workers: int,
    n_lat: int,
    nx: int,
    dt: float,
    max_steps: int,
    grain_latitude_mode: str = "global",
    q_tidal_scale: float = 1.0,
) -> str:
    """Run one Monte Carlo scenario and return the result path."""
    scenario = get_scenario(scenario_name)
    print(f"\n=== {scenario.name}: {scenario.citation} ===")
    print(f"  {scenario.description}")
    if grain_latitude_mode != "global":
        print(f"  grain_latitude_mode: {grain_latitude_mode}")

    runner = MonteCarloRunner2D(
        n_iterations=iterations,
        seed=seed,
        n_workers=n_workers,
        n_lat=n_lat,
        nx=nx,
        dt=dt,
        use_convection=True,
        max_steps=max_steps,
        ocean_pattern=scenario.ocean_pattern,
        q_star=scenario.q_star if scenario.q_star > 0 else None,
        grain_latitude_mode=grain_latitude_mode,
        q_tidal_scale=q_tidal_scale,
    )
    results = runner.run()

    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"mc_2d_{scenario.name}_{iterations}.npz")
    save_results_2d(results, output_path)
    return output_path


if __name__ == "__main__":
    mp.freeze_support()
    args = _parse_args()

    results_dir = os.path.join(_PROJECT_DIR, "results")
    scenario_names = list_scenarios() if args.scenario == "all" else (args.scenario,)

    saved_paths = []
    for offset, scenario_name in enumerate(scenario_names):
        output_path = run_mc_scenario(
            scenario_name=scenario_name,
            results_dir=results_dir,
            iterations=args.iterations,
            seed=args.seed + offset * 10000,
            n_workers=args.n_workers,
            n_lat=args.n_lat,
            nx=args.nx,
            dt=args.dt,
            max_steps=args.max_steps,
            grain_latitude_mode=args.grain_mode,
        )
        saved_paths.append(output_path)

    print("\nSaved MC outputs:")
    for path in saved_paths:
        print(f"  - {path}")
