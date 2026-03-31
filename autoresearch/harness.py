"""Experiment harness for autoresearch — run, score, log."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add Europa2D/src and EuropaProjectDJ/src to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'EuropaProjectDJ' / 'src'))

from objectives import compute_solver_score, compute_physics_score, compute_latitude_score


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class ExperimentHarness:
    """Run-score-log harness for autoresearch experiments."""

    # Reference solver parameters (equatorial, Howell defaults)
    _REF_PARAMS = {
        'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
        'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
        'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
        'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
    }
    _REF_T_SURF = 104.0
    _REF_Q_OCEAN = 0.025
    _REF_NX = 31
    _REF_DT = 5e12
    _REF_THICKNESS = 20e3

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or Path(__file__).parent)
        self.ref_dir = self.base_dir / 'reference'
        self.best_path = self.base_dir / 'best.json'
        self.log_path = self.base_dir / 'experiments.jsonl'

    def init(self):
        """Initialize reference artifacts and best.json."""
        self.ref_dir.mkdir(parents=True, exist_ok=True)

        # Solver baseline
        solver_result = self._run_single_solver()
        solver_score, solver_metrics = compute_solver_score(solver_result, solver_result)
        self._write_json(self.ref_dir / 'solver_ref.json', solver_result)

        # Physics baseline
        mc = self._run_mc_ensemble("uniform", n_samples=250, n_workers=8)
        physics_dict = self._mc_to_dict(mc)
        physics_score, physics_metrics = compute_physics_score(physics_dict)
        self._write_json(self.ref_dir / 'physics_ref.json', physics_metrics)

        # Latitude baseline
        latitude_score, latitude_metrics = self._run_latitude_experiment(250, 8)
        self._write_json(self.ref_dir / 'latitude_ref.json', latitude_metrics)

        best = {
            'solver': {'score': solver_score, 'metrics': solver_metrics},
            'physics': {'score': physics_score, 'metrics': physics_metrics},
            'latitude': {'score': latitude_score, 'metrics': latitude_metrics},
        }
        self._write_json(self.best_path, best)
        print("=== BASELINE INITIALIZED ===")
        print(f"  Solver score:   {solver_score:.4f}")
        print(f"  Physics score:  {physics_score:.4f}")
        print(f"  Latitude score: {latitude_score:.4f}")

    def run(self, mode: str, tag: str, n_samples: int = 250, n_workers: int = 8):
        """Run an experiment, score it, log the result."""
        try:
            if mode == 'solver':
                score, metrics = self._run_solver_experiment()
            elif mode == 'physics':
                score, metrics = self._run_physics_experiment(n_samples, n_workers)
            elif mode == 'latitude':
                score, metrics = self._run_latitude_experiment(n_samples, n_workers)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Compare to best
            best = self._load_best()
            prev_score = best.get(mode, {}).get('score', float('inf'))
            delta = score - prev_score
            improved = delta < 0

            # Log
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': mode,
                'tag': tag,
                'status': 'ok',
                'score': score,
                'delta': delta,
                'improved': improved,
                'metrics': metrics,
                'git_sha': self._get_git_sha(),
            }
            self._append_log(entry)

            # Update best if improved
            if improved:
                best[mode] = {'score': score, 'metrics': metrics}
                self._write_json(self.best_path, best)

            # Print structured result
            self._print_result(mode, tag, score, prev_score, delta, improved, metrics)

        except Exception:
            tb = traceback.format_exc()
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': mode,
                'tag': tag,
                'status': 'failed',
                'score': float('inf'),
                'error': tb[-500:],
            }
            self._append_log(entry)
            print(f"=== EXPERIMENT FAILED ===")
            print(f"Mode: {mode}")
            print(f"Tag: {tag}")
            print(f"Error:\n{tb}")

    # --- Solver mode ---

    def _run_solver_experiment(self):
        """Run solver 5 times, score against frozen reference."""
        ref_data = self._load_json(self.ref_dir / 'solver_ref.json')
        times = []
        last_result = None
        for _ in range(5):
            result = self._run_single_solver()
            times.append(result['time'])
            last_result = result
        last_result['time'] = float(np.median(times))
        return compute_solver_score(last_result, ref_data)

    def _run_single_solver(self):
        """Run one solver evaluation at reference params."""
        from latitude_profile import LatitudeProfile
        from axial_solver import AxialSolver2D

        profile = LatitudeProfile(
            T_eq=self._REF_T_SURF,
            epsilon_eq=self._REF_PARAMS['epsilon_0'],
            epsilon_pole=self._REF_PARAMS['epsilon_0'],
            q_ocean_mean=self._REF_Q_OCEAN,
            ocean_pattern="uniform",
        )
        solver = AxialSolver2D(
            n_lat=1, nx=self._REF_NX, dt=self._REF_DT,
            latitude_profile=profile, physics_params=dict(self._REF_PARAMS),
            use_convection=True, initial_thickness=self._REF_THICKNESS,
        )
        t0 = time.perf_counter()
        result = solver.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False)
        elapsed = time.perf_counter() - t0

        return {
            'time': elapsed,
            'steps': result['steps'],
            'T_2d': result['T_2d'],
            'H_profile_km': result['H_profile_km'],
        }

    # --- Physics mode ---

    def _run_physics_experiment(self, n_samples: int, n_workers: int):
        """Run physics-mode MC ensemble and score."""
        mc_results = self._run_mc_ensemble("uniform", n_samples=n_samples, n_workers=n_workers)
        result_dict = self._mc_to_dict(mc_results)
        return compute_physics_score(result_dict)

    # --- Latitude mode ---

    def _run_latitude_experiment(self, n_samples: int, n_workers: int):
        """Run 3 scenarios + 1D/2D calibration, then score."""
        scenario_configs = [
            ('uniform', 'uniform', None),
            ('polar', 'polar_enhanced', 0.455),
            ('equator', 'equator_enhanced', 0.4),
        ]
        scenarios = {}
        for name, pattern, q_star in scenario_configs:
            mc = self._run_mc_ensemble(pattern, n_samples=n_samples, n_workers=n_workers, q_star=q_star)
            scenarios[name] = self._mc_to_dict(mc)

        consistency_error = self._run_calibration_check()
        return compute_latitude_score(scenarios, consistency_error)

    def _run_calibration_check(self) -> float:
        """Fixed 1D/2D consistency check at equatorial reference params."""
        from latitude_profile import LatitudeProfile
        from axial_solver import AxialSolver2D
        from Solver import Thermal_Solver
        from Boundary_Conditions import FixedTemperature

        params = dict(self._REF_PARAMS)
        params['T_surf'] = self._REF_T_SURF

        # 1D solve
        bc_1d = FixedTemperature(temperature=self._REF_T_SURF)
        solver_1d = Thermal_Solver(
            nx=self._REF_NX, thickness=self._REF_THICKNESS, dt=1e11,
            surface_bc=bc_1d, use_convection=True, physics_params=params,
        )
        for _ in range(500):
            v = solver_1d.solve_step(self._REF_Q_OCEAN)
            if abs(v) < 1e-12:
                break
        H_1d = solver_1d.H / 1000.0

        # 2D single-column solve
        profile = LatitudeProfile(
            T_eq=self._REF_T_SURF,
            epsilon_eq=self._REF_PARAMS['epsilon_0'],
            epsilon_pole=self._REF_PARAMS['epsilon_0'],
            q_ocean_mean=self._REF_Q_OCEAN,
            ocean_pattern="uniform",
        )
        solver_2d = AxialSolver2D(
            n_lat=1, nx=self._REF_NX, dt=self._REF_DT,
            latitude_profile=profile, physics_params=dict(self._REF_PARAMS),
            use_convection=True, initial_thickness=self._REF_THICKNESS,
        )
        result_2d = solver_2d.run_to_equilibrium(threshold=1e-12, max_steps=500, verbose=False)
        H_2d = float(result_2d['H_profile_km'][0])

        if H_1d <= 0:
            return 1.0
        return abs(H_2d - H_1d) / H_1d

    def _run_mc_ensemble(self, ocean_pattern: str, n_samples: int, n_workers: int,
                         q_star: Optional[float] = None):
        """Run one MC ensemble via MonteCarloRunner2D."""
        from monte_carlo_2d import MonteCarloRunner2D

        runner = MonteCarloRunner2D(
            n_iterations=n_samples,
            seed=42,
            n_workers=n_workers,
            ocean_pattern=ocean_pattern,
            q_star=q_star,
            verbose=False,
        )
        return runner.run()

    def _mc_to_dict(self, mc) -> Dict[str, Any]:
        """Extract scoring-relevant fields from MonteCarloResults2D."""
        return {
            'D_cond_profiles': np.asarray(mc.D_cond_profiles),
            'D_conv_profiles': np.asarray(mc.D_conv_profiles),
            'Ra_profiles': np.asarray(mc.Ra_profiles),
            'H_profiles': np.asarray(mc.H_profiles),
            'n_valid': mc.n_valid,
            'n_iterations': mc.n_iterations,
            'latitudes_deg': np.asarray(mc.latitudes_deg),
        }

    # --- Utilities ---

    def _load_best(self) -> Dict:
        if self.best_path.exists():
            return self._load_json(self.best_path)
        return {}

    def _get_git_sha(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            return result.stdout.strip()
        except Exception:
            return 'unknown'

    def _append_log(self, entry: Dict):
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry, cls=_NumpyEncoder) + '\n')

    def _write_json(self, path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, cls=_NumpyEncoder, indent=2)

    def _load_json(self, path: Path) -> Dict:
        with open(path) as f:
            return json.load(f)

    def _print_result(self, mode, tag, score, prev, delta, improved, metrics):
        status = "IMPROVED" if improved else "no improvement"
        print(f"=== EXPERIMENT RESULT ===")
        print(f"Mode: {mode}")
        print(f"Tag: {tag}")
        print(f"Score: {score:.4f} (prev: {prev:.4f}, delta: {delta:.4f}, {status})")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description='Autoresearch experiment harness')
    parser.add_argument('--init', action='store_true', help='Initialize reference artifacts')
    parser.add_argument('--mode', choices=['solver', 'physics', 'latitude'], help='Experiment mode')
    parser.add_argument('--tag', default='unnamed', help='Experiment tag/description')
    parser.add_argument('--n-samples', type=int, default=250, help='MC samples per ensemble')
    parser.add_argument('--n-workers', type=int, default=8, help='Parallel workers')
    args = parser.parse_args()

    harness = ExperimentHarness()

    if args.init:
        harness.init()
    elif args.mode:
        harness.run(args.mode, args.tag, args.n_samples, args.n_workers)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
