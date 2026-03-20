"""
Monte Carlo framework for 2D axisymmetric Europa ice shell model.

Each MC iteration samples shared parameters, builds a LatitudeProfile,
runs AxialSolver2D to equilibrium, and collects the H(phi) profile.
"""
import sys
import os

# Prevent thread thrashing: force single-threaded BLAS/LAPACK per worker.
# Without this, N workers × M OpenBLAS threads can saturate the CPU with
# context-switching overhead, causing apparent deadlocks on Windows.
# Must be set before importing NumPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
import time
import multiprocessing as mp
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from functools import partial

from constants import Thermal, Planetary
from latitude_profile import LatitudeProfile, OceanPattern
from latitude_sampler import LatitudeParameterSampler
from axial_solver import AxialSolver2D
from profile_diagnostics import band_mean_samples, LOW_LAT_BAND, HIGH_LAT_BAND


@dataclass(frozen=True)
class MonteCarloResults2D:
    """Immutable container for 2D Monte Carlo results."""
    H_profiles: npt.NDArray[np.float64]         # (n_valid, n_lat)
    latitudes_deg: npt.NDArray[np.float64]       # (n_lat,)
    n_iterations: int
    n_valid: int
    H_median: npt.NDArray[np.float64]            # (n_lat,)
    H_mean: npt.NDArray[np.float64]              # (n_lat,)
    H_sigma_low: npt.NDArray[np.float64]         # (n_lat,)
    H_sigma_high: npt.NDArray[np.float64]        # (n_lat,)
    runtime_seconds: float
    ocean_pattern: str
    ocean_amplitude: float
    T_floor: float = 46.0  # Ashkenazy (2019) annual-mean polar floor at Q=0.05 W/m²
    q_star: float = 0.0
    mantle_tidal_fraction: float = 0.5
    D_cond_profiles: Optional[npt.NDArray[np.float64]] = None   # (n_valid, n_lat)
    D_conv_profiles: Optional[npt.NDArray[np.float64]] = None
    Ra_profiles: Optional[npt.NDArray[np.float64]] = None
    Nu_profiles: Optional[npt.NDArray[np.float64]] = None
    lid_fraction_profiles: Optional[npt.NDArray[np.float64]] = None

    # D_cond aggregate statistics (n_lat,) — same shape as H_median
    D_cond_median: Optional[npt.NDArray[np.float64]] = None
    D_cond_mean: Optional[npt.NDArray[np.float64]] = None
    D_cond_sigma_low: Optional[npt.NDArray[np.float64]] = None   # 15.87th percentile
    D_cond_sigma_high: Optional[npt.NDArray[np.float64]] = None  # 84.13th percentile

    # D_conv aggregate statistics (n_lat,)
    D_conv_median: Optional[npt.NDArray[np.float64]] = None
    D_conv_mean: Optional[npt.NDArray[np.float64]] = None
    D_conv_sigma_low: Optional[npt.NDArray[np.float64]] = None
    D_conv_sigma_high: Optional[npt.NDArray[np.float64]] = None

    # Convective fraction aggregate statistics (n_lat,)
    # Fraction of shell thickness occupied by the convective sublayer: D_conv / H_total
    conv_fraction_median: Optional[npt.NDArray[np.float64]] = None
    conv_fraction_mean: Optional[npt.NDArray[np.float64]] = None
    conv_fraction_sigma_low: Optional[npt.NDArray[np.float64]] = None
    conv_fraction_sigma_high: Optional[npt.NDArray[np.float64]] = None

    # Latitude-band distributions: (n_valid,) — one value per MC sample
    # Low band = area-weighted mean over 0-10° latitude
    # High band = area-weighted mean over 80-90° latitude
    H_low_band: Optional[npt.NDArray[np.float64]] = None
    H_high_band: Optional[npt.NDArray[np.float64]] = None
    D_cond_low_band: Optional[npt.NDArray[np.float64]] = None
    D_cond_high_band: Optional[npt.NDArray[np.float64]] = None


def _run_single_2d_sample(
    sample_id: int,
    base_seed: int,
    n_lat: int,
    nx: int,
    dt: float,
    use_convection: bool,
    max_steps: int,
    eq_threshold: float,
    initial_thickness: float,
    ocean_pattern: str,
    ocean_amplitude: Optional[float],
    q_star: Optional[float],
    rannacher_steps: int,
    coordinate_system: str,
) -> Optional[Dict[str, Any]]:
    """Worker function for one 2D MC iteration."""
    try:
        sampler = LatitudeParameterSampler(
            seed=base_seed + sample_id,
            ocean_pattern=ocean_pattern,
            ocean_amplitude=ocean_amplitude,
            q_star=q_star,
        )
        shared_params, profile = sampler.sample()
        D_H2O = shared_params['D_H2O']

        # Warm start: conductive estimate with mild convection correction.
        # Convective shells are only ~1.3-1.5x thicker than the conductive
        # estimate; the old 8x factor overshot to the 100 km clip for every
        # sample, wasting hundreds of convergence steps.
        q_mean = profile.q_ocean_mean
        if q_mean > 0:
            k_mean = float(Thermal.conductivity(190.0))
            H_guess = k_mean * 170.0 / q_mean
            if use_convection:
                H_guess *= 1.5
            H_guess = np.clip(H_guess, 5e3, 80e3)
        else:
            H_guess = initial_thickness

        solver = AxialSolver2D(
            n_lat=n_lat,
            nx=nx,
            dt=dt,
            latitude_profile=profile,
            physics_params=shared_params,
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

        H_km = result['H_profile_km']
        D_H2O_km = D_H2O / 1000.0

        # Filter: reject if >50% of columns are non-physical
        valid_mask = (H_km > 0.5) & (H_km < D_H2O_km * 0.99) & (H_km < 200)
        if np.sum(valid_mask) < len(H_km) * 0.5:
            return None

        # For invalid columns, interpolate from neighbors
        if not np.all(valid_mask):
            lats = np.degrees(solver.latitudes)
            H_km = np.interp(
                lats, lats[valid_mask], H_km[valid_mask],
            )

        # Extract diagnostics
        diag = result['diagnostics']
        D_cond = np.array([d['D_cond_km'] for d in diag])
        D_conv = np.array([d['D_conv_km'] for d in diag])
        Ra = np.array([d['Ra'] for d in diag])
        Nu = np.array([d['Nu'] for d in diag])
        lid_frac = np.array([d['lid_fraction'] for d in diag])

        return {
            'H_km': H_km,
            'D_cond_km': D_cond,
            'D_conv_km': D_conv,
            'Ra': Ra,
            'Nu': Nu,
            'lid_fraction': lid_frac,
        }

    except Exception:
        return None


class MonteCarloRunner2D:
    """Monte Carlo runner for 2D axisymmetric model."""

    def __init__(
        self,
        n_iterations: int = 100,
        seed: Optional[int] = None,
        n_workers: Optional[int] = None,
        n_lat: int = 19,
        nx: int = 31,
        dt: float = 5e12,
        use_convection: bool = True,
        max_steps: int = 500,
        eq_threshold: float = 1e-12,
        initial_thickness: float = 20e3,
        ocean_pattern: OceanPattern = "uniform",
        ocean_amplitude: Optional[float] = None,
        q_star: Optional[float] = None,
        T_floor: float = 46.0,
        mantle_tidal_fraction: float = 0.5,
        verbose: bool = True,
        rannacher_steps: int = 4,
        coordinate_system: str = 'auto',
    ):
        self.n_iterations = n_iterations
        self.seed = seed if seed is not None else int(time.time())
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.n_lat = n_lat
        self.nx = nx
        self.dt = dt
        self.use_convection = use_convection
        self.max_steps = max_steps
        self.eq_threshold = eq_threshold
        self.initial_thickness = initial_thickness
        self.ocean_pattern = ocean_pattern
        self.ocean_amplitude = ocean_amplitude
        self.q_star = q_star
        self.T_floor = T_floor
        self.mantle_tidal_fraction = mantle_tidal_fraction
        self.verbose = verbose
        self.rannacher_steps = rannacher_steps
        self.coordinate_system = coordinate_system

    def run(self) -> MonteCarloResults2D:
        if self.verbose:
            print("=" * 60)
            print("2D MONTE CARLO: Europa Ice Shell Thickness Profile")
            print("=" * 60)
            print(f"Iterations: {self.n_iterations}, Workers: {self.n_workers}")
            print(f"Columns: {self.n_lat}, Nodes/col: {self.nx}")
            print("-" * 60)

        start_time = time.time()

        worker = partial(
            _run_single_2d_sample,
            base_seed=self.seed,
            n_lat=self.n_lat,
            nx=self.nx,
            dt=self.dt,
            use_convection=self.use_convection,
            max_steps=self.max_steps,
            eq_threshold=self.eq_threshold,
            initial_thickness=self.initial_thickness,
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=self.ocean_amplitude,
            q_star=self.q_star,
            rannacher_steps=self.rannacher_steps,
            coordinate_system=self.coordinate_system,
        )

        # Sequential or parallel execution
        if self.n_workers > 1:
            with mp.Pool(self.n_workers) as pool:
                chunksize = max(10, self.n_iterations // (self.n_workers * 4))
                results = []
                for i, result in enumerate(pool.imap_unordered(worker, range(self.n_iterations), chunksize=chunksize)):
                    results.append(result)
                    if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                        valid = sum(1 for r in results if r is not None)
                        print(f"  Progress: {100 * (i + 1) / self.n_iterations:5.1f}% | Valid: {valid}/{i + 1}")
        else:
            results = []
            for i in range(self.n_iterations):
                results.append(worker(i))
                if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                    valid = sum(1 for r in results if r is not None)
                    print(f"  Progress: {100 * (i + 1) / self.n_iterations:5.1f}% | Valid: {valid}/{i + 1}")

        runtime = time.time() - start_time
        valid_results = [r for r in results if r is not None]

        if len(valid_results) == 0:
            raise RuntimeError("No valid 2D solutions. Check parameter distributions.")

        # Stack results
        H_profiles = np.array([r['H_km'] for r in valid_results])
        D_cond = np.array([r['D_cond_km'] for r in valid_results])
        D_conv = np.array([r['D_conv_km'] for r in valid_results])
        Ra = np.array([r['Ra'] for r in valid_results])
        Nu = np.array([r['Nu'] for r in valid_results])
        lid_frac = np.array([r['lid_fraction'] for r in valid_results])

        latitudes_deg = np.linspace(0, 90, self.n_lat)

        # D_cond aggregate statistics
        D_cond_median = np.median(D_cond, axis=0)
        D_cond_mean = np.mean(D_cond, axis=0)
        D_cond_sigma_low = np.percentile(D_cond, 15.87, axis=0)
        D_cond_sigma_high = np.percentile(D_cond, 84.13, axis=0)

        # D_conv aggregate statistics
        D_conv_median = np.median(D_conv, axis=0)
        D_conv_mean = np.mean(D_conv, axis=0)
        D_conv_sigma_low = np.percentile(D_conv, 15.87, axis=0)
        D_conv_sigma_high = np.percentile(D_conv, 84.13, axis=0)

        # Latitude-band mean distributions: (n_valid,)
        H_low_band = band_mean_samples(latitudes_deg, H_profiles, LOW_LAT_BAND)
        H_high_band = band_mean_samples(latitudes_deg, H_profiles, HIGH_LAT_BAND)
        D_cond_low_band = band_mean_samples(latitudes_deg, D_cond, LOW_LAT_BAND)
        D_cond_high_band = band_mean_samples(latitudes_deg, D_cond, HIGH_LAT_BAND)

        # Convective fraction: D_conv / H_total per sample, then aggregate
        conv_fraction_stack = np.where(
            H_profiles > 0, D_conv / H_profiles, 0.0
        )
        conv_fraction_median = np.median(conv_fraction_stack, axis=0)
        conv_fraction_mean = np.mean(conv_fraction_stack, axis=0)
        conv_fraction_sigma_low = np.percentile(conv_fraction_stack, 15.87, axis=0)
        conv_fraction_sigma_high = np.percentile(conv_fraction_stack, 84.13, axis=0)

        _meta_profile = LatitudeProfile(
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=self.ocean_amplitude,
            q_star=self.q_star,
            mantle_tidal_fraction=self.mantle_tidal_fraction,
            T_floor=self.T_floor,
            strict_q_star=False,
        )

        mc_results = MonteCarloResults2D(
            H_profiles=H_profiles,
            latitudes_deg=latitudes_deg,
            n_iterations=self.n_iterations,
            n_valid=len(valid_results),
            H_median=np.percentile(H_profiles, 50, axis=0),
            H_mean=np.mean(H_profiles, axis=0),
            H_sigma_low=np.percentile(H_profiles, 15.87, axis=0),
            H_sigma_high=np.percentile(H_profiles, 84.13, axis=0),
            runtime_seconds=runtime,
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=_meta_profile.resolved_ocean_amplitude(),
            T_floor=self.T_floor,
            q_star=_meta_profile.resolved_q_star(),
            mantle_tidal_fraction=self.mantle_tidal_fraction,
            D_cond_profiles=D_cond,
            D_conv_profiles=D_conv,
            Ra_profiles=Ra,
            Nu_profiles=Nu,
            lid_fraction_profiles=lid_frac,
            D_cond_median=D_cond_median,
            D_cond_mean=D_cond_mean,
            D_cond_sigma_low=D_cond_sigma_low,
            D_cond_sigma_high=D_cond_sigma_high,
            D_conv_median=D_conv_median,
            D_conv_mean=D_conv_mean,
            D_conv_sigma_low=D_conv_sigma_low,
            D_conv_sigma_high=D_conv_sigma_high,
            conv_fraction_median=conv_fraction_median,
            conv_fraction_mean=conv_fraction_mean,
            conv_fraction_sigma_low=conv_fraction_sigma_low,
            conv_fraction_sigma_high=conv_fraction_sigma_high,
            H_low_band=H_low_band,
            H_high_band=H_high_band,
            D_cond_low_band=D_cond_low_band,
            D_cond_high_band=D_cond_high_band,
        )

        if self.verbose:
            print("-" * 60)
            print(f"Valid: {mc_results.n_valid}/{self.n_iterations}")
            print(f"Runtime: {runtime:.1f}s")
            print(f"H range: [{mc_results.H_median.min():.1f}, {mc_results.H_median.max():.1f}] km (median)")
            print("=" * 60)

        return mc_results


def save_results_2d(results: MonteCarloResults2D, filepath: str) -> None:
    """Save 2D MC results to NumPy archive."""
    save_dict = {
        'H_profiles': results.H_profiles,
        'latitudes_deg': results.latitudes_deg,
        'n_iterations': results.n_iterations,
        'n_valid': results.n_valid,
        'H_median': results.H_median,
        'H_mean': results.H_mean,
        'H_sigma_low': results.H_sigma_low,
        'H_sigma_high': results.H_sigma_high,
        'runtime_seconds': results.runtime_seconds,
        'ocean_pattern': np.array(results.ocean_pattern),
        'ocean_amplitude': results.ocean_amplitude,
        'T_floor': results.T_floor,
        'q_star': results.q_star,
        'mantle_tidal_fraction': results.mantle_tidal_fraction,
    }
    optional_arrays = [
        'D_cond_profiles', 'D_conv_profiles', 'Ra_profiles', 'Nu_profiles', 'lid_fraction_profiles',
        'D_cond_median', 'D_cond_mean', 'D_cond_sigma_low', 'D_cond_sigma_high',
        'D_conv_median', 'D_conv_mean', 'D_conv_sigma_low', 'D_conv_sigma_high',
        'conv_fraction_median', 'conv_fraction_mean', 'conv_fraction_sigma_low', 'conv_fraction_sigma_high',
        'H_low_band', 'H_high_band', 'D_cond_low_band', 'D_cond_high_band',
    ]
    for name in optional_arrays:
        val = getattr(results, name)
        if val is not None:
            save_dict[name] = val

    np.savez(filepath, **save_dict)
    print(f"Results saved to: {filepath}")
