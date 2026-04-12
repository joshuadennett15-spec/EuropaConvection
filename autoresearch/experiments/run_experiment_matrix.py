"""Phase 5 experiment matrix: closure x preset x ocean x grain.

Stages:
  0  smoke    --samples 10   (verify all combos run)
  1  pilot    --samples 50   (closure screen)
  2  core     --samples 150  (full grid, thesis-facing)
  3  rerun    --samples 500  (shortlisted combos only)

Each run records Phase 6 standardised diagnostics:
  D_cond@35, H_total, D_cond, D_conv, convective fraction,
  band means (low 0-10, high 80-90), full metadata.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))

from monte_carlo_2d import MonteCarloRunner2D
from profile_diagnostics import band_mean_samples, LOW_LAT_BAND, HIGH_LAT_BAND

# ── Grid dimensions ──────────────────────────────────────────────────────

CLOSURES = ["green", "isoviscous_benchmark"]
PRESETS = ["ashkenazy_low_q", "ashkenazy_high_q"]
OCEAN_PATTERNS = [
    ("uniform", None),
    ("polar_enhanced", 0.455),
    ("equator_enhanced", 0.4),
]
GRAIN_CENTERS_MM = [0.6, 1.0]

# ── Physics feature-flag dimensions ──────────────────────────────────
PHYSICS_GRID = {
    "conductivity": ["Howell", "Carnahan"],
    "creep": ["diffusion", "composite_gbs"],
    "nu_scaling": ["green", "dv2021"],
    "grain_mode": ["sampled", "wattmeter"],
}

JUNO_LATITUDE_DEG = 35.0


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _find_lat_index(lats_deg, target_deg):
    return int(np.argmin(np.abs(lats_deg - target_deg)))


def _percentile_stats(arr, axis=0):
    """Median + 1-sigma percentile envelope."""
    return {
        'median': np.median(arr, axis=axis),
        'p16': np.percentile(arr, 15.87, axis=axis),
        'p84': np.percentile(arr, 84.13, axis=axis),
        'mean': np.mean(arr, axis=axis),
        'std': np.std(arr, axis=axis),
    }


def run_combo(closure, preset, ocean_name, ocean_pattern, q_star,
              grain_mm, n_samples, seed, n_workers, n_lat,
              conductivity_model="Carnahan", creep_model="diffusion",
              nu_scaling_flag="green", grain_mode="sampled"):
    """Run one experiment combo and return standardised diagnostics."""
    # nu_scaling_flag overrides the closure-based default when the physics
    # grid is active; fall back to closure for backward compat.
    effective_nu = nu_scaling_flag if nu_scaling_flag else closure
    runner = MonteCarloRunner2D(
        n_iterations=n_samples,
        seed=seed,
        n_workers=n_workers,
        n_lat=n_lat,
        nx=31,
        ocean_pattern=ocean_pattern,
        q_star=q_star,
        surface_preset=preset,
        grain_center_mm=grain_mm,
        nu_scaling=effective_nu,
        conductivity_model=conductivity_model,
        creep_model=creep_model,
        grain_mode=grain_mode,
        verbose=False,
        max_steps=1500,
        eq_threshold=1e-12,
    )

    t0 = time.time()
    mc = runner.run()
    elapsed = time.time() - t0

    lats_deg = mc.latitudes_deg
    idx_35 = _find_lat_index(lats_deg, JUNO_LATITUDE_DEG)

    # Per-sample diagnostics at Juno latitude
    D_cond_35 = mc.D_cond_profiles[:, idx_35]
    D_conv_35 = mc.D_conv_profiles[:, idx_35]
    H_35 = mc.H_profiles[:, idx_35]

    # Convective fraction: D_conv / H_total
    conv_frac = mc.D_conv_profiles / np.clip(mc.H_profiles, 1e-6, None)

    # Band means (0-10 deg, 80-90 deg)
    D_cond_low = band_mean_samples(lats_deg, mc.D_cond_profiles, LOW_LAT_BAND)
    D_cond_high = band_mean_samples(lats_deg, mc.D_cond_profiles, HIGH_LAT_BAND)
    H_low = band_mean_samples(lats_deg, mc.H_profiles, LOW_LAT_BAND)
    H_high = band_mean_samples(lats_deg, mc.H_profiles, HIGH_LAT_BAND)

    # Fraction of samples that are convecting at each latitude
    convecting_frac = np.mean(mc.D_conv_profiles > 0.5, axis=0)

    diagnostics = {
        # Metadata
        'closure': closure,
        'surface_preset': preset,
        'ocean_pattern': ocean_name,
        'grain_center_mm': grain_mm,
        'conductivity_model': conductivity_model,
        'creep_model': creep_model,
        'nu_scaling': effective_nu,
        'grain_mode': grain_mode,
        'seed': seed,
        'n_samples': n_samples,
        'n_valid': int(mc.n_valid),
        'n_lat': int(len(lats_deg)),
        'runtime_seconds': elapsed,

        # Latitude grid
        'latitudes_deg': lats_deg,

        # D_cond @ Juno latitude
        'D_cond_35_median': float(np.median(D_cond_35)),
        'D_cond_35_mean': float(np.mean(D_cond_35)),
        'D_cond_35_std': float(np.std(D_cond_35)),

        # H_total @ Juno latitude
        'H_35_median': float(np.median(H_35)),

        # D_conv @ Juno latitude
        'D_conv_35_median': float(np.median(D_conv_35)),

        # Convective fraction @ Juno latitude
        'conv_frac_35_median': float(np.median(conv_frac[:, idx_35])),

        # Latitude profiles (median + 1-sigma)
        'H_profile': _percentile_stats(mc.H_profiles),
        'D_cond_profile': _percentile_stats(mc.D_cond_profiles),
        'D_conv_profile': _percentile_stats(mc.D_conv_profiles),
        'conv_frac_profile': _percentile_stats(conv_frac),

        # Band means
        'D_cond_low_band_median': float(np.median(D_cond_low)),
        'D_cond_high_band_median': float(np.median(D_cond_high)),
        'H_low_band_median': float(np.median(H_low)),
        'H_high_band_median': float(np.median(H_high)),

        # Convecting fraction per latitude
        'convecting_fraction': convecting_frac,
    }

    return diagnostics


def make_tag(closure, preset, ocean, grain_mm,
             conductivity="Carnahan", creep="diffusion",
             nu_scaling="green", grain_mode="sampled"):
    preset_short = preset.replace('ashkenazy_', 'ash_')
    grain_tag = f'g{grain_mm:.1f}'.replace('.', '')
    # Abbreviate physics flags for readable filenames
    cond_tag = conductivity[:3].lower()   # "how" / "car"
    creep_tag = "gbs" if creep == "composite_gbs" else "dif"
    nu_tag = nu_scaling[:3].lower()       # "gre" / "dv2"
    gm_tag = "wm" if grain_mode == "wattmeter" else "sp"
    return (f'{closure}__{preset_short}__{ocean}__{grain_tag}'
            f'__{cond_tag}_{creep_tag}_{nu_tag}_{gm_tag}')


def main():
    parser = argparse.ArgumentParser(description='Phase 5 experiment matrix')
    parser.add_argument('--samples', type=int, default=10,
                        help='MC samples per combo (10=smoke, 50=pilot, 150=core)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--n-lat', type=int, default=19)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--closures', nargs='+', default=None,
                        help=f'Subset of closures (default: all). Valid: {CLOSURES}')
    parser.add_argument('--presets', nargs='+', default=None,
                        help=f'Subset of presets (default: all). Valid: {[p for p in ["ashkenazy_low_q", "ashkenazy_high_q"]]}')
    parser.add_argument('--oceans', nargs='+', default=None,
                        help='Subset of ocean patterns (default: all). Valid: uniform, polar_enhanced, equator_enhanced')
    parser.add_argument('--grains', nargs='+', type=float, default=None,
                        help=f'Subset of grain centers in mm (default: all). Valid: {GRAIN_CENTERS_MM}')
    parser.add_argument('--conductivities', nargs='+', default=None,
                        help=f'Subset of conductivity models. Valid: {PHYSICS_GRID["conductivity"]}')
    parser.add_argument('--creeps', nargs='+', default=None,
                        help=f'Subset of creep models. Valid: {PHYSICS_GRID["creep"]}')
    parser.add_argument('--nu-scalings', nargs='+', default=None,
                        help=f'Subset of Nu scaling laws. Valid: {PHYSICS_GRID["nu_scaling"]}')
    parser.add_argument('--grain-modes', nargs='+', default=None,
                        help=f'Subset of grain modes. Valid: {PHYSICS_GRID["grain_mode"]}')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: override samples=5, limit combos')
    parser.add_argument('--combos', type=int, default=None,
                        help='Limit to first N combos (for smoke testing)')
    args = parser.parse_args()

    if args.smoke:
        args.samples = min(args.samples, 5)

    closures = args.closures or CLOSURES
    presets = args.presets or PRESETS
    ocean_cfgs = OCEAN_PATTERNS
    if args.oceans:
        ocean_map = {name: (name, qs) for name, qs in OCEAN_PATTERNS}
        ocean_cfgs = [(name, ocean_map[name][1]) for name in args.oceans]
    grains = args.grains or GRAIN_CENTERS_MM

    conductivities = args.conductivities or PHYSICS_GRID["conductivity"]
    creeps = args.creeps or PHYSICS_GRID["creep"]
    nu_scalings = args.nu_scalings or PHYSICS_GRID["nu_scaling"]
    grain_modes = args.grain_modes or PHYSICS_GRID["grain_mode"]

    stage = 'smoke' if args.samples <= 10 else 'pilot' if args.samples <= 50 else 'core' if args.samples <= 200 else 'rerun'
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / f'matrix_{stage}_s{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(product(closures, presets, ocean_cfgs, grains,
                          conductivities, creeps, nu_scalings, grain_modes))
    if args.combos is not None:
        combos = combos[:args.combos]
    n_combos = len(combos)

    print(f'\n{"="*70}')
    print(f'  PHASE 5 EXPERIMENT MATRIX — {stage.upper()}')
    print(f'  {n_combos} combos x {args.samples} samples, seed={args.seed}')
    print(f'  Output: {out_dir}')
    print(f'{"="*70}\n')

    summary_rows = []

    for idx, (closure, preset, (ocean_name, q_star), grain_mm,
              cond_model, creep_model, nu_scale, gm) in enumerate(combos):
        tag = make_tag(closure, preset, ocean_name, grain_mm,
                       cond_model, creep_model, nu_scale, gm)
        out_path = out_dir / f'{tag}.json'

        # Skip if already done
        if out_path.exists():
            print(f'[{idx+1}/{n_combos}] SKIP {tag} (exists)')
            with open(out_path) as f:
                diag = json.load(f)
            summary_rows.append(diag)
            continue

        print(f'[{idx+1}/{n_combos}] {tag} ...', end=' ', flush=True)

        try:
            diag = run_combo(
                closure=closure,
                preset=preset,
                ocean_name=ocean_name,
                ocean_pattern=ocean_name,
                q_star=q_star,
                grain_mm=grain_mm,
                n_samples=args.samples,
                seed=args.seed,
                n_workers=args.workers,
                n_lat=args.n_lat,
                conductivity_model=cond_model,
                creep_model=creep_model,
                nu_scaling_flag=nu_scale,
                grain_mode=gm,
            )
            diag['tag'] = tag
            diag['timestamp'] = datetime.now(timezone.utc).isoformat()

            with open(out_path, 'w') as f:
                json.dump(diag, f, cls=_Enc, indent=2)

            print(f'D_cond@35={diag["D_cond_35_median"]:.1f} km  '
                  f'H@35={diag["H_35_median"]:.1f} km  '
                  f'D_conv@35={diag["D_conv_35_median"]:.1f} km  '
                  f'yield={diag["n_valid"]}/{diag["n_samples"]}  '
                  f'({diag["runtime_seconds"]:.0f}s)')

            summary_rows.append(diag)

        except Exception as e:
            print(f'FAILED: {e}')
            summary_rows.append({
                'tag': tag, 'closure': closure, 'surface_preset': preset,
                'ocean_pattern': ocean_name, 'grain_center_mm': grain_mm,
                'conductivity_model': cond_model, 'creep_model': creep_model,
                'nu_scaling': nu_scale, 'grain_mode': gm,
                'error': str(e),
            })

    # ── Summary table ────────────────────────────────────────────────────

    print(f'\n{"="*70}')
    print(f'  SUMMARY — {stage.upper()} ({n_combos} combos)')
    print(f'{"="*70}')
    print(f'\n  {"Tag":50s}  {"Dcond35":>7s}  {"H35":>6s}  {"Dconv35":>7s}  {"CF35":>5s}  {"Yield":>6s}')
    print(f'  {"-"*50}  {"-"*7}  {"-"*6}  {"-"*7}  {"-"*5}  {"-"*6}')

    for row in summary_rows:
        if 'error' in row:
            print(f'  {row["tag"]:50s}  ** FAILED: {row["error"][:30]} **')
            continue
        print(f'  {row["tag"]:50s}  '
              f'{row["D_cond_35_median"]:7.1f}  '
              f'{row["H_35_median"]:6.1f}  '
              f'{row["D_conv_35_median"]:7.1f}  '
              f'{row["conv_frac_35_median"]:5.2f}  '
              f'{row["n_valid"]:3d}/{row["n_samples"]:3d}')

    # Save summary
    summary_path = out_dir / 'summary.json'
    # Strip non-serializable arrays from summary
    summary_clean = []
    for row in summary_rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, dict) and 'median' in v:
                # Profile stats — store only scalar summaries
                clean[k + '_median_mean'] = float(np.mean(v['median']))
            elif isinstance(v, np.ndarray):
                clean[k] = v.tolist()
            else:
                clean[k] = v
        summary_clean.append(clean)

    with open(summary_path, 'w') as f:
        json.dump(summary_clean, f, cls=_Enc, indent=2)
    print(f'\n  Summary saved: {summary_path}')

    print(f'\n{"="*70}')
    print(f'  MATRIX COMPLETE')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
