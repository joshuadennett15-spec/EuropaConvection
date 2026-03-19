"""Generate thesis-anchor result table from 2D MC archives."""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src

import numpy as np

from literature_scenarios import list_scenarios, get_scenario
from profile_diagnostics import (
    LOW_LAT_BAND,
    HIGH_LAT_BAND,
    area_weighted_band_mean,
    band_mean_samples,
)


def _pct(arr, p):
    return float(np.percentile(arr, p))


def generate_table(results_dir: str, iterations: int) -> str:
    """Build a formatted result table from all scenario NPZ files."""
    lines = []
    header = (
        f"{'Scenario':<32} {'n_valid':>7} {'H_eq':>8} {'H_pole':>8} "
        f"{'DeltaH':>8} {'D_cond_eq':>9} {'D_cond_pole':>11} "
        f"{'conv_frac':>9} {'q_star':>7} {'T_floor':>7} {'mtf':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for name in list_scenarios():
        path = os.path.join(results_dir, f"mc_2d_{name}_{iterations}.npz")
        if not os.path.exists(path):
            lines.append(f"{name:<32} -- file not found --")
            continue

        data = np.load(path)
        n_valid = int(data["n_valid"])
        n_iter = int(data["n_iterations"])
        lats = data["latitudes_deg"]
        h_profiles = data["H_profiles"]
        h_median = data["H_median"]

        # Band means on median profile
        h_eq = area_weighted_band_mean(lats, h_median, LOW_LAT_BAND)
        h_pole = area_weighted_band_mean(lats, h_median, HIGH_LAT_BAND)
        delta_h = h_pole - h_eq

        # D_cond band means (median across samples, then band-average)
        if "D_cond_profiles" in data:
            d_cond_median = np.percentile(data["D_cond_profiles"], 50, axis=0)
            d_cond_eq = area_weighted_band_mean(lats, d_cond_median, LOW_LAT_BAND)
            d_cond_pole = area_weighted_band_mean(lats, d_cond_median, HIGH_LAT_BAND)
        else:
            d_cond_eq = d_cond_pole = float("nan")

        # Convective fraction (median Nu > 1.01 at any latitude)
        if "Nu_profiles" in data:
            nu_profiles = data["Nu_profiles"]
            conv_frac = float(np.mean(nu_profiles > 1.01))
        else:
            conv_frac = float("nan")

        # New metadata (with backward-compat defaults)
        q_star = float(data["q_star"]) if "q_star" in data else -1.0
        t_floor = float(data["T_floor"]) if "T_floor" in data else -1.0
        mtf = float(data["mantle_tidal_fraction"]) if "mantle_tidal_fraction" in data else -1.0

        lines.append(
            f"{name:<32} {n_valid:>4}/{n_iter:<3} {h_eq:>7.1f}  {h_pole:>7.1f}  "
            f"{delta_h:>+7.1f}  {d_cond_eq:>8.1f}  {d_cond_pole:>10.1f}  "
            f"{conv_frac:>8.2f}  {q_star:>6.3f}  {t_floor:>6.1f}  {mtf:>4.2f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print thesis result table from MC archives.")
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    results_dir = os.path.join(_PROJECT_DIR, "results")
    table = generate_table(results_dir, args.iterations)
    print()
    print(table)
    print()
