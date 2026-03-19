"""Plot 2D MC thickness profiles with literature-aware diagnostics."""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src

import matplotlib.pyplot as plt
import numpy as np

from latitude_profile import LatitudeProfile
from literature_scenarios import DEFAULT_SCENARIO, get_scenario, list_scenarios
from profile_diagnostics import (
    HIGH_LAT_BAND,
    LOW_LAT_BAND,
    band_mean_samples,
    compute_profile_diagnostics,
    ocean_pattern_metadata,
)


def _shade_latitude_bands(ax) -> None:
    """Highlight the latitude bands used in the scientific summary."""
    ax.axvspan(*LOW_LAT_BAND, color="tab:blue", alpha=0.06)
    ax.axvspan(*HIGH_LAT_BAND, color="tab:red", alpha=0.06)


def _extract_pattern(data: np.lib.npyio.NpzFile) -> dict:
    """Read forcing metadata from the NPZ archive, with backward-compatible defaults.

    Returns a dict with keys: pattern, amplitude, T_floor, q_star, mantle_tidal_fraction.
    Old archives that lack the new fields get safe defaults.
    """
    pattern = (
        str(np.asarray(data["ocean_pattern"]).item())
        if "ocean_pattern" in data
        else "polar_enhanced"
    )
    amplitude = float(data["ocean_amplitude"]) if "ocean_amplitude" in data else None
    T_floor = float(data["T_floor"]) if "T_floor" in data else 52.0
    q_star = float(data["q_star"]) if "q_star" in data else None
    mantle_tidal_fraction = (
        float(data["mantle_tidal_fraction"])
        if "mantle_tidal_fraction" in data
        else 0.5
    )
    return {
        "pattern": pattern,
        "amplitude": amplitude,
        "T_floor": T_floor,
        "q_star": q_star,
        "mantle_tidal_fraction": mantle_tidal_fraction,
    }


def _percentile_summary(samples: np.ndarray) -> str:
    """Return median and 1sigma interval as a short annotation string."""
    median = float(np.percentile(samples, 50))
    low = float(np.percentile(samples, 15.87))
    high = float(np.percentile(samples, 84.13))
    return f"{median:.2f} [{low:.2f}, {high:.2f}] km"


def plot_thickness_profile(filepath: str, output_dir: str | None = None) -> str | None:
    """Create a science-oriented MC summary figure."""
    data = np.load(filepath)
    lats = data["latitudes_deg"]
    h_profiles = data["H_profiles"]
    h_median = data["H_median"]
    h_mean = data["H_mean"]
    h_low = data["H_sigma_low"]
    h_high = data["H_sigma_high"]
    n_valid = int(data["n_valid"])
    n_iter = int(data["n_iterations"])

    meta = _extract_pattern(data)
    profile = LatitudeProfile(
        q_ocean_mean=1.0,
        ocean_pattern=meta["pattern"],
        ocean_amplitude=meta["amplitude"],
        T_floor=meta["T_floor"],
        q_star=meta["q_star"],
        mantle_tidal_fraction=meta["mantle_tidal_fraction"],
        strict_q_star=False,
    )
    metadata = ocean_pattern_metadata(profile)

    nu_profiles = data["Nu_profiles"] if "Nu_profiles" in data else np.ones_like(h_profiles)
    lid_profiles = data["lid_fraction_profiles"] if "lid_fraction_profiles" in data else np.ones_like(h_profiles)
    d_cond_profiles = data["D_cond_profiles"] if "D_cond_profiles" in data else None
    d_conv_profiles = data["D_conv_profiles"] if "D_conv_profiles" in data else None

    nu_median = np.percentile(nu_profiles, 50, axis=0)
    convective_fraction = np.mean(nu_profiles > 1.01, axis=0)
    lid_median = np.percentile(lid_profiles, 50, axis=0)
    diagnostics = compute_profile_diagnostics(lats, h_median, profile, nu_profile=nu_median)

    low_band_samples = band_mean_samples(lats, h_profiles, LOW_LAT_BAND)
    high_band_samples = band_mean_samples(lats, h_profiles, HIGH_LAT_BAND)
    delta_band_samples = high_band_samples - low_band_samples

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.1, 1.0]},
    )
    ax_h, ax_struct, ax_state = axes

    for ax in axes:
        _shade_latitude_bands(ax)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 90)

    ax_h.fill_between(lats, h_low, h_high, alpha=0.30, color="steelblue", label="1sigma profile band")
    ax_h.plot(lats, h_median, color="tab:blue", linewidth=2.2, label="Median profile")
    ax_h.plot(lats, h_mean, color="tab:red", linestyle="--", linewidth=1.4, label="Mean profile")
    ax_h.plot(
        lats[-1],
        h_median[-1],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="tab:blue",
        markeredgewidth=1.5,
        linestyle="None",
        label="90 deg boundary node",
    )
    ax_h.set_ylabel("Ice Shell Thickness (km)")
    ax_h.set_title(
        f"Europa 2D MC profile: {metadata.title} ({metadata.citation})\n"
        f"N = {n_valid}/{n_iter} valid samples",
        fontsize=13,
    )
    ax_h.legend(loc="upper left", fontsize=9)

    q_star_label = (
        f"q* = {meta['q_star']:.3f}"
        if meta["q_star"] is not None
        else "q* = N/A (legacy)"
    )
    summary_lines = [
        metadata.summary,
        f"H(0-10 deg): {_percentile_summary(low_band_samples)}",
        f"H(80-90 deg): {_percentile_summary(high_band_samples)}",
        f"Delta H_high-low: {_percentile_summary(delta_band_samples)}",
        f"H_min,median = {diagnostics.min_thickness_km:.2f} km at {diagnostics.min_latitude_deg:.1f} deg",
        f"q_pole/q_eq = {diagnostics.q_ratio_pole_over_eq:.2f}",
        q_star_label,
        f"T_floor = {meta['T_floor']:.1f} K",
        "Interpret 90 deg as a symmetry boundary node.",
    ]
    ax_h.text(
        0.985,
        0.97,
        "\n".join(summary_lines),
        transform=ax_h.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
    )

    if d_cond_profiles is not None and d_conv_profiles is not None:
        ax_struct.plot(lats, np.percentile(d_cond_profiles, 50, axis=0), color="tab:green", linewidth=2, label="Median D_cond")
        ax_struct.plot(lats, np.percentile(d_conv_profiles, 50, axis=0), color="tab:orange", linewidth=2, label="Median D_conv")
    ax_struct.set_ylabel("Shell Structure (km)")
    ax_struct.set_title("Median conductive and convective layer thickness", fontsize=11)
    if d_cond_profiles is not None and d_conv_profiles is not None:
        ax_struct.legend(loc="upper left", fontsize=9)

    ax_state.plot(lats, nu_median, color="tab:brown", linewidth=2, label="Median effective Nu")
    ax_state.set_ylabel("Effective Nu")
    ax_state.set_xlabel("Latitude (degrees)")
    ax_state.set_ylim(bottom=0.95)
    ax_state_right = ax_state.twinx()
    ax_state_right.plot(
        lats,
        convective_fraction,
        color="tab:purple",
        linewidth=2,
        linestyle="--",
        label="Fraction with Nu > 1.01",
    )
    ax_state_right.plot(
        lats,
        lid_median,
        color="tab:olive",
        linewidth=1.8,
        linestyle="-.",
        label="Median lid fraction",
    )
    ax_state_right.set_ylabel("Fraction")
    ax_state_right.set_ylim(0.0, 1.05)
    handles_left, labels_left = ax_state.get_legend_handles_labels()
    handles_right, labels_right = ax_state_right.get_legend_handles_labels()
    ax_state.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=9)
    ax_state.set_title("Convection activity and lid fraction", fontsize=11)

    fig.text(
        0.5,
        0.015,
        f"{metadata.caution} Reference: {metadata.reference_url}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    saved_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        saved_path = os.path.join(output_dir, f"{base_name}.png")
        fig.savefig(saved_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {saved_path}")

    plt.close(fig)
    return saved_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot 2D MC outputs for literature scenarios.")
    parser.add_argument(
        "--filepath",
        default="",
        help="Direct path to a Monte Carlo NPZ file. Overrides --scenario if provided.",
    )
    parser.add_argument(
        "--scenario",
        choices=list_scenarios(),
        default=DEFAULT_SCENARIO,
        help="Scenario name used to construct the default results filename.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Iteration count used in the default results filename.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results_dir = os.path.join(_PROJECT_DIR, "results")
    figures_dir = os.path.join(_PROJECT_DIR, "figures")

    if args.filepath:
        filepath = args.filepath
    else:
        scenario = get_scenario(args.scenario)
        filepath = os.path.join(results_dir, f"mc_2d_{scenario.name}_{args.iterations}.npz")

    if os.path.exists(filepath):
        plot_thickness_profile(filepath, figures_dir)
    else:
        print(f"No results found at {filepath}. Run run_2d_mc.py first.")
