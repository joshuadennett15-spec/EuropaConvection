"""
Single deterministic 2D axisymmetric runs for literature-backed scenarios.

By default this script runs the core literature presets and saves
scenario-specific figures for each case.
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
import src  # triggers import path setup

import matplotlib.pyplot as plt
import numpy as np

from axial_solver import AxialSolver2D
from literature_scenarios import DEFAULT_SCENARIO, SCENARIOS, get_scenario, list_scenarios
from profile_diagnostics import (
    HIGH_LAT_BAND,
    LOW_LAT_BAND,
    compute_profile_diagnostics,
    format_diagnostic_lines,
    ocean_pattern_metadata,
)


FIXED_PARAMS = {
    "d_grain": 1e-3,
    "Q_v": 59.4e3,
    "Q_b": 49.0e3,
    "mu_ice": 3.3e9,
    "D0v": 9.1e-4,
    "D0b": 8.4e-4,
    "d_del": 7.13e-10,
    "f_porosity": 0.1,
    "f_salt": 0.03,
    "B_k": 1.0,
    "T_phi": 150.0,
}


def _shade_latitude_bands(ax) -> None:
    """Highlight the latitude bands used in the scientific summary."""
    ax.axvspan(*LOW_LAT_BAND, color="tab:blue", alpha=0.06)
    ax.axvspan(*HIGH_LAT_BAND, color="tab:red", alpha=0.06)


def _plot_thickness_profile(
    latitudes_deg: np.ndarray,
    thickness_km: np.ndarray,
    profile,
    solver_diagnostics: list[dict],
    output_path: str,
) -> None:
    """Save a multi-panel thickness figure with forcing and shell diagnostics."""
    nu_profile = np.array([d["Nu"] for d in solver_diagnostics], dtype=float)
    lid_fraction = np.array([d["lid_fraction"] for d in solver_diagnostics], dtype=float)
    diagnostics = compute_profile_diagnostics(
        latitudes_deg=latitudes_deg,
        thickness_km=thickness_km,
        profile=profile,
        nu_profile=nu_profile,
    )
    metadata = ocean_pattern_metadata(profile)

    phi = np.radians(latitudes_deg)
    q_ocean = np.array([profile.ocean_heat_flux(val) for val in phi], dtype=float)
    t_surf = np.array([profile.surface_temperature(val) for val in phi], dtype=float)
    epsilon = np.array([profile.tidal_strain(val) for val in phi], dtype=float)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.2, 1.0]},
    )
    ax_h, ax_forcing, ax_state = axes

    for ax in axes:
        _shade_latitude_bands(ax)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 90)

    ax_h.plot(
        latitudes_deg,
        thickness_km,
        color="tab:blue",
        marker="o",
        linewidth=2.2,
        markersize=4.5,
        label="Equilibrium thickness",
    )
    ax_h.plot(
        latitudes_deg[-1],
        thickness_km[-1],
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="tab:blue",
        markeredgewidth=1.5,
        linestyle="None",
        label="90 deg boundary node",
    )
    ax_h.hlines(
        diagnostics.low_band_mean_km,
        LOW_LAT_BAND[0],
        LOW_LAT_BAND[1],
        color="tab:cyan",
        linestyle="--",
        linewidth=2,
        label="0-10 deg area mean",
    )
    ax_h.hlines(
        diagnostics.high_band_mean_km,
        HIGH_LAT_BAND[0],
        HIGH_LAT_BAND[1],
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label="80-90 deg area mean",
    )
    ax_h.scatter(
        diagnostics.min_latitude_deg,
        diagnostics.min_thickness_km,
        color="black",
        zorder=5,
        label="Minimum thickness",
    )
    ax_h.set_ylabel("Ice Shell Thickness (km)")
    ax_h.set_title(
        f"Europa 2D Shell Profile: {metadata.title} ({metadata.citation})",
        fontsize=13,
    )
    ax_h.legend(loc="upper left", fontsize=9)

    diag_lines = format_diagnostic_lines(metadata, diagnostics)
    ax_h.text(
        0.985,
        0.97,
        "\n".join(diag_lines),
        transform=ax_h.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    ax_forcing.plot(latitudes_deg, t_surf / diagnostics.ts_eq_k, color="tab:red", linewidth=2, label="T_surf / T_eq")
    ax_forcing.plot(
        latitudes_deg,
        q_ocean / profile.q_ocean_mean,
        color="tab:purple",
        linewidth=2,
        label="q_ocean / q_mean",
    )
    ax_forcing.plot(
        latitudes_deg,
        epsilon / diagnostics.epsilon_eq,
        color="tab:green",
        linewidth=2,
        label="epsilon_0 / epsilon_eq",
    )
    ax_forcing.set_ylabel("Normalized Forcing")
    ax_forcing.legend(loc="upper left", ncol=3, fontsize=9)
    ax_forcing.set_title("Forcing profiles used by the axisymmetric shell model", fontsize=11)

    ax_state.plot(latitudes_deg, nu_profile, color="tab:brown", linewidth=2, label="Effective Nu")
    ax_state.set_ylabel("Effective Nu")
    ax_state.set_xlabel("Latitude (degrees)")
    ax_state.set_ylim(bottom=0.95)
    ax_state_right = ax_state.twinx()
    ax_state_right.plot(
        latitudes_deg,
        lid_fraction,
        color="tab:olive",
        linewidth=2,
        linestyle="--",
        label="Lid fraction",
    )
    ax_state_right.set_ylabel("D_cond / H")
    ax_state_right.set_ylim(0.0, 1.05)
    handles_left, labels_left = ax_state.get_legend_handles_labels()
    handles_right, labels_right = ax_state_right.get_legend_handles_labels()
    ax_state.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=9)
    ax_state.set_title("Shell-state diagnostics", fontsize=11)

    fig.text(
        0.5,
        0.015,
        (
            f"{metadata.summary} {metadata.caution} "
            f"Reference: {metadata.reference_url}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_temperature_cross_section(
    latitudes_deg: np.ndarray,
    thickness_km: np.ndarray,
    temperature_2d: np.ndarray,
    profile,
    solver_diagnostics: list[dict],
    output_path: str,
) -> None:
    """Save a physical-depth temperature cross-section with interpretation notes."""
    diagnostics = compute_profile_diagnostics(
        latitudes_deg=latitudes_deg,
        thickness_km=thickness_km,
        profile=profile,
        nu_profile=np.array([d["Nu"] for d in solver_diagnostics], dtype=float),
    )
    metadata = ocean_pattern_metadata(profile)

    n_lat, nx = temperature_2d.shape
    lat_grid = np.broadcast_to(latitudes_deg, (nx, n_lat))
    depth_grid_km = np.array(
        [np.linspace(0.0, thickness_km[j], nx) for j in range(n_lat)],
        dtype=float,
    ).T

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    t_plot = np.clip(temperature_2d.T, 40.0, 275.0)
    mesh = ax.pcolormesh(
        lat_grid,
        depth_grid_km,
        t_plot,
        shading="auto",
        cmap="inferno",
        vmin=40,
        vmax=275,
    )
    _shade_latitude_bands(ax)
    ax.plot(latitudes_deg, thickness_km, color="white", linewidth=1.5, alpha=0.9)
    ax.set_xlim(0, 90)
    ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Depth Below Surface (km)")
    ax.set_title(f"Temperature cross-section: {metadata.title}", fontsize=12)
    ax.invert_yaxis()
    cbar = fig.colorbar(mesh, ax=ax, label="Temperature (K)")
    cbar.ax.tick_params(labelsize=9)

    ax.text(
        0.985,
        0.03,
        (
            f"{metadata.citation}\n"
            f"H_min = {diagnostics.min_thickness_km:.2f} km at {diagnostics.min_latitude_deg:.1f} deg\n"
            f"H(80-90 deg) - H(0-10 deg) = {diagnostics.high_minus_low_km:+.2f} km\n"
            f"max Nu = {diagnostics.max_nu:.2f}\n"
            "90 deg is a symmetry boundary node"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round", "facecolor": "black", "alpha": 0.45, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_single_scenario(
    scenario_name: str,
    output_dir: str,
    q_ocean_mean: float,
    n_lat: int,
    nx: int,
    dt: float,
    max_steps: int,
) -> dict:
    """Run one deterministic literature scenario and save its figures."""
    scenario = get_scenario(scenario_name)
    profile = scenario.build_profile(
        T_eq=96.0,
        epsilon_eq=6e-6,
        epsilon_pole=1.2e-5,
        q_ocean_mean=q_ocean_mean,
    )

    print(f"\n=== {scenario.name}: {scenario.citation} ===")
    print(f"  {scenario.description}")

    solver = AxialSolver2D(
        n_lat=n_lat,
        nx=nx,
        dt=dt,
        latitude_profile=profile,
        physics_params=dict(FIXED_PARAMS),
        use_convection=True,
        initial_thickness=25e3,
        rannacher_steps=4,
    )

    result = solver.run_to_equilibrium(
        threshold=1e-12,
        max_steps=max_steps,
        verbose=True,
    )

    thickness_km = result["H_profile_km"]
    latitudes_deg = result["latitudes_deg"]
    temperature_2d = result["T_2d"]
    solver_diagnostics = result["diagnostics"]

    os.makedirs(output_dir, exist_ok=True)
    thickness_path = os.path.join(output_dir, f"thickness_profile_{scenario.name}.png")
    cross_section_path = os.path.join(output_dir, f"temperature_cross_section_{scenario.name}.png")

    _plot_thickness_profile(
        latitudes_deg,
        thickness_km,
        profile,
        solver_diagnostics,
        thickness_path,
    )
    _plot_temperature_cross_section(
        latitudes_deg,
        thickness_km,
        temperature_2d,
        profile,
        solver_diagnostics,
        cross_section_path,
    )

    print(f"Saved figure to {thickness_path}")
    print(f"Saved figure to {cross_section_path}")
    return {
        "scenario": scenario.name,
        "thickness_path": thickness_path,
        "cross_section_path": cross_section_path,
        "result": result,
    }


def run_benchmark_suite() -> None:
    """Run three-scenario benchmark and print band diagnostics.

    Scenarios: uniform_transport, soderlund2014_equator, lemasquerier2023_polar
    All run with fixed reference parameters (not MC).
    """
    scenario_names = [
        "uniform_transport",
        "soderlund2014_equator",
        "lemasquerier2023_polar",
    ]

    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]
        profile = scenario.build_profile(
            T_eq=96.0,
            epsilon_eq=6e-6,
            epsilon_pole=1.2e-5,
            q_ocean_mean=0.02,
            T_floor=46.0,
        )

        solver = AxialSolver2D(
            n_lat=37,
            nx=31,
            dt=5e12,
            latitude_profile=profile,
            use_convection=True,
        )

        result = solver.run_to_equilibrium(threshold=1e-12, max_steps=500)

        latitudes_deg = result["latitudes_deg"]
        thickness_km = result["H_profile_km"]

        diagnostics = compute_profile_diagnostics(latitudes_deg, thickness_km, profile)

        print(f"=== {scenario_name} ===")
        print(f"  H_low (0-10 deg):   {diagnostics.low_band_mean_km:.2f} km")
        print(f"  H_high (80-90 deg): {diagnostics.high_band_mean_km:.2f} km")
        print(f"  dH:                 {diagnostics.high_minus_low_km:.2f} km")
        print(f"  Min H:              {diagnostics.min_thickness_km:.2f} km at {diagnostics.min_latitude_deg:.1f} deg")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run deterministic 2D literature scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["all", *list_scenarios()],
        default="all",
        help="Scenario preset to run. Default: all literature presets.",
    )
    parser.add_argument(
        "--q-ocean-mean",
        type=float,
        default=0.025,
        help="Global-mean ocean heat flux in W/m^2.",
    )
    parser.add_argument("--n-lat", type=int, default=37, help="Number of latitude columns.")
    parser.add_argument("--nx", type=int, default=31, help="Radial nodes per column.")
    parser.add_argument("--dt", type=float, default=5e12, help="Time step in seconds.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum number of time steps.")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark_suite()
