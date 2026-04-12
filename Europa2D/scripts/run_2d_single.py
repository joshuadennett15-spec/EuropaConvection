"""
Single deterministic 2D axisymmetric runs for literature-backed scenarios.

Produces two publication-quality figures per scenario:
  1. Three-panel thickness / forcing / shell-state profile
  2. Latitude--depth temperature cross-section

By default runs the core literature presets with the corrected dt=1e12.
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
import matplotlib.ticker as mticker
import numpy as np

from constants import Thermal
from axial_solver import AxialSolver2D
from literature_scenarios import DEFAULT_SCENARIO, SCENARIOS, get_scenario, list_scenarios
from profile_diagnostics import (
    HIGH_LAT_BAND,
    LOW_LAT_BAND,
    compute_profile_diagnostics,
    format_diagnostic_lines,
    ocean_pattern_metadata,
)
from pub_style import apply_style, PAL, label_panel, save_fig, add_minor_gridlines, figsize_double


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

# Latitude-band shading colours (colorblind-safe, subtle)
_BAND_EQ_COLOR = PAL.alpha(PAL.CYAN, 0.08)
_BAND_PO_COLOR = PAL.alpha(PAL.RED, 0.08)


def _shade_bands(ax) -> None:
    """Add subtle shading for equatorial and polar diagnostic bands."""
    ax.axvspan(*LOW_LAT_BAND, color=_BAND_EQ_COLOR, zorder=0)
    ax.axvspan(*HIGH_LAT_BAND, color=_BAND_PO_COLOR, zorder=0)


def _plot_thickness_profile(
    latitudes_deg: np.ndarray,
    thickness_km: np.ndarray,
    profile,
    solver_diagnostics: list[dict],
    output_path: str,
) -> None:
    """Publication three-panel figure: thickness, forcing, shell state."""
    apply_style()

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
        3, 1,
        figsize=(figsize_double(aspect=0.65)[0], 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.0], "hspace": 0.08},
    )
    ax_h, ax_f, ax_s = axes

    for ax in axes:
        _shade_bands(ax)
        add_minor_gridlines(ax)
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(15))

    # ── Panel (a): Thickness profile ─────────────────────────────────────
    ax_h.plot(
        latitudes_deg, thickness_km,
        color=PAL.BLUE, lw=1.6, marker="o", ms=3, zorder=3,
        label="Equilibrium thickness",
    )
    ax_h.plot(
        latitudes_deg[-1], thickness_km[-1],
        marker="o", ms=5, mfc="white", mec=PAL.BLUE, mew=1.2,
        ls="None", zorder=4,
        label=r"90$\degree$ boundary node",
    )
    ax_h.hlines(
        diagnostics.low_band_mean_km,
        LOW_LAT_BAND[0], LOW_LAT_BAND[1],
        color=PAL.CYAN, ls="--", lw=1.2,
        label=r"$\langle H \rangle_{0\text{--}10\degree}$",
    )
    ax_h.hlines(
        diagnostics.high_band_mean_km,
        HIGH_LAT_BAND[0], HIGH_LAT_BAND[1],
        color=PAL.RED, ls="--", lw=1.2,
        label=r"$\langle H \rangle_{80\text{--}90\degree}$",
    )
    ax_h.scatter(
        diagnostics.min_latitude_deg, diagnostics.min_thickness_km,
        marker="v", s=30, color=PAL.BLACK, zorder=5,
        label=f"Min H = {diagnostics.min_thickness_km:.1f} km",
    )
    ax_h.set_ylabel("Ice shell thickness (km)")
    ax_h.legend(loc="upper left", ncol=2)
    label_panel(ax_h, "a")

    # Compact annotation box
    ann = (
        f"{metadata.title}\n"
        f"{metadata.citation}\n"
        f"$\\Delta H_{{\\mathrm{{high-low}}}}$ = {diagnostics.high_minus_low_km:+.1f} km\n"
        f"$q^*$ = {diagnostics.q_star:.3f}   "
        f"$q_{{\\mathrm{{pole}}}}/q_{{\\mathrm{{eq}}}}$ = {diagnostics.q_ratio_pole_over_eq:.2f}"
    )
    ax_h.text(
        0.98, 0.97, ann,
        transform=ax_h.transAxes, ha="right", va="top",
        fontsize=6.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="#bbb", lw=0.4),
    )

    # ── Panel (b): Normalized forcing profiles ───────────────────────────
    ax_f.plot(
        latitudes_deg, t_surf / diagnostics.ts_eq_k,
        color=PAL.RED, lw=1.2,
        label=r"$T_{\mathrm{s}} / T_{\mathrm{eq}}$",
    )
    ax_f.plot(
        latitudes_deg, q_ocean / profile.q_ocean_mean,
        color=PAL.PURPLE, lw=1.2,
        label=r"$q_{\mathrm{ocean}} / \langle q \rangle$",
    )
    ax_f.plot(
        latitudes_deg, epsilon / diagnostics.epsilon_eq,
        color=PAL.GREEN, lw=1.2,
        label=r"$\varepsilon_0 / \varepsilon_{\mathrm{eq}}$",
    )
    ax_f.set_ylabel("Normalized forcing")
    ax_f.legend(loc="center left", ncol=3)
    ax_f.set_ylim(bottom=0)
    label_panel(ax_f, "b")

    # ── Panel (c): Shell-state diagnostics ───────────────────────────────
    ln1 = ax_s.plot(
        latitudes_deg, nu_profile,
        color=PAL.ORANGE, lw=1.2,
        label="Effective Nu",
    )
    ax_s.set_ylabel("Effective Nu")
    ax_s.set_xlabel(r"Latitude ($\degree$)")
    ax_s.set_ylim(bottom=0.9)

    ax_r = ax_s.twinx()
    ln2 = ax_r.plot(
        latitudes_deg, lid_fraction,
        color=PAL.CYAN, lw=1.0, ls="--",
        label=r"$D_{\mathrm{cond}} / H$",
    )
    ax_r.set_ylabel(r"$D_{\mathrm{cond}} / H$")
    ax_r.set_ylim(0.0, 1.05)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_linewidth(0.4)

    lines = ln1 + ln2
    ax_s.legend(lines, [l.get_label() for l in lines], loc="upper left")
    label_panel(ax_s, "c")

    save_fig(fig, os.path.splitext(os.path.basename(output_path))[0],
             os.path.dirname(output_path), formats=("png", "pdf"))


def _plot_temperature_cross_section(
    latitudes_deg: np.ndarray,
    thickness_km: np.ndarray,
    temperature_2d: np.ndarray,
    profile,
    solver_diagnostics: list[dict],
    output_path: str,
) -> None:
    """Publication latitude--depth temperature cross-section."""
    apply_style()

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

    fig, ax = plt.subplots(figsize=figsize_double(aspect=0.45))

    t_plot = np.clip(temperature_2d.T, 40.0, 275.0)
    mesh = ax.pcolormesh(
        lat_grid, depth_grid_km, t_plot,
        shading="gouraud", cmap="inferno", vmin=40, vmax=275, rasterized=True,
    )

    # Ice--ocean interface
    ax.plot(
        latitudes_deg, thickness_km,
        color="white", lw=1.0, alpha=0.85, zorder=3,
    )

    # Conductive lid base (where T = T_c)
    T_c_arr = np.array([d.get("T_c", 0.0) for d in solver_diagnostics], dtype=float)
    z_c_km = np.array([d.get("z_c_km", 0.0) for d in solver_diagnostics], dtype=float)
    conv_mask = z_c_km > 0
    if np.any(conv_mask):
        ax.plot(
            latitudes_deg[conv_mask], z_c_km[conv_mask],
            color=PAL.CYAN, lw=0.8, ls="--", alpha=0.8, zorder=3,
            label=r"$z_c$ (lid base)",
        )

    ax.set_xlim(0, 90)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(15))
    ax.set_xlabel(r"Latitude ($\degree$)")
    ax.set_ylabel("Depth below surface (km)")
    ax.invert_yaxis()

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("Temperature (K)")

    # Annotation
    ann = (
        f"{metadata.title} ({metadata.citation})\n"
        f"$H_{{\\min}}$ = {diagnostics.min_thickness_km:.1f} km "
        f"at {diagnostics.min_latitude_deg:.0f}$\\degree$     "
        f"$\\Delta H$ = {diagnostics.high_minus_low_km:+.1f} km\n"
        f"max Nu = {diagnostics.max_nu:.1f}     "
        f"$q^*$ = {diagnostics.q_star:.3f}"
    )
    ax.text(
        0.98, 0.04, ann,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6, color="white", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5, ec="0.6", lw=0.3),
    )

    if np.any(conv_mask):
        ax.legend(loc="lower left", fontsize=6, labelcolor="white",
                  facecolor="black", framealpha=0.4, edgecolor="0.6")

    save_fig(fig, os.path.splitext(os.path.basename(output_path))[0],
             os.path.dirname(output_path), formats=("png", "pdf"))


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
        T_eq=Thermal.SURFACE_TEMP_MEAN,
        epsilon_eq=6e-6,
        epsilon_pole=1.2e-5,
        q_ocean_mean=q_ocean_mean,
        T_floor=46.0,
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
            T_eq=Thermal.SURFACE_TEMP_MEAN,
            epsilon_eq=6e-6,
            epsilon_pole=1.2e-5,
            q_ocean_mean=0.02,
            T_floor=46.0,
        )

        solver = AxialSolver2D(
            n_lat=37,
            nx=31,
            dt=1e12,
            latitude_profile=profile,
            use_convection=True,
        )

        result = solver.run_to_equilibrium(threshold=1e-12, max_steps=1500)

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
    parser.add_argument("--dt", type=float, default=1e12, help="Time step in seconds.")
    parser.add_argument("--max-steps", type=int, default=1500, help="Maximum number of time steps.")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark_suite()
