"""
Lateral ice flow diagnostic for Europa's ice shell.

Applies the Ashkenazy, Sayag & Tziperman (2018) thin-film gravity current
model as a post-hoc diagnostic to completed MC ensemble H(phi) profiles.

For each H(phi) profile the script:
  1. Computes Glen diffusivity D(phi) at the basal temperature.
  2. Estimates relaxation timescale tau = L^2 / mean(D).
  3. Integrates the nonlinear diffusion equation to steady state.
  4. Reports original delta_H, relaxed delta_H_eq, and the reduction factor.

Reference:
    Ashkenazy, Sayag & Tziperman (2018), Nature Astronomy 2, 43-49.
    doi:10.1038/s41550-017-0326-7

Usage:
    # Synthetic test profile
    python Europa2D/scripts/run_lateral_flow_diagnostic.py --test

    # Process a saved MC ensemble file
    python Europa2D/scripts/run_lateral_flow_diagnostic.py \
        --input Europa2D/results/mc_2d_uniform_transport_250.npz

    # Process all npz files in the default results directory
    python Europa2D/scripts/run_lateral_flow_diagnostic.py --all
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
# Insert EuropaProjectDJ/src so constants is importable
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
import src  # triggers Europa2D/src/__init__ path setup

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Europa planetary values (Planetary class in EuropaProjectDJ/src/constants.py)
RADIUS_EUROPA_M: float = 1_561_000.0     # m
GRAVITY_EUROPA: float = 1.315            # m/s^2

# Ice properties
RHO_ICE: float = 917.0                  # kg/m^3  (reference density, CBE constant)
N_GLEN: int = 3                          # Glen flow law exponent

# Glen rate factor A for cold ice.
#   Goldsby & Kohlstedt (2001), grain-boundary sliding regime at ~180 K:
#   A ~ 1e-25 Pa^-3 s^-1 for cold (polar/basal) ice.
#   Users can override with --glen-A.
GLEN_A_DEFAULT: float = 1.0e-25          # Pa^-3 s^-1

# Convergence criterion for explicit diffusion integration
CONVERGENCE_TOL: float = 1.0e-6         # km — max change in H per step

# Maximum diffusion integration steps (safety cap)
MAX_DIFFUSION_STEPS: int = 200_000

# ---------------------------------------------------------------------------
# Glen diffusivity
# ---------------------------------------------------------------------------


def glen_diffusivity(
    H_m: NDArray[np.float64],
    A: float = GLEN_A_DEFAULT,
    rho: float = RHO_ICE,
    g: float = GRAVITY_EUROPA,
    n: int = N_GLEN,
) -> NDArray[np.float64]:
    """
    Compute the nonlinear Glen diffusivity at each latitude column.

    From Ashkenazy et al. (2018), Eq. (1):
        D = (2*A*(rho*g)^n / (n+2)) * H^(n+2)

    Parameters
    ----------
    H_m : array, shape (n_lat,)
        Ice thickness in metres.
    A : float
        Glen rate factor in Pa^-n s^-1.
    rho : float
        Ice density in kg/m^3.
    g : float
        Surface gravity in m/s^2.
    n : int
        Glen flow law exponent (default 3).

    Returns
    -------
    D : array, shape (n_lat,)
        Diffusivity in m^(n+2) / (m^n s^-1) = m^2 s^-1 (for n=3: m^5 / m^3 s^-1).
        Units: m^2 s^-1  — valid because H carries the remaining m^n.
        Dimensionally: [Pa^-n s^-1] * [(kg m^-2 s^-2)^n] * [m^(n+2)]
                     = [Pa^-n s^-1] * [Pa^n] * [m^(n+2)] / m^n
                     = m^2 s^-1.  (Correct.)
    """
    prefactor = (2.0 * A * (rho * g) ** n) / (n + 2)
    return prefactor * H_m ** (n + 2)


# ---------------------------------------------------------------------------
# Explicit thin-film diffusion solver (1-D latitude)
# ---------------------------------------------------------------------------


def solve_thin_film_diffusion(
    latitudes_deg: NDArray[np.float64],
    H_km: NDArray[np.float64],
    A: float = GLEN_A_DEFAULT,
    *,
    max_steps: int = MAX_DIFFUSION_STEPS,
    tol_km: float = CONVERGENCE_TOL,
) -> tuple[NDArray[np.float64], float]:
    """
    Integrate the 1-D nonlinear diffusion equation on the latitude grid.

        dH/dt = d/dphi [ D(H) * dH/dphi ] / (R * cos(phi))   (latitude form)

    Mass is conserved (Neumann BCs: zero flux at both ends).

    Parameters
    ----------
    latitudes_deg : array, shape (n_lat,)
        Latitude in degrees (0 = equator, 90 = pole).
    H_km : array, shape (n_lat,)
        Initial ice thickness in km.
    A : float
        Glen rate factor in Pa^-n s^-1.
    max_steps : int
        Maximum number of explicit time steps.
    tol_km : float
        Convergence threshold in km (max |dH| per step to stop).

    Returns
    -------
    H_eq_km : array, shape (n_lat,)
        Steady-state ice thickness in km.
    tau_yr : float
        Estimated relaxation timescale in years.
    """
    H_m = H_km * 1.0e3                                # km -> m
    phi_rad = np.deg2rad(latitudes_deg)
    dphi = phi_rad[1] - phi_rad[0]                    # uniform grid assumed
    L = RADIUS_EUROPA_M * dphi                        # arc-length step in m

    # Diffusivity on the initial profile (for timescale estimate)
    D0 = glen_diffusivity(H_m, A=A)
    D_mean = float(np.mean(D0))
    L_total = RADIUS_EUROPA_M * (phi_rad[-1] - phi_rad[0])  # total arc length
    tau_s = L_total ** 2 / D_mean if D_mean > 0.0 else np.inf
    tau_yr = tau_s / (3.15576e7)                     # s -> years

    # CFL-stable time step: dt <= 0.5 * L^2 / D_max
    D_max = float(np.max(D0))
    if D_max <= 0.0:
        return H_km.copy(), tau_yr

    dt_s = 0.4 * L ** 2 / D_max                     # slightly below CFL limit

    H = H_m.copy()
    n_lat = len(H)

    for _ in range(max_steps):
        D = glen_diffusivity(H, A=A)
        # Face-centred diffusivity (average of adjacent nodes)
        D_face = 0.5 * (D[:-1] + D[1:])             # shape (n_lat-1,)

        # Flux at interior faces
        dH_dphi = np.diff(H) / L                     # shape (n_lat-1,)
        flux = D_face * dH_dphi                      # shape (n_lat-1,)

        # Divergence: (flux_right - flux_left) / L
        dH = np.zeros(n_lat)
        dH[1:-1] = (flux[1:] - flux[:-1]) / L       # interior nodes
        # Neumann BCs: zero flux at both ends (equator and pole)
        dH[0] = flux[0] / L
        dH[-1] = -flux[-1] / L

        H_new = H + dt_s * dH

        # Enforce H >= 0 (physically required)
        H_new = np.maximum(H_new, 0.0)

        # Re-CFL if D grows
        D_new_max = float(np.max(glen_diffusivity(H_new, A=A)))
        if D_new_max > 0.0:
            dt_s = min(dt_s, 0.4 * L ** 2 / D_new_max)

        change_km = float(np.max(np.abs(H_new - H))) * 1.0e-3
        H = H_new

        if change_km < tol_km:
            break

    return H * 1.0e-3, tau_yr   # m -> km


# ---------------------------------------------------------------------------
# Per-profile diagnostic
# ---------------------------------------------------------------------------


def analyse_profile(
    latitudes_deg: NDArray[np.float64],
    H_km: NDArray[np.float64],
    T_c_K: NDArray[np.float64] | None = None,
    A: float = GLEN_A_DEFAULT,
) -> dict:
    """
    Run the thin-film diagnostic for a single H(phi) profile.

    Parameters
    ----------
    latitudes_deg : array, shape (n_lat,)
    H_km : array, shape (n_lat,)
        Ice thickness in km.
    T_c_K : array, shape (n_lat,) or None
        Basal (conductive lid base) temperatures in K.  Currently informational;
        future versions can use this to make A temperature-dependent.
    A : float
        Glen rate factor.

    Returns
    -------
    dict with keys:
        delta_H_orig_km, delta_H_eq_km, reduction_factor,
        tau_yr, H_eq_km, H_orig_km
    """
    H_eq_km, tau_yr = solve_thin_film_diffusion(latitudes_deg, H_km, A=A)

    delta_H_orig = float(np.max(H_km) - np.min(H_km))
    delta_H_eq = float(np.max(H_eq_km) - np.min(H_eq_km))

    if delta_H_orig > 0.0:
        reduction = delta_H_eq / delta_H_orig
    else:
        reduction = 1.0

    return {
        "delta_H_orig_km": delta_H_orig,
        "delta_H_eq_km": delta_H_eq,
        "reduction_factor": reduction,
        "tau_yr": tau_yr,
        "H_eq_km": H_eq_km,
        "H_orig_km": H_km.copy(),
    }


# ---------------------------------------------------------------------------
# Ensemble analysis
# ---------------------------------------------------------------------------


def analyse_ensemble(
    latitudes_deg: NDArray[np.float64],
    H_profiles: NDArray[np.float64],
    T_c_profiles: NDArray[np.float64] | None = None,
    A: float = GLEN_A_DEFAULT,
) -> list[dict]:
    """
    Analyse all profiles in an MC ensemble.

    Parameters
    ----------
    H_profiles : array, shape (n_samples, n_lat)
    T_c_profiles : array, shape (n_samples, n_lat) or None
    """
    n_samples = H_profiles.shape[0]
    results = []
    for i in range(n_samples):
        T_c = T_c_profiles[i] if T_c_profiles is not None else None
        r = analyse_profile(latitudes_deg, H_profiles[i], T_c_K=T_c, A=A)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_ensemble_table(results: list[dict], label: str = "") -> None:
    """Print a summary table of the ensemble diagnostic results."""
    header = f"{'Sample':>7}  {'dH_orig (km)':>12}  {'dH_eq (km)':>10}  {'Reduction':>9}  {'tau (yr)':>10}"
    sep = "-" * len(header)
    print(f"\n=== Lateral Flow Diagnostic{': ' + label if label else ''} ===")
    print(header)
    print(sep)
    for i, r in enumerate(results):
        print(
            f"{i+1:>7}  "
            f"{r['delta_H_orig_km']:>12.3f}  "
            f"{r['delta_H_eq_km']:>10.3f}  "
            f"{r['reduction_factor']:>9.4f}  "
            f"{r['tau_yr']:>10.3e}"
        )
    print(sep)
    orig_arr = np.array([r["delta_H_orig_km"] for r in results])
    eq_arr = np.array([r["delta_H_eq_km"] for r in results])
    red_arr = np.array([r["reduction_factor"] for r in results])
    tau_arr = np.array([r["tau_yr"] for r in results])
    print(
        f"{'MEDIAN':>7}  "
        f"{float(np.median(orig_arr)):>12.3f}  "
        f"{float(np.median(eq_arr)):>10.3f}  "
        f"{float(np.median(red_arr)):>9.4f}  "
        f"{float(np.median(tau_arr)):>10.3e}"
    )
    print(
        f"{'MEAN':>7}  "
        f"{float(np.mean(orig_arr)):>12.3f}  "
        f"{float(np.mean(eq_arr)):>10.3f}  "
        f"{float(np.mean(red_arr)):>9.4f}  "
        f"{float(np.mean(tau_arr)):>10.3e}"
    )
    print(sep)
    print(
        f"\nConclusion: lateral flow reduces thickness contrast by a median "
        f"factor of {float(np.median(red_arr)):.3f} "
        f"({100.0*(1-float(np.median(red_arr))):.1f}% reduction) "
        f"on a timescale tau ~ {float(np.median(tau_arr)):.2e} yr."
    )


def plot_diagnostic(
    latitudes_deg: NDArray[np.float64],
    results: list[dict],
    figures_dir: str,
    label: str = "diagnostic",
) -> None:
    """
    Produce two figures:
    1. Before/after thickness profiles for each sample (shaded ensemble).
    2. Distribution of reduction factors across the ensemble.
    """
    orig_arr = np.array([r["H_orig_km"] for r in results])   # (n, n_lat)
    eq_arr = np.array([r["H_eq_km"] for r in results])       # (n, n_lat)
    red_arr = np.array([r["reduction_factor"] for r in results])

    os.makedirs(figures_dir, exist_ok=True)

    # ---- Figure 1: before/after profiles ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    # Panel A: median ± 1-sigma band
    orig_med = np.median(orig_arr, axis=0)
    orig_lo = np.percentile(orig_arr, 16, axis=0)
    orig_hi = np.percentile(orig_arr, 84, axis=0)
    eq_med = np.median(eq_arr, axis=0)
    eq_lo = np.percentile(eq_arr, 16, axis=0)
    eq_hi = np.percentile(eq_arr, 84, axis=0)

    ax = axes[0]
    ax.fill_between(latitudes_deg, orig_lo, orig_hi, alpha=0.25, color="#2166ac", label="Original 68%")
    ax.plot(latitudes_deg, orig_med, color="#2166ac", lw=1.8, label="Original median")
    ax.fill_between(latitudes_deg, eq_lo, eq_hi, alpha=0.25, color="#d6604d", label="Relaxed 68%")
    ax.plot(latitudes_deg, eq_med, color="#d6604d", lw=1.8, ls="--", label="Relaxed (steady state)")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Ice thickness H (km)")
    ax.set_title("Before / After Lateral Relaxation")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 90)

    # Panel B: reduction factor distribution
    ax2 = axes[1]
    ax2.hist(red_arr, bins=20, color="#4dac26", edgecolor="white", linewidth=0.5)
    ax2.axvline(float(np.median(red_arr)), color="black", ls="--", lw=1.5,
                label=f"Median = {float(np.median(red_arr)):.3f}")
    ax2.set_xlabel("Reduction factor (dH_eq / dH_orig)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Thickness-Contrast Reduction")
    ax2.legend(fontsize=9)

    fig.suptitle(f"Ashkenazy 2018 Thin-Film Diagnostic  —  {label}", fontsize=10)
    fig.tight_layout()

    fname = os.path.join(figures_dir, f"lateral_flow_{label}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {fname}")
    plt.close(fig)

    # ---- Figure 2: tau distribution ----
    tau_arr = np.array([r["tau_yr"] for r in results])
    fig2, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(np.log10(tau_arr), bins=20, color="#762a83", edgecolor="white", linewidth=0.5)
    ax3.axvline(float(np.median(np.log10(tau_arr))), color="black", ls="--", lw=1.5,
                label=f"Median tau = {float(np.median(tau_arr)):.2e} yr")
    ax3.set_xlabel("log10(tau) [yr]")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Relaxation Timescale — {label}")
    ax3.legend(fontsize=9)
    fig2.tight_layout()
    fname2 = os.path.join(figures_dir, f"lateral_flow_tau_{label}.png")
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    print(f"  Saved figure: {fname2}")
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Synthetic test profile
# ---------------------------------------------------------------------------


def make_test_profile(n_lat: int = 37) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Construct a synthetic H(phi) profile for --test mode.

    H(phi) = H_eq + (H_pole - H_eq) * sin^2(phi)

    Using H_equator = 25 km, H_pole = 55 km (large contrast to make relaxation
    clearly visible).
    """
    H_EQ_KM = 25.0
    H_POLE_KM = 55.0
    lats = np.linspace(0.0, 90.0, n_lat)
    phi = np.deg2rad(lats)
    H = H_EQ_KM + (H_POLE_KM - H_EQ_KM) * np.sin(phi) ** 2
    return lats, H


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lateral ice flow diagnostic based on Ashkenazy et al. (2018). "
            "Applies thin-film gravity-current relaxation to MC ensemble H profiles."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--test",
        action="store_true",
        help="Run on a synthetic cosine test profile (equator=25 km, pole=55 km).",
    )
    mode.add_argument(
        "--input",
        metavar="FILE.npz",
        help="Path to a saved MC ensemble .npz file.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        dest="all_files",
        help="Process all .npz files in Europa2D/results/.",
    )
    parser.add_argument(
        "--glen-A",
        type=float,
        default=GLEN_A_DEFAULT,
        metavar="A",
        help=f"Glen rate factor in Pa^-3 s^-1 (default: {GLEN_A_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--n-lat",
        type=int,
        default=37,
        help="Number of latitude nodes for --test mode (default: 37).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        metavar="DIR",
        help="Output directory for figures (default: Europa2D/figures/).",
    )
    return parser.parse_args()


def _figures_dir_default() -> str:
    return os.path.join(_PROJECT_DIR, "figures")


def _run_on_file(npz_path: str, glen_A: float, figures_dir: str, no_plot: bool) -> None:
    """Load an MC .npz file and run the lateral flow diagnostic."""
    label = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"\nLoading: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)

    if "H_profiles" not in data or "latitudes_deg" not in data:
        print(f"  WARNING: {npz_path} missing 'H_profiles' or 'latitudes_deg' — skipping.")
        return

    H_profiles = data["H_profiles"]       # (n_valid, n_lat)
    latitudes_deg = data["latitudes_deg"]  # (n_lat,)
    T_c_profiles = data["T_c_profiles"] if "T_c_profiles" in data else None

    print(f"  {H_profiles.shape[0]} samples x {H_profiles.shape[1]} latitudes")

    results = analyse_ensemble(
        latitudes_deg, H_profiles, T_c_profiles=T_c_profiles, A=glen_A
    )
    print_ensemble_table(results, label=label)

    if not no_plot:
        plot_diagnostic(latitudes_deg, results, figures_dir=figures_dir, label=label)


def main() -> None:
    args = _parse_args()
    figures_dir = args.figures_dir or _figures_dir_default()

    if args.test:
        print("=== Test mode: synthetic cosine profile (H_eq=25 km, H_pole=55 km) ===")
        lats, H_test = make_test_profile(n_lat=args.n_lat)
        result = analyse_profile(lats, H_test, A=args.glen_A)

        print(f"\n  Original delta_H : {result['delta_H_orig_km']:.3f} km")
        print(f"  Relaxed  delta_H : {result['delta_H_eq_km']:.3f} km")
        print(f"  Reduction factor : {result['reduction_factor']:.4f}")
        print(f"  tau              : {result['tau_yr']:.3e} yr")

        # Single-profile table output
        print_ensemble_table([result], label="synthetic_test")

        if not args.no_plot:
            plot_diagnostic(lats, [result], figures_dir=figures_dir, label="synthetic_test")

    elif args.input:
        _run_on_file(args.input, glen_A=args.glen_A, figures_dir=figures_dir, no_plot=args.no_plot)

    else:  # --all
        results_dir = os.path.join(_PROJECT_DIR, "results")
        npz_files = sorted(
            f for f in os.listdir(results_dir) if f.endswith(".npz")
        )
        if not npz_files:
            print(f"No .npz files found in {results_dir}")
            return
        print(f"Found {len(npz_files)} .npz files in {results_dir}")
        for fname in npz_files:
            _run_on_file(
                os.path.join(results_dir, fname),
                glen_A=args.glen_A,
                figures_dir=figures_dir,
                no_plot=args.no_plot,
            )


if __name__ == "__main__":
    main()
