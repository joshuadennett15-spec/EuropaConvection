# Design Spec: 2D Axisymmetric Europa Ice Shell Model

## Overview

Extend the existing 1D transient thermal evolution model (EuropaProjectDJ) into a 2D axisymmetric (z, θ) coupled-column model that produces continuous ice shell thickness profiles H(θ) from equator to pole with Monte Carlo uncertainty quantification.

## Motivation

Current Europa ice shell thickness estimates are either:
- 1D point estimates at discrete regions (Howell 2021, our existing model)
- Full 3D convection simulations without UQ (Mitri & Showman, Barr & Showman)

No published model produces a **continuous latitude-dependent thickness profile with uncertainty bounds**. This model fills that gap, producing directly testable predictions for Europa Clipper's ice-penetrating radar (REASON instrument).

## Architecture

### Directory Structure

```
EuropaConvection/
├── EuropaProjectDJ/          # Existing 1D model (untouched)
│   └── src/
└── Europa2D/                 # New 2D axisymmetric model
    ├── src/
    │   ├── __init__.py
    │   ├── latitude_profile.py    # Continuous θ-dependent parameters
    │   ├── axial_solver.py        # Coupled-column 2D solver
    │   ├── monte_carlo_2d.py      # 2D Monte Carlo framework
    │   └── latitude_sampler.py    # Parameter sampler for 2D runs
    ├── scripts/
    │   ├── run_2d_single.py       # Single deterministic 2D run
    │   ├── run_2d_mc.py           # Full 2D Monte Carlo
    │   └── plot_thickness_profile.py
    ├── results/
    ├── figures/
    ├── tests/
    └── docs/
```

### Dependency on EuropaProjectDJ

Europa2D imports from the existing 1D codebase:
- `constants.py` — Physical constants, material properties
- `Physics.py` — IcePhysics static methods (viscosity, tidal heating, Stefan velocity)
- `Convection.py` — IceConvection parameterization (Ra, Nu, conductivity profiles)
- `Boundary_Conditions.py` — FixedTemperature surface BC
- `Solver.py` — Thermal_Solver (each latitude column is a 1D solver instance)

No modifications to EuropaProjectDJ are required.

### Import Path Setup

Europa2D discovers EuropaProjectDJ via `sys.path` insertion at the top of each module:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))
```
This inherits `EuropaProjectDJ/src/config.json` via the existing `ConfigManager` singleton. All config values (RA_CRIT, NU_PREFACTOR, USE_GREEN_METHOD, etc.) are shared between 1D and 2D models.

## Coordinate Convention

**φ denotes geographic latitude:** φ = 0° at the equator, φ = 90° at the pole. All latitude-dependent functions, grids, and outputs use this convention consistently.

The lateral diffusion operator in geographic latitude coordinates is:
```
(k / R² cosφ) ∂/∂φ [cosφ ∂T/∂φ]
```

## Governing Equation

The 2D axisymmetric heat equation in (z, φ) coordinates:

```
ρ(T) cp(T) ∂T/∂t = (1/G) ∂/∂z [G · k_eff(T,z) ∂T/∂z]
                   + (k / R² cosφ) ∂/∂φ [cosφ ∂T/∂φ]
                   + q_tidal(T, φ)
```

where G = r² (spherical) or 1 (Cartesian), R is Europa's radius.

### Operator Splitting Justification

Lateral thermal diffusion timescale across 90° of latitude (~2400 km):
- τ_lateral ~ L² / κ ~ (2.4×10⁶)² / 10⁻⁶ ≈ 6×10¹⁸ s ≈ 200 billion years

This is far longer than Europa's age, so lateral coupling is weak. Lie-Trotter operator splitting is physically justified:
1. **Radial step (implicit):** Each column solves 1D heat equation independently
2. **Lateral step (explicit):** Diffuse heat between adjacent columns

### Lateral Step Stability

For explicit diffusion, the CFL condition is: `dt · κ / (R² · dφ²) < 0.5`

With κ ~ 10⁻⁶ m²/s, R ~ 1.56×10⁶ m, dφ ~ 5° ~ 0.087 rad:
- dt_max = 0.5 × R² × dφ² / κ ≈ 9.2×10¹⁵ s

This is ~10,000× larger than the radial timestep (dt = 10¹² s), so the explicit lateral step is unconditionally stable. No sub-cycling is needed.

## Module Specifications

### 1. `latitude_profile.py` — Latitude-Dependent Parameters

Provides continuous functions for parameters that vary with geographic latitude φ.

**Classes:**
- `LatitudeProfile` — frozen dataclass container for all latitude-dependent functions

**Functions:**

Surface temperature (Ojakangas & Stevenson 1989):
```
T_s(φ) = T_eq · max(cos(φ), cos(85°))^(1/4)
```
- T_eq sampled from MC (replaces fixed T_surf)
- Floor at 85° prevents singularity at pole
- At φ=0° (equator): T_s = T_eq ≈ 110 K
- At φ=85° (near pole): T_s ≈ 50 K

Tidal strain amplitude (simplified parameterization inspired by Tobie et al. 2003):
```
ε₀(φ) = ε_eq + (ε_pole - ε_eq) · sin²(φ)
```
- ε_eq ~ 6×10⁻⁶, ε_pole ~ 1.2×10⁻⁵
- Note: This is a first-order approximation of the full spherical harmonic tidal strain pattern. The full Tobie et al. pattern involves e₂,₀ and e₂,₂ components with non-monotonic latitude dependence. This simplification is acceptable for the parameterized convection model.

Ocean heat flux patterns:
```
q_ocean(φ, pattern) where pattern ∈ {uniform, polar_enhanced, equator_enhanced}
```
- All patterns integrate to the same global total P_total / A_surface
- polar_enhanced: q ∝ 1 + α·sin²(φ), following Soderlund et al. (2014)
- equator_enhanced: q ∝ 1 + α·cos²(φ)
- Normalization ensures ∫₀^{π/2} q(φ) cosφ dφ = q_global_mean

### 2. `axial_solver.py` — Coupled-Column 2D Solver

**Class: `AxialSolver2D`**

Constructor:
```python
AxialSolver2D(
    n_lat: int = 19,           # Columns from 0° to 90° (every 5°)
    nx: int = 31,              # Radial nodes per column
    dt: float = 1e12,          # Time step (s)
    latitude_profile: LatitudeProfile,
    physics_params: Dict[str, float],  # Shared MC-sampled params
    use_convection: bool = True,
    initial_thickness: float = 20e3,   # Initial H for all columns (m)
)
```

**Column construction:** During `__init__`, for each latitude φ_j:
1. Build per-column `physics_params` by copying shared params and overriding:
   - `params['epsilon_0'] = latitude_profile.epsilon_0(phi_j)`
   - `params['T_surf'] = latitude_profile.T_s(phi_j)`
2. Create `FixedTemperature(latitude_profile.T_s(phi_j))` as surface BC
3. Instantiate `Thermal_Solver(nx=nx, thickness=initial_thickness, surface_bc=bc, physics_params=params, ...)`

Key methods:
- `solve_step(q_ocean_profile)` → array of db/dt per column
  1. Radial step: call each column's `Thermal_Solver.solve_step(q_ocean_profile[j])` independently
  2. Lateral step: explicit finite difference on φ for each depth level
  3. Writeback: apply `columns[j].T[i] += dT_lat[j, i]` directly (pragmatic, avoids modifying 1D code)
- `run_to_equilibrium(threshold, max_steps)` → Dict with H(φ), T(z,φ), diagnostics
  - Convergence criterion: `max(abs(db_dt)) < threshold` across all columns
- `get_thickness_profile()` → `np.array([col.H for col in self.columns])` (columns[j].H is authoritative)

Internal state:
- `self.columns: List[Thermal_Solver]` — one per latitude, each owns its own H
- `self.latitudes: np.ndarray` — geographic latitude values in radians (0 to π/2)

**Thickness authority:** Each `Thermal_Solver` instance owns its own `H` via the Stefan condition. `get_thickness_profile()` reads from the solvers directly. There is no separate `H_profile` array.

Lateral diffusion (explicit, in geographic latitude):
```python
# For each depth index i:
# Half-node metric factors: cosφ[j+½] = cos((φ[j] + φ[j+1]) / 2)
dT_lat[j] = (k * dt / (R² * dφ²)) * (
    cosφ[j+½] * (T[j+1,i] - T[j,i]) - cosφ[j-½] * (T[j,i] - T[j-1,i])
) / cosφ[j]
```
Boundary conditions in φ: symmetric (dT/dφ = 0) at equator (φ=0) and pole (φ=π/2).

**Grid mismatch acknowledgment:** Each column uses a normalized grid ξ ∈ [0,1] mapped to physical depths [0, H_j]. When columns have different H_j, the same node index i corresponds to different physical depths. Since lateral diffusion is extremely weak (τ_lateral ~ 200 Gyr), this approximation introduces errors that are second-order in an already negligible correction. The lateral step's role is primarily to ensure smooth transitions between columns, not to transport significant heat.

### 3. `latitude_sampler.py` — 2D Parameter Sampler

**Class: `LatitudeParameterSampler`**

Extends the existing `HowellParameterSampler` logic:
- Samples all shared parameters identically (grain size, activation energies, etc.)
- Additionally samples amplitude parameters for latitude profiles:
  - `T_eq`: equatorial surface temperature (normal, 110 ± 5 K)
  - `epsilon_eq`, `epsilon_pole`: strain amplitude endpoints
  - `P_tidal_total`: total silicate tidal power (log-normal, same as Howell)
  - `ocean_pattern`: categorical choice of heat flux pattern

Returns a dict with shared params + a `LatitudeProfile` instance.

### 4. `monte_carlo_2d.py` — 2D Monte Carlo Framework

**Class: `MonteCarloRunner2D`**

Similar interface to existing `MonteCarloRunner`:
```python
runner = MonteCarloRunner2D(n_iterations=1000, seed=42, n_workers=7)
results = runner.run()
```

**Class: `MonteCarloResults2D`**

Frozen dataclass containing:
- `H_profiles`: (n_valid, n_lat) array — thickness at each latitude for each sample
- `latitudes`: (n_lat,) array — latitude values in degrees
- `H_median`, `H_mean`: (n_lat,) arrays — statistics at each latitude
- `H_sigma_low`, `H_sigma_high`: (n_lat,) arrays — 1σ bounds
- `D_cond_profiles`, `D_conv_profiles`: (n_valid, n_lat) arrays
- `Ra_profiles`, `Nu_profiles`: (n_valid, n_lat) arrays
- Standard scalar statistics (runtime, n_valid, etc.)

Worker function:
1. Sample shared params + build LatitudeProfile
2. Create AxialSolver2D with those params
3. Run to equilibrium
4. Apply physical filters per-column
5. Return H(θ) profile + diagnostics (or None if >50% columns fail filters)

## Performance Estimate

- 1D solve: ~0.1s per column
- 2D iteration (19 columns): ~0.5-2s (columns parallelizable within iteration)
- 1000 MC iterations: ~15-30 minutes
- 5000 MC iterations: ~1-2.5 hours

## Key Outputs

1. **H(θ) with uncertainty bands** — continuous thickness profile with 1σ shading
2. **Convective structure vs. latitude** — D_cond(θ), D_conv(θ) showing regime transitions
3. **Ocean circulation comparison** — how different Soderlund patterns change the profile
4. **T(z, θ) cross-section** — 2D heatmap of the shell's thermal structure

## Implementation Phases

1. `latitude_profile.py` + tests (~2 days)
2. `axial_solver.py` + single-run script + tests (~1.5 weeks)
3. `monte_carlo_2d.py` + MC run script (~3-4 days)
4. Plotting scripts + analysis (~2-3 days)
5. Validation against 1D results at equator/pole (~2 days)
6. Thesis figures and writeup integration (~1.5 weeks)

Total: ~5-6 weeks with buffer.

### Scope Cutbacks (if behind schedule by week 4)
- Reduce MC iterations to 1000
- Limit ocean patterns to 2 (uniform + polar_enhanced)
- Drop grid convergence study (use n_lat=19 as default)

## Validation Strategy

- Run 2D model at a single latitude → must match 1D model output exactly
- Compare equator and pole columns against existing regional MC results
- Grid convergence: vary n_lat (9, 19, 37) and verify H(θ) converges
- Energy conservation: verify total heat flux integrates correctly over the sphere
