# Minimal Latitude-Column Model

**Date**: 2026-03-22
**Status**: Draft
**Scope**: Clean latitude-indexed extension of the audited 1D shell model, with no extra 2D closures

---

## 1. Purpose

Define the stripped-down model that the project should have had as the first
latitude-dependent production path:

- one audited 1D Monte Carlo realization,
- reused across many latitude columns,
- with only three latitude-varying forcings:
  - `T_surf(phi)`
  - `epsilon_0(phi)`
  - `q_basal(phi)`

This model is not a 2D PDE solver. It is a set of independent 1D columns
indexed by latitude.

The goal is to preserve the trusted `EuropaProjectDJ` thermal solver and
convection decomposition while adding only the minimum physics needed to study
latitude structure.

---

## 2. Core Contract

For Monte Carlo realization `i`:

1. Draw one audited shared shell-physics parameter set.
2. Build a latitude grid `phi_j`.
3. For each latitude `phi_j`, create one 1D column using the same audited draw.
4. Override only:
   - `T_surf -> T_surf(phi_j)`
   - `epsilon_0 -> epsilon_0(phi_j)`
   - bottom boundary flux -> `q_basal(phi_j)`
5. Run the ordinary 1D solver for each column to equilibrium.
6. Store per-latitude outputs for that realization.

Everything else remains identical to the audited 1D model.

---

## 3. Shared vs Latitude-Varying Inputs

### 3.1 Shared across all latitudes within one realization

These must be sampled once and reused unchanged across every latitude column:

- `d_grain`
- `d_del`
- `D0v`
- `D0b`
- `mu_ice`
- `D_H2O`
- `Q_v`
- `Q_b`
- `H_rad`
- `f_porosity`
- `f_salt`
- `T_phi`
- `B_k`
- rheology model
- solver numerics: `nx`, `dt`, `total_time`, `rannacher_steps`, `eq_threshold`, `max_steps`

This preserves the meaning of one Monte Carlo realization as one shell-physics
 draw.

### 3.2 Latitude-varying within one realization

These are the only permitted latitude-dependent fields:

- `T_surf(phi)`
- `epsilon_0(phi)`
- `q_basal(phi)`

No other model parameter may vary with latitude in the minimal model.

---

## 4. Bottom Boundary Definition

Use `q_basal(phi)` as the authoritative name for the bottom boundary flux seen
by each latitude column.

Recommended decomposition:

`q_basal(phi) = q_radiogenic + q_tidal(phi)`

where:

- `q_radiogenic` is spatially uniform within one realization
- `q_tidal(phi)` is the redistributed latitude-dependent component

This avoids the naming confusion in the current code where `q_ocean(phi)` is
effectively being used as the local basal flux.

### 4.1 Shared radiogenic term

For one audited draw:

- compute `q_radiogenic` from `H_rad` and `D_H2O`
- hold it fixed for all latitudes in that realization

### 4.2 Latitude-varying tidal term

The latitude structure enters only through `q_tidal(phi)`.

This may be parameterized in either of two equivalent ways:

1. Direct local flux profile:
   - define `q_tidal(phi)` explicitly

2. Global-mean plus normalized shape:
   - define `q_tidal_mean`
   - define a normalized shape `f(phi)` with area-weighted mean 1
   - set `q_tidal(phi) = q_tidal_mean * f(phi)`

If the second form is used, normalization must preserve the area-weighted global
mean exactly.

---

## 5. Allowed Latitude Physics

The minimal model allows only the following latitude-structure modules.

### 5.1 Surface temperature field

Provide a deterministic or sampled surface-temperature law:

`T_surf(phi)`

Examples:

- endpoint proxy mode
- band-mean mode
- calibrated radiative-equilibrium mode

The model should not care which is used, as long as it returns one scalar
surface temperature per latitude.

### 5.2 Tidal strain field

Provide one latitude law:

`epsilon_0(phi)`

Again, the model only requires one scalar strain amplitude per latitude.

### 5.3 Basal heat-flux field

Provide one latitude law:

`q_basal(phi)`

This is the only bottom-boundary input passed to the 1D solver.

---

## 6. Explicit Non-Goals

The following are forbidden in the minimal latitude-column model:

- lateral diffusion between columns
- any `q_tidal_scale` or other 2D-only global uplift
- cold-column convection ramp
- latitude-dependent grain size
- latitude-dependent porosity, salinity, or rheology constants
- special per-latitude solver tolerances
- new convection metrics that do not already exist in the 1D workflow
- mixing global-mean and latitude-averaged summary definitions under the same label

If any of these are added, the model is no longer the minimal latitude-column
model and must be treated as a separate branch.

---

## 7. Numerical Identity Requirement

Each latitude column must be the ordinary audited 1D solver.

That means:

- same `Thermal_Solver`
- same convection code
- same `D_cond` / `D_conv` decomposition
- same `Ra` and `Nu` definitions
- same physical filters
- same `reject_subcritical` policy everywhere in the latitude campaign

The latitude model is a driver/wrapper around the 1D solver, not a new thermal
core.

---

## 8. Minimal Algorithm

For each MC realization `i`:

1. Sample one audited shared parameter set.
2. Compute shared derived quantities:
   - `q_radiogenic`
   - `q_tidal_mean` or equivalent tidal normalization
3. Build latitude fields:
   - `T_surf(phi_j)`
   - `epsilon_0(phi_j)`
   - `q_basal(phi_j)`
4. For each latitude `phi_j`:
   - copy the shared parameter dict
   - override `T_surf`
   - override `epsilon_0`
   - run one 1D solver with `q_basal(phi_j)`
5. Save per-column diagnostics:
   - `H`
   - `D_cond`
   - `D_conv`
   - `Ra`
   - `Nu`
   - convergence flag
6. Aggregate by latitude over realizations.

No column should know anything about neighboring columns.

---

## 9. Validation Requirements

### 9.1 Single-column parity

With `n_lat = 1` and a chosen latitude `phi_0`:

- the latitude-column model must match a standalone audited 1D run with the
  same:
  - shared draw
  - `T_surf`
  - `epsilon_0`
  - `q_basal`
  - solver settings

Outputs should agree within normal solver tolerance:

- `H`
- `D_cond`
- `D_conv`
- `Ra`
- `Nu`

### 9.2 Uniform-field identity

If:

- `T_surf(phi)` is constant,
- `epsilon_0(phi)` is constant,
- `q_basal(phi)` is constant,

then every latitude column must produce the same result.

This is the clean bookkeeping test that should have been locked before adding
any extra latitude physics.

---

## 10. Output Contract

For one campaign, store:

- `latitudes_deg`
- per-latitude arrays of:
  - `H_profile_km`
  - `D_cond_profile_km`
  - `D_conv_profile_km`
  - `Ra_profile`
  - `Nu_profile`
- optional per-realization raw arrays with shape `(n_samples, n_lat)`
- the shared audited parameter arrays for traceability
- the three forcing fields per realization:
  - `T_surf_profile`
  - `epsilon_0_profile`
  - `q_basal_profile`

The forcing profiles must be stored explicitly. They should not have to be
reconstructed later from ambiguous metadata.

---

## 11. Recommended Implementation Boundary

This model should live outside the current `Europa2D` branch so its contract is
not polluted by later 2D-specific experiments.

Recommended path:

- `EuropaProjectDJ/src/latitude_column_fields.py`
- `EuropaProjectDJ/src/latitude_column_runner.py`
- `EuropaProjectDJ/scripts/run_latitude_column_mc.py`

Suggested responsibilities:

- `latitude_column_fields.py`
  - build `T_surf(phi)`, `epsilon_0(phi)`, `q_basal(phi)`
- `latitude_column_runner.py`
  - orchestrate shared draws plus many independent 1D columns
- `run_latitude_column_mc.py`
  - campaign entry point for production runs

---

## 12. Relationship to Europa2D

The minimal latitude-column model is the green-path baseline.

`Europa2D` should be treated as a later experimental branch that adds extra
closures beyond this baseline, including:

- lateral diffusion
- cold-column convection ramp
- optional grain-latitude coupling
- extra calibration choices such as `q_tidal_scale`

Those are not part of the minimal model.

The correct development order is:

1. minimal latitude-column model
2. parity validation against 1D
3. only then optional extra 2D closures as separate experiments

---

## 13. Short Version

The clean target is:

- audited 1D physics
- shared draw per realization
- many independent latitude columns
- only `T_surf(phi)`, `epsilon_0(phi)`, and `q_basal(phi)` vary
- nothing else

If a feature changes anything beyond those three forcing fields, it belongs in a
different branch.
