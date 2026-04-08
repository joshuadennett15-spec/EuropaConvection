# Minimal Latitude-Column Model -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the stripped-down latitude-column model defined in `docs/superpowers/specs/2026-03-22-minimal-latitude-column-model.md`: one audited shared 1D shell-physics draw per Monte Carlo realization, then many independent 1D latitude columns where only `T_surf(phi)`, `epsilon_0(phi)`, and `q_basal(phi)` vary.

**Architecture:** New `EuropaProjectDJ` modules only. No dependency on `Europa2D` runtime code. The implementation is a wrapper around the existing `Thermal_Solver` and audited 1D physics. A small field builder generates latitude forcing profiles, a runner executes one shared draw across many columns, and a script saves campaign outputs and summary statistics.

**Tech Stack:** Python 3.x, NumPy, pytest, existing `EuropaProjectDJ/src/` modules

**Spec:** `docs/superpowers/specs/2026-03-22-minimal-latitude-column-model.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `EuropaProjectDJ/src/latitude_column_fields.py` | Build `T_surf(phi)`, `epsilon_0(phi)`, and `q_basal(phi)` from a shared audited draw |
| `EuropaProjectDJ/src/latitude_column_runner.py` | Execute one Monte Carlo realization as many independent 1D columns and aggregate campaign results |
| `EuropaProjectDJ/scripts/run_latitude_column_mc.py` | Entry-point script for production campaigns |
| `EuropaProjectDJ/tests/test_latitude_column_fields.py` | Unit tests for field generation and normalization |
| `EuropaProjectDJ/tests/test_latitude_column_runner.py` | Runner parity, identity, and data-shape tests |

This keeps the minimal model entirely inside `EuropaProjectDJ` and prevents accidental coupling to `Europa2D` closures.

---

## Phase A -- Forcing Field Builder

This phase implements the three allowed latitude-dependent inputs and nothing else.

---

### Task 1: Create `latitude_column_fields.py`

**Files:**
- Create: `EuropaProjectDJ/src/latitude_column_fields.py`
- Create: `EuropaProjectDJ/tests/test_latitude_column_fields.py`

The module should expose pure functions or a small dataclass-based API that builds:

- `T_surf_profile`
- `epsilon_0_profile`
- `q_basal_profile`

for a supplied latitude grid and one shared audited draw.

- [ ] **Step 1: Define a minimal config surface**

Create lightweight dataclasses or typed dicts for:

- latitude grid metadata
- surface-temperature field controls
- tidal-strain field controls
- basal-flux field controls

Do not add controls for lateral diffusion, convection ramps, grain-size latitude scaling, or any other non-minimal feature.

- [ ] **Step 2: Implement latitude grid helpers**

Implement helpers to:

- construct `latitudes_deg`
- construct `latitudes_rad`
- validate monotonic increasing latitude grids
- support both:
  - explicit latitude arrays
  - simple `n_lat`, `lat_min_deg`, `lat_max_deg` generation

Recommended default campaign grid:

- `lat_min_deg = 0`
- `lat_max_deg = 90`
- `n_lat = 19` or `37`

- [ ] **Step 3: Implement `T_surf(phi)` field generation**

Support the following modes:

- `uniform`
- `endpoint_proxy`
- `radiative_calibrated`

Behavior:

- `uniform`: one scalar surface temperature everywhere
- `endpoint_proxy`: user-provided equator and pole anchors interpolated by an explicit latitude law
- `radiative_calibrated`: calibrated `T_eq` / `T_floor` law intended for continuous latitude studies

The function must return one scalar per latitude and must not alter any other parameter.

- [ ] **Step 4: Implement `epsilon_0(phi)` field generation**

Support the following modes:

- `uniform`
- `endpoint_proxy`
- `mantle_core`

Behavior:

- `uniform`: one scalar strain amplitude everywhere
- `endpoint_proxy`: explicit equator and pole anchors mapped across latitude
- `mantle_core`: monotonic whole-shell latitude law

The result must be one scalar strain amplitude per latitude.

- [ ] **Step 5: Implement `q_basal(phi)` field generation**

The bottom boundary must be built as:

- `q_basal(phi) = q_radiogenic + q_tidal(phi)`

Implementation requirements:

- compute `q_radiogenic` once from the audited shared draw
- derive `q_tidal_mean` from the audited shared draw
- support:
  - `uniform`
  - `equator_enhanced`
  - `polar_enhanced`
- preserve the area-weighted global mean exactly when using patterned fluxes

The API should expose both:

- `q_radiogenic`
- `q_tidal_profile`
- `q_basal_profile`

so diagnostics do not have to infer them later.

- [ ] **Step 6: Write field-builder tests**

Required tests:

- latitude-grid helper returns correct endpoints
- `uniform` surface temperature returns a constant profile
- `uniform` tidal strain returns a constant profile
- `uniform` basal flux returns a constant profile
- patterned `q_basal(phi)` preserves the area-weighted mean
- `q_basal_profile - q_radiogenic == q_tidal_profile` elementwise
- no field builder mutates the shared audited input dict

- [ ] **Step 7: Run tests**

Run:

```bash
cd EuropaProjectDJ
python -m pytest tests/test_latitude_column_fields.py -v
```

- [ ] **Step 8: Commit**

```bash
git add EuropaProjectDJ/src/latitude_column_fields.py EuropaProjectDJ/tests/test_latitude_column_fields.py
git commit -m "feat: add minimal latitude forcing field builder"
```

---

## Phase B -- Column Runner

This phase implements the shared-draw-plus-many-columns execution path using the existing audited 1D solver.

---

### Task 2: Create the single-column execution helper

**Files:**
- Modify: `EuropaProjectDJ/src/latitude_column_runner.py`
- Optional minimal reuse from: `EuropaProjectDJ/src/Monte_Carlo.py`

Implement one internal helper that runs a single 1D column from:

- a shared audited parameter dict
- one `T_surf`
- one `epsilon_0`
- one `q_basal`
- one `SolverConfig`

This helper must use:

- the ordinary `Thermal_Solver`
- the ordinary convection code
- the ordinary physical filters

It must not recompute its own latitude logic.

- [ ] **Step 1: Decide the reuse boundary**

Preferred path:

- implement a local helper in `latitude_column_runner.py`

Alternative:

- extract a reusable helper from `Monte_Carlo.py` if that reduces duplication cleanly

Do not refactor `Monte_Carlo.py` broadly. Keep the change set narrow.

- [ ] **Step 2: Return complete per-column diagnostics**

Each column execution must return:

- `sample_id`
- `latitude_deg`
- `valid`
- `converged`
- `H_km`
- `D_cond_km`
- `D_conv_km`
- `Ra`
- `Nu`
- `T_surf`
- `epsilon_0`
- `q_basal`
- `q_radiogenic`
- `q_tidal`

If a column is invalid, preserve its location in the output with:

- `valid = False`
- numerical outputs as `NaN`

Do not drop invalid columns silently.

---

### Task 3: Create `latitude_column_runner.py`

**Files:**
- Create: `EuropaProjectDJ/src/latitude_column_runner.py`
- Create: `EuropaProjectDJ/tests/test_latitude_column_runner.py`

This module should orchestrate the entire campaign.

- [ ] **Step 1: Define campaign result containers**

Create dataclasses for:

- one realization worth of latitude-column outputs
- full campaign outputs

The campaign container should include:

- `latitudes_deg`
- raw arrays with shape `(n_samples, n_lat)` for:
  - `H_km`
  - `D_cond_km`
  - `D_conv_km`
  - `Ra`
  - `Nu`
  - `T_surf`
  - `epsilon_0`
  - `q_basal`
  - `q_radiogenic`
  - `q_tidal`
  - `valid_mask`
  - `converged_mask`
- per-latitude summaries:
  - median
  - mean
  - sigma low
  - sigma high
  - valid count

- [ ] **Step 2: Implement one-realization execution**

For realization `i`:

1. sample one audited shared draw
2. build the three forcing profiles
3. loop over latitude columns
4. run the 1D column helper for each latitude
5. return all raw arrays for that realization

All columns in the realization must use the same shared audited draw.

- [ ] **Step 3: Implement campaign execution**

Outer-loop over `n_iterations`.

Recommended first implementation:

- parallelize by realization, not by column

Reason:

- one realization is the unit of pairing
- all latitude columns inside that realization share one audited draw

Store `sample_id` explicitly so raw outputs remain pairable even if some columns are invalid.

- [ ] **Step 4: Implement summary statistics**

Compute per-latitude summaries using `NaN`-aware logic:

- median
- mean
- 15.87 percentile
- 84.13 percentile
- valid counts

Do not collapse the campaign to only all-valid realizations by default.

Optional extra summary:

- `all_valid_realization_mask`

for later strict paired analyses.

- [ ] **Step 5: Write runner tests**

Required tests:

- campaign output arrays have shape `(n_samples, n_lat)`
- `sample_id` ordering is preserved
- invalid columns remain represented as `NaN` plus `valid_mask=False`
- a single-column latitude run matches a standalone 1D run with the same forcing
- when `T_surf(phi)`, `epsilon_0(phi)`, and `q_basal(phi)` are uniform, every column produces the same answer within tolerance

- [ ] **Step 6: Run tests**

Run:

```bash
cd EuropaProjectDJ
python -m pytest tests/test_latitude_column_runner.py -v
```

- [ ] **Step 7: Commit**

```bash
git add EuropaProjectDJ/src/latitude_column_runner.py EuropaProjectDJ/tests/test_latitude_column_runner.py
git commit -m "feat: add minimal latitude-column Monte Carlo runner"
```

---

## Phase C -- Validation Gates

These are not optional. The minimal model is not ready for production campaigns until both pass.

---

### Task 4: Single-column parity validation

**Files:**
- Modify: `EuropaProjectDJ/tests/test_latitude_column_runner.py`

- [ ] **Step 1: Add the parity test**

Construct one shared audited draw manually or with a fixed seed.

Run:

- standalone audited 1D column
- latitude-column runner with `n_lat = 1`

using exactly the same:

- parameter draw
- `T_surf`
- `epsilon_0`
- `q_basal`
- `SolverConfig`

Assert agreement for:

- `H_km`
- `D_cond_km`
- `D_conv_km`
- `Ra`
- `Nu`

- [ ] **Step 2: Run the parity test**

Run:

```bash
cd EuropaProjectDJ
python -m pytest tests/test_latitude_column_runner.py -k parity -v
```

- [ ] **Step 3: Fix discrepancies before proceeding**

If parity fails:

- stop feature work
- reconcile the helper path until parity passes

Do not proceed to campaign scripting with an unresolved parity mismatch.

---

### Task 5: Uniform-field identity validation

**Files:**
- Modify: `EuropaProjectDJ/tests/test_latitude_column_runner.py`

- [ ] **Step 1: Add the identity test**

Build a multi-latitude run where:

- `T_surf(phi)` is constant
- `epsilon_0(phi)` is constant
- `q_basal(phi)` is constant

Assert that every latitude column matches every other latitude column within solver tolerance for:

- `H_km`
- `D_cond_km`
- `D_conv_km`
- `Ra`
- `Nu`

- [ ] **Step 2: Run the identity test**

Run:

```bash
cd EuropaProjectDJ
python -m pytest tests/test_latitude_column_runner.py -k identity -v
```

- [ ] **Step 3: Fix bookkeeping bugs before proceeding**

If columns diverge under uniform forcing, treat that as a framework bug.

---

## Phase D -- Campaign Script and Persistence

This phase makes the model usable for real runs.

---

### Task 6: Create `run_latitude_column_mc.py`

**Files:**
- Create: `EuropaProjectDJ/scripts/run_latitude_column_mc.py`

The script should expose a clean production entry point for the minimal model.

- [ ] **Step 1: Add explicit campaign presets**

Include a small set of named presets for:

- latitude grid
- surface-temperature mode
- strain mode
- basal-flux scenario
- number of iterations

Keep presets minimal and transparent. Do not hide calibration features inside them.

- [ ] **Step 2: Add output saving**

Save:

- raw arrays
- summary arrays
- forcing profiles
- campaign metadata

Recommended output path:

- `EuropaProjectDJ/results/latitude_column/`

Recommended filename pattern:

- `latitude_column_<campaign_name>_<n_iter>.npz`

- [ ] **Step 3: Add a simple textual summary**

Print:

- number of iterations
- latitude grid
- valid counts by latitude
- median `H`
- median `D_cond`
- median `D_conv`

No new derived “global convective fraction” metric should be introduced here.

- [ ] **Step 4: Add a smoke-run mode**

Support a tiny run for local validation:

- `n_iterations = 5`
- `n_lat = 5`

This should be the first executable path used in development.

- [ ] **Step 5: Run a smoke test**

Run the script with a tiny preset and confirm:

- output file written successfully
- raw arrays present
- forcing profiles present
- latitudes and shapes correct

- [ ] **Step 6: Commit**

```bash
git add EuropaProjectDJ/scripts/run_latitude_column_mc.py
git commit -m "feat: add minimal latitude-column campaign script"
```

---

## Phase E -- End-to-End Verification

This phase confirms the implementation is stable enough for larger runs.

---

### Task 7: Full targeted test pass

- [ ] **Step 1: Run the dedicated test files**

```bash
cd EuropaProjectDJ
python -m pytest tests/test_latitude_column_fields.py tests/test_latitude_column_runner.py -v
```

- [ ] **Step 2: Fix failures until green**

Do not move to production campaigns with known failures.

---

### Task 8: Small end-to-end campaign

- [ ] **Step 1: Run a small campaign**

Suggested first run:

- `n_iterations = 20`
- `n_lat = 19`
- one simple scenario, preferably `uniform`

- [ ] **Step 2: Inspect outputs**

Check:

- no missing forcing profiles
- valid counts are sensible
- medians vary smoothly with latitude
- no obvious discontinuities under smooth forcing laws

- [ ] **Step 3: Archive the first small result**

Save the produced NPZ in `EuropaProjectDJ/results/latitude_column/` and note the campaign configuration.

---

## Implementation Notes

- Keep this branch independent from `Europa2D`.
- Do not import `Europa2D/src/latitude_profile.py` into the runtime path.
- Reuse `SolverConfig` if possible, but do not inherit any `Europa2D`-specific assumptions.
- Use one common `reject_subcritical` policy for the whole campaign. Recommended default: `False`.
- Persist `sample_id` explicitly so future paired comparisons remain possible.
- Preserve raw forcing profiles in the output archive. They are part of the scientific result.

---

## Exit Criteria

The minimal latitude-column model is complete when all of the following are true:

- field-builder tests pass
- runner tests pass
- single-column parity passes
- uniform-field identity passes
- smoke campaign runs successfully
- raw NPZ output contains explicit forcing profiles and per-latitude diagnostics
- no forbidden extra closures were added

---

## Short Version

Build three small pieces:

1. a forcing-profile builder
2. a shared-draw latitude-column runner
3. a campaign script

Then refuse to proceed until:

1. `n_lat=1` matches standalone 1D
2. uniform forcing gives identical columns

That is the correct green path for this model.
