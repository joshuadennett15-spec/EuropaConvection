# Paired 1D Band Suite Design

**Date:** 2026-03-22
**Goal:** Run 1D MC simulations with parameter sets consistent with equatorial (0-30°) and polar (60-90°) latitude bands under three ocean transport scenarios, to cross-check 2D conductive lid thickness estimates.

## Motivation

The 2D model appears to overestimate conductive lid thickness. By running the trusted 1D solver with band-representative parameters, we can isolate whether the discrepancy comes from the 2D solver/geometry, the 1.20x tidal uplift, or something else.

## New Sampler: AuditedBandSampler

**File:** `EuropaProjectDJ/src/audited_band_sampler.py`

Inherits from `AuditedShellSampler`. Constructor accepts band-specific overrides for T_surf, epsilon_0, and P_tidal fraction. All other parameters (d_grain, Q_v, Q_b, rheology, composition) come from the audited prior unchanged.

### Constructor parameters

- `T_surf_mean`, `T_surf_std`, `T_surf_clip` — surface temperature distribution
- `epsilon_0_log_center`, `epsilon_0_log_sigma`, `epsilon_0_clip` — tidal strain distribution
- `p_fraction` — fraction of audited P_tidal assigned to this band
- `seed`

### sample() method

1. Call `super().sample()` to get full audited parameter set
2. Override `T_surf` with band-specific normal draw, clipped
3. Override `epsilon_0` with band-specific lognormal draw, clipped
4. Scale `P_tidal` by `p_fraction`
5. Return params

### Band parameter presets

| Parameter | Equatorial (0-30°) | Polar (60-90°) |
|-----------|-------------------|----------------|
| T_surf | N(110, 5) K, [95, 120] | N(50, 5) K, [45, 80] |
| epsilon_0 | lognormal(6e-6, 0.2 dex), [2e-6, 2e-5] | lognormal(1.2e-5, 0.2 dex), [2e-6, 3.4e-5] |

### Scenario P_tidal fractions

| Scenario | Equatorial | Polar | Reference |
|----------|-----------|-------|-----------|
| Uniform | 0.500 | 0.500 | Ashkenazy & Tziperman (2021) |
| Equator-enhanced | 0.583 | 0.417 | Soderlund et al. (2014) |
| Polar-enhanced | 0.333 | 0.667 | Lemasquerier et al. (2023) |

P_tidal for each band = `fraction * P_tidal_from_audited_prior` (no uplift, no fixed budget).

### Notes on parameter choices

- **Equatorial T_surf N(110, 5)** matches the 2D `latitude_sampler.py` T_eq and the `AuditedEquatorialSampler`, not the older `budget_samplers.py` N(108, 2).
- **Polar T_surf clip [45, 80]** aligns with the 2D model's T_floor=46 K default. The older `budget_samplers.py` used [35, 70] which predates the audited T_floor calibration.

### Named subclasses (Windows pickling)

Each subclass hard-codes its band + scenario and accepts only `seed` as a constructor parameter (required by `MonteCarloRunner._run_single_sample`):

1. `UniformEqBandSampler`
2. `UniformPoleBandSampler`
3. `SoderlundEqBandSampler`
4. `SoderlundPoleBandSampler`
5. `LemasquerierEqBandSampler`
6. `LemasquerierPoleBandSampler`

## Driver Script

**File:** `EuropaProjectDJ/scripts/run_paired_band_suite.py`

Runs 6 ensembles (3 scenarios x 2 bands):

| Run | Band | Scenario | Output file |
|-----|------|----------|-------------|
| 1 | Equatorial | Uniform | `band_uniform_eq_andrade.npz` |
| 2 | Polar | Uniform | `band_uniform_pole_andrade.npz` |
| 3 | Equatorial | Soderlund | `band_soderlund_eq_andrade.npz` |
| 4 | Polar | Soderlund | `band_soderlund_pole_andrade.npz` |
| 5 | Equatorial | Lemasquerier | `band_lemasquerier_eq_andrade.npz` |
| 6 | Polar | Lemasquerier | `band_lemasquerier_pole_andrade.npz` |

### Run configuration

- N = 5,000 iterations per ensemble
- Shared seed across all 6 runs for paired comparison
- `reject_subcritical=False` for all runs
- Andrade rheology (asserted from config.json)
- Named sampler subclasses for Windows multiprocessing pickling
- Results to `EuropaProjectDJ/results/`
- No q_tidal_scale uplift (1.0x, raw audited prior)

## Key design decisions

1. **AuditedShellSampler base** — not HowellParameterSampler. Consistent with 2026 audited priors and the 2D model.
2. **No tidal uplift** — q_tidal_scale=1.0. The 2D model's 1.20x uplift is suspected of contributing to thick lids; keeping 1.0x isolates that.
3. **reject_subcritical=False everywhere** — no asymmetric branch handling between bands.
4. **Shared seed** — same base seed for all 6 runs so rheology draws are from the same RNG sequence, reducing sampler-side noise in cross-band comparisons.
