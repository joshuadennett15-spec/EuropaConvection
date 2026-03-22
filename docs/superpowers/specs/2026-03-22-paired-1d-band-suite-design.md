# 1D Equator/Pole Endmember Proxy Suite

**Date:** 2026-03-22
**Goal:** Run standalone 1D MC simulations at equatorial and polar endmember conditions under three ocean heat transport scenarios to characterize how shell thickness varies with latitude forcing.

## Motivation

Produce publication-quality 1D thickness distributions for equatorial and polar endmember proxies. Each scenario (uniform, equator-enhanced, polar-enhanced ocean transport) controls the local tidal flux seen at each endmember. This is a standalone 1D production suite — not a 2D validation exercise.

## New Sampler: AuditedEndmemberSampler

**File:** `EuropaProjectDJ/src/audited_endmember_sampler.py`

Inherits from `AuditedShellSampler`. Constructor accepts endmember-specific overrides for T_surf, epsilon_0, and a local tidal flux multiplier. All other parameters (d_grain, Q_v, Q_b, rheology, composition) come from the audited prior unchanged.

### Constructor parameters

- `T_surf_mean`, `T_surf_std`, `T_surf_clip` — surface temperature distribution
- `epsilon_0_log_center`, `epsilon_0_log_sigma`, `epsilon_0_clip` — tidal strain distribution
- `q_tidal_multiplier` — local tidal flux relative to global mean (1.0 = no redistribution)
- `seed`

### sample() method

1. Call `super().sample()` to get full audited parameter set
2. Override `T_surf` with endmember-specific normal draw, clipped
3. Override `epsilon_0` with endmember-specific lognormal draw, clipped
4. Scale `P_tidal *= q_tidal_multiplier` (local flux adjustment, not a budget split)
5. Return params

### Endmember parameter presets

| Parameter | Equatorial proxy | Polar proxy |
|-----------|-----------------|-------------|
| T_surf | N(110, 5) K, [95, 120] | N(50, 5) K, [45, 80] |
| epsilon_0 | lognormal(6e-6, 0.2 dex), [2e-6, 2e-5] | lognormal(1.2e-5, 0.2 dex), [2e-6, 3.4e-5] |

These are endpoint proxies representing equatorial and polar forcing conditions, not area-weighted band averages.

### Scenario q_tidal_multiplier values

Derived from literature redistribution fractions normalized to the uniform case (fraction / 0.5):

| Scenario | Eq multiplier | Pole multiplier | Derivation | Reference |
|----------|--------------|----------------|------------|-----------|
| Uniform | 1.00 | 1.00 | 0.500/0.500 | Ashkenazy & Tziperman (2021) |
| Equator-enhanced | 1.17 | 0.83 | 0.583/0.500, 0.417/0.500 | Soderlund et al. (2014) |
| Polar-enhanced | 0.67 | 1.33 | 0.333/0.500, 0.667/0.500 | Lemasquerier et al. (2023) |

The multiplier scales the audited P_tidal directly. The 1D solver computes `q_tidal = P_tidal / A_surface`, so a multiplier of 1.17 means the equator sees 17% more local tidal flux than the global mean. Radiogenic heating is unaffected (spatially uniform in the mantle).

### Notes on parameter choices

- **Equatorial T_surf N(110, 5)** matches the `AuditedEquatorialSampler`, not the older `budget_samplers.py` N(108, 2).
- **Polar T_surf clip [45, 80]** aligns with the audited T_floor=46 K calibration. The older `budget_samplers.py` used [35, 70] which predates this.

### Named subclasses (Windows pickling)

Each subclass hard-codes its endmember + scenario and accepts only `seed` as a constructor parameter (required by `MonteCarloRunner._run_single_sample`):

1. `UniformEqSampler`
2. `UniformPoleSampler`
3. `SoderlundEqSampler`
4. `SoderlundPoleSampler`
5. `LemasquerierEqSampler`
6. `LemasquerierPoleSampler`

## Driver Script

**File:** `EuropaProjectDJ/scripts/run_endmember_suite.py`

Runs 6 ensembles (3 scenarios x 2 endmembers):

| Run | Endmember | Scenario | Output file |
|-----|-----------|----------|-------------|
| 1 | Equatorial | Uniform | `endmember_uniform_eq_andrade.npz` |
| 2 | Polar | Uniform | `endmember_uniform_pole_andrade.npz` |
| 3 | Equatorial | Soderlund | `endmember_soderlund_eq_andrade.npz` |
| 4 | Polar | Soderlund | `endmember_soderlund_pole_andrade.npz` |
| 5 | Equatorial | Lemasquerier | `endmember_lemasquerier_eq_andrade.npz` |
| 6 | Polar | Lemasquerier | `endmember_lemasquerier_pole_andrade.npz` |

### Run configuration

- N = 5,000 iterations per ensemble
- Shared base seed across all 6 runs (reproducible ensembles, not per-sample pairing — `imap_unordered` and invalid-run filtering break index alignment across runs)
- `reject_subcritical=False` for all runs
- Andrade rheology (asserted from config.json)
- Named sampler subclasses for Windows multiprocessing pickling
- Results to `EuropaProjectDJ/results/`

## Key design decisions

1. **AuditedShellSampler base** — consistent with 2026 audited priors.
2. **q_tidal_multiplier, not p_fraction** — the 1D solver uses `q_tidal = P_tidal / A_surface` for local flux. A budget fraction would halve both endmembers in the uniform case. Multipliers preserve the correct local forcing.
3. **reject_subcritical=False everywhere** — no asymmetric branch handling between endmembers.
4. **Shared base seed** — same base seed for all 6 runs so audited prior draws (d_grain, Q_v, Q_b, etc.) come from the same RNG sequence, reducing sampler-side noise in cross-endmember comparisons. This is ensemble-level reproducibility, not sample-wise pairing (invalid-run filtering breaks index alignment).
5. **Endpoint proxies, not band averages** — T_surf and epsilon_0 distributions represent equatorial and polar forcing conditions directly, not area-weighted integrals over latitude bands.
