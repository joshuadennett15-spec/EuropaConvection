# Europa Ice Shell Model Physics Improvements

**Date:** 2026-04-11
**Status:** Draft
**Approach:** Feature-flagged modular development (Approach C)
**Scope:** Both 1D (EuropaProjectDJ) and 2D (Europa2D) solvers

## Motivation

Claude Deep Research identified seven areas where EuropaConvection can be improved
against recent literature. A critical audit of those recommendations against the
actual codebase found: (1) some are trivial — the code already exists but isn't
defaulted, (2) one major gap the report missed — GBS creep is absent despite
grain sizes squarely in the GBS-dominated regime, (3) Andrade rheology and Sobol
infrastructure are already partially built. This spec captures every actionable
improvement, with precise equations, parameter values, and references.

## Architecture: Feature Flags

Every improvement is behind a config flag so it can be toggled independently.
This enables factorial experiment matrices and isolated impact analysis.

All new flags have legacy defaults that reproduce current behavior. Existing
config files without the new keys fall back via `ConfigManager.get(section, key, default)`.

### New Config Flags

```json
{
    "thermal": {
        "CONDUCTIVITY_MODEL": "Carnahan"
    },
    "rheology": {
        "CREEP_MODEL": "composite_gbs",
        "GRAIN_MODE": "sampled",
        "WATTMETER_P": 2.0
    },
    "convection": {
        "NU_SCALING": "dv2021",
        "FK_CORRECTION": true,
        "GEOMETRY_CORRECTION": 1.0
    }
}
```

| Flag | Section | Values | Legacy default | Improved default |
|------|---------|--------|---------------|-----------------|
| `CONDUCTIVITY_MODEL` | thermal | `"Howell"`, `"Carnahan"`, `"CBE"` | `"Howell"` | `"Carnahan"` |
| `CREEP_MODEL` | rheology | `"diffusion"`, `"composite_gbs"` | `"diffusion"` | `"composite_gbs"` |
| `GRAIN_MODE` | rheology | `"sampled"`, `"wattmeter"` | `"sampled"` | `"sampled"` |
| `WATTMETER_P` | rheology | float (grain growth exponent) | `2.0` | `2.0` |
| `NU_SCALING` | convection | `"green"`, `"howell"`, `"dv2021"` | `"green"` | `"dv2021"` |
| `FK_CORRECTION` | convection | `true`, `false` | `false` | `true` |
| `GEOMETRY_CORRECTION` | convection | float multiplier | `1.0` | `1.0` |

---

## Section 1: Thermal Conductivity Default Swap

### Problem

The current default `k(T) = 567/T` (Klinger 1980 / Howell 2021) incorporates
contaminated low-quality data from Dillard & Timmerhaus (1966).

### Solution

Change default from `"Howell"` to `"Carnahan"` in `constants.py:45`.

### Equation

```
k(T) = 612 / T   [W/m/K]
```

### References

- Carnahan, Wolfenbarger, Jordan & Hesse (2021), EPSL 567, 116996
- Wolfenbarger et al. (2021), Data in Brief 36, 107079
- Confirmed by PlanetProfile (Vance et al.): uses identical `612/T` formula

### Implementation

**File:** `EuropaProjectDJ/src/constants.py:45`
- Wire `ConfigManager.get("thermal", "CONDUCTIVITY_MODEL", "Carnahan")` into the
  `model` parameter of `Thermal.conductivity()`
- All callers that currently use the default will automatically pick up Carnahan

**Impact:** ~8% increase in k(T). Slightly thicker conductive lids. All regression
test baselines will shift.

**Effort:** ~1 hour

---

## Section 2: GBS Composite Creep Model

### Problem

The current viscosity (`Physics.py:36`) implements only Nabarro-Herring + Coble
diffusion creep (grain-size exponent p=2). The sampled grain size range
(0.05-4 mm, centered at 0.6 mm) is squarely in the GBS-dominated regime
(0.4 mm - 30 mm), where GBS has p=1.4 and n=1.8. The absence of GBS means
viscosity is systematically overestimated, suppressing convection.

### Solution

Implement the full Goldsby & Kohlstedt (2001) composite flow law with four
parallel creep mechanisms.

### Approach: Reduced GBS + Dislocation Composite

The full four-mechanism Goldsby-Kohlstedt law (diffusion + GBS + basal slip +
dislocation) is stress-dependent and cannot be dropped into the current
`composite_viscosity(T, d, ...)` API without a stress argument. Instead, we
implement a **reduced two-mechanism composite** (GBS + dislocation) that is
sufficient for the grain-size range of interest (0.05-4 mm at 240-270 K):

**Strain-rate formulation** (additive, parallel mechanisms):

```
eps_dot_total = eps_dot_GBS + eps_dot_disl
             = A_GBS * d^(-m) * sigma^n_GBS * exp(-Q_GBS/(RT))
             + A_disl * sigma^n_disl * exp(-Q_disl/(RT))
```

**Effective viscosity** at a given stress sigma:

```
eta_eff = sigma / (2 * eps_dot_total)
```

**Flow law parameters** (Goldsby & Kohlstedt 2001, Table 3):

ALL VALUES CONVERTED TO SI (Pa, m, s). The original paper uses MPa-based units.
Conversion: A_SI = A_MPa * 1e-6^n (e.g., for GBS: 1e-6^1.8 = 10^-10.8).

| Mechanism | n | m (grain) | A_SI (Pa^-n m^m s^-1) | Q (kJ/mol) | T range |
|-----------|---|-----------|----------------------|------------|---------|
| GBS | 1.8 | 1.4 | 6.2e-14 * 1e-10.8 = ~9.8e-25 | 49 | T < 255 K |
| GBS | 1.8 | 1.4 | 3.9e-3 * 1e-10.8 = ~6.2e-14 | 192 | T > 255 K |
| Dislocation | 4.0 | 0.0 | 4.0e5 * 1e-24 = 4.0e-19 | 60 | all |

**CRITICAL: Unit conversion.** The A values in Goldsby & Kohlstedt (2001) are in
MPa^-n s^-1. The conversion to Pa^-n s^-1 is A_SI = A_MPa * (1e-6)^n. For
n=1.8 this is a factor of 10^-10.8; for n=4 this is 10^-24. Copying MPa values
directly into SI code would produce strain rates wrong by factors of 10^(6n).

**Stress convention and linearization:**

The current API returns viscosity without a stress argument. For the composite
law, we adopt the standard approach for parameterized convection models:

1. Estimate convective stress from boundary-layer scaling:
   ```
   sigma_conv = rho * g * alpha * DT_rh * delta_rh
   ```
   where `DT_rh = 2.24 * R * T_i^2 / E_a` and `delta_rh` is the rheological
   boundary layer thickness.

2. Compute effective viscosity at that stress:
   ```
   eta_eff(T, d, sigma) = sigma / (2 * eps_dot_total(T, d, sigma))
   ```

3. Use eta_eff in the Rayleigh number computation.

This linearization is evaluated once per Picard iteration, not per grid node.
The stress estimate updates as T_i converges.

**API change:** `composite_viscosity()` gains an optional `sigma` parameter.
When `CREEP_MODEL="diffusion"`, sigma is ignored (current behavior). When
`CREEP_MODEL="composite_gbs"`, sigma is required and defaults to the
boundary-layer estimate if not provided.

### Config flag

```json
"rheology": { "CREEP_MODEL": "composite_gbs" }
```

`"diffusion"` preserves current behavior. `"composite_gbs"` activates the
reduced GBS + dislocation composite with explicit stress linearization.

### Implementation

**File:** `EuropaProjectDJ/src/Physics.py:36`
- Add `creep_model` and `sigma` parameters to `composite_viscosity()`
- When `"composite_gbs"`: compute GBS + dislocation strain rates at given sigma,
  return effective viscosity. Validate sigma is provided.
- When `"diffusion"`: existing code path unchanged, sigma ignored
- Add `convective_stress()` helper for boundary-layer stress estimation
- All A values stored in SI units with conversion documented in comments

**Both solvers:** Physics.py is shared; change propagates to 1D and 2D automatically.

**Note on basal slip:** Omitted from the reduced composite. Basal slip (n=2.4)
acts as an upper bound on GBS strain rate at high stress but is not the
rate-limiting mechanism at Europa's convective stresses (~1-100 kPa). If needed
later, it can be added as a series element with GBS without changing the API.

### References

- Goldsby & Kohlstedt (2001), JGR 106, 11017-11030
- Barr & McKinnon (2007), JGR 112, E02012
- Harel et al. (2020), Icarus 338, 113448

**Effort:** ~3-4 hours

---

## Section 3: Deschamps & Vilella 2021 Mixed-Heating Scaling

### Problem

The current Green et al. (2021) scaling `Nu = 0.3446 * Ra^(1/3)` was derived for
bottom-heated-only stagnant-lid convection. Europa has both internal (tidal) and
basal (ocean) heating. Four stub locations in `Convection.py` (lines 871, 1031,
1050, 1181) raise `NotImplementedError` for `"dv2021"`.

### Solution

Implement the DV2021 mixed-heating scaling in all four stub locations.

### Equations (DV2021, JGR Planets, Tables 1-2)

**IMPORTANT:** The DV2021 scaling laws are for nondimensional quantities —
interior temperature, surface heat flux, and lid thickness — as functions of
effective Rayleigh number Ra_eff, viscosity contrast gamma, and nondimensional
internal heating rate h. They are NOT a direct Nu(Ra_i, H) law.

The published fitted laws (DV2021 Table 2, Mendeley dataset 4hxsj8rw86/1):

```
Theta_i  = a1 * Ra_eff^b1 * gamma^c1 * (1 + h)^d1     [nondim. interior T]
Phi_s    = a2 * Ra_eff^b2 * gamma^c2 * (1 + h)^d2     [nondim. surface heat flux]
delta_lid = a3 * Ra_eff^b3 * gamma^c3 * (1 + h)^d3     [nondim. lid thickness]
```

where:
- `Ra_eff` = effective Rayleigh number (defined with reference viscosity at
  some characteristic temperature, per the paper's convention)
- `gamma` = total viscosity contrast across the layer (= exp(theta) in FK)
- `h` = nondimensional internal heating rate = H_vol * D^2 / (k * DT),
  where H_vol is volumetric heating (W/m^3) — this is internal heating
  normalized by the conductive heat flux scale, NOT internal/(internal+basal)
- Fitted coefficients (a, b, c, d) from Mendeley dataset for each geometry

**Urey ratio / internal heating rate:**

The paper's nondimensional h is:
```
h = H_vol * D^2 / (k * DT)
```
where H_vol = volumetric tidal dissipation (W/m^3) from `IcePhysics.tidal_heating()`,
D = convecting layer thickness, k = thermal conductivity, DT = temperature drop
across the convecting layer.

**Basal heat flux:** Do NOT use the global `HeatFlux.RADIOGENIC_FLUX +
TIDAL_SILICATE_FLUX` constants. Use the actual per-run basal flux already
flowing through the solver (from the sampler via `q_basal` parameter). The
latitude-dependent 2D sampler (`latitude_sampler.py:143`) computes run-specific
q_basal that may differ from the global constants.

**FK correction** (Harel et al. 2020; Grigne 2023): When `FK_CORRECTION=true`,
apply 0.7x multiplier to the heat flux scaling to correct the ~30% FK
overestimate vs Arrhenius.

### Calibration consistency with composite_gbs

DV2021 was calibrated on FK-style temperature-dependent Newtonian viscosity,
NOT on a grain-size/stress-dependent non-Newtonian composite law. When using
`CREEP_MODEL="composite_gbs"` together with `NU_SCALING="dv2021"`:

- The GBS composite viscosity is used for the **thermal profile** computation
  (conduction through the lid, convective heat transport)
- The **Ra for the DV2021 scaling law** must still use FK-equivalent viscosity
  (eta_FK at T_i), to match the calibration basis of the scaling law
- This is consistent with how the current code already handles the Green
  scaling: `Convection.py:1021` explicitly selects FK viscosity for Ra when
  using `nu_scaling="green"`
- Add an explicit mapping rule: `Ra_for_scaling = Ra(eta_FK)` regardless of
  which creep model is active for the thermal profile

### Self-consistent iteration

DV2021 requires T_i to compute Ra_eff and gamma, but T_i is an output. This
slots into the existing Picard iteration in `_find_transition_temperatures()` —
same loop structure, different equations inside. The nondimensional h also
depends on D (layer thickness), creating a coupled system that the iteration
must resolve.

### Config flag

Already exists as `NU_SCALING: "dv2021"`. Implementation fills the four stubs.

### Data & References

- Paper: [Deschamps & Vilella 2021, JGR Planets](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JE006963)
- Data: [Mendeley dataset 4hxsj8rw86/1](https://data.mendeley.com/datasets/4hxsj8rw86/1)
- FK correction: Harel et al. (2020), Icarus 338, 113448; Grigne (2023), GJI 235, 2410

**Effort:** ~4-5 hours (implement once in helper, call from all four stubs)

---

## Section 4: Behn et al. (2021) Wattmeter Equilibrium Grain Size

### Problem

Grain size is currently a free parameter sampled as LogNormal(0.6 mm, sigma=0.35),
clipped [0.05, 4.0 mm]. The Behn et al. (2021) wattmeter provides a physics-based
equilibrium calculation that collapses this dimension.

### Solution

New module `EuropaProjectDJ/src/wattmeter.py` implementing the ice paleowattmeter.
Dual-mode operation: `"sampled"` (current) vs `"wattmeter"` (physics-constrained).

### Equations (Behn et al. 2021, The Cryosphere 15, 4589-4605)

**Grain growth** (Eq. 13, growth term):

```
dd/dt|_growth = K_gg / (p * d^(p-1)) * exp(-Q_gg / (R*T))
```

**Grain reduction** (Eq. 13, reduction term):

```
dd/dt|_red = [lambda_GBS*(1-beta) + lambda_disl*beta] * (d^2 / (c*gamma)) * sigma * eps_dot
```

where `beta = W_dot_disl / W_dot_total` (dislocation work fraction).

**Equilibrium grain size** (Eq. 14, setting dd/dt = 0):

```
d_ss^(1+p) = K_gg * exp(-Q_gg/(R*T)) * c * gamma / ((p-1) * lambda_eff * sigma * eps_dot)
```

```
d_ss = [K_gg * exp(-Q_gg/(R*T)) * c * gamma / ((p-1) * lambda_eff * sigma * eps_dot)]^(1/(1+p))
```

With p=6.03, the exponent is 1/7.03 ~ 0.14. Grain size is weakly sensitive to
stress — good for convergence.

### Ice Ih Parameters

Two parameter sets are available, depending on impurity assumptions:

**Bubble-free / pure ice (DEFAULT for this model):**

| Parameter | Value | Source |
|-----------|-------|--------|
| p (grain growth exponent) | 2.0 | Standard normal grain growth (no pinning) |
| K_gg (growth rate constant) | 1.1e-8 m^2/s | Azuma et al. 2012 (pure ice fit) |
| Q_gg (growth activation energy) | 40 kJ/mol | Azuma et al. 2012 |
| lambda (work fraction coefficient) | 0.01 (range 0.005-0.015) | Behn 2021 |
| c (geometric constant) | pi ~ 3.14 | Austin & Evans 2007 |
| gamma (grain boundary energy) | 0.065 J/m^2 | Ketcham & Hobbs 1969 |

**Bubble-rich / impurity-drag (sensitivity branch):**

| Parameter | Value | Source |
|-----------|-------|--------|
| p (grain growth exponent) | 6.03 +/- 0.25 | Behn 2021 (joint fit to bubble-rich cores) |
| K_gg (growth rate constant) | 6.18e-6 m^p/s | Behn 2021 |
| Q_gg (growth activation energy) | 49.6 kJ/mol | Behn 2021 |
| lambda, c, gamma | same as above | same |

**Rationale for p=2 default:** The rest of the model carries no explicit
impurity/bubble physics (f_salt=0, pure-ice baseline per PARAMETER_PRIOR_AUDIT_2026.md).
Using p=6.03 would silently bake in impurity-drag grain-growth suppression that
is inconsistent with the pure-ice assumption elsewhere. The p=6.03 branch is
available as a sensitivity experiment via config:

```json
"rheology": { "WATTMETER_P": 2.0 }
```

Set to 6.03 for impurity-drag sensitivity runs.

**Note:** Both parameter sets differ significantly from silicate defaults
(ASPECT: p=3, lambda=0.1, gamma=1.0). Using silicate parameters would give
wrong grain sizes for ice.

### Iteration strategy (wattmeter mode)

1. Start with d_guess = 1 mm (or sampled value as seed)
2. Compute eta(T_i, d_guess) using composite creep (Section 2)
3. Compute sigma_conv from boundary layer scaling
4. Compute eps_dot from composite flow law (GBS + dislocation)
5. Compute beta = W_dot_disl / W_dot_total
6. Compute d_eq from wattmeter equation
7. If |d_eq - d_guess| / d_guess > 0.01: d_guess = d_eq, goto 2
8. Typically converges in 3-5 iterations (exponent 1/7.03 is a contraction)

### Config flag

```json
"rheology": { "GRAIN_MODE": "wattmeter" }
```

`"sampled"` preserves current behavior. `"wattmeter"` computes d_eq per MC realization.

### What MC still samples in wattmeter mode

Everything except d_grain: T_surf, epsilon_0, q_basal, Q_v, Q_b, mu_ice, D_H2O.
Grain size becomes a derived quantity varying per realization.

### References

- Paper: [Behn, Goldsby & Hirth 2021, The Cryosphere 15, 4589-4605](https://tc.copernicus.org/articles/15/4589/2021/)
- Reference impl: [ASPECT grain_size_evolution.cc](https://github.com/geodynamics/aspect/blob/main/source/material_model/grain_size.cc)
- Ice grain boundary energy: Ketcham & Hobbs (1969), J. Glaciology
- Grain growth kinetics: Azuma et al. (2012)

**Effort:** ~5-6 hours

---

## Section 5: Sobol Global Sensitivity Analysis

### Problem

No published Sobol sensitivity analysis exists for Europa thermal models. The
existing test infrastructure (`test_sobol_workflow.py`) validates the pipeline at
small N but hasn't been run in production.

### Solution

Production Sobol analysis across three model configurations, integrated with the
existing `thesis_stats.py` statistical framework.

### Parameter space (continuous, per configuration)

| Parameter | Distribution | Sobol group |
|-----------|-------------|-------------|
| d_grain (sampled mode) | LogNormal(0.6mm, sigma=0.35) | grain |
| epsilon_0 (tidal strain) | LogNormal(1.2e-5, sigma=0.3) | tidal |
| q_radiogenic | TruncNormal(7 mW/m^2, sigma=1) | basal_heat |
| q_tidal_silicate | LogUniform(2-20 mW/m^2) | basal_heat |
| T_surf | Normal(104 K, sigma=7) | surface |
| D_H2O | Normal(127 km, sigma=21) | ocean |
| mu_ice | TruncNormal(3.5 GPa, sigma=0.5) | rigidity |
| Q_v, Q_b | Normal(+/-5%) | activation |

### Three configurations (discrete, across feature flags)

1. **baseline** — Howell k, diffusion creep, Green scaling, sampled grain
2. **improved** — Carnahan k, GBS creep, DV2021 scaling, sampled grain
3. **wattmeter** — Carnahan k, GBS creep, DV2021 scaling, wattmeter grain

### Outputs per configuration

- S1 (first-order) and ST (total-order) indices with 95% CI
- Target QoIs: H_total, D_cond, D_conv, lid_fraction, convective_flag
- S2 (second-order interaction) matrix for top 4 parameters
- Convergence plot: S1/ST vs N (128, 256, 512, 1024)

### Sample budget

- Saltelli with k=8 grouped parameters: N*(2k+2) = 1024*18 = 18,432 evals/config
- At ~0.3 s/eval on 12 CPUs: ~25 min/config
- 3 configs x 25 min = ~75 min total compute

### Statistical synthesis (integration with thesis_stats.py)

The Sobol outputs feed into the existing statistical machinery:

| Tool | Application |
|------|-------------|
| Kendall W | Parameter ranking concordance across 3 configurations |
| Cliff's delta | Effect size of each physics improvement on output distributions |
| Jonckheere-Terpstra | Monotonic trend test: baseline -> improved -> wattmeter |
| Pairwise + FDR | Which configuration pairs differ significantly? |
| Quantile regression | Does grain size control the median or just the tails? |
| Bootstrap CI | Precision of Sobol index estimates |

### Implementation

**New scripts:**
- `autoresearch/experiments/run_sobol_analysis.py` — production runner
- `autoresearch/experiments/run_sobol_synthesis.py` — statistical synthesis

**Existing infrastructure used:**
- SALib (`saltelli.sample()`, `sobol.analyze()`)
- `test_sobol_workflow.py` patterns promoted to production
- `thesis_stats.py` statistical functions

### References

- SALib: [github.com/SALib/SALib](https://github.com/SALib/SALib)
- Saltelli et al. (2010), Computer Physics Communications 181, 259-270
- Howell (2021), PSJ 2, 129 — Monte Carlo but no Sobol decomposition

**Effort:** ~4-5 hours

---

## Section 6: Seven Pitfalls Audit

### Audit Matrix

| # | Pitfall | Status | Action |
|---|---------|--------|--------|
| 1 | Wrong reference viscosity for Ra | CLEAN: uses eta(T_mean_conv) | Add assertion guard |
| 2 | Inconsistent FK parameter | VERIFY: confirm theta = E_a/(R*T_i^2) | Read _find_transition_temperatures(), fix if wrong |
| 3 | FK overestimate (~20-30%) | ADDRESSED: FK_CORRECTION flag in Section 3 | Config-driven correction |
| 4 | Wrong scaling regime (beta) | CLEAN: hardcoded beta=1/3 (chaotic) | Document as verified |
| 5 | Implicit coupling not iterated | CLEAN: Picard 3 iters, 0.01 K tol | Consider increasing to 5 for GBS nonlinearity |
| 6 | 2D vs 3D geometry correction | NOT APPLIED | Add GEOMETRY_CORRECTION multiplier |
| 7 | Enhanced k_eff application | VERIFY: Nu*k only in convecting region | Add regression test |

### Implementation

- Pitfalls 1, 4, 5: Add assert guards + docstring references
- Pitfall 2: Verify FK formula, fix if incorrect
- Pitfall 3: Already covered by FK_CORRECTION
- Pitfall 6: `GEOMETRY_CORRECTION` config flag (default 1.0, set 1.2 for 3D estimate)
- Pitfall 7: New regression test verifying Nu enhancement is zero above interface

### References

- Solomatov (1995); Davaille & Jaupart (1993)
- Reese et al. (1999); Harel et al. (2020)
- Deschamps & Lin (2014), PEPI 234, 27
- Grigne (2023), GJI 235, 2410

**Effort:** ~2-3 hours

---

## Section 6b: Tidal Dissipation Validation (TidalPy Benchmark)

### Problem

The Andrade implementation (`Physics.py:265-282`) is active with alpha=0.2,
zeta=1.0 but has not been validated against an independent community tool.

### Solution

Add a validation test comparing our tidal dissipation against TidalPy.

### Verification results (from research)

Our Andrade formulation matches TidalPy (Renaud & Henning 2018) exactly:

```
J*(w) = J_U - i/(eta*w) + J_U * (i*J_U*eta*zeta*w)^(-alpha) * Gamma(1+alpha)
```

Parameters: alpha=0.2 matches paper Table 1 nominal. zeta=1.0 is the standard
value (Castillo-Rogez et al. 2011). TidalPy code defaults to alpha=0.3 but the
paper uses 0.2. Our choice is correct for ice Ih.

### Implementation

- `pip install TidalPy` as dev dependency
- New test in `EuropaProjectDJ/tests/test_tidal_validation.py`:
  - Reference config: T=250 K, eta=5e13 Pa*s, mu=3.3 GPa, epsilon_0=1e-5
  - Compare our `IcePhysics.tidal_heating()` vs TidalPy Andrade compliance
  - Pass criterion: <1% relative error

### References

- Paper: Renaud & Henning (2018), ApJ 857:98
- Code: [github.com/jrenaud90/TidalPy](https://github.com/jrenaud90/TidalPy)
- Bierson (2024), Icarus — zeta sensitivity analysis

**Effort:** ~1-2 hours

---

## Section 7: Lateral Ice Flow Diagnostic

### Problem

The model computes independent latitude columns with no meridional coupling.
Ashkenazy et al. (2018) showed lateral ice flow reduces the equator-to-pole
thickness contrast by ~10x for soft ice.

### Solution

Post-hoc diagnostic script (no solver changes). Option to upgrade later.

### Equations (Ashkenazy, Sayag & Tziperman 2018, Nature Astronomy)

**Thin-film gravity current:**

```
dH/dt = div(D * grad(H)) + S(phi)
D = (2*A*(rho*g)^n / (n+2)) * H^(n+2)
```

where A = Glen flow law rate factor, n=3, S(phi) = net freeze/melt source.

### Implementation

**Script:** `Europa2D/scripts/run_lateral_flow_diagnostic.py`

For each MC realization's H(phi) profile:
1. Compute diffusivity D(phi) from Glen flow law at basal temperature
2. Estimate relaxation timescale tau = L^2 / D
3. Solve diffusion to steady state for H_eq(phi)
4. Report: original DeltaH, relaxed DeltaH_eq, reduction factor

**Output:** Diagnostic table + figure showing before/after profiles and
distribution of reduction factors across the MC ensemble.

### References

- Ashkenazy, Sayag & Tziperman (2018), Nature Astronomy 2, 43-49

**Effort:** ~2 hours

---

## Section 8: Experiment Matrix Extension

### Design

Extend `run_experiment_matrix.py` to include the new feature flags as grid
dimensions. Recommended factorial for thesis:

```
conductivity:  [Howell, Carnahan]         — 2
creep:         [diffusion, composite_gbs]  — 2
nu_scaling:    [green, dv2021]             — 2
grain_mode:    [sampled, wattmeter]        — 2
```

16 physics combinations. Combined with existing ocean pattern (3) and surface
preset (2) dimensions: 16 * 6 = 96 total combinations.

At N=150 samples/combo and ~0.3 s/eval on 12 CPUs:
- Per combo: ~4 seconds
- Total: 96 * 150 * 0.3 / 12 ~ 6 minutes

Manageable on local hardware. For N=500 (thesis-grade): ~20 minutes.

### Implementation

- Extend grid definition in `run_experiment_matrix.py`
- Each combo writes a JSON diagnostics file (existing pattern)
- Add summary aggregation script for cross-combo comparison

**Effort:** ~1-2 hours

---

## Validation Strategy

### Regression tests (update baselines)

Every section triggers regression test updates. Run full suite after each change:
- `Europa2D/tests/` — 35+ tests
- `EuropaProjectDJ/tests/` — 30+ tests

### Literature benchmarks

| Target | Source | Expected range |
|--------|--------|---------------|
| Shell thickness | Juno MWR (Levin et al. 2026) | 29 +/- 10 km |
| Convective fraction | Howell (2021) | ~1/3 of shell |
| Equilibrium thickness | Green & Montesi (2021) | 13-25 km (equator) |
| Polar shell | Quick & Marsh (2015) | ~17 km (with tidal) |

### Cross-validation

- TidalPy benchmark (Section 6b): <1% error on Andrade dissipation
- 1D vs 2D parity: single-column 2D must match 1D within 2%
- Carnahan k(T) vs PlanetProfile: exact match (both 612/T)

---

## Implementation Order

Audit/plumbing work comes first to ensure the foundation is correct before
building new physics on top.

1. **Pitfalls audit** (Section 6) — verify Ra/transition/heating plumbing is
   correct before any new physics changes. Fix FK parameter if wrong, add
   assertion guards, verify k_eff application.
2. **Conductivity swap** (Section 1) — trivial foundation change, no dependencies
3. **GBS composite creep** (Section 2) — reduced GBS+dislocation with correct
   SI units and explicit stress linearization. Highest physics impact.
4. **TidalPy validation** (Section 6b) — quick, independent, confirms tidal
   heating is correct before DV2021 needs it for the Urey ratio
5. **DV2021 scaling** (Section 3) — depends on verified plumbing (step 1) and
   correct viscosity framework (step 3). Must use FK-equivalent Ra for scaling
   law even when composite_gbs is active for thermal profile.
6. **Wattmeter grain size** (Section 4) — depends on GBS creep (step 3) for
   the dislocation strain rate in the wattmeter equation
7. **Config & experiment matrix** (Section 8) — wires all flags together
8. **Sobol analysis** (Section 5) — runs on final model with all flags
9. **Lateral flow diagnostic** (Section 7) — post-hoc, no dependencies

Total estimated effort: ~24-30 hours of implementation.

---

## Out of Scope (Future Work)

- **Runtime lateral ice flow** — meridional coupling between columns (Section 7 is diagnostic-only)
- **Bercovici-Ricard two-phase grain damage** — most complete theory but unadapted to ice
- **Full 2D/3D numerical convection** — ASPECT/StagYY scale; outside parameterized model scope
- **Porosity-dependent conductivity** — relevant for near-surface only, not convective dynamics
- **Clathrate hydrate conductivity** — would require compositional model not currently in scope
