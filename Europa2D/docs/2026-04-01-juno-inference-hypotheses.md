# Competing Hypotheses: What Controls Europa's Conductive Lid Thickness?

**Date:** 2026-04-01
**Context:** 2D axisymmetric ice shell model with 4 ocean transport scenarios, corrected dt=1e12

---

## Core Question

Juno MWR measured D_cond = 29 +/- 10 km at ~35 deg latitude (Levin et al. 2025, pure
water ice). Our 2D model with broad priors (0.6 mm grain center, Howell 2021 audited
distributions) produces D_cond median ~19 km at 35 deg -- a ~10 km underprediction.

Three questions follow:
1. What physical parameter or mechanism is responsible for the discrepancy?
2. Can the Juno constraint discriminate between ocean transport regimes?
3. What is the most scientifically defensible way to report model results against Juno?

---

## Observations From This Study

| Observation | Source |
|---|---|
| D_cond(35) median = 19.3 km with 0.6mm grain prior | 250-sample MC, uniform scenario |
| D_cond(35) median = 26.8 km with 1.0mm grain prior | autoresearch experiment 2 |
| D_cond(35) median = 29.7 km with 1.5mm grain prior | autoresearch experiment 4 |
| Juno reweighting of 0.6mm prior shifts D_cond to 24.5 km | importance reweighting |
| Juno-reweighted samples have higher Nu|conv (7.6) than 1.5mm prior (4.8) | conditional diagnostics |
| All 4 ocean scenarios produce D_cond ~19-20 km at 35 deg (0.6mm prior) | indistinguishable at single point |
| Ocean scenarios ARE distinguishable in latitude structure (H_eq, H_pole) | 4-scenario comparison |
| Reducing q_tidal_scale 1.20 -> 1.00 worsened the score (more thin shells) | autoresearch experiment 1 |

---

## Competing Hypotheses

### H1: Grain Size is the Primary Control on D_cond

The audited 0.6 mm grain center (Howell 2021) underestimates Europa's equilibrium
grain size. Barr & McKinnon (2007) show equilibrium recrystallization grains of
1-80 mm for Europa conditions. Murioz-Iglesias et al. (2023) find 30-80 mm in
absence of impurities. A larger grain size produces higher ice viscosity, suppresses
convection onset, and thickens the conductive lid.

**Prediction:** D_cond increases monotonically with grain size center. The relationship
is approximately D_cond ~ 19 + 7 * log10(d_grain/0.6mm) km at 35 deg.

**What this result means:** The Juno measurement primarily constrains Europa's ice
grain size (or equivalently, the effective viscosity of the ice shell) rather than
the ocean transport regime.

**Concern:** Larger grains suppress convective vigor globally. At 1.5 mm, only 27%
of equatorial samples convect (vs 54% at 0.6 mm), and conditional Nu drops from
6.4 to 4.8. If Europa has active convection (inferred from surface geology --
chaos terrain, double ridges), the 1.5 mm prior may overpredict lid thickness.

**What would weaken this hypothesis:** If grain sizes > 1 mm produce shell structures
inconsistent with other observational constraints (surface heat flux, elastic
thickness, chaos terrain distribution).

### H2: Bayesian Selection Within the Broad Prior

The 0.6 mm prior is the more defensible starting point because it encompasses the
full audited parameter range. Juno simply selects a subset of parameter space --
the samples with thicker conductive lids. These samples achieve thick D_cond through
specific combinations of parameters (likely larger grains, lower tidal heating,
higher viscosity) while maintaining vigorous convection when it occurs.

**Prediction:** Juno-reweighted posterior has D_cond ~24-25 km with conditional
Nu ~7-8, compared to 1.5 mm prior D_cond ~28 km with Nu ~4-5. The reweighted
posterior preserves more convective vigor.

**What this result means:** The 0.6 mm prior already contains Juno-consistent
samples. The constraint narrows the posterior but doesn't require changing the
prior. The posterior convection properties (higher Nu, lower convecting fraction)
may be more realistic than the flat 1.5 mm prior.

**Concern:** The reweighted posterior only reaches D_cond ~24.5 km, still 4.5 km
below Juno's center. The effective sample size drops (N_eff ~ 167/250), and the
posterior width remains large. The 29 +/- 10 km uncertainty is so broad that
the reweighting is mild.

**What would weaken this hypothesis:** If the reweighted posterior produces
convection properties inconsistent with surface geological constraints.

### H3: A Compromise -- Modest Grain Shift + Bayesian Reweighting

Neither extreme (0.6 mm flat prior, 1.5 mm flat prior) is optimal. A modest
grain size increase to ~1.0 mm (still conservative relative to Barr & McKinnon)
shifts the prior closer to Juno, then Bayesian reweighting does the remaining
adjustment. This preserves more convective vigor than H1 while achieving better
Juno agreement than H2.

**Prediction:** 1.0 mm grain + Juno reweighting produces D_cond ~27-28 km with
conditional Nu ~6-7 -- balancing observational fit and convection realism.

**What this result means:** The optimal reporting strategy may be to present
multiple grain priors with Juno reweighting, showing sensitivity to this
poorly-constrained parameter.

**Concern:** Introduces a subjective choice (why 1.0 mm and not 0.8 or 1.2?).

**What would weaken this hypothesis:** If the 1.0 mm + Juno result is not
materially different from either 0.6 mm + Juno or 1.5 mm flat.

### H4: Missing Physics in the Model

The model lacks a mechanism that naturally thickens D_cond without suppressing
convection. Candidates include:
- Lateral ice flow (Ashkenazy, Sayag & Tziperman 2018) could redistribute
  thickness, reducing latitude contrast and potentially thickening mid-latitude D_cond
- The convection closure (Nu = 0.3446 * Ra^1/3) may be too efficient at
  marginal Rayleigh numbers, producing too-thin conductive lids
- Tidal dissipation partitioning between mantle and shell may need refinement
- Temperature-dependent grain size recrystallization (not yet in 2D model)
  could produce a self-consistent latitude-dependent viscosity structure

**Prediction:** No combination of existing model parameters fully matches Juno
D_cond while simultaneously matching convection constraints from surface geology.

**What this result means:** The model needs additional physics before quantitative
Juno comparison is meaningful.

**What would weaken this hypothesis:** If any parameter combination in the current
model matches both Juno D_cond AND produces realistic convection (30-60%
convecting, Nu 4-8).

### H5: Ocean Transport Regime Matters at 35 deg

Different ocean transport scenarios produce different D_cond at 35 deg because
q_basal affects the shell structure even at mid-latitudes. The correct ocean
regime naturally produces thicker D_cond without needing grain size changes.

**Prediction:** At least one ocean scenario produces D_cond > 25 km at 35 deg
with the 0.6 mm grain prior.

**What this result means:** Juno could in principle discriminate ocean circulation
regimes through D_cond at a single latitude.

**What would weaken this hypothesis:** If all scenarios produce D_cond within
2 km of each other at 35 deg. (Initial evidence suggests this is the case --
all scenarios gave ~19-20 km at 35 deg with 0.6 mm grains.)

---

## Critical Evaluation of Our Methodology

### Identified Biases

1. **Optimization bias:** The autoresearch experiment loop was explicitly designed
   to minimize a score based on D_cond at 35 deg. This biases the search toward
   parameter changes that affect D_cond, potentially missing changes that improve
   other aspects of the model.

2. **Single-observable focus:** We optimized against one number (D_cond at 35 deg)
   when the model produces rich latitude-dependent output. A multi-constraint
   approach using thickness profile shape, convection fraction, and heat flux
   would be more robust.

3. **Prior sensitivity:** The grain size prior spans 3 orders of magnitude in the
   literature (0.05-80 mm). Our results are highly sensitive to where we center
   this prior. This uncertainty should be reported, not hidden.

4. **Numerical path dependence:** The dt fix (5e12 -> 1e12) fundamentally changed
   the model's equilibrium states. Results from pre-fix runs are not comparable
   and should not inform parameter choices.

5. **Sample size limitations:** 250 MC samples with importance reweighting gives
   N_eff ~ 150-180. Some conditional statistics (Nu|convecting at high latitudes)
   are based on small subsets.

### What We Cannot Resolve With Current Data

- Whether Europa's ice shell is actually convecting (we can only predict the
  parameter space consistent with convection, not confirm it observationally)
- The true grain size distribution (no direct measurement exists)
- Whether the Juno MWR measurement is representative of Europa's global
  shell or a local anomaly
- Whether the pure-ice or salty-ice correction applies

### Recommended Reporting Strategy

Present results as a parameter sensitivity study, not as "the answer":

1. Show all grain size priors (0.6, 1.0, 1.5 mm) with and without Juno reweighting
2. Report each ocean scenario separately -- they represent different physical assumptions
3. Highlight what Juno constrains (grain size / viscosity) vs what it doesn't
   (ocean transport regime at a single latitude)
4. Note that latitude-resolved shell thickness observations would provide the
   discriminating power that single-point D_cond lacks
