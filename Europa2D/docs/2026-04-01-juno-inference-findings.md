# Juno Inference Study: Findings

**Date:** 2026-04-01
**Based on:** 250-sample MC ensembles across 4 ocean scenarios x 3 grain priors (0.6, 1.0, 1.5 mm), with and without Juno D_cond importance reweighting.

---

## 1. Summary

The Juno MWR measurement of D_cond = 29 +/- 10 km at 35 deg latitude primarily constrains Europa's effective ice grain size (and hence viscosity), not the ocean heat transport regime. All four ocean transport scenarios produce statistically indistinguishable D_cond at 35 deg within each grain prior, but they produce clearly distinguishable latitude-dependent shell structure. The most scientifically defensible reporting approach is a parameter sensitivity analysis showing all grain priors with Juno reweighting.

---

## 2. Hypothesis Evaluation

### H1: Grain Size Dominance — SUPPORTED

D_cond at 35 deg scales monotonically with grain size prior center:

| Grain prior | D_cond(35) median | Conv% equator | Nu|convecting |
|---|---|---|---|
| 0.6 mm | 19.3 km | 54% | 6.4 |
| 1.0 mm | 24.7 km | 37% | 5.9 |
| 1.5 mm | 27.9 km | 27% | 4.8 |

Increasing grain size from 0.6 to 1.5 mm shifts D_cond by +8.6 km, sufficient to reach Juno's center. However, convecting fraction drops from 54% to 27% and conditional Nu from 6.4 to 4.8. The grain size prior is the single most powerful lever on D_cond in the model.

### H2: Bayesian Selection — SUPPORTED

Juno reweighting of the 0.6mm prior shifts D_cond from 19.3 to 24.5 km while increasing conditional Nu from 6.4 to 7.6. The reweighting selects for samples with thick conductive lids that still convect vigorously underneath. This is a fundamentally different path to thick D_cond than increasing grain size — it preserves convective vigor.

The limitation is that the reweighted posterior only reaches 24.5 km, 4.5 km below Juno's center. The Juno uncertainty is broad enough (sigma_eff = 10.4 km) that this is still well within the constraint, but the posterior does not center on 29 km.

### H3: Combined Approach — SUPPORTED AS BEST COMPROMISE

The 1.0mm grain prior with Juno reweighting produces:
- D_cond(35) = 26.2 km (within 2.8 km of Juno center)
- Conv% equator = 25%
- Nu|convecting = 6.4

This balances observational agreement with convective realism. The 1.0mm center is physically motivated (still conservative relative to Barr & McKinnon 2007 equilibrium estimates of 1-80 mm) and the Juno reweighting does the fine adjustment.

### H4: Missing Physics — INCONCLUSIVE

The current model can match Juno D_cond with grain size adjustments within the literature range, so missing physics is not required to explain the discrepancy. However, the sensitivity to grain size (a poorly constrained parameter) suggests the model's predictive power for D_cond is limited by our ignorance of Europa's grain size distribution, not by the fidelity of the thermal solver.

Candidate missing physics (lateral ice flow, temperature-dependent recrystallization) would modify the grain size sensitivity and could either strengthen or weaken the current agreement.

### H5: Ocean Transport Sensitivity at 35 deg — NOT SUPPORTED

D_cond at 35 deg varies by less than 2 km across all four ocean transport scenarios within each grain prior:

| Grain | Uniform | Eq-enhanced | Polar-enh. | Strong polar |
|---|---|---|---|---|
| 0.6mm | 19.3 | 20.8 | 19.8 | 19.7 |
| 1.5mm | 27.9 | 27.8 | 28.0 | 28.0 |

A single-latitude D_cond measurement cannot discriminate ocean circulation regimes. The scenarios ARE distinguishable in latitude structure — equator-enhanced produces H_eq = 25 km vs strong polar H_eq = 32 km at 0.6mm — but this information is not available from a single Juno flyby at 35 deg.

---

## 3. What Juno Actually Constrains

**Constrains strongly:**
- Ice grain size / effective viscosity: a factor of 2.5x change in grain center (0.6 to 1.5 mm) produces a factor of 1.45x change in D_cond
- The fraction of parameter space that convects: Juno-consistent samples have 20-35% convecting at the equator
- The character of convection when it occurs: Juno selects for high-Ra, high-Nu samples (vigorous but beneath a thick lid)

**Does not constrain:**
- Ocean transport regime: all four scenarios produce identical D_cond at 35 deg
- Absolute convective vigor: depends on which grain prior is assumed
- Polar shell structure: no Juno data at high latitudes

**Would constrain with additional data:**
- A second D_cond measurement at a different latitude (e.g., equator or >60 deg) would immediately discriminate ocean scenarios via the latitude gradient
- A tighter D_cond uncertainty (e.g., +/- 5 km) would more strongly select between grain size hypotheses
- The salty-ice correction (24 +/- 10 km vs 29 +/- 10 km) shifts all results by ~5 km but does not change the relative rankings

---

## 4. Recommended Reporting for Thesis

### Primary results (present all, don't privilege one):

1. **Table of all scenario x grain x view combinations** (the comprehensive comparison table above). This is the complete parameter sensitivity analysis.

2. **4-scenario latitude structure figure** (2x2 thickness panels) using the 1.5mm grain prior — shows distinguishable ocean transport effects.

3. **Three-way comparison figure** (4x4 panels) — shows how prior, Juno reweighting, and grain shift affect shell structure and convection across all scenarios.

4. **Prior vs posterior figure** (2x2 D_cond distributions at 35 deg) — shows the Bayesian shrinkage for the 0.6mm prior.

### Narrative structure:

> "Our 2D model with audited parameter priors (Howell 2021, d_grain = 0.6 mm center) produces D_cond = 19.3 +/- 13 km at 35 deg latitude, below the Juno MWR measurement of 29 +/- 10 km. Importance reweighting by the Juno likelihood shifts the posterior to 24.5 km while selecting for samples with vigorous stagnant-lid convection (Nu = 7.6 when convecting). Adjusting the grain size prior to 1.0-1.5 mm — within the range supported by dynamic recrystallization studies (Barr & McKinnon 2007) — brings the model into closer agreement with Juno (D_cond = 26-28 km) but reduces convective vigor. We interpret the Juno constraint as primarily constraining Europa's ice grain size rather than the ocean heat transport regime, which controls the latitude structure of the shell but not the conductive lid thickness at a single latitude."

### Key caveats to include:

- The grain size prior is the dominant uncertainty and is poorly constrained
- All results assume pure water ice (f_salt = 0); the salty-ice correction would shift D_cond down by ~5 km
- 250 MC samples with importance reweighting gives N_eff ~ 167-186; some conditional statistics are based on small subsets
- The model does not include lateral ice flow, which could modify the latitude structure
- Juno MWR measured at a specific location during a single flyby; spatial variability is unknown

---

## 5. Future Work

1. **Europa Clipper flybys** will provide D_cond at multiple latitudes — this will directly test H5 and discriminate ocean transport regimes
2. **Grain size recrystallization modeling** (temperature and strain-dependent) would make the grain size self-consistent rather than a free parameter
3. **Larger MC campaigns** (1000+ samples) would improve N_eff for Bayesian analysis and reduce noise in conditional statistics
4. **Joint inference** across D_cond + surface heat flux + elastic thickness would provide multi-constraint parameter estimation
5. **Salty-ice model runs** would quantify how the f_salt assumption affects all conclusions
