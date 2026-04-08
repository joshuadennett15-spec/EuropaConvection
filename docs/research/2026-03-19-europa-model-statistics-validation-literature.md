# Europa Model Statistics, Validation, and Physics:
## Literature-Grounded Guidance for the Equatorial-Proxy Suite

Date: 2026-03-19

## Scope

This memo is targeted at the current `EuropaProjectDJ` workflow, especially the
equatorial-proxy Monte Carlo suite and the Juno MWR comparison.

Relevant local files:

- `EuropaProjectDJ/src/audited_sampler.py`
- `EuropaProjectDJ/src/audited_equatorial_sampler.py`
- `EuropaProjectDJ/src/Convection.py`
- `EuropaProjectDJ/src/Physics.py`
- `EuropaProjectDJ/scripts/run_equatorial_suite.py`
- `EuropaProjectDJ/scripts/bayesian_inversion_juno.py`
- `EuropaProjectDJ/scripts/bayesian_equatorial_juno.py`
- `EuropaProjectDJ/scripts/sobol_analysis.py`

The goal here is not just "what statistics can be run", but what the Europa and
broader planetary-science literature treats as credible inference, validation,
and physics for this kind of model.

## Executive View

The current repo is already doing three things that are directionally correct:

1. Propagating uncertain physics with Monte Carlo.
2. Reweighting against a new observation rather than collapsing to a single best fit.
3. Running several sensitivity metrics instead of relying on one ranking.

However, literature-grade planetary inference would usually push the workflow
further in five ways:

1. Distinguish clearly between code verification, model validation, and scientific calibration.
2. Treat the Juno `29 +/- 10 km` result as a compressed data product, not the raw observation.
3. Carry model discrepancy explicitly, because the dominant uncertainty is often the forward model, not the instrument.
4. Compare model families with full predictive evidence, not evidence conditional on "solver-valid draws only".
5. Validate against multiple independent observables, not just one shell-thickness proxy.

## What The Current Literature Says

### 1. What Juno actually constrains

Levin et al. (published 2025-12-17; Nature Astronomy 2026) infer that for pure
water ice, the Juno MWR data are consistent with a *thermally conductive* shell
thickness of `29 +/- 10 km` in the observed region, and that modest shell
salinity would reduce that estimate by about `5 km`. They also note that a
convective layer beneath the conductive shell remains possible, meaning the
total shell thickness can be larger than the conductive lid thickness.

This matters for your statistics:

- `D_cond` is the right target if you are using the published Juno summary.
- `H_total` should not be used as the direct likelihood target for that datum.
- A one-number Gaussian likelihood on `D_cond` is a reasonable first-pass
  approximation, but it discards most of the original observation model.

The Juno paper is based on six MWR frequency channels and 129 measurements per
channel over the observed swath. The paper also states that lateral brightness
temperature differences are precise and that discrepancies are dominated by
modeling assumptions more than measurement noise. That is a strong argument for
including an explicit model-discrepancy term in your likelihood.

### 2. Equatorial enhancement is physically plausible, but only as a proxy

Soderlund et al. (Nature Geoscience, 2014) modeled thermal convection in
Europa's rotating ocean and found low-latitude heat delivery stronger than at
higher latitudes, with longitude-time averaged heat flux varying by about 40%
and peaking near the equator. That supports the existence of an "equatorial
enhancement" effect.

What it does *not* support is the idea that a single scalar multiplier is a
complete physical model. In the literature, the enhancement arises from ocean
dynamics, rotation, and shell-ocean coupling. A `1.2x` or `1.5x` factor is
therefore best treated as a proxy model or sensitivity branch, not as a final
mechanistic description.

### 3. Shell thickness, tidal dissipation, and melting are strongly rheology-dependent

Vilella et al. (JGR Planets, 2020) show that melting and shell thermal state are
highly sensitive to shell thickness, tidal heating rate, and viscosity/rheology.
They emphasize that different rheological assumptions can materially change heat
transfer and inferred shell state.

Ruiz et al. (Icarus, 2007) specifically show that Europa shell heat flow and
thickness are sensitive to grain-size-dependent rheology, with grain size
becoming a controlling parameter rather than a cosmetic nuisance parameter.

Tobie et al. (Space Science Reviews, 2025) review tidal deformation and
dissipation in icy worlds and make the broader point that:

- Maxwell vs Andrade vs more complex anelastic rheologies can change dissipation.
- Dissipation can reside in the shell, ocean, or silicate interior.
- Thermal evolution and dissipation are strongly coupled through feedbacks.

For this project, that means Andrade vs Maxwell is not just a robustness check;
it is a distinct model family.

### 4. Time evolution may matter, not just steady state

Shibley and Goodman (2023 preprint, later 2024 Icarus article) show that a
Europan shell can remain out of steady state for long intervals depending on
ice-ocean heat flux and rheology, and may still be growing in some parameter
regimes.

That does not invalidate a steady-state Monte Carlo model, but it means the
steady-state assumption should be treated as a model choice and, ideally, as one
branch in a model hierarchy.

### 5. Independent validation targets exist beyond Juno MWR

Recent and classic Europa literature suggests at least five other observables
that can act as validation targets:

1. Latitudinal heat-flow structure.
2. Geographic concentration of chaos terrain and cryovolcanic indicators.
3. Shallow-water or sill interpretations of double ridges / chaos terrains.
4. Tidal response quantities such as Love numbers.
5. Whole-body structure quantities such as moment of inertia or ocean/shell
   thickness combinations from coupled interior models.

Rhoden et al. (Icarus, 2026) argue that surface heat-flow measurements, if
surface temperature is known, can constrain shell thickness and that the
magnitude of latitudinal heat-flow variation helps distinguish dominant heat
sources.

Lesage et al. (Nature Communications, 2025) explicitly frame future Europa
interpretation as a *combined* use of salinity, surface temperature, radar, and
shell-thickness measurements to distinguish deep-ocean vs shallow-reservoir
sources of erupted material.

This is the main planetary-science lesson: validation is multi-observable.

## How Planetary Science Usually Does The Statistics

### A. Prior propagation Monte Carlo is common for underconstrained interiors

Howell's Europa probabilistic work and the 2022 LPSC follow-up use large Monte
Carlo ensembles to propagate uncertainty in heat flux, viscosities, layer
properties, and structural parameters through a simplified physical model.

That is close to the role played by your current `AuditedShellSampler` and
`AuditedEquatorialSampler`.

### B. Full inversion is used when conditioning on observations becomes central

In broader planetary modeling, more formal inverse methods are used once the
goal becomes "infer parameters from data" rather than "propagate uncertainty".

Examples:

- Drilleau et al. (2021) use MCMC for Martian interior inversion and compare a
  classical parameterization with a geodynamically constrained one.
- Otegi et al. (A&A, 2020) use Bayesian inference with nested sampling for
  exoplanet interior structure, explicitly accounting for observational and
  model uncertainty.

The key pattern is:

- Monte Carlo for uncertainty propagation and prior exploration.
- MCMC or nested sampling for posterior inference and model comparison.

### C. Planetary papers usually compare model classes, not just parameter values

The literature commonly compares:

- different rheologies,
- different heating partitions,
- different parameterizations,
- and different physical assumptions.

For your project, the natural model classes are:

- global audited shell,
- equatorial proxy with transport multiplier,
- different rheologies,
- possibly steady-state vs growth models,
- pure-ice vs salted-shell variants.

### D. Strong studies use external constraints and synthetic experiments

The Mars inversion literature is especially instructive here: they test
recovery under synthetic data scenarios, vary data quality, and compare
parameterizations under known truth settings.

That is much stronger than only running the inversion on real data once.

## Recommended Statistical Workflow For This Repo

### 1. Use a model hierarchy, not a single "best" model

At minimum define:

- `M0`: audited global 1D shell.
- `M1`: equatorial-proxy 1D shell with transport multiplier.
- `M2`: same as `M1` but with alternative rheology.
- `M3`: optional non-steady-state / growth branch.

Then separate:

- parameter uncertainty within a model,
- structural uncertainty between models.

### 2. Split the observation model into "summary-likelihood" and "raw-data-likelihood"

For now, keep the current summary likelihood:

- `D_cond ~ Normal(D_obs, sigma_total)`.

But document it explicitly as:

- a compressed likelihood derived from Levin et al. rather than the original
  Juno radiometry data.

Longer term, the literature-grade target is a hierarchical observation model:

`physical shell model -> temperature / structure profile -> microwave forward model -> brightness temperatures`

and then compare to the actual multi-frequency MWR observations.

That would let you infer not only `D_cond`, but also nuisance terms related to
scatterers, salinity, and thermal structure.

### 3. Carry model discrepancy explicitly

Right now `sigma_model = 3 km` is a fixed tuning constant. That is better than
pretending the forward model is exact, but it should ideally become:

- an inferred hyperparameter, or
- a structured discrepancy term by model family.

Why:

- Juno's paper says the mismatches are dominated by modeling assumptions.
- Otegi et al. emphasize that theoretical/model uncertainty can dominate as data improve.

Recommended change:

- Treat `sigma_model` as a prior-distributed hyperparameter and infer it jointly.

### 4. Keep importance reweighting, but add proper diagnostics

Importance reweighting is defensible when:

- you have prior draws already,
- the observation is low-dimensional,
- the posterior is not far from the prior,
- and the effective sample size is still healthy.

That fits your current use case reasonably well.

But add:

- PSIS `k-hat` diagnostics,
- tail ESS,
- Monte Carlo standard errors on posterior summaries,
- and failure thresholds that force a switch to direct sampling when reweighting collapses.

Kish ESS alone is not enough to diagnose unstable importance ratios.

### 5. For model comparison, do not stop at conditional Bayes factors

Your current `bayesian_equatorial_juno.py` already notes the important caveat:
the Bayes factors are conditional on solver-valid draws.

That means the current quantity is:

`p(data | model, valid)`

not

`p(data | model)`.

If valid-draw yield differs across modes, the full model evidence should include
the probability of producing a physically/solver-valid realization:

`p(data | model) = p(valid | model) * p(data | model, valid) + p(invalid | model) * p(data | model, invalid)`

In practice, for this workflow the minimal fix is:

- record invalid draws and failure reasons,
- estimate `p(valid | model)`,
- and include that factor when comparing models.

If Bayes factors become central rather than illustrative, move to nested
sampling or another dedicated evidence method instead of relying on reused prior
draws alone.

### 6. Separate prior sensitivity from posterior sensitivity

Your repo already computes a strong set of global sensitivity measures:

- Spearman,
- PRCC,
- SRC,
- RF permutation importance,
- delta indices,
- SHAP,
- mutual information,
- conditional KS / regime PRCC.

That is good.

The next improvement is to compute them in two spaces:

- prior predictive space,
- posterior-weighted space.

This tells you which parameters drive raw model spread versus which ones still
matter *after* the data are applied.

### 7. Use grouped sensitivity by physical block

In addition to per-parameter sensitivity, define grouped blocks:

- shell rheology,
- tidal forcing,
- surface thermal boundary condition,
- basal heat flux partition,
- composition / porosity,
- ocean-transport proxy.

Planetary models often suffer from parameter degeneracy inside a block; group
level importance is often more interpretable than ranking 15-20 individual
parameters.

## Validation Ladder For These Models

### Level 1: Code verification

This is "does the implementation solve the equations it claims to solve?"

Required tests:

1. Pure conduction benchmark:
   - compare against analytic conductive-thickness solutions when convection and
     internal heating are disabled.
2. Convection benchmark:
   - verify `Ra`, `Nu`, `D_cond`, and `D_conv` against published scaling-law
     expectations from the Barr/Showman, Green, and Deschamps/Vilella framework.
3. Rheology benchmark:
   - verify Maxwell and Andrade dissipation curves against the cited source
     formulas over relevant temperature/viscosity ranges.
4. Numerical convergence:
   - grid-size, timestep, and iteration-count convergence.
5. Reproducibility:
   - same seed, same outputs; stable sample IDs across parallel execution.

### Level 2: Inference validation

This is "does the statistical machinery recover truth when truth is known?"

Required tests:

1. Synthetic twin experiments:
   - generate pseudo-observations from a known model / known parameter setting,
     then recover them.
2. Model discrimination tests:
   - simulate under baseline / moderate / strong equatorial modes and test if
     your Bayes-factor machinery actually recovers the generating mode.
3. Simulation-based calibration:
   - if you move to direct Bayesian sampling, run SBC to catch coding and
     inference bugs.
4. Reweighting diagnostics:
   - compare importance reweighting to direct posterior sampling on a reduced
     version of the model.

### Level 3: Physics validation

This is "does the model reproduce physically meaningful observables not used in calibration?"

Recommended external targets:

1. Juno MWR conductive shell thickness in the observed swath.
2. Geographic tendency for enhanced low-latitude activity if equatorial heat
   delivery is invoked.
3. Plausible shell total thickness and convective-lid partition.
4. Heat-flow magnitudes and latitudinal variations.
5. Optional whole-body constraints such as Love number / MoI ranges from coupled
   interior structure studies.

### Level 4: Predictive validation

This is the hardest level and the one most aligned with planetary mission use.

Examples:

- Predict latitudinal heat-flow patterns for future Clipper thermal data.
- Predict shell-thickness / heat-flow / cryovolcanic relationships for combined
  radar and topography comparisons.
- Predict what observables should differ between `1.0x`, `1.2x`, and `1.5x`
  transport modes beyond `D_cond`.

## Physics Gaps Between The Repo And The Literature

These are not "bugs". They are model-scope limits that should be stated openly.

### 1. Ocean transport is proxied, not modeled

The literature basis for low-latitude enhancement comes from 3D rotating ocean
dynamics. The current equatorial suite replaces that with a scalar multiplier on
the tidal component of basal heating. That is acceptable as a sensitivity proxy
but should not be described as a mechanistic ocean model.

### 2. Juno is being assimilated through a derived shell-thickness summary

The raw Juno information lives in microwave brightness temperatures and a
radiative/scattering forward model. Your current likelihood uses only the
published shell-thickness summary.

This is acceptable for a first inversion but not the terminal form of the model.

### 3. Salinity and marine ice are largely outside the current model

Levin et al. show salinity shifts the shell-thickness estimate by less than the
published uncertainty but not by zero. Lesage et al. argue combined salinity,
radar, thermal, and shell-thickness measurements are important for interpreting
cryovolcanic pathways.

If you want literature-grade physics, salinity should be at least a model branch.

### 4. Thermal evolution is mostly collapsed to present-day steady state

Shibley and Goodman show that shell-growth history can matter. If you only run
present-day steady-state shells, you are imposing a structural assumption that
should be compared against alternatives.

### 5. Grain size likely evolves, but is treated as static

Ruiz et al. and later tidal-rheology work show grain size can materially control
viscosity and heat transport. A static prior on `d_grain` is fine for now, but
it is still a reduced-order approximation to a dynamic microphysical process.

## Repo-Specific Assessment

### What is already good

1. `run_equatorial_suite.py` uses shared seeds across modes, which is the right
   idea for reducing comparison noise.
2. `bayesian_inversion_juno.py` uses importance reweighting rather than a naive
   hard cut.
3. `bayesian_equatorial_juno.py` explicitly warns that the Bayes factors are
   conditional on valid draws.
4. `sobol_analysis.py` already contains a stronger sensitivity stack than many
   planetary papers publish.

### What is the biggest statistical weakness right now

The biggest weakness is not the Gaussian `D_cond` likelihood itself. The bigger
problem is that model comparison is still conditional on valid draws, while the
choice of mode may change valid-draw yield.

The second biggest weakness is that the Juno likelihood is a compressed summary
of a much richer observation model.

### What is the biggest physics weakness right now

The current equatorial enhancement is a good proxy experiment but not a physical
ocean-shell coupling model. That is fine if presented honestly as a proxy suite.

## Recommended Near-Term Work

### Immediate

1. Add PSIS diagnostics to the importance-reweighting scripts.
2. Record sample IDs and invalid-draw reasons across all modes.
3. Report full evidence with valid-yield correction, not only conditional evidence.
4. Compute posterior-weighted sensitivity in addition to prior sensitivity.
5. Turn `sigma_model` into an inferred or at least sensitivity-tested quantity.

### Next

1. Add synthetic twin tests for mode recovery.
2. Add a benchmark suite for conduction, convection, and rheology formulas.
3. Add one or more independent validation targets beyond Juno `D_cond`.

### Longer-term

1. Replace scalar equatorial enhancement with a calibrated latent transport field
   or a reduced-order emulator trained on 3D ocean results.
2. Add shell salinity and non-steady-state growth as explicit model branches.
3. Move from summary-likelihood inversion to a hierarchical Juno radiometry
   likelihood if the forward observation model can be implemented.

## Sources

Europa / icy-shell / ocean-world physics:

- Levin et al., "Europa's ice thickness and subsurface structure characterized by the Juno microwave radiometer" (Nature Astronomy, published 2025-12-17): https://www.nature.com/articles/s41550-025-02718-0
- Soderlund et al., "Ocean-driven heating of Europa's icy shell at low latitudes" (Nature Geoscience, 2014 PDF): https://website.whoi.edu/gfd/wp-content/uploads/sites/14/2018/10/Soderlund_Ocean_ngeo_2014-1_233284.pdf
- Vilella et al., "Tidally Heated Convection and the Occurrence of Melting in Icy Satellites: Application to Europa" (JGR Planets, 2020 PDF): https://idv.sinica.edu.tw/fdeschamps/Publi/Vilella-et-al_JGR-Planets_2020.pdf
- Shibley and Goodman, "Europa's Coupled Ice-Ocean System: Temporal Evolution of a Pure Ice Shell" (arXiv 2309.16821 / later Icarus): https://arxiv.org/abs/2309.16821
- Tobie et al., "Tidal Deformation and Dissipation Processes in Icy Worlds" (Space Science Reviews, 2025): https://link.springer.com/article/10.1007/s11214-025-01136-y
- Ruiz et al., "Heat flow and thickness of a convective ice shell on Europa for grain size-dependent rheologies" (Icarus, 2007): https://www.sciencedirect.com/science/article/abs/pii/S0019103507001133
- Howell and Hay, "Europa's Icy Structure: Statistical Modeling as a Bridge to Physical Modeling" (LPSC 2022 PDF): https://www.hou.usra.edu/meetings/lpsc2022/pdf/2601.pdf
- Rhoden et al., "The diagnostic power of surface heat flow measurements at Europa" (Icarus, 2026 abstract page): https://m.booksci.cn/literature/144892814.htm
- Schmidt et al., "Active formation of 'chaos terrain' over shallow subsurface water on Europa" (Nature, 2011 abstract page): https://medluna.com/article/22089135
- Lesage et al., "Identifying signatures of past and present cryovolcanism on Europa" (Nature Communications, 2025 abstract page): https://www.lifescience.net/publications/1193455/identifying-signatures-of-past-and-present-cryovol/

Planetary / Bayesian / inversion workflow:

- Drilleau et al., "Bayesian inversion of the Martian structure using geodynamic constraints" (Geophysical Journal International, 2021): https://academic.oup.com/gji/article/226/3/1615/6169708
- Otegi et al., "Impact of the measured parameters of exoplanets on the inferred internal structure" (A&A, 2020): https://www.aanda.org/articles/aa/full_html/2020/08/aa38006-20/aa38006-20.html
- Gelman et al., "Bayesian Workflow" (arXiv 2011.01808): https://arxiv.org/abs/2011.01808
- Talts et al., "Validating Bayesian Inference Algorithms with Simulation-Based Calibration" (arXiv 1804.06788): https://arxiv.org/abs/1804.06788
- Vehtari et al., "Pareto Smoothed Importance Sampling" (arXiv 1507.02646 / JMLR 2024): https://arxiv.org/abs/1507.02646
