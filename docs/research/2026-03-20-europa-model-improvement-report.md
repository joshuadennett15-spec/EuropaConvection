# Europa Model Improvement Report

Date: 2026-03-20

## Scope

This note synthesizes primary literature on Europa's ice shell thickness, ocean dynamics,
tidal heating, and ice-ocean coupling, and maps those results onto the current
`EuropaProjectDJ` and `Europa2D` implementations.

The goal is not to produce another literature summary. The goal is to answer a narrower
question:

- What does the literature say that the current models already capture well?
- Where do the current models diverge from a more rigorous treatment?
- Which improvements matter most for shell-thickness inference, latitude structure, and
  Juno-facing model comparison?

## Executive Summary

- The strongest current part of the repo is the 1D audited shell workflow used as a
  probabilistic forward model, especially when it compares to Juno via `D_cond` rather
  than `H_total`.
- The current 2D model is scientifically useful as a benchmark shell-response model, but
  it is not a resolved ocean-ice system. It prescribes latitude-dependent forcing rather
  than solving circulation, salinity, phase change, or topography feedbacks.
- The biggest immediate improvement is not "more 15k runs." It is tightening the model
  logic and reporting:
  - keep `H_total`, `D_cond`, and convective fraction separate,
  - make the 2D boundary conditions internally consistent,
  - report latitude-band means rather than over-reading the pole node,
  - and keep Juno comparison strictly attached to `D_cond`.
- The highest-value physics upgrades are:
  - better shell tidal-heating patterns,
  - mixed-heating diagnostics,
  - stronger handling of surface-boundary uncertainty,
  - and eventually a reduced coupled ocean module or at least a more defensible transport
    closure than a prescribed latitude-only `q_ocean(phi)`.

## Paper-by-Paper Synthesis

### 1. Levin et al. (2026), "Europa's ice thickness and subsurface structure characterized by the Juno microwave radiometer"

Source: https://www.nature.com/articles/s41550-025-02718-0

Key takeaways:

- Juno constrains a local conductive shell thickness, not the global shell thickness.
- For pure water ice, the observed swath is consistent with a thermally conductive shell
  of `29 +/- 10 km`.
- The same paper also infers shallow scattering structures in the upper shell, but the
  deep-shell quantity relevant here is still the conductive thickness summary.

Model implications:

- `D_cond` is the correct data-facing quantity for the current model hierarchy.
- `H_total` should not be presented as "the Juno-comparable output."
- Any Bayes factor or likelihood statement must be phrased as a comparison to a compressed
  data product, not to the raw microwave radiometry.

Repo implications:

- The current Juno comparison logic in `EuropaProjectDJ/scripts/bayesian_equatorial_juno.py`
  is pointing in the right direction because it reweights `D_cond`.
- The main remaining issue is language and interpretation, not the existence of the
  comparison itself.

### 2. Quick and Marsh (2015), "Constraining the thickness of Europa's water-ice shell: Insights from tidal dissipation and conductive cooling"

Source: https://doi.org/10.1016/j.icarus.2015.02.016

Key takeaways:

- A Stefan-style conductive cooling framework gives an average shell thickness of roughly
  `28 km` per `1 TW` of dissipated energy.
- Without tidal heating, Europa's ocean would crystallize in about `64 Myr`.
- The framework is globally averaged and conductive by construction.

Model implications:

- This paper is valuable as a global energy-balance and timescale sanity check.
- It is not a substitute for lid-convection partitioning, but it is a strong check on
  whether the global heat budget and implied shell thickness are in the right regime.

Repo implications:

- Add a simple global benchmark check: for a given globally integrated tidal power, does
  the 1D audited model produce a shell thickness broadly consistent with Quick and Marsh
  in the conductive limit?
- Add an optional transient "ocean freezing timescale" diagnostic to stop the current
  models from drifting too far away from globally plausible states while still fitting
  local `D_cond`.

### 3. Billings and Kattenhorn (2005), "The great thickness debate: Ice shell thickness models for Europa and comparisons with estimates based on flexure at ridges"

Source: https://doi.org/10.1016/j.icarus.2005.03.013

Key takeaways:

- Estimates of Europa's shell thickness span from sub-kilometer elastic thicknesses to
  tens of kilometers of total shell thickness.
- Many of the "thin shell" estimates refer only to the elastic or mechanically active
  portion, not to the full ice shell.

Model implications:

- This is the paper that explains why shell-thickness reporting becomes confused so
  easily.
- Any thesis chapter that mixes elastic thickness, conductive thickness, and total shell
  thickness without explicitly separating them will recreate the historical ambiguity.

Repo implications:

- Report `D_cond`, `H_total`, and any mechanically inferred thickness as distinct layers.
- Never treat a single thickness number as universal unless the observable being matched
  is also clearly defined.

### 4. Ashkenazy and Tziperman (2021), "Dynamic Europa ocean shows transient Taylor columns and convection driven by ice melting and salinity"

Source: https://www.nature.com/articles/s41467-021-26710-0

Key takeaways:

- The ocean circulation changes substantially when the model uses prescribed bottom heat
  flux instead of prescribed temperature.
- Salinity and melting/freezing feedbacks matter.
- The meridional heat transport is strong enough to support a nearly uniform shell
  thickness in their framework.
- Their 2D and 3D ocean models produce Taylor columns, eddies, and low-latitude freezing
  behavior linked to tangent-cylinder geometry and salinity.

Model implications:

- A latitude-only prescribed `q_ocean(phi)` is not the same thing as solving ocean
  circulation.
- If the model is meant to test shell response to transport families, that is fine.
- If the model is meant to claim an actual circulation mechanism, it is missing the key
  dynamics that this paper shows matter: salinity, latent heat, tangent-cylinder effects,
  and resolved eddies.

Repo implications:

- `Europa2D` should continue to be framed as a benchmark shell-response model, not as a
  resolved ocean model.
- A future reduced-order ocean module should use:
  - prescribed bottom heat flux rather than prescribed ocean temperature,
  - a top boundary with freezing/melting feedback,
  - at least a parameterized meridional heat transport closure that depends on shell
    geometry and salinity regime.

### 5. Zhang, Kang, and Marshall (2025 preprint), "How does ice shell geometry shape ocean dynamics on icy moons?"

Source: https://doi.org/10.48550/arXiv.2510.25988

Key takeaways:

- A poleward-thinning shell can drive circulation because the pressure-dependent freezing
  point imposes a meridional temperature gradient at the ice-ocean interface.
- The sign and strength of the circulation depend on salinity and topographic slope.
- Baroclinic eddies dominate meridional heat transport in many cases.
- Sloped topography is not a small correction; it changes both stratification and the
  available potential energy.

Model implications:

- The shell geometry is not just an output. It can also be an input to ocean dynamics.
- A prescribed `q_ocean(phi)` that does not know the shell geometry cannot capture this
  feedback.

Repo implications:

- The next-generation 2D logic should introduce at least one geometry-sensitive ocean
  closure.
- Even if a full ocean model is not feasible, the ocean forcing should be allowed to
  depend on shell slope or shell-thickness contrast rather than being only an externally
  chosen pattern family.

### 6. Kihoulou et al. (2025), "Subduction-like process in Europa's ice shell triggered by enhanced eccentricity periods"

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12136034/

Key takeaways:

- Subduction-like recycling occurs only for sufficiently thin shells, roughly
  `<= 10 km`, in their simulations.
- The mechanism is linked to transient episodes of enhanced eccentricity and therefore
  to time-dependent forcing, not to a static thermal equilibrium alone.

Model implications:

- A steady-state shell model is not the right tool for strong claims about episodic
  subduction or tectonic recycling.
- If the shell is tens of kilometers thick in the current runs, this paper weakens any
  claim that present-day subduction-like behavior should follow automatically.

Repo implications:

- Keep tectonic or subduction interpretations out of the main inference unless a
  time-dependent branch exists.
- If episodic tectonics become part of the thesis, build a separate transient/orbital
  branch rather than trying to force the steady-state shell model to answer it.

### 7. Deschamps and Vilella (2021), "Scaling Laws for Mixed-Heated Stagnant-Lid Convection and Application to Europa"

Source: https://doi.org/10.1029/2021JE006963

Key takeaways:

- Internal heating changes stagnant-lid convection in ways that are not captured by pure
  basal-heating scalings.
- As internal heating increases, the lid stiffens but thins, and the bottom heat flux can
  turn negative above a critical heating rate.
- The paper provides mixed-heating scaling laws for surface heat flux, interior
  temperature, and lid thickness.

Model implications:

- The ratio of internal heating to basal heating should be treated as a first-class
  physical diagnostic.
- Not all heat that enters the shell does the same job. Shell tidal heating changes the
  lid and convective transport differently from basal heat.

Repo implications:

- Add explicit diagnostics for:
  - basal versus internal heating partition,
  - proximity to the mixed-heating critical regime,
  - and whether the bottom heat flux is near sign reversal.
- This matters in both 1D and 2D because the current models interpret many scenarios
  primarily through net lower-boundary heat, but the literature shows that internally
  distributed heat can alter the lid logic substantially.

### 8. Beuthe (2013), "Spatial patterns of tidal heating"

Source: https://doi.org/10.1016/j.icarus.2012.11.020

Key takeaways:

- Tidal heating patterns in a stratified body are a linear combination of three angular
  functions.
- The pattern depends strongly on where the dissipation occurs.
- For eccentricity tides, mantle-dominated dissipation can peak at the poles, while thin
  soft layers can produce equatorial maxima and degree-4 structure.

Model implications:

- There is no single universal latitude pattern for tidal heating.
- A monotonic pole-enhanced strain profile is a usable proxy, but only for a restricted
  subset of dissipation geometries.

Repo implications:

- The current 2D strain parameterization should be treated as a benchmark family, not as
  the physically complete tidal response.
- The next improvement is to add multiple shell-tidal pattern families tied to
  dissipation geometry:
  - shell-dominated,
  - mantle/core-dominated,
  - and a higher-order pattern with non-monotonic latitude structure.

### 9. Deschamps (2021), "Stagnant lid convection with temperature-dependent thermal conductivity and the thermal evolution of icy worlds"

Source: https://doi.org/10.1093/gji/ggaa540

Key takeaways:

- Ice Ih conductivity varying as `1/T` thickens the stagnant lid and reduces heat
  transfer.
- For Europa-like surface temperatures, temperature-dependent conductivity thickens the
  lid by roughly a factor `1.2-1.4` relative to constant-conductivity treatments.

Model implications:

- This is not optional physics if the model is trying to interpret lid thickness.

Repo implications:

- This is one area where the current code is already stronger than many simplified shell
  models: the 1D and 2D solvers do use temperature-dependent conductivity.
- The improvement need is not "add variable k." It is "verify that all fast diagnostics,
  initialization logic, and interpretation remain consistent with variable k."

### 10. Ashkenazy (2019), "The surface temperature of Europa"

Source: https://doi.org/10.1016/j.heliyon.2019.e01908

Key takeaways:

- A more complete seasonal energy balance gives Europa surface temperatures near
  `96 K` at the equator and `46 K` at the pole for a representative internal-heating
  case.
- High-latitude surface temperature is significantly affected by internal heating.

Model implications:

- The cold-pole boundary condition is physically important, not cosmetic.
- The surface boundary should not be treated as a generic cosine law if the goal is to
  compare subtle lid-thickness differences.

Repo implications:

- The current 2D defaults are numerically close to Ashkenazy's values, but the repo has
  an internal consistency problem:
  - comments and design notes refer to a `52 K` floor,
  - while code defaults and the MC runner currently use `46 K`.
- That mismatch should be fixed before using 2D outputs for strong interpretation.

### 11. Lawrence et al. (2024), "Ice-Ocean Interactions on Ocean Worlds Influence Ice Shell Topography"

Source: https://doi.org/10.1029/2023JE008036

Key takeaways:

- Pressure-dependent freezing point and shell draft variations drive an "ice pump"
  process: deeper ice melts, buoyant water upwells, and refreezes shallower.
- This process links shell topography to ocean temperature and tends to smooth the shell
  base on Europa-like worlds.

Model implications:

- Topography is not just a passive indicator of thickness. It feeds back on basal melting
  and freezing.
- A shell model without phase-boundary redistribution is incomplete for interpreting
  topography or linking shell slope to ocean temperature.

Repo implications:

- The current models do not include basal melt/freeze redistribution or topography
  evolution.
- Therefore:
  - shell-thickness outputs are still useful,
  - but topographic predictions or ocean-temperature inferences are not yet safe.

### 12. Soderlund et al. (2024), "The Physical Oceanography of Ice-Covered Moons"

Source: https://www.annualreviews.org/doi/10.1146/annurev-marine-040323-101355

Key takeaways:

- Ocean circulation on icy moons can be driven by buoyancy, tides, libration, precession,
  and electromagnetic pumping.
- The interaction of these mechanisms matters as much as any single mechanism alone.

Model implications:

- A single transport-multiplier axis is too narrow to stand in for all physically
  plausible ocean states.

Repo implications:

- Keep using `uniform`, `equator-enhanced`, and `polar-enhanced` benchmark families, but
  describe them as a partial span of ocean possibilities, not as a full basis.
- If the model grows, the next forcing families to add are mechanically or
  electromagnetically driven patterns rather than only more thermal-convection variants.

### 13. Soderlund et al. (2014), "Ocean-driven heating of Europa's icy shell at low latitudes"

Source: https://doi.org/10.1038/ngeo2021

Key takeaways:

- The 3D ocean model predicts low-latitude enhancement of upward heat delivery.
- The time- and longitude-averaged heat flux varies by about `40%` and peaks near the
  equator.
- This is a major basis for equatorial thinning and low-latitude geologic activity.

Model implications:

- Equatorial enhancement is a literature-backed benchmark, but it arises from solved
  circulation in a rotating ocean model, not from a scalar multiplier by itself.

Repo implications:

- The equatorial 1D suite is scientifically defensible as a proxy experiment.
- It should not be described as a solved Soderlund circulation model.
- The logical next step is to calibrate the 1D proxy to full heat-flux fields or
  endpoint ratios instead of a generic enhancement factor alone.

### 14. Howell (2021), "The Likely Thickness of Europa's Icy Shell"

Source: https://doi.org/10.3847/PSJ/abfe10

Key takeaways:

- Howell's work is one of the cleanest Monte Carlo uncertainty treatments for Europa's
  shell thickness.
- It supports a broad tens-of-kilometers shell-thickness regime, not a single sharp
  answer.
- The 2022 follow-up summary reports a current best estimate near `24.3 km` with a broad
  upper tail.

Model implications:

- This paper is the strongest precedent for your general methodology: probabilistic shell
  structure rather than one deterministic thickness.

Repo implications:

- The current 1D audited framework is closest in spirit to Howell and should remain the
  core inference engine.
- Use Howell-style probabilistic framing as the main methodological anchor, then treat the
  equatorial and 2D branches as structured extensions rather than replacements.

### 15. Kamata and Nimmo (2017), "Interior thermal state of Enceladus inferred from the viscoelastic state of the ice shell"

Source: comparison framework summarized at
https://cir.nii.ac.jp/crid/1050306506452913920

Key takeaways:

- In Enceladus, thermal equilibrium requires balancing basal melting and viscous
  relaxation timescales.
- The broader lesson is that shell state is constrained by coupled mechanical and thermal
  timescales, not only by steady heat balance.

Model implications:

- This is not a direct Europa calibration paper, but it matters for methodology.
- It highlights a class of missing physics in the current models: viscoelastic
  topography-relaxation and melting timescale competition.

Repo implications:

- Relevant mainly for future topography or shell-state evolution work.
- Not a priority for the Juno-facing `D_cond` inference, but important if the thesis moves
  toward topography or long-timescale shell evolution.

## Current Repo Audit Against the Literature

### What the current code already does well

1. It keeps a clear distinction between total thickness and conductive-lid structure in
   the underlying data model.

2. The 1D and 2D solvers use temperature-dependent conductivity through
   `Thermal.conductivity(T)` in:
   - `EuropaProjectDJ/src/constants.py`
   - `EuropaProjectDJ/src/Convection.py`
   - `EuropaProjectDJ/src/Solver.py`
   - `Europa2D/src/axial_solver.py`

3. The 2D model normalizes latitude-dependent ocean heat-flux patterns to preserve the
   global mean in `Europa2D/src/latitude_profile.py`.

4. The 2D branch has a single-column validation target against the 1D solver in
   `Europa2D/tests/test_validation.py`.

5. The equatorial Juno comparison uses `D_cond`, which is the right observable under the
   Levin et al. summary-likelihood approach.

### Where the current model logic diverges from a more rigorous treatment

#### 1. Prescribed versus solved ocean forcing

Current state:

- In `Europa2D/src/latitude_sampler.py`, `q_ocean_mean` is inherited from the audited 1D
  basal flux and then redistributed with a chosen latitude pattern.
- In `Europa2D/src/latitude_profile.py`, the ocean forcing is one of three prescribed
  families: `uniform`, `equator_enhanced`, or `polar_enhanced`.
- In `Europa2D/src/literature_scenarios.py`, these families are mapped to literature
  scenarios.

Why this is weaker than the literature:

- Ashkenazy and Tziperman (2021) show that salinity, tangent-cylinder geometry, freezing,
  and eddies all alter the direction and strength of heat transport.
- Zhang et al. (2025) show that shell geometry itself can drive or suppress ocean
  transport.
- The review by Soderlund et al. (2024) shows that thermal convection is not the only
  driver.

Practical consequence:

- The current 2D model is safe for relative shell-response tests across forcing families.
- It is not safe for strong claims about the true ocean circulation mechanism on Europa.

#### 2. Shell tidal-heating parameterization

Current state:

- `Europa2D/src/latitude_profile.py` uses a monotonic strain-amplitude law anchored at
  `epsilon_eq` and `epsilon_pole` with a `sin^2(phi)` dependence.
- Default polar strain is larger than equatorial strain.
- The 1D equatorial branch in `EuropaProjectDJ/src/audited_equatorial_sampler.py` lowers
  `epsilon_0` and scales only the tidal component of basal forcing.

Why this is weaker than the literature:

- Beuthe (2013) shows that tidal-heating patterns depend on dissipation geometry and are
  not universally monotonic or universally polar-amplified.
- Deschamps and Vilella (2021) show that internal heating changes convection differently
  from basal heat.

Practical consequence:

- The current tidal-heating pattern is a useful benchmark proxy, not a complete tidal
  physics model.

#### 3. Surface boundary parameterization

Current state:

- `Europa2D/src/latitude_profile.py` uses
  `T_s(phi) = ((T_eq^4 - T_floor^4) * cos(phi) + T_floor^4)^(1/4)`.
- The comments cite Ashkenazy (2019), but the file defaults are currently
  `T_eq = 96 K` and `T_floor = 46 K`.
- `Europa2D/src/latitude_sampler.py` samples `T_eq ~ N(96, 5)` and `T_floor ~ N(46, 4)`.
- Some design notes in the repo still refer to `T_eq = 110 K` or `T_floor = 52 K`.

Why this is weaker than the literature:

- Ashkenazy (2019) supports a more careful surface boundary, but the repo has code/doc
  drift.
- High-latitude surface temperature is sensitive to internal heating, which the current
  implementation partly brackets but does not fully couple.

Practical consequence:

- The cold-pole logic is scientifically defensible.
- The exact numeric defaults are not yet internally clean enough for strong claims.

#### 4. Convection and regime logic

Current state:

- `Europa2D/src/axial_solver.py` uses a convection ramp that is zero below `60 K`, full
  above `80 K`, and smoothly blended in between.
- Lateral diffusion is explicitly described as extremely weak.
- In the 1D MC logic, subcritical samples are either rejected or kept as purely
  conductive depending on run configuration.

Why this is weaker than the literature:

- The literature treats stagnant-lid structure as an emergent result of the full thermal
  problem, not as a piecewise ramp in surface temperature alone.
- The current ramp is a pragmatic closure, not a direct literature-based scaling law.

Practical consequence:

- The model can classify regimes robustly enough for ensemble work.
- Exact threshold behavior near onset should be treated cautiously.

#### 5. Output interpretation

Safe outputs:

- 1D prior-predictive distributions of `H_total`, `D_cond`, `lid_fraction`, and
  convective fraction.
- Juno comparison performed on `D_cond`.
- 2D relative response across benchmark forcing families.
- 2D area-weighted low-latitude and high-latitude band means.
- Latitude of minimum thickness as a comparative diagnostic.

Unsafe outputs:

- Exact polar-node interpretation.
- Claims about real Europa ocean circulation geometry.
- Longitude-specific or chaos-terrain-specific predictions.
- Topography or ocean-temperature inference from shell thickness alone.
- Strong tectonic or subduction claims from the current steady-state runs.

## Priority Improvement Roadmap

### P0: Fix interpretation and internal consistency before more large runs

1. Reconcile `T_eq` and `T_floor` defaults and comments across:
   - `Europa2D/src/latitude_profile.py`
   - `Europa2D/src/latitude_sampler.py`
   - `Europa2D/src/monte_carlo_2d.py`
   - the associated design notes

2. Make the current 2D default scenario explicit in every plot and archive.
   Right now `polar_enhanced` is easy to treat as a neutral baseline when it is not.

3. Split all reporting of shell structure into:
   - `H_total`
   - `D_cond`
   - `D_conv`
   - convective fraction

4. Report latitude-band means instead of the single `90 deg` node.

5. Keep Juno comparison strictly attached to `D_cond`.

### P1: Improve the physics without building a full ocean GCM

1. Add more than one shell-tidal pattern family, motivated by Beuthe (2013):
   - polar-enhanced mantle/core dissipation,
   - equatorial or higher-order shell dissipation,
   - and a non-monotonic benchmark.

2. Add mixed-heating diagnostics from Deschamps and Vilella (2021):
   - internal-to-basal heating ratio,
   - sign and magnitude of basal heat flux,
   - and a "near-critical mixed heating" flag.

3. Add a surface-boundary sensitivity mode:
   - Ashkenazy-like low internal-heating floor,
   - higher internal-heating floor,
   - and an older `110/52` style comparison only as a sensitivity check.

4. Improve the 1D equatorial proxy by storing a literature-facing transport diagnostic:
   - endpoint ratio,
   - `q_star`,
   - or equivalent,
   rather than only a scalar enhancement factor.

5. Add regime-split `D_cond` plots, because total-thickness panels are easy to
   misinterpret when the shell thickens mainly by growing `D_conv`.

### P2: Next-generation coupled improvements

1. Add a reduced-order ocean closure with:
   - prescribed bottom heat flux,
   - salinity state or salinity family,
   - and a top boundary that allows melt/freeze feedback.

2. Introduce geometry-sensitive ocean forcing so that shell slope or shell-thickness
   contrast can alter transport.

3. Add a transient/orbital branch for episodic eccentricity forcing if tectonic
   implications are needed.

4. Add topography or phase-boundary evolution only if the project explicitly moves toward
   shell-shape interpretation.

## Concrete Experiments Worth Running Next

1. A strict apples-to-apples 2D validation:
   - `n_lat = 1`
   - `uniform` forcing
   - 2D defaults
   - and then the same with the warmer documented `T_eq/T_floor` pair

2. A three-scenario 2D benchmark suite:
   - `uniform_transport`
   - `soderlund2014_equator`
   - `lemasquerier2023_polar`

3. For each 2D scenario, report:
   - `H_low` over `0-10 deg`
   - `H_high` over `80-90 deg`
   - `D_cond_low`
   - `D_cond_high`
   - `DeltaH = H_high - H_low`
   - latitude of minimum thickness

4. For the 1D equatorial suite, keep the main text focused on:
   - `H_total`
   - `D_cond`
   - `lid_fraction`
   - convective fraction
   - Juno comparison

5. If depleted modes are kept, treat them as literature-alternative sensitivity cases
   rather than burying them in the same monotonic enhancement trend figure.

## Bottom Line

The current project is scientifically strongest when framed as a model hierarchy:

- audited 1D shell inference,
- equatorial transport proxy comparison against Juno `D_cond`,
- and 2D latitude-response benchmarks across plausible forcing families.

It is scientifically weakest when it is asked to behave like a self-consistent coupled
ocean-shell geodynamics model. The literature now shows that salinity, phase change,
ocean geometry, multiple forcing mechanisms, and dissipation geometry all matter.

So the right improvement strategy is:

- clean up interpretation first,
- upgrade the benchmark physics second,
- and only then consider larger Monte Carlo campaigns.
