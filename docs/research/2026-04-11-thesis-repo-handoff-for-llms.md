# EuropaConvection: External LLM Handoff for Thesis Review

Date: 2026-04-11

## Purpose

This document is meant to be given to external online LLMs so they can review the
thesis research design, repository structure, model workflow, physical
assumptions, and likely improvement opportunities without needing to infer the
project from scratch.

The goal is not to ask for generic code feedback. The goal is to ask for useful,
domain-aware critique of:

- the physical model hierarchy,
- the numerical implementation,
- the uncertainty workflow,
- the observational inference logic,
- and the highest-value next improvements.

## One-Paragraph Summary

This repository models Europa's ice shell using a hierarchy of thermal models.
The trusted baseline is a 1D transient shell solver with parameterized
stagnant-lid convection and Monte Carlo uncertainty quantification
(`EuropaProjectDJ`). A newer 2D latitude-column model (`Europa2D`) reuses the
same 1D thermal/convection core at each latitude, adds latitude-dependent
surface temperature, tidal strain, and ocean heat-flux patterns, and produces
continuous shell-thickness profiles `H(phi)` with uncertainty bounds. The code
also contains a Sobol sensitivity workflow, Juno-facing conductive-lid
comparison logic, plotting/reporting scripts, and an `autoresearch/` harness for
AI-guided experiments on the 2D branch.

## What This Repository Is Trying to Do

The scientific intent is:

- infer plausible equilibrium ice shell thicknesses for Europa,
- separate total shell thickness `H_total` from conductive lid thickness
  `D_cond`,
- quantify how uncertain physical inputs affect those outputs,
- compare the model to Juno microwave-radiometer constraints using the most
  appropriate observable,
- and explore whether latitude-dependent forcing can produce distinguishable
  shell structures under different ocean transport scenarios.

The project is strongest when interpreted as a model hierarchy:

1. audited 1D shell inference,
2. Juno-facing comparison via `D_cond`,
3. 2D latitude-response benchmarks across prescribed forcing families.

It is weaker when treated as a fully self-consistent coupled ocean-shell
circulation model.

## Top-Level Repository Layout

```text
EuropaConvection/
|-- docs/             Research notes, plans, methods, reviews
|-- tools/            Small repo-level utilities
|-- Europa2D/         Active 2D latitude-column shell model
|-- EuropaProjectDJ/  1D shell model, thesis material, sensitivity workflows
|-- autoresearch/     AI-guided experiment harness for Europa2D
`-- results/          Scratch output
```

## The Three Most Important Subprojects

### 1. `EuropaProjectDJ/`

This is the original and still most trusted model branch.

Core role:

- 1D transient thermal evolution of Europa's shell,
- parameterized stagnant-lid convection,
- Monte Carlo uncertainty propagation,
- Sobol sensitivity analysis,
- thesis-facing figures and tables.

Important directories:

- `EuropaProjectDJ/src/`
- `EuropaProjectDJ/scripts/`
- `EuropaProjectDJ/tests/`
- `EuropaProjectDJ/docs/thesis/`
- `EuropaProjectDJ/results/`
- `EuropaProjectDJ/figures/`

### 2. `Europa2D/`

This is the active latitude-dependent extension.

Core role:

- builds one 1D column per latitude,
- applies latitude-dependent boundary/forcing profiles,
- runs each column to equilibrium,
- lightly couples columns with lateral diffusion,
- aggregates Monte Carlo profile statistics,
- supports attribution and literature-scenario experiments.

Important directories:

- `Europa2D/src/`
- `Europa2D/scripts/`
- `Europa2D/tests/`
- `Europa2D/docs/`
- `Europa2D/results/`
- `Europa2D/figures/`

### 3. `autoresearch/`

This is an automation layer for AI-guided experimentation on the 2D model.

Core role:

- define experiment objectives,
- run structured hypothesis-driven edits,
- score changes,
- and keep the AI from modifying the wrong files.

This is not the physical model itself. It is a controlled experimentation wrapper
around `Europa2D`.

## Scientific Design at a Glance

### Main outputs

- `H_total`: total shell thickness
- `D_cond`: conductive lid thickness
- `D_conv`: convective sublayer thickness
- `lid_fraction = D_cond / H_total`
- `Ra`: Rayleigh number of convective layer
- `Nu`: Nusselt number
- latitude structure metrics such as low-latitude vs high-latitude band means

### Observational anchor currently emphasized

The repo treats the Juno microwave-radiometer constraint as primarily a
constraint on `D_cond`, not on total shell thickness.

The project docs repeatedly use a Juno-facing benchmark near:

- `D_cond ~ 29 +/- 10 km`
- at roughly `35 deg` latitude

That distinction matters because `D_cond` is the most directly data-facing output
for this inference workflow, while `H_total` is more geodynamically meaningful.

## Core Numerical / Physical Architecture

## 1D Solver Backbone: `EuropaProjectDJ/src/`

### `constants.py`

This file defines the physical constants and main parameter families:

- `Planetary`
- `Thermal`
- `Rheology`
- `HeatFlux`
- `Convection`
- `Porosity`

Important details:

- Europa radius, mass, gravity, and orbital frequency are centralized here.
- Thermal conductivity defaults to `k(T) = 567 / T` in the Howell-style model.
- Density is temperature-dependent.
- A temperature-dependent `specific_heat(T)` function exists.
- The checked-in config currently sets:
  - `rheology.model = "Andrade"`
  - `convection.NU_SCALING = "green"`
  - `RA_CRIT = 1000`

Configuration is loaded through `ConfigManager` from
`EuropaProjectDJ/src/config.json`.

### `Physics.py`

This is the stateless physics engine.

Key responsibilities:

- composite diffusion/grain-boundary viscosity,
- simpler Frank-Kamenetskii viscosity,
- effective conductivity with porosity and salt corrections,
- thermal diffusivity and inertia,
- volumetric tidal dissipation,
- basal melting point via Clausius-Clapeyron,
- Stefan-condition quantities.

Important physics choices:

- Composite viscosity follows a Howell-style grain-size-sensitive diffusion creep
  framework.
- Tidal heating supports both Maxwell and Andrade rheology through the global
  rheology setting.
- Porosity reduces conductivity in the cold upper shell.
- Salt can modify conductivity via a scaling factor.

### `Convection.py`

This file implements parameterized stagnant-lid convection.

Important ideas:

- the shell is split into conductive lid plus convective sublayer,
- the code dynamically identifies the conductive/convective interface from the
  temperature profile,
- `T_c` and `Ti` are estimated using Green / Deschamps-Vilella-style logic,
- `Ra` is computed for the warm sublayer,
- `Nu` scales with `Ra`,
- conductivity is enhanced in the convecting part of the shell.

Important numerical design:

- dynamic interface detection rather than fixed lid fraction,
- harmonic mean conductivity at interfaces to preserve flux when `k` changes
  sharply,
- explicit storage of convection diagnostics per run.

### `Solver.py`

This is the main 1D transient thermal solver.

Important numerical methods:

- Crank-Nicolson time stepping,
- Rannacher startup with backward-Euler half steps,
- flux-conservative differencing,
- optional spherical geometry for thick shells,
- Stefan-condition shell growth/melt at the base,
- optional convection parameterization integrated into matrix assembly.

High-level workflow per step:

1. build material-property and conductivity profile,
2. compute tidal heating source term,
3. assemble tridiagonal diffusion system,
4. apply surface and basal boundary conditions,
5. solve for the updated temperature field,
6. update shell thickness from the Stefan condition,
7. repeat to equilibrium.

## 1D Governing Physics

The model is solving a transient heat equation of the form:

```text
rho(T) cp(T) dT/dt = (1/G) d/dz [G k_eff(T,z) dT/dz] + q_tidal(T)
```

with:

- fixed or radiative surface boundary condition,
- pressure-dependent basal melting temperature,
- shell-thickness evolution from basal energy balance.

Important boundary logic:

- surface is usually prescribed through `FixedTemperature`,
- basal temperature is `T_melt(P)`,
- basal shell growth rate depends on conductive flux minus prescribed ocean heat
  flux.

## Uncertainty and Inference in 1D

### `Monte_Carlo.py`

This is the generic Monte Carlo framework.

Key responsibilities:

- parameter sampling,
- multiprocessing,
- repeated 1D equilibrium solves,
- result aggregation,
- PDF / CBE / percentile summaries,
- convection diagnostics,
- subpopulation statistics for conductive vs convective branches.

The framework is generic with respect to sampler class.

### `HowellParameterSampler`

This is the classic baseline prior sampler.

It samples parameters like:

- grain size,
- tidal strain amplitude,
- surface temperature,
- shell/ocean depth scale,
- rigidity,
- activation energies,
- radiogenic heating,
- silicate tidal power,
- porosity,
- salt-related terms.

### `audited_sampler.py`

This is the more thesis-mature prior set.

Important audited changes:

- decomposes basal flux into radiogenic plus silicate tidal components,
- tightens and literature-centers the grain-size prior,
- sets pure-ice baseline with `f_salt = 0`,
- narrows porosity,
- tightens `epsilon_0`,
- fixes `T_phi = 150 K`,
- keeps `T_surf` near `104 +/- 7 K`,
- truncates nonphysical draws.

This audited sampler is important because much of the newer work is built around
it rather than the original Howell-like defaults.

### `sobol_workflow.py`

This is a true Saltelli/Sobol sensitivity workflow, not just random MC with
post-hoc ranking.

Important responsibilities:

- generate Sobol designs in unit space,
- map designs into audited physical priors,
- evaluate the thermal model,
- compute first-order and total-order indices,
- preserve convergence checkpoints,
- separate physical failures from successful evaluations.

This is one of the strongest anti-"vibecoding" parts of the repo because it
formalizes global sensitivity analysis rather than relying on ad hoc intuition.

## 2D Model Design: `Europa2D/src/`

## Architectural Idea

The 2D model is not a fully resolved 2D continuum solver in the style of a
general FEM package. It is an axisymmetric latitude-column model:

- each latitude column is a full 1D `Thermal_Solver`,
- shared shell physics are held constant within a realization,
- latitude dependence is introduced through profiles for forcing and boundary
  conditions,
- columns may be weakly coupled by lateral diffusion,
- the solver runs all columns to equilibrium and returns `H(phi)`.

This is scientifically defensible for Europa because horizontal heat diffusion is
extremely weak compared with vertical heat transport, but it remains a shell
response model rather than a full ocean-shell GCM.

### `latitude_profile.py`

This file defines the latitude-dependent forcing and boundary functions.

Important fields:

- `T_eq`
- `T_floor`
- `epsilon_eq`
- `epsilon_pole`
- `q_ocean_mean`
- `ocean_pattern`
- `q_star`
- `mantle_tidal_fraction`
- `tidal_pattern`
- `surface_temp_exponent`
- `surface_pattern`
- grain-latitude mode settings

Important physics encoded here:

- surface temperature is

```text
T_s(phi) = ((T_eq^4 - T_floor^4) cos(phi)^p + T_floor^4)^(1/4)
```

- default `p = 1.25` is a calibrated Ashkenazy-style surface law,
- tidal strain can follow:
  - `mantle_core`
  - `shell_dominated`
  - `non_monotonic`
- ocean heat flux can follow:
  - `uniform`
  - `equator_enhanced`
  - `polar_enhanced`

Important interpretation:

- the ocean heat-flux patterns are literature-inspired proxies,
- they are not solved ocean circulation states.

### `literature_scenarios.py`

This file packages named scenario presets for the 2D model.

Important scenario families:

- `uniform_transport`
- `soderlund2014_equator`
- `lemasquerier2023_polar`
- `lemasquerier2023_polar_strong`

Important surface presets:

- `ashkenazy_low_q` -> `T_eq = 96 K`, `T_floor = 46 K`
- `ashkenazy_high_q` -> `T_eq = 96 K`, `T_floor = 53 K`
- `legacy_110_52` -> older sensitivity-only surface values

These presets are important because they capture the thesis-facing literature
cases in a reproducible form.

### `latitude_sampler.py`

This wraps the audited 1D sampler and adds latitude structure.

Important design choice:

- shared shell parameters are sampled once per realization,
- latitude dependence is added afterward through `LatitudeProfile`.

Important current behavior:

- no implicit tidal uplift in the sampler default path,
- `q_tidal_scale = 1.0` by default in the current source,
- the equatorial temperature anchor is inherited from the audited 1D draw,
- `T_floor` is sampled around the chosen surface preset,
- `q_ocean_mean` inherits the global basal heat budget from the audited 1D model.

### `axial_solver.py`

This is the 2D axisymmetric solver wrapper.

Important design:

- constructs one 1D `Thermal_Solver` per latitude,
- sets per-column `T_surf` and `epsilon_0`,
- optionally modifies grain size by latitude mode,
- uses the shared 1D thermal/convection logic directly,
- applies either explicit or implicit lateral diffusion between columns.

Important interpretation:

- the columns are authoritative for shell thickness,
- lateral diffusion is intentionally weak and mostly a smoothing/completeness
  term,
- the main physics still comes from each column's local vertical energy balance.

### `convection_2d.py`

This provides experimental "hypothesis adjusters" for 2D-only convection logic.

It is important because it creates a controlled place to test alternative 2D
closure ideas without rewriting the 1D solver.

Current mechanisms include:

- `heat_balance`
- `ra_onset`
- `ra_onset_heatbal`
- `tidal_viscosity`

This is best understood as an experimentation hook, not a finalized physical
theory layer.

### `monte_carlo_2d.py`

This is the 2D Monte Carlo runner.

Important responsibilities:

- sample shared physics plus latitude structure,
- choose a warm-start thickness,
- run a full latitude-column solve,
- reject clearly nonphysical profiles,
- aggregate `H(phi)`, `D_cond(phi)`, `D_conv(phi)`, `Ra(phi)`, `Nu(phi)`,
- compute latitude-band means,
- save NPZ outputs for plotting and thesis tables.

Important 2D outputs:

- profile medians and percentiles,
- low-latitude and high-latitude band distributions,
- `D_cond` and `D_conv` profile statistics,
- convective fraction profile,
- `T_c` and `Ti` profile statistics.

### `profile_diagnostics.py`

This is important for thesis-facing interpretation.

It centralizes rules like:

- use area-weighted band means,
- avoid over-interpreting the `90 deg` symmetry boundary node,
- report low-band vs high-band contrasts,
- keep literature metadata attached to proxy forcing patterns.

This is one of the places where the repo is actively trying to be more rigorous
about interpretation, not just calculation.

## 2D Physics / Interpretation in Plain Language

The 2D model is asking:

- if the surface is colder at high latitude,
- if tidal strain changes with latitude,
- and if ocean heat delivery varies with latitude,
- then what equilibrium shell-thickness profile emerges?

What it is not yet doing:

- solving ocean circulation dynamically,
- solving salinity evolution,
- evolving topography,
- solving a fully coupled 2D/3D momentum problem in the shell,
- or inferring longitude structure.

## Important Physics and Model Choices

### 1. Thermal conductivity is strongly temperature-dependent

The repo relies heavily on:

```text
k(T) = 567 / T
```

This makes cold polar ice much more conductive than warmer equatorial ice.
That is one major reason surface temperature strongly controls `D_cond`.

### 2. Basal heat flux is decomposed physically

The audited workflow treats basal heat flux as:

- radiogenic silicate heating,
- plus silicate tidal heating,
- rather than one opaque lumped parameter.

### 3. Tidal heating is volumetric

The model computes shell tidal heating as an internal source term, not just a
boundary flux.

### 4. Convection is parameterized, not fully resolved

The shell is split into:

- conductive lid,
- convective sublayer.

`Nu` enhances transport in the convecting layer, but this remains a scaling-law
closure rather than a resolved convective flow simulation.

### 5. Grain size is extremely important

The newer research notes repeatedly show that grain size is one of the strongest
controls on viscosity and therefore on `D_cond`, convective fraction, and Juno
agreement.

### 6. Surface temperature can dominate latitude structure

The repo's own analysis shows that in many cases the latitude variation of
surface temperature drives shell-thickness contrast more strongly than the
prescribed ocean forcing or tidal strain pattern.

This is an important conclusion for reviewers because it means the 2D model can
be physically meaningful while still remaining strongly boundary-condition
dominated.

## Thesis Workflow and Typical Analysis Path

## Workflow A: 1D global shell-thickness inference

Typical sequence:

1. sample uncertain physical parameters,
2. run the 1D solver to equilibrium,
3. collect `H_total`, `D_cond`, `D_conv`, `Ra`, `Nu`,
4. aggregate many samples into PDFs and summary statistics,
5. compare prior / posterior / conditional behavior.

Representative files:

- `EuropaProjectDJ/src/Monte_Carlo.py`
- `EuropaProjectDJ/src/audited_sampler.py`
- `EuropaProjectDJ/scripts/run_global_updated.py`

## Workflow B: Juno-facing inference

Typical sequence:

1. run prior-predictive ensembles,
2. extract `D_cond` at the appropriate latitude,
3. compare to the Juno MWR constraint,
4. importance-reweight or otherwise analyze the subset of consistent samples,
5. report how the posterior shifts.

Representative files:

- `EuropaProjectDJ/scripts/run_midlat_juno_refit.py`
- `EuropaProjectDJ/scripts/bayesian_inversion_juno.py`
- `Europa2D/docs/2026-04-01-juno-inference-findings.md`

## Workflow C: 2D latitude-structure experiments

Typical sequence:

1. choose literature-backed forcing scenario,
2. choose surface preset and grain prior,
3. sample shared shell physics,
4. construct latitude profiles,
5. run `AxialSolver2D`,
6. collect profile ensembles,
7. compare equator/high-latitude band means,
8. separate `H_total` from `D_cond`,
9. plot and interpret scenario differences.

Representative files:

- `Europa2D/scripts/run_2d_single.py`
- `Europa2D/scripts/run_2d_mc.py`
- `Europa2D/scripts/run_2d_attribution.py`
- `Europa2D/src/profile_diagnostics.py`

## Workflow D: Sensitivity analysis

Typical sequence:

1. generate dedicated Sobol design,
2. map to audited priors,
3. evaluate model outputs,
4. compute first-order and total-order indices,
5. track convergence and validity.

Representative files:

- `EuropaProjectDJ/src/sobol_workflow.py`
- `EuropaProjectDJ/scripts/run_sobol_suite.py`

## Workflow E: AI-guided experiment loop

Typical sequence:

1. state one hypothesis,
2. modify one targeted part of `Europa2D`,
3. run the harness in solver / physics / latitude mode,
4. score the change,
5. run smoke/full tests if it improves,
6. keep only defensible improvements.

Representative files:

- `autoresearch/program.md`
- `autoresearch/harness.py`
- `autoresearch/objectives.py`

## Important Scripts for an External Reviewer to Know

### Highest-value run / analysis scripts

- `EuropaProjectDJ/scripts/run_global_updated.py`
- `EuropaProjectDJ/scripts/run_midlat_juno_refit.py`
- `EuropaProjectDJ/scripts/run_sobol_suite.py`
- `Europa2D/scripts/run_2d_single.py`
- `Europa2D/scripts/run_2d_mc.py`
- `Europa2D/scripts/run_2d_attribution.py`
- `Europa2D/scripts/result_table.py`

### Important plotting / reporting scripts

- `EuropaProjectDJ/scripts/plot_shell_structure.py`
- `EuropaProjectDJ/scripts/plot_sobol.py`
- `EuropaProjectDJ/scripts/generate_pub_figures.py`
- `Europa2D/scripts/plot_thickness_profile.py`
- `Europa2D/scripts/generate_pub_figures.py`

## Validation and Test Coverage

### 1D tests

The 1D branch includes:

- regression anchors for subcritical and supercritical cases,
- convection diagnostics checks,
- NPZ metadata round-trip checks,
- other workflow-specific tests across the `tests/` tree.

Important file:

- `EuropaProjectDJ/tests/test_regression.py`

### 2D tests

The 2D branch includes:

- single-column parity against the 1D solver,
- regression checks for a 3-column benchmark,
- diagnostics tests for area-weighted band means,
- literature scenario tests,
- Monte Carlo output consistency tests.

Important files:

- `Europa2D/tests/test_validation.py`
- `Europa2D/tests/test_regression_2d.py`
- `Europa2D/tests/test_literature_scenarios.py`
- `Europa2D/tests/test_monte_carlo_2d.py`

### Why the tests matter

For an outside reviewer, this is strong evidence that the repo is not just a set
of loosely connected scripts. The project has:

- numerical regression anchors,
- parity tests between 1D and 2D,
- metadata preservation,
- explicit scientific diagnostics checks.

## Current Strengths

- Clear separation between total thickness and conductive lid thickness.
- Reuse of the same 1D solver core inside the 2D branch.
- Audited priors rather than only legacy Howell-like defaults.
- Real Monte Carlo infrastructure with immutable result objects.
- Proper Sobol workflow rather than faux sensitivity analysis.
- Literature-labeled 2D forcing scenarios.
- Explicit thesis-facing diagnostic logic for latitude-band means.
- AI experimentation harness constrained by documented rules.

## Current Limitations / Approximations

- The 2D model prescribes ocean heat-flux families instead of solving ocean
  circulation.
- Latitude structure may be dominated by the surface boundary condition.
- Grain size is highly influential but still weakly constrained physically.
- The model does not include lateral ice flow or shell topography evolution.
- Longitude structure is not represented.
- Ocean salinity and melt/freeze feedbacks are not dynamically solved.
- Some docs and scripts still reflect older assumptions or defaults.

## Known Review Targets and Possible Improvement Areas

These are especially good places for an external LLM to focus.

### 1. Prescribed ocean forcing vs solved ocean physics

The 2D ocean heat-flux patterns are literature-backed proxies, but they are not
resolved circulation models. A reviewer should assess whether the current proxy
families are sufficient for the thesis claims.

### 2. Surface boundary condition dominance

The repo's own diagnostics suggest that `T_surf(phi)` may dominate latitude
structure. A reviewer should assess whether the project's claims about ocean
transport remain appropriately modest.

### 3. Grain-size sensitivity

The Juno-facing inference seems highly sensitive to grain size. A reviewer should
assess whether grain-size treatment is the correct next physics target.

### 4. Mixed physical-property treatment

Some parts of the code use temperature-dependent `cp(T)` and density, while other
parts use fixed values in simplified lateral or diagnostic terms. A reviewer
should assess whether this inconsistency matters materially.

### 5. 2D experimental closure hooks

`Europa2D/src/convection_2d.py` is a structured experimentation layer. A reviewer
should assess whether the current hypothesis mechanisms are physically coherent
and whether the coupling point is the right abstraction.

### 6. Default / script drift

There are signs of historical drift between docs, plots, and current source.
One example worth checking carefully:

- the current 2D sampler default is `q_tidal_scale = 1.0`,
- some docs and figure labels still refer to `1.20`,
- `Europa2D/scripts/run_2d_mc.py` exposes a CLI `--q-tidal-scale` default of
  `1.20`,
- and the main loop should be checked to confirm whether that CLI argument is
  actually forwarded in all code paths.

This is exactly the kind of issue an external reviewer can catch quickly.

### 7. Surface preset drift across old and new notes

The code now supports named presets like `ashkenazy_low_q`, `ashkenazy_high_q`,
and `legacy_110_52`. A reviewer should assess whether all reporting and figures
are clearly labeling which preset is active.

### 8. Safe vs unsafe interpretation

The repo already has internal notes distinguishing safe outputs from unsafe
claims. A reviewer should focus on whether the thesis narrative stays within
those limits.

## Safe Interpretations vs Risky Interpretations

### Usually safe

- prior-predictive and posterior distributions of `H_total`, `D_cond`, and
  convective diagnostics,
- Juno comparison framed around `D_cond`,
- relative comparison between literature proxy scenarios,
- low-latitude vs high-latitude band means,
- sensitivity-analysis ranking of influential parameters.

### Riskier

- strong claims about the true ocean circulation mechanism,
- exact polar-node interpretation at `90 deg`,
- longitude-specific predictions,
- topography or tectonics claims from the current steady-state shell model,
- strong ocean-temperature inference from shell thickness alone.

## Most Important Files for a New Reviewer

If an external LLM can only read a limited subset, this is the best reading
order.

### Read first

- `README.md`
- `Europa2D/README.md`
- `EuropaProjectDJ/README.md`
- `docs/research/2026-03-20-europa-model-improvement-report.md`
- `docs/research/2026-03-19-1d-vs-2d-methodology-vetting.md`

### Then the core 1D model

- `EuropaProjectDJ/src/constants.py`
- `EuropaProjectDJ/src/Physics.py`
- `EuropaProjectDJ/src/Convection.py`
- `EuropaProjectDJ/src/Solver.py`
- `EuropaProjectDJ/src/Monte_Carlo.py`
- `EuropaProjectDJ/src/audited_sampler.py`
- `EuropaProjectDJ/src/sobol_workflow.py`

### Then the core 2D model

- `Europa2D/src/latitude_profile.py`
- `Europa2D/src/literature_scenarios.py`
- `Europa2D/src/latitude_sampler.py`
- `Europa2D/src/axial_solver.py`
- `Europa2D/src/convection_2d.py`
- `Europa2D/src/monte_carlo_2d.py`
- `Europa2D/src/profile_diagnostics.py`

### Then verification

- `EuropaProjectDJ/tests/test_regression.py`
- `Europa2D/tests/test_validation.py`
- `Europa2D/tests/test_regression_2d.py`

## What I Want an External LLM to Evaluate

Please evaluate the repository at three levels:

1. Physics:
   Are the governing assumptions and closures scientifically defensible for the
   claims being made?

2. Numerics:
   Are there implementation or workflow choices that could produce misleading
   results even if the equations are reasonable?

3. Research design:
   What improvements would most increase scientific rigor per unit effort?

## Desired Style of Feedback

I want feedback prioritized like this:

1. likely bugs or implementation mismatches,
2. scientifically important approximations,
3. thesis/reporting risks,
4. high-ROI next improvements,
5. lower-priority polish.

Please be concrete. If you think something is weak, say exactly:

- what is weak,
- why it matters,
- how serious it is,
- and what file or workflow it affects.

## Copy-Paste Prompt for Online LLMs

Use this prompt with the repository or with this handoff note:

```text
I am working on a thesis project that models Europa's ice shell using a 1D
audited thermal solver and a 2D latitude-column extension. Please review the
research design, code architecture, physical assumptions, and workflow.

Focus on:
1. whether the physics and closures are appropriate for the claims,
2. whether there are likely implementation drifts or hidden inconsistencies,
3. what parts are strongest and most publishable,
4. what parts are weakest or most vulnerable to reviewer criticism,
5. the highest-value improvements I should make next.

Please prioritize concrete findings with file-level references or workflow-level
references when possible. Do not just suggest a total rewrite into a giant FEM
code unless you can explain why that is actually the highest-ROI next step.
```

## Bottom Line

This repo is not just a loose pile of scripts. It contains:

- a real 1D thermal-convection solver,
- an audited uncertainty workflow,
- a 2D latitude-column extension,
- literature-backed forcing scenarios,
- Juno-facing inference logic,
- Sobol sensitivity analysis,
- automated experiment scaffolding,
- and meaningful regression / parity tests.

The most useful external review is therefore not "write it from scratch." The
useful review is:

- identify implementation drift,
- tighten the weakest physical closures,
- confirm which outputs are truly defensible,
- and recommend the highest-value next additions to the current hierarchy.
