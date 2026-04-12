# Framework Integration Notes for Future Revisit

Date: 2026-04-11

## Context

This note captures the recommendation from the April 11, 2026 discussion about
whether adding external modelling frameworks would make the Europa ice-shell
project more rigorous and less "vibecoded".

Short answer:

- Yes, but mostly through targeted validation and physics import.
- No, not by rewriting the whole project into a heavyweight external framework.

## Current Repo State

The current repository already has a real model stack, not just an improvised
prototype:

- `EuropaProjectDJ` is the audited 1D shell model and still the strongest
  inference engine.
- `Europa2D` reuses the 1D solver stack rather than inventing a separate thermal
  core.
- There is a 1D/2D single-column parity test in `Europa2D/tests/test_validation.py`.
- The old ad hoc 2D-only tidal uplift is no longer implicit by default:
  `q_tidal_scale = 1.0` in `Europa2D/src/latitude_sampler.py`.
- The 2D branch now includes a calibrated surface-temperature law and multiple
  tidal-pattern families in `Europa2D/src/latitude_profile.py`.

What is still approximate:

- `Europa2D` is still best interpreted as a shell-response benchmark model.
- It prescribes latitude-dependent forcing instead of solving a fully coupled
  ocean-shell circulation problem.

## Main Recommendation

If the goal is to make the project more defensible, the best path is:

1. Improve the forcing and benchmark physics.
2. Add one or two independent validation paths.
3. Avoid a full framework migration unless the science scope changes a lot.

In other words:

- Replace the weakest closures first.
- Benchmark the current solver against an external FEM tool in a reduced case.
- Do not pay the cost of a total rewrite until there is a clear scientific need.

## Framework Priority

### 1. TidalPy

Best near-term return on effort.

Why:

- It is designed for tidal dissipation and orbit-spin calculations for rocky and
  icy worlds.
- It is much closer to the current weak point of the repo than a full FEM
  rewrite.
- It could improve the quality of tidal-heating inputs, forcing families, and
  rheology-facing diagnostics.

Best use here:

- Replace or calibrate the current hand-shaped tidal forcing.
- Generate more defensible shell-heating scenarios.
- Improve the `D_cond`-facing physical story without replacing the thermal
  solver.

### 2. FEniCS or Elmer/Ice

Best validation tools if an external benchmark is desired.

Why:

- They provide independent finite-element solutions.
- They are good for reduced benchmark problems where the current solver should
  agree on lid thickness, conductive structure, and convection onset behavior.

Best use here:

- Build one reduced 2D benchmark case.
- Compare against the existing `Europa2D` solution.
- Use the result as a scientific validation check, not as an immediate full
  migration target.

### 3. ASPECT or ISSM

Powerful, but likely too heavy for the current project phase.

Why:

- These are serious large-scale community codes.
- They become more attractive if the project moves toward fully resolved
  convection, large-scale geodynamics, or a dedicated planetary FEM branch.
- They are probably overkill for the current question, which is still dominated
  by forcing closures and interpretation discipline.

Best use here:

- Only revisit if the project scope expands into full convection or deeper
  coupled geodynamics.

### 4. PINNICLE

Interesting, but low immediate ROI for this repo.

Why:

- It is more compelling for inverse problems and data-assimilation-style work.
- The current project is primarily a forward shell-physics and uncertainty
  workflow.

Best use here:

- Revisit only if the project starts solving inversion problems from surface or
  radar-derived constraints directly.

### 5. COMSOL

Useful for quick prototyping, but not the best long-term scientific backbone.

Why:

- It can be fast for exploratory models.
- It is weaker on reproducibility, portability, and repo-native workflows than
  the open-source options.

Best use here:

- Only for a quick exploratory sanity check if needed.
- Not recommended as the main long-term platform.

## Suggested Roadmap

### P1: Best next upgrades

- Add a `TidalPy`-backed forcing adapter or at least a calibration workflow for
  tidal-heating scenarios.
- Add one reduced external FEM benchmark in `FEniCS` or `Elmer/Ice`.
- Keep the current 1D audited model as the main inference engine.

### P2: If the science scope expands

- Revisit `ASPECT` for resolved convection-focused work.
- Revisit `ISSM` only if a true planetary FEM branch becomes a core objective.

### P3: Only for a different project mode

- Revisit `PINNICLE` if the work becomes inversion-heavy.
- Use `COMSOL` only for very fast prototypes, not the main code path.

## Practical Bottom Line

The highest-value way to make this project less "vibecoded" is not to import the
biggest framework available. It is to:

- strengthen the tidal/ocean forcing,
- benchmark the existing solver externally,
- and keep the model hierarchy honest about what is solved vs prescribed.

That should improve scientific defensibility much faster than a full rewrite.

## Revisit Trigger

Re-open this note if any of the following become true:

- the project needs a fully resolved convection code,
- the project needs direct inversion from observations,
- reviewers demand independent FEM verification,
- or the shell/ocean forcing becomes the dominant scientific bottleneck.

## External References

- TidalPy: <https://pypi.org/project/TidalPy/0.3.0.dev8/>
- FEniCS: <https://fenicsproject.org/>
- Elmer/Ice example paper: <https://gmd.copernicus.org/articles/11/4563/2018/gmd-11-4563-2018.html>
- ASPECT: <https://aspect.geodynamics.org/>
- ISSM: <https://issm.jpl.nasa.gov/>

## Date-Sensitive Note

As of January 27, 2026, NASA/JPL publicly reported a Juno-based Europa ice-shell
result centered near `28.9 km` conductive thickness. That makes better
`D_cond`-facing forcing and validation especially worth prioritizing in future
model updates.
