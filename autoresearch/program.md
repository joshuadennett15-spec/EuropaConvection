# Autoresearch: Europa Ice Shell Optimization

You are an AI research agent running autonomous experiments on a 2D axisymmetric Europa ice shell model. Your goal is to improve the model's physical realism, observational agreement, and computational performance through systematic experimentation.

## Project Overview

Europa (Jupiter's moon) has a global ocean beneath an ice shell. This project models the ice shell's thermal structure using a 2D latitude-dependent solver. Two codebases:

- **Europa2D/** — Active 2D axisymmetric model (your primary target)
- **EuropaProjectDJ/** — 1D reference solver (trusted baseline, modify cautiously)

### Current State

The 2D model's shell thickness is dominated by surface temperature variation (96 K equator, 46 K pole). The conductive lid at poles is ~14 km thicker than at the equator. The convective sublayer is nearly uniform (~2 km) regardless of latitude, ocean pattern, or tidal strain. This means ocean transport scenarios are barely distinguishable — a known limitation you should try to improve.

## Observational Constraints

| Observable | Value | Uncertainty | Source | Notes |
|---|---|---|---|---|
| D_cond at 35 deg lat | 29 km | +/- 10 km | Levin et al. 2025, Juno MWR | Single flyby at ~35 deg latitude |
| H_total minimum | > 15 km | hard bound | Wakita et al. 2024 | Impact basin formation |
| H_total preferred | > 20 km | soft bound | Wakita et al. 2024 | Multiring formation |
| Surface heat flux (equil.) | 20-50 mW/m2 | order of magnitude | Hussmann & Spohn 2002 | For equilibrium shells |
| T_surf equatorial | 96 K | +/- 5 K | Ashkenazy 2019 | Annual mean |
| T_surf polar | 46 K | +/- 5 K | Ashkenazy 2019 | Annual mean |

**Critical:** The Juno D_cond is measured at ~35 deg latitude, not globally averaged. Apply it at the correct latitude bin.

## Physics Primer

- **D_cond scaling:** D_cond ~ integral[T_surf to T_interior] k(T)/q dT. k(T) = 567/T W/m-K.
- **Why T_surf dominates:** 50 K equator-to-pole contrast creates ~14 km D_cond variation. Typical q_basal heterogeneity creates ~5 km. k(T) amplifies: k at 46 K is ~2x k at 96 K.
- **Ocean transport regimes:**
  - Soderlund (2014): equatorial enhancement via Hadley cells (q* = 0.4)
  - Lemasquerier (2023): polar enhancement, ocean transposes mantle tidal pattern
  - Ashkenazy & Tziperman (2021): uniform (efficient meridional mixing)
- **q* parameter:** q* = 0.91 x mantle_tidal_fraction. Pure tidal gives q* = 0.91.
- **Tidal strain:** Mantle-core pattern: eps(phi) ~ sqrt(1 + 3*sin^2(phi)), 4:1 pole-to-equator dissipation.
- **Convection:** Nu = 0.3446 x Ra^(1/3). Stagnant-lid, ~10 K sublayer delta-T. Ra_crit = 1000.
- **Grain recrystallization:** Higher tidal strain -> smaller grains -> lower viscosity.

## Experiment Protocol

1. Read `autoresearch/best.json` to understand current best scores.
2. Formulate a hypothesis: "If I change X, I expect Y because Z."
3. Make ONE focused code change per experiment.
4. Commit the change: `git add <files> && git commit -m "autoresearch: <hypothesis>"`
5. Run: `python autoresearch/harness.py --mode <mode> --tag "<description>"`
6. Evaluate: did the score improve? Does the physics make sense?
7. If improved: run smoke test `python -m pytest Europa2D/tests/test_validation.py -x`
8. If smoke passes AND score improved: run full suite `python -m pytest Europa2D/tests/`
9. Commit the result: `git commit -m "autoresearch(<mode>): <tag> -- score X.XX (delta Y.YY, IMPROVED/no improvement)"`
10. If failed or tests broken: `git checkout -- Europa2D/src/` to revert, then try next idea.

## Experiment Modes

### `--mode solver`
Optimize wall-clock time and convergence without breaking accuracy.
- Modify: `Europa2D/src/axial_solver.py`
- Budget: ~30 seconds per experiment
- Key metric: wall-clock time (median of 5 runs)
- Guardrail: max temperature error < 0.1 K vs frozen reference

### `--mode physics`
Improve agreement with Juno and other observational constraints.
- Modify: `Europa2D/src/latitude_sampler.py`, `latitude_profile.py`, `literature_scenarios.py`, `axial_solver.py`
- Budget: ~5-10 minutes per experiment (250-sample MC)
- Key metric: negative log-likelihood of D_cond at 35 deg against Juno MWR

### `--mode latitude`
Make the 2D model produce realistic latitude-dependent shell structure.
- Modify: same as physics mode
- Budget: ~15-20 minutes per experiment (3 x 250-sample MC)
- Key metrics: D_conv contrast, scenario discriminability (JS divergence), Ra gradient
- This is the hardest and most valuable mode.

## Research Questions (Prioritized)

### Priority 1 -- Latitude realism
1. Can q_basal patterns be made strong enough to compete with T_surf without violating Juno?
2. Does the convection closure need latitude-dependent parameters?
3. What q_tidal_scale value produces the most realistic latitude structure?
4. Should mantle_tidal_fraction have a narrower or shifted prior?

### Priority 2 -- Physics optimization
5. Are sampling distributions well-centered on observational constraints?
6. Does the 1.20x tidal flux scale have the right magnitude?
7. Can additional constraints tighten the posterior?

### Priority 3 -- Solver performance
8. Can the radial grid be coarsened without losing accuracy?
9. Is the convergence criterion too tight or too loose?
10. Can lateral diffusion be skipped (tau ~ 200 Gyr)?

## Files You May Modify

| Scope | File | Purpose |
|---|---|---|
| Always | `Europa2D/src/axial_solver.py` | Solver numerics |
| Always | `Europa2D/src/latitude_sampler.py` | Sampling distributions |
| Always | `Europa2D/src/latitude_profile.py` | Latitude parameterizations |
| Always | `Europa2D/src/literature_scenarios.py` | Scenario configs |
| Cautious | `Europa2D/src/monte_carlo_2d.py` | MC runner (needs rationale) |
| Cautious | `Europa2D/src/profile_diagnostics.py` | Diagnostics (needs rationale) |
| Cautious | `EuropaProjectDJ/src/` | 1D solver (needs strong justification + test both suites) |

## Files You Must NOT Modify

- `autoresearch/harness.py`
- `autoresearch/objectives.py`
- `autoresearch/program.md` (this file)
- Test files (unless adding NEW tests for NEW physics)

## Stopping Criteria

Stop experimenting when:
- You have reached the maximum experiment count
- 3 consecutive experiments show no improvement
- A test failure you cannot fix after 2 attempts
