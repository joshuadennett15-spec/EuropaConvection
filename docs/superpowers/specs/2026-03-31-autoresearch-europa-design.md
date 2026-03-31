# Autoresearch for EuropaConvection — Design Spec

**Date:** 2026-03-31
**Status:** Draft
**Goal:** Adapt Karpathy's autoresearch paradigm so an AI agent can autonomously run experiments on the Europa ice shell model — optimizing solver performance, parameter fits to observations, and latitude-dependent realism.

---

## 1. Overview

Autoresearch provides a tight experiment loop: an AI agent modifies code, runs an experiment, evaluates a scalar metric, and iterates. The original framework targets ML training (modify `train.py` → train 5 min → check `val_bpb`). This adaptation targets computational geophysics: modify solver/sampler/profile code → run simulation or MC ensemble → evaluate against observational constraints and physical realism.

The system operates fully autonomously on a dedicated git branch. The user kicks it off, walks away, and returns to a structured experiment log with results, diffs, and scores.

### What the agent can modify

| Scope | Files | When |
|-------|-------|------|
| Always allowed | `Europa2D/src/axial_solver.py` | Solver numerics, grid, algorithms |
| Always allowed | `Europa2D/src/latitude_sampler.py` | Sampling distributions, ranges, scales |
| Always allowed | `Europa2D/src/latitude_profile.py` | Latitude-dependent parameterizations |
| Always allowed | `Europa2D/src/literature_scenarios.py` | Scenario configurations |
| Cautious | `Europa2D/src/monte_carlo_2d.py` | MC runner logic (if needed) |
| Cautious | `Europa2D/src/profile_diagnostics.py` | Diagnostic definitions (if needed) |
| Cautious | `EuropaProjectDJ/src/` | 1D solver code (only if physics improvement justified) |

"Cautious" means the agent should have a clear physical rationale before modifying, and must verify 1D/2D validation tests still pass.

### What the agent must not modify

- `autoresearch/harness.py` (experiment infrastructure)
- `autoresearch/objectives.py` (scoring functions)
- `autoresearch/program.md` (its own instructions)
- Test files (unless adding new tests for new physics)

---

## 2. Directory Structure

```
autoresearch/
├── program.md              # Agent instructions — physics primer, protocol, research questions
├── harness.py              # Experiment runner — wraps existing model code
├── objectives.py           # Composite scoring functions per mode
├── reference/              # Immutable per-mode reference artifacts (created on first run)
│   ├── solver_ref.json     # Reference solution for solver accuracy comparison
│   ├── physics_ref.json    # Reference MC results for physics mode
│   └── latitude_ref.json   # Reference triple-scenario results for latitude mode
├── best.json               # Mutable best-so-far scores per mode (updated on improvement)
├── experiments.jsonl        # Append-only experiment log
└── run.py                  # Entry point — launches autonomous agent loop
```

**Reference vs best-so-far separation:** The `reference/` directory is written once on baseline initialization and never overwritten. It captures the frozen starting-point solution that accuracy guardrails compare against (e.g., solver temperature error is always measured against `solver_ref.json`, not against the last accepted run). The `best.json` file tracks the highest-scoring experiment per mode so the agent knows what it's trying to beat. This prevents drift accumulation where a chain of individually-acceptable experiments gradually degrades accuracy.

All experiments run on a git branch: `autoresearch/run-YYYY-MM-DD-HHMMSS`. The `master` branch is never modified.

---

## 3. Experiment Modes

### 3.1 Solver Optimization (`--mode solver`)

**Purpose:** Make the 2D axial solver faster or more accurate without breaking physics.

**Experiment unit:** Single simulation at a fixed reference parameter set (equatorial, Howell defaults).

**Budget:** ~30 seconds per experiment (includes multiple runs for timing stability).

**Metrics:**
- Wall-clock time (median of 5 runs)
- Convergence iteration count (from `result['steps']`)
- Max absolute temperature error vs frozen reference solution in `reference/solver_ref.json` (K)

All three metrics are directly observable from the `AxialSolver2D.run_to_equilibrium()` return dict (`steps`, `T_2d`, `H_profile_km`). Energy conservation residual and Stefan velocity are not currently exposed by the solver API and are deferred to a future instrumentation pass.

**Modifiable files:** `axial_solver.py`

**Guardrails:**
- Temperature error must stay < 0.1 K vs frozen reference (measured against `reference/solver_ref.json`, not `best.json`)
- All existing tests must pass (smoke suite: `test_validation.py` per experiment; full `pytest` on promoted improvements only)

### 3.2 Parameter/Physics Optimization (`--mode physics`)

**Purpose:** Find parameter distributions and physics formulations that best match observational constraints.

**Experiment unit:** 250-sample MC ensemble on the `uniform_transport` scenario.

**Budget:** ~5-10 minutes per experiment.

**Metrics (composite negative log-likelihood, lower = better):**
- D_cond at 35° latitude vs Juno MWR: Gaussian likelihood with μ = 29 km, σ = sqrt(10² + 3²) km. The 35° constraint reflects Juno's actual flyby geometry — not a global average. Observable from `D_cond_profiles[:, lat_idx_35]`.
- H_total hard bound: penalty if any significant fraction of samples produce H_total < 15 km (Wakita et al. 2024 multiring basin constraint). Observable from `H_profiles`.
- Convergence quality: fraction of samples that converged within `max_steps` (proxy for thermal equilibrium — samples that don't converge are already filtered as invalid by the MC runner).
- Valid sample yield: `n_valid / n_iterations`. Higher yield = better-posed parameter space. Directly available from `MonteCarloResults2D`.

**Modifiable files:** `latitude_sampler.py`, `latitude_profile.py`, `literature_scenarios.py`, `axial_solver.py`

**Guardrails:**
- Must not violate hard physical bounds (negative thicknesses, T > T_melt in conductive lid)
- 1D/2D validation test must still pass
- Sampling distributions must remain physically motivated (no arbitrary narrowing to hit targets)

### 3.3 Latitude Realism (`--mode latitude`)

**Purpose:** Get the 2D model to produce physically realistic latitude-dependent shell structure, rather than being dominated by surface temperature with a flat convective layer.

**Experiment unit:** Three consecutive 250-sample MC ensembles — one per ocean transport scenario (uniform, polar-enhanced, equator-enhanced).

**Budget:** ~15-20 minutes per experiment.

**Context — the problem:**
The current 2D model produces shell structure dominated by surface temperature. The conductive lid at poles (T_s = 46 K) is ~14 km thicker than at the equator (T_s = 96 K). The convective sublayer is nearly uniform (~2 km) regardless of latitude, ocean pattern, or tidal strain. This means:
- Ocean transport scenarios are barely distinguishable in the output
- Convective vigor doesn't vary meaningfully with latitude
- The model can't test hypotheses about what controls Europa's latitude structure

**What "realistic" looks like (from literature):**
- Pole-equator total thickness contrast: 2-8 km (sign depends on ocean regime)
- Convective layer thickness should vary with latitude (responds to local Ra)
- Different ocean patterns should produce distinguishable shell structures
- Equatorial Ra should be higher than polar Ra (thinner lid → stronger temperature contrast across convective layer)
- Total shell 20-30 km, consistent with Juno D_cond = 29 ± 10 km at 35°

**Key physics the agent should investigate:**

1. **T_surf parameterization:** The cos^1.25 form is well-calibrated to Ashkenazy (2019), but its interaction with the convection closure may suppress realistic mid-latitude behavior. Agent should explore whether the functional form or exponent needs adjustment for the convective response.

2. **q_basal magnitude and patterns:** The 1.20x tidal flux scale was chosen to nudge away from lid-dominated parameter space. The agent should explore whether stronger scaling, different ocean pattern amplitudes (the `a` parameter in polar/equator enhancement), or different mantle_tidal_fraction ranges can make ocean patterns visible against the T_surf background without violating Juno.

3. **Convection closure latitude sensitivity:** The Nusselt-Rayleigh scaling (Nu = 0.3446 × Ra^1/3) is applied identically at every latitude. The agent should investigate whether latitude-dependent rheology (grain size, activation energy, strain-dependent viscosity) can produce more realistic convective vigor variation.

4. **1D/2D consistency:** The equatorial 2D column should produce similar results to a 1D run at the same forcing. Divergence > 5% signals a boundary condition or parameterization mismatch, not a physical difference.

**Metrics:**
- D_conv latitude contrast: max minus min of `D_conv_median` across latitude (reward > 1 km). Directly observable from `D_conv_profiles`.
- Ra latitude gradient: ratio of median equatorial to median polar Rayleigh number (should be > 1 for uniform ocean). Observable from `Ra_profiles`.
- Scenario discriminability: Jensen-Shannon divergence on `D_cond` at 35° latitude across the three ocean scenarios. Computed as the minimum pairwise JS divergence (of the three pairs), using histograms with fixed 1 km bins from 5-60 km. JS divergence is bounded [0, 1] and symmetric, avoiding the zero-bin and asymmetry problems of raw KL. With 250 samples per scenario this gives stable estimates. Higher = ocean pattern produces more distinguishable shell structure.
- 1D/2D equatorial consistency: a **fixed calibration check** run once per experiment (not per MC sample). One 2D single-column solve at equatorial reference params, one 1D solve at identical params, compare H_total. Metric: `|H_2d - H_1d| / H_1d`. Penalize > 5%. This adds ~3 seconds, not a budget concern. Uses the same validation path as `test_validation.py`.
- Physics sanity: positive thicknesses, sensible Ra/Nu ranges (Ra > 0, Nu ≥ 1), D_cond at 35° within Juno bounds (soft check, not dominant)

**Modifiable files:** `latitude_profile.py`, `axial_solver.py`, `latitude_sampler.py`, `literature_scenarios.py`

**Agent reasoning guidance:** The agent should use physical judgment, not just minimize a score. The `program.md` instructions tell it to reason about why a change did or didn't work — e.g., "increasing q_tidal_scale to 1.5 made polar shells thinner but broke energy balance because..." The experiment log should read like a research notebook.

---

## 4. Composite Scoring (`objectives.py`)

Each mode computes a single scalar score. Lower is better. The score guides the agent but doesn't replace physical reasoning.

### 4.1 Solver Score

```
score = w_time * (t / t_ref) + w_err * (max_T_err / 0.1) + w_iter * (iters / iters_ref)
```

Default weights: w_time = 0.5, w_err = 0.35, w_iter = 0.15.

Hard penalty: score += 1000 if max_T_err > 0.1 K (measured against frozen `reference/solver_ref.json`, not `best.json`).

All inputs directly observable: `t` from wall-clock timing, `max_T_err` from `np.max(np.abs(T_2d - T_ref))`, `iters` from `result['steps']`.

### 4.2 Physics Score

```
score = -log L(D_cond_35 | 29, σ_eff) + penalty(H_total < 15 km) + w_yield * (1 - valid_fraction)
```

Where σ_eff = sqrt(10² + 3²) ≈ 10.44 km (Juno measurement uncertainty + model discrepancy in quadrature).

Penalty for H_total: +100 per percent of samples with H_total < 15 km.

All inputs observable: `D_cond_35` from `D_cond_profiles[:, lat_idx_35]`, `H_total` from `H_profiles`, `valid_fraction` from `n_valid / n_iterations`.

### 4.3 Latitude Score

```
score = -w_dconv * D_conv_contrast_km
        - w_disc * min_pairwise_JS_D_cond_35
        - w_ra * log(Ra_eq_median / Ra_pole_median)
        + w_1d2d * max(0, consistency_error - 0.05) * 100
        + w_juno * max(0, |D_cond_35_median - 29| - 10) * 10
        + w_sanity * sanity_penalty
```

Default weights: w_dconv = 1.0, w_disc = 2.0, w_ra = 0.5, w_1d2d = 5.0, w_juno = 3.0, w_sanity = 10.0.

The negative terms reward desirable properties (more D_conv contrast, better scenario discriminability, steeper Ra gradient). The positive terms penalize constraint violations.

**JS divergence details:** `min_pairwise_JS_D_cond_35` is the minimum of the three pairwise Jensen-Shannon divergences computed on `D_cond` at the 35° latitude bin across the three ocean scenarios. Histograms use fixed 1 km bins from 5-60 km. JS ∈ [0, 1]. Using the minimum (not mean) ensures the agent doesn't get credit for separating only two scenarios while the third collapses onto one of them.

**1D/2D consistency:** `consistency_error` comes from the fixed calibration check described in Section 3.3. One solve, ~3 seconds, not per-sample.

---

## 5. Experiment Harness (`harness.py`)

Invocation:

```bash
python autoresearch/harness.py --mode physics --tag "wider_q_basal_prior" [--n-samples 250] [--n-workers 8]
```

Execution flow:

1. **Run:** Execute the appropriate experiment inside a top-level try/except:
   - `solver`: Import `axial_solver`, run 5 timed single-sim evaluations at reference params.
   - `physics`: Import `monte_carlo_2d`, run 250-sample ensemble on `uniform_transport`.
   - `latitude`: Import `monte_carlo_2d`, run 250-sample ensemble on each of three scenarios sequentially, plus one fixed 1D/2D calibration check.
2. **Score:** Call `objectives.compute_score(mode, results)` → scalar score + metric dict.
3. **Compare:** Load `best.json`, compute delta. For accuracy guardrails, compare against frozen `reference/` artifacts.
4. **Log:** Append to `experiments.jsonl`:
   ```json
   {
     "timestamp": "2026-03-31T22:15:00Z",
     "mode": "latitude",
     "tag": "higher_q_tidal_scale",
     "score": 3.42,
     "delta": -0.58,
     "improved": true,
     "metrics": { "D_conv_contrast": 2.1, "JS_discrim": 0.34 },
     "git_sha": "abc1234",
     "changed_files": ["Europa2D/src/latitude_sampler.py"],
     "notes": "Increased q_tidal_scale from 1.20 to 1.50"
   }
   ```
5. **Update best:** If score improved, update the relevant mode entry in `best.json`.
6. **Print:** Structured output for the agent to parse:
   ```
   === EXPERIMENT RESULT ===
   Mode: latitude
   Tag: higher_q_tidal_scale
   Score: 3.42 (prev: 4.00, delta: -0.58, IMPROVED)
   D_conv_contrast: 2.1 km (prev: 0.8 km)
   JS_discriminability: 0.34 (prev: 0.21)
   ```

### Failure handling

The harness wraps the entire experiment in a try/except. If the run fails (e.g., `RuntimeError: No valid 2D solutions` from `monte_carlo_2d.py:301`, import errors from a bad code change, or any other exception):

- The experiment is logged with `"status": "failed"`, `"score": Infinity`, `"error": "<traceback summary>"`.
- The harness prints a structured `EXPERIMENT FAILED` result with the error message.
- The harness exits with code 0 (not 1) so the agent loop continues rather than crashing.
- The agent sees the failure, reasons about what went wrong, reverts or fixes the change, and tries again.

Failed experiments never update `best.json`.

### Git commit responsibility

The harness does **not** commit. All git operations (branch creation, pre/post-experiment commits) are the responsibility of `run.py` and the AI agent. The harness is a pure run-score-log tool. This avoids dirty-state auto-commits polluting the experiment history.

### Baseline initialization

On first run, `harness.py --init` runs all three modes with current code to create the `reference/` directory and initial `best.json`. The reference artifacts are frozen and never overwritten. `best.json` starts with the same scores and is updated as experiments improve.

---

## 6. Agent Instructions (`program.md`)

The `program.md` file is the agent's primary context. It contains:

### 6.1 Project Context
- What Europa is, what the ice shell model does
- The two codebases (Europa2D and EuropaProjectDJ) and their relationship
- Current state of the model (T_surf dominance finding, weak convective variation)

### 6.2 Observational Constraints
- Juno MWR: D_cond = 29 ± 10 km at ~35° latitude (Levin et al. 2025, Nature Astronomy)
- H_total > 15-20 km (Wakita et al. 2024, impact basins)
- Surface heat flux: 20-50 mW/m² for equilibrium shells
- T_surf: 96 K equatorial, 46 K polar (Ashkenazy 2019)
- Thermal equilibrium requirement: q_surface ≈ q_basal + q_tidal_shell

### 6.3 Physics Primer
Key relationships the agent must understand:

- **D_cond scaling:** D_cond ~ ∫[T_surf to T_interior] k(T)/q dT. Surface temperature drives this through both the integration bounds and the temperature-dependent conductivity k(T) = 567/T.
- **Why T_surf dominates:** 50 K equator-to-pole contrast creates ~14 km D_cond variation. Typical q_basal heterogeneity creates ~5 km. The k(T) dependence amplifies the polar effect (k at 46 K is ~2x k at 96 K).
- **Ocean transport regimes:** Soderlund (2014) equatorial enhancement (Hadley cells), Lemasquerier (2023) polar enhancement (ocean transposes mantle tidal pattern), Ashkenazy & Tziperman (2021) uniform (efficient meridional mixing).
- **q* parameter:** q* = 0.91 × mantle_tidal_fraction. Pure tidal mantle heating gives q* = 0.91; radiogenic dilution reduces it.
- **Tidal strain latitude dependence:** Mantle-core pattern: ε(φ) ~ √(1 + 3sin²φ), giving 4:1 pole-to-equator dissipation ratio (Beuthe 2013).
- **Convection scaling:** Nu = 0.3446 × Ra^(1/3). Stagnant-lid regime with ~10 K sublayer ΔT. Ra_crit = 1000.
- **Grain size recrystallization:** Higher tidal strain → smaller grains → lower viscosity → potentially more vigorous convection at high latitudes.

### 6.4 Experiment Protocol
1. Read `baseline.json` to understand current state.
2. Formulate a hypothesis: "If I change X, I expect Y because Z."
3. Make the code change (single focused modification per experiment).
4. Run `python autoresearch/harness.py --mode <mode> --tag "<description>"`.
5. Evaluate: did the score improve? Does the physics make sense?
6. Log reasoning in the git commit message.
7. Decide next experiment based on results.

### 6.5 Research Questions (Prioritized)

**Priority 1 — Latitude realism:**
1. Can q_basal patterns be made strong enough to compete with T_surf without violating Juno?
2. Does the convection closure need latitude-dependent parameters (grain size, activation energy)?
3. What q_tidal_scale value produces the most physically realistic latitude structure?
4. Should mantle_tidal_fraction have a narrower or shifted prior?

**Priority 2 — Physics optimization:**
5. Are the current sampling distributions well-centered on observational constraints?
6. Does the 1.20x tidal flux scale have the right magnitude?
7. Can additional observational constraints (surface heat flux, H_total range) tighten the posterior?

**Priority 3 — Solver performance:**
8. Can the radial grid be coarsened without losing accuracy?
9. Is the convergence criterion too tight or too loose?
10. Can the lateral diffusion step be skipped (given τ ~ 200 Gyr)?

### 6.6 Guardrails
- **Smoke suite per experiment:** Run `python -m pytest Europa2D/tests/test_validation.py -x` after any physics change. This takes ~10 seconds and catches 1D/2D divergence and basic physics breakage.
- **Full suite on promotion:** When an experiment improves the score and the agent wants to keep it, run the full `python -m pytest Europa2D/tests/` before committing. If full tests fail, revert.
- Never narrow sampling distributions just to hit a target — distributions must be physically motivated.
- Commit before AND after each experiment. The agent (not the harness) is responsible for all git operations.
- If a change breaks tests, revert it before trying the next experiment.
- If modifying `EuropaProjectDJ/src/`, run its tests too (`python -m pytest EuropaProjectDJ/tests/`).

---

## 7. Autonomous Loop (`run.py`)

Entry point:

```bash
python autoresearch/run.py --max-experiments 20 --mode latitude
```

Behavior:

1. Create branch `autoresearch/run-YYYY-MM-DD-HHMMSS` from current HEAD.
2. If `reference/` doesn't exist, run `harness.py --init` to create baseline.
3. Launch Claude Code (or compatible AI agent) with `program.md` as context.
4. Agent autonomously runs experiments. The agent owns all git operations:
   - Commits code changes before running harness (clean working tree)
   - Commits results after evaluating (structured commit message with score)
   - Reverts failed experiments via `git checkout` on affected files
5. Agent stops when:
   - `--max-experiments` reached, OR
   - 3 consecutive experiments with no improvement (diminishing returns), OR
   - A test failure it can't fix in 2 attempts.
6. On completion, print summary: total experiments, best score, best delta, branch name.

The user can also run modes sequentially:

```bash
python autoresearch/run.py --max-experiments 10 --mode solver
python autoresearch/run.py --max-experiments 20 --mode latitude
python autoresearch/run.py --max-experiments 15 --mode physics
```

Each run creates a new branch so experiments are isolated.

---

## 8. Traceability

### Git history
Every experiment produces a commit on the autoresearch branch:
```
autoresearch(latitude): higher_q_tidal_scale — score 3.42 (Δ-0.58, IMPROVED)
```

### Experiment log (`experiments.jsonl`)
Machine-readable, one JSON object per line. Contains all metrics, git SHA, changed files, and agent notes.

### Merging improvements
When the user reviews the experiment log and wants to keep a change:
```bash
git cherry-pick <sha>   # Pick specific winning experiments
# or
git merge autoresearch/run-2026-03-31-221500  # Merge entire run
```

---

## 9. Dependencies

- Python 3.10+ (already satisfied)
- Existing Europa2D dependencies (numpy, scipy)
- No new external dependencies required — the harness imports existing model code directly
- AI agent runtime: Claude Code CLI or compatible agent that can read `program.md`, modify files, and run shell commands

---

## 10. Scope Boundaries

**In scope:**
- `autoresearch/` directory with harness, objectives, program, runner
- `baseline.json` initialization from current model state
- Three experiment modes (solver, physics, latitude)
- Git branch isolation and experiment logging

**Out of scope:**
- Bayesian optimization frameworks (Optuna, Ax) — the agent IS the optimizer
- Dashboard or web UI for results — `experiments.jsonl` + git log suffice
- Multi-GPU or cloud execution — runs on the user's local machine
- Modifications to the 1D validation test suite structure
