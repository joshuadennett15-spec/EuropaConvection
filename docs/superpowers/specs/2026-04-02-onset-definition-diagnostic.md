# Onset-Definition Diagnostic

## Purpose

This diagnostic exists to test one specific model-definition hypothesis:

> The 2D solver may be suppressing convection because it applies a hard local-`Ra` onset gate after identifying a valid rheological transition temperature `T_c`.

That is a narrower claim than "the convection physics is wrong." The diagnostic is meant to separate:

- a shell that is genuinely fully conductive
- a shell that contains a warm rheological sublayer but is collapsed by the `Ra >= RA_CRIT` gate
- a shell that becomes active once the onset gate is relaxed

This is a diagnostic and interpretation tool. It is not a production-model endorsement.

## Why The Existing Quick Tests Were Not Enough

The older quick tests and validation passes were useful for ranking experiments, but they were not built to answer the onset-definition question cleanly:

- They summarized `D_cond`, `D_conv`, and `Ra`, but did not measure whether a latent transition depth existed before collapse.
- Once a column is collapsed to `D_conv = 0`, the aggregate MC arrays no longer preserve the pre-collapse interface depth.
- They could tell us that convection looked weak, but not whether the weakness came from the hard onset gate or from the underlying shell physics.

So the right test must inspect explicit solver samples and reconstruct the latent transition depth from the equilibrium temperature profile itself.

## What The New Runner Does

The new runner is:

- [run_onset_definition_diagnostic.py](/c:/Users/Joshu/.cursor/projects/EuropaConvection/autoresearch/experiments/run_onset_definition_diagnostic.py)

It runs explicit 2D samples for each scenario and records, at every latitude:

- `D_cond`: actual conductive thickness used by the solver
- `D_conv`: actual convective thickness used by the solver
- `latent_D_conv`: the sublayer thickness implied by the temperature crossing `T_c`, regardless of whether the solver later collapsed it
- `transition_fraction`: fraction of samples with a rheological transition inside the shell
- `active_fraction`: fraction of samples the solver treats as convecting
- `collapsed_fraction`: fraction of samples with a latent transition but no active convective layer
- `upper_clamp_fraction`: fraction of samples hitting the `0.95H` conductive clamp
- `Ra_median`, `Nu_median`

It then feeds the same `D_cond` / `D_conv` / `H` / `Ra` arrays into the existing latitude scorer so the onset test can still be compared against the usual Juno-facing metrics.

## Experiment Matrix

Default experiment sweep:

- `baseline`
- `ra_onset_1000`
- `ra_onset_100`
- `ra_onset_10`
- `ra_onset_1`

Interpretation:

- `baseline` is the current production behavior.
- `ra_onset_1000` is the control. It should match baseline closely. If it does not, the hypothesis hook is perturbing more than intended.
- `ra_onset_100` and `ra_onset_10` show whether the result changes gradually or only when the gate is almost removed.
- `ra_onset_1` is the near-disabled diagnostic. It is not a claim that the physical critical Rayleigh number is 1.

## Key Metrics And What They Mean

### `transition_fraction`

Fraction of columns where the equilibrium temperature profile crosses `T_c` above the base, so a rheological transition exists inside the shell.

This is the closest diagnostic proxy for:

> "Green/Deschamps says a warm mobile sublayer exists here."

### `active_fraction`

Fraction of columns where the solver ends with `state.is_convecting = True` and a nontrivial `D_conv`.

This is:

> "The production solver actually uses convective transport here."

### `collapsed_fraction`

Fraction of columns where:

- a latent transition exists
- but the final solver state has no meaningful active convective sublayer

This is the smoking-gun metric for the onset mismatch. If this is large in baseline and collapses toward zero as `ra_crit_override` is reduced, the hard onset gate is a primary suppressor.

### `latent_D_conv`

This is not the solver's final `D_conv`. It is the warm-sublayer thickness implied by the equilibrium temperature crossing `T_c`.

Use it to distinguish:

- `latent_D_conv ~ 0`: genuinely conductive
- `latent_D_conv > 0` but `D_conv ~ 0`: collapsed by onset logic
- `latent_D_conv > 0` and `D_conv > 0`: active warm sublayer

### `upper_clamp_fraction`

A sanity metric. If a relaxed onset threshold only wakes convection by driving most columns into the `0.95H` clamp, the result is not clean.

## Decision Logic

### Outcome A: Strong support for onset-mismatch hypothesis

This means:

- `transition_fraction` is substantial in baseline
- `collapsed_fraction` is also substantial in baseline
- lowering `ra_crit_override` causes `active_fraction` to rise sharply
- `collapsed_fraction` drops sharply
- `D_cond@35` stays in or near the Juno window

Interpretation:

> The hard local-`Ra` onset gate is likely suppressing shells that the rheological transition criterion would otherwise treat as active or marginally active.

### Outcome B: Weak support

This means:

- lowering `ra_crit_override` changes little
- or only tiny pockets become active

Interpretation:

> The onset gate is not the primary reason convection is weak; the underlying shell/rheology is already too conduction-dominated.

### Outcome C: Activation without useful shell states

This means:

- `active_fraction` rises
- but `D_cond@35` moves far away from Juno
- or clamp fractions become extreme

Interpretation:

> The onset gate matters, but removing it alone does not yield a physically attractive Juno-compatible solution.

## Recommended Commands

Fast diagnostic:

```powershell
python autoresearch/experiments/run_onset_definition_diagnostic.py --samples 8 --workers 8
```

More stable comparison:

```powershell
python autoresearch/experiments/run_onset_definition_diagnostic.py --samples 20 --seeds 42 137 --workers 12
```

Save machine-readable output for downstream analysis:

```powershell
python autoresearch/experiments/run_onset_definition_diagnostic.py --samples 20 --output-json autoresearch/experiments/results/onset_definition_diagnostic.json
```

## What To Look For First

In the printed report, the highest-signal lines are:

1. `Scenario overview`
2. `Delta vs baseline`
3. The latitude tables at `0-20 deg`, `35 deg`, and `70-90 deg`

Most important pattern:

- large `transition_fraction`
- low `active_fraction`
- high `collapsed_fraction`
- then a strong reversal of those numbers under `ra_onset_10` or `ra_onset_1`

## Thesis-Safe Interpretation

If this diagnostic shows strong rescue under relaxed onset thresholds, the safe claim is:

> The baseline 2D implementation appears sensitive to the choice of onset criterion. A substantial fraction of columns contain a rheologically defined warm sublayer but are classified as non-convecting under the hard local-`Ra` threshold. Relaxing that threshold activates additional sublayers, indicating that convection suppression in the current implementation is partly a model-definition issue rather than purely a lack of thermal driving.

Avoid the stronger claim:

> "Therefore Europa definitely convects at Juno thicknesses."

That conclusion requires a separate physical argument.
