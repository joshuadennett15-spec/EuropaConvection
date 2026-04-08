# Sobol Workflow For EuropaProjectDJ

## Bottom Line

For this project, a good Sobol run is not “run SALib on the existing Monte Carlo
archive.” The literature is consistent that a proper Sobol study needs:

- a dedicated low-discrepancy design,
- one output value per design row in preserved order,
- independent factor definitions that match the scientific prior,
- convergence checks,
- confidence intervals,
- explicit handling of failed or non-physical runs.

That is now implemented in:

- `EuropaProjectDJ/src/sobol_workflow.py`
- `EuropaProjectDJ/scripts/run_sobol_suite.py`

## Why This Design

### 1. Dedicated Sobol design

Sobol indices are defined for a structured sampling design, not arbitrary Monte
Carlo draws. The existing `EuropaProjectDJ/scripts/sobol_analysis.py` is useful
screening, but it is not a true Sobol run because it analyzes saved MC draws
after the fact.

### 2. Reduced audited parameter set

The first Sobol pass uses 10 interpretable inputs:

- `q_basal_target_mW_m2`
- `d_grain_mm`
- `epsilon_0`
- `T_surf_K`
- `D_H2O_km`
- `mu_ice_GPa`
- `Q_v_kJ_mol`
- `Q_b_kJ_mol`
- `H_rad_pW_kg`
- `f_porosity`

This is deliberate. The audited baseline already fixes `f_salt = 0`,
`B_k = 1`, and `T_phi = 150 K`, and the remaining diffusion-prefactor terms are
better treated as a second-stage extension than as part of the first
production Sobol run.

### 3. Inverse-CDF mapping to the audited priors

The workflow samples the unit hypercube and maps it to the scientific priors
with inverse transforms:

- uniform for `q_basal_target_mW_m2` and `f_porosity`,
- truncated normal for `T_surf`, `D_H2O`, `mu_ice`, `Q_v`, `Q_b`, `H_rad`,
- truncated log-normal in log10 space for `d_grain` and `epsilon_0`.

That keeps the Sobol design mathematically clean while still matching the
audited prior intent.

### 4. Ordered evaluation with explicit validity flags

True Sobol analysis requires the evaluation order to be preserved. The main MC
runner uses unordered multiprocessing and filters invalid draws, which is fine
for posterior-style ensembles but not for Sobol estimators.

The Sobol runner therefore evaluates rows in design order and records:

- `numerical_success`
- `physical_flag`
- `valid_flag`
- `subcritical_flag`

By default it does **not** silently drop non-physical rows. That matches the
literature emphasis on exploring the full input space rather than moving along
filtered one-dimensional corridors.

The runner now makes that choice explicit with `--physical-output-policy`:

- `keep` for a first-pass response-surface study,
- `nan` if you want the run to skip any QoI touched by non-physical outputs.

### 5. First-order and total-order first

The default run computes `S1` and `ST`. Second-order indices are available, but
they are not the default because they expand the cost substantially and are not
usually the first thing you need for thesis-quality ranking.

### 6. Convergence checkpoints

The runner stores Sobol results at increasing powers of two. For `N = 512`, the
default checkpoints are:

- `128`
- `256`
- `512`

This follows the convergence-and-validation guidance from the environmental and
geoscience sensitivity-analysis literature.

## Recommended Runs

### Main scientific run

Use the audited global baseline first:

```powershell
python EuropaProjectDJ/scripts/run_sobol_suite.py --scenario global_audited --base-samples 512 --n-workers 8
```

If your Python entrypoint is different on your machine, keep the script
arguments and swap the launcher only.

### Grouped interpretability run

After the parameter-level run, do a grouped pass:

```powershell
python EuropaProjectDJ/scripts/run_sobol_suite.py --scenario global_audited --base-samples 512 --grouped --n-workers 8
```

The grouped factors are:

- `basal_flux`
- `shell_rheology`
- `shell_tides`
- `surface_boundary`
- `porosity`

### Equatorial follow-up

Only after the global baseline is stable:

```powershell
python EuropaProjectDJ/scripts/run_sobol_suite.py --scenario equatorial_baseline --base-samples 512 --n-workers 8
```

Then repeat for:

- `equatorial_moderate`
- `equatorial_strong`

That tells you whether the ranking itself changes under the equatorial Juno
proxy assumptions.

## Outputs To Report

The default outputs analyzed are:

- `valid_flag`
- `physical_flag`
- `convective_flag`
- `thickness_km`
- `D_cond_km`
- `D_conv_km`
- `lid_fraction`
- `Ra`
- `Nu`

For the thesis, the most important ones are:

- `thickness_km`
- `D_cond_km`
- `lid_fraction`
- `convective_flag`
- `valid_flag`

## Practical Interpretation Rules

- Treat `ST` as the main screening metric.
- Use `S1` to identify mainly additive controls.
- Large `ST - S1` implies interaction structure.
- If the ranking changes materially between `N = 256` and `N = 512`, do not
  trust the final order yet.
- If `valid_flag` is highly sensitive to one factor, that factor is controlling
  where the model itself becomes non-physical. That is scientifically important,
  not just a nuisance.
- Record the SALib version in the run manifest so the sampling implementation is
  reproducible.

## When To Escalate Beyond Direct Sobol

If `N = 512` or `1024` is too expensive, the literature points to two sensible
fallbacks:

- use existing MC screening metrics to reduce the factor set further,
- or fit an emulator / surrogate and run Sobol on the surrogate.

Do not fake a Sobol analysis by running `SALib.analyze.sobol` on arbitrary Monte
Carlo rows.

## Sources

- Sobol, I. M. (2001). *Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates*. https://doi.org/10.1016/S0378-4754(00)00270-6
- Saltelli, A. et al. (2010). *Variance based sensitivity analysis of model output*. https://doi.org/10.1016/j.cpc.2009.09.018
- SALib documentation. https://salib.readthedocs.io/en/latest/api/SALib.html
- Saltelli, A. et al. (2019). *Why so many published sensitivity analyses are false*. https://doi.org/10.1016/j.envsoft.2019.01.012
- Sarrazin, F. J., Pianosi, F., and Wagener, T. (2016). *Global Sensitivity Analysis of environmental models: Convergence and validation*. https://doi.org/10.1016/j.envsoft.2016.02.005
- Ryan, E. et al. (2018). *Fast sensitivity analysis methods for computationally expensive models*. https://doi.org/10.5194/gmd-11-3131-2018
