# MC Statistical Comparison for Thesis Chapter

## Overview

Statistical analysis framework for comparing the four production Monte Carlo
ensembles (Global Audited, Equatorial 1.0x/1.2x/1.5x) in a single thesis
chapter. Produces AGU-style tables, publication figures, and all intermediate
results as structured JSON/CSV.

## Inputs

Four NPZ archives, each ~15,000 valid samples with Andrade rheology:

| Label | File | Enhancement | N_valid |
|-------|------|-------------|---------|
| Global Audited | `mc_15000_optionA_v2_andrade.npz` | 1.0x (global) | 15,000 |
| Eq Baseline | `eq_baseline_andrade.npz` | 1.0x (equatorial priors) | 14,998 |
| Eq Moderate | `eq_moderate_andrade.npz` | 1.2x | 14,998 |
| Eq Strong | `eq_strong_andrade.npz` | 1.5x | 14,997 |

### QoIs

NPZ key names (plural form) mapped to internal analysis names:

| NPZ key | Internal name | Description |
|---------|--------------|-------------|
| `thicknesses_km` | `thickness_km` | Total ice shell thickness |
| `D_cond_km` | `D_cond_km` | Conductive lid thickness |
| `D_conv_km` | `D_conv_km` | Convective layer thickness |
| `lid_fractions` | `lid_fraction` | D_cond / thickness |
| `Ra_values` | `Ra` | Rayleigh number |
| `Nu_values` | `Nu` | Nusselt number |

`load_scenario()` maps the NPZ keys to the internal names so all downstream
code uses the standardized names.

For Ra and Nu, descriptive statistics are also computed in log10 space
(`log10_Ra`, `log10_Nu`) since these span 6+ orders of magnitude.

### Parameters (for ranking)

All `param_*` keys in the archives (16 in Global Audited, 17 in equatorial
archives which additionally include `param_eq_enhancement`). Zero-variance
constants are auto-detected and excluded from ranking:

| Excluded | Value | Reason |
|----------|-------|--------|
| `param_B_k` | 1.0 | Fixed in audited baseline |
| `param_T_phi` | 150.0 | Fixed in audited baseline |
| `param_f_salt` | 0.0 | Pure-ice baseline |
| `param_eq_enhancement` | 1.0/1.2/1.5 | Constant within each archive; routed to metadata |

The remaining 13 parameters are ranked:

`param_P_tidal`, `param_d_grain`, `param_Q_v`, `param_Q_b`, `param_epsilon_0`,
`param_T_surf`, `param_D_H2O`, `param_mu_ice`, `param_H_rad`,
`param_f_porosity`, `param_D0v`, `param_D0b`, `param_d_del`

`load_scenario()` auto-detects zero-variance columns using a relative threshold
(`std / max(|mean|, 1e-30) < 1e-6`) and filters them with a logged warning.
This is more robust than an absolute threshold for parameters stored at small
magnitudes (e.g., H_rad ~ 1e-12).

## Outputs

- `results/thesis_stats/comparison_results.json` — all statistics, CIs, p-values
- `results/thesis_stats/summary_tables.csv` — AGU-formatted tables
- `results/thesis_stats/parameter_rankings.csv` — Spearman ranks + Kendall's W
- `figures/thesis_stats/` — publication-quality PNG (300 DPI) + PDF

## Statistical Methods

### Block 1: Descriptive Summary

Per scenario, per QoI:

- Mean, median, CBE (PDF mode via Savitzky-Golay), standard deviation
- Percentiles: P5, P16, P25, P50, P75, P84, P95
- IQR, skewness, excess kurtosis
- Bootstrap 95% CIs on median and mean (BCa, 10,000 resamples)
- Conductive fraction: percentage of samples with lid_fraction >= 0.999
  (reported as a standalone summary statistic per scenario)

For Ra and Nu, statistics are computed on both raw and log10-transformed values.

Output: one AGU-style summary table per QoI.

### Block 2: Pairwise Distribution Comparison

For every scenario pair (6 pairs x 6 QoIs = 36 tests):

- **Two-sample KS test** — distribution shape difference (D statistic, p-value)
- **Mann-Whitney U** — location shift with rank-biserial correlation (r_rb) as
  effect size
- **Cliff's delta** — non-parametric effect size, computed directly from
  pairwise comparisons: `cliff_d = (sum(x_i > y_j) - sum(x_i < y_j)) / (n1*n2)`.
  More appropriate than Cohen's d for skewed, bimodal data.
- **Cohen's d** (pooled SD) — standardized mean difference, retained for
  comparison with parametric literature
- **Benjamini-Hochberg FDR** correction applied within each QoI (6 pairwise
  tests per QoI, corrected separately). This preserves statistical power while
  controlling FDR within each physical quantity.

Output: comparison table with D-stat, U-stat, r_rb, cliff_d, cohen_d,
corrected p-values.

### Block 3: Enhancement Sweep Trend

Across the 3 equatorial scenarios (1.0x, 1.2x, 1.5x), per QoI:

- **Kruskal-Wallis** — omnibus test for any difference
- **Jonckheere-Terpstra** — ordered-alternatives test for monotonic trend.
  One-sided expected directions:
  - Decreasing: thickness, D_conv, Ra, Nu
  - Increasing: lid_fraction
  - Two-sided: D_cond (direction depends on competing effects)

  JT is not in scipy; implemented as a sum of pairwise Mann-Whitney U
  statistics across the ordered groups (~20 lines). Included in test suite.

- **Quantile regression** (statsmodels `QuantReg`) at P5, P50, P95 with
  enhancement factor as predictor. With only 3 distinct x-values, slopes are
  reported as descriptive summaries ("the median thins by X km per 0.1x step"),
  not as inferential tests. Bootstrap CIs on slopes (resample within each group,
  refit, 5,000 resamples) provide proper uncertainty estimates.

Output: trend table with H-stat, J-stat, quantile regression slopes + bootstrap CIs.

### Block 4: Parameter Control Ranking

Per scenario, per QoI:

- **Spearman rho** for each of the 13 non-constant parameters, sorted by |rho|
- Significance at p < 0.001, Bonferroni-corrected for 13 parameters
- **Kendall's W** (coefficient of concordance) across the 4 scenarios — tests
  whether all scenarios agree on parameter importance ordering.
  Not in scipy; implemented directly from the formula
  `W = 12 * SS / (k^2 * (n^3 - n))` (~15 lines). Included in test suite.
- Where W is low, report which parameters swap rank and between which scenarios

Output: ranking table + concordance summary.

### Block 5: Bootstrap Stability

Per scenario, for median and P5/P95 of `thickness_km` and conductive fraction:

- Subsample at n = 500, 1000, 2000, 5000, 10000, 15000
- Bootstrap 95% CI at each subsample size
- Report sample size at which median CI width < 1 km

Conductive fraction is included because binary proportions can converge at
different rates than continuous QoIs.

This is the "are 15,000 samples enough?" evidence.

### Block 6: Shell Structure Partitioning

**Whole-population analysis:**
- Lid fraction distributions compared via KS/MW framework from Block 2
- Conductive fraction (lid_fraction >= 0.999) with bootstrap CIs per scenario

**Convective-subpopulation analysis:**
- Filter to samples with lid_fraction < 0.999 (active convective layer)
- Repeat Block 2 pairwise comparison on thickness, D_cond, D_conv, Ra, Nu
  for the convective subpopulation only (BH FDR within each QoI, same as Block 2)
- Ra and Nu conditional statistics (mean, median, IQR) in both raw and log10
  space, per scenario
- D_cond vs D_conv Pearson r per scenario

This separation is necessary because the whole-population KS tests on D_conv,
Ra, and Nu are dominated by the point mass at the conductive boundary
(D_conv=0, Ra sub-critical, Nu=1), which trivially differs across scenarios.
The physically meaningful question is whether the convective sub-population
itself changes shape.

## Code Architecture

Two modules in `EuropaProjectDJ/scripts/`:

### `thesis_stats.py` — Analysis module

```
load_scenario(path) -> dict
    Maps NPZ keys to standardized names. Auto-detects and excludes
    zero-variance parameters. Returns dict with keys:
      "qois": {name: array}, "params": {name: array}, "metadata": {...}

descriptive_summary(data, n_boot=10000) -> dict
pairwise_comparison(data_a, data_b, qois) -> dict
fdr_correct(results_dict) -> dict
enhancement_trend(eq_scenarios, qois) -> dict
parameter_ranking(data, qois) -> dict
ranking_concordance(rankings_by_scenario, qois) -> dict
bootstrap_convergence(data, qois, subsample_sizes) -> dict
shell_structure(data_by_scenario) -> dict
run_all(scenario_paths) -> dict
save_results(results, output_dir)

# Custom statistical implementations (~35 lines total)
_jonckheere_terpstra(groups, alternative) -> (J_stat, p_value)
_kendall_w(rankings_matrix) -> (W, chi2, p_value)
```

### `thesis_figures.py` — Plotting module

Reads `comparison_results.json` and the original NPZ archives.
One function per figure, each returns a `matplotlib.Figure`.

Planned figures:

1. **4-panel thickness PDF overlay** — all scenarios, CBE/median/1-sigma annotated
2. **Quantile shift plot** — P5/P50/P95 vs enhancement factor with bootstrap
   regression lines
3. **Parameter ranking heatmap** — |Spearman rho| for top-5 params x 4 scenarios
4. **Shell structure comparison** — D_cond vs D_conv scatter, 4 panels
5. **Bootstrap convergence** — median +/- CI vs sample size
6. **Enhancement trend summary** — bar chart of median thickness with CI whiskers,
   Jonckheere p annotated
7. **Conductive fraction comparison** — bar chart with bootstrap CIs per scenario

All figures: 300 DPI PNG + PDF, colorblind-safe palette (IBM or Tol),
AGU single/double column widths.

### Dependencies

scipy, numpy, matplotlib, statsmodels (`QuantReg` for quantile regression).

**Note:** `statsmodels` is a new dependency for this project. Install with
`pip install statsmodels`. It is used only for `QuantReg`; all other
statistics use scipy/numpy. Benjamini-Hochberg FDR correction uses
`statsmodels.stats.multitest.multipletests`.

## Test Strategy

### Unit tests (`tests/test_thesis_stats.py`)

- **Descriptive summary:** known array, assert percentiles match scipy
- **KS/MW wrappers:** identical arrays -> D=0, p=1; separated normals -> D~1, p~0
- **FDR correction:** known p-values, assert matches statsmodels multipletests
- **Jonckheere-Terpstra:** monotonically shifted samples -> p < 0.05;
  identical samples -> p > 0.05. Also: reversed order -> p > 0.5 (one-sided)
- **Quantile regression:** y = 2x + noise, assert slope CI contains 2.0
- **Kendall's W:** identical rankings -> W=1.0; random rankings -> W near 0;
  partially concordant -> 0 < W < 1
- **Cliff's delta:** identical distributions -> 0.0; fully separated -> +/-1.0
- **Zero-variance filter:** feed array with constant column, assert excluded

### Smoke test

`run_all()` on the four real archives completes without error and produces
non-empty JSON/CSV.

## Audience and Reporting Style

- Planetary science thesis committee
- AGU-style table formatting
- Statistical choices justified in physical terms (e.g., "KS because
  distributions are non-normal and we care about shape, not just location")
- No over-explanation of standard methods
