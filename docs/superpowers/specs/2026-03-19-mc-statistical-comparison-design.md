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

`thickness_km`, `D_cond_km`, `D_conv_km`, `lid_fractions`, `Ra_values`,
`Nu_values`

### Parameters (for ranking)

`param_P_tidal`, `param_d_grain`, `param_Q_v`, `param_Q_b`, `param_epsilon_0`,
`param_T_surf`, `param_D_H2O`, `param_mu_ice`, `param_H_rad`,
`param_f_porosity`

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

Output: one AGU-style summary table per QoI.

### Block 2: Pairwise Distribution Comparison

For every scenario pair (6 pairs x 6 QoIs = 36 tests):

- **Two-sample KS test** — distribution shape difference (D statistic, p-value)
- **Mann-Whitney U** — location shift with rank-biserial correlation (r_rb) as
  effect size
- **Cohen's d** (pooled SD) — standardized mean difference
- **Benjamini-Hochberg FDR** correction across the 36 tests per QoI

Output: comparison table with D-stat, U-stat, r_rb, d, corrected p-values.

### Block 3: Enhancement Sweep Trend

Across the 3 equatorial scenarios (1.0x, 1.2x, 1.5x), per QoI:

- **Kruskal-Wallis** — omnibus test for any difference
- **Jonckheere-Terpstra** — ordered-alternatives test for monotonic trend
  (one-sided: decreasing for thickness/D_conv/Ra/Nu, increasing for
  lid_fraction)
- **Quantile regression** (statsmodels `QuantReg`) at P5, P50, P95 with
  enhancement factor as predictor — slope +/- SE gives "the median thins by
  X +/- Y km per 0.1x enhancement step"

Output: trend table with H-stat, J-stat, quantile regression slopes.

### Block 4: Parameter Control Ranking

Per scenario, per QoI:

- **Spearman rho** for each of the 10 parameters, sorted by |rho|
- Significance at p < 0.001, Bonferroni-corrected for 10 parameters
- **Kendall's W** (coefficient of concordance) across the 4 scenarios — tests
  whether all scenarios agree on parameter importance ordering
- Where W is low, report which parameters swap rank and between which scenarios

Output: ranking table + concordance summary.

### Block 5: Bootstrap Stability

Per scenario, for median and P5/P95 of `thickness_km`:

- Subsample at n = 500, 1000, 2000, 5000, 10000, 15000
- Bootstrap 95% CI at each subsample size
- Report sample size at which median CI width < 1 km

This is the "are 15,000 samples enough?" evidence.

### Block 6: Shell Structure Partitioning

- Lid fraction distributions compared via KS/MW framework from Block 2
- Ra and Nu conditional statistics (mean, median, IQR) per scenario
- D_cond vs D_conv Pearson r per scenario

## Code Architecture

Two modules in `EuropaProjectDJ/scripts/`:

### `thesis_stats.py` — Analysis module

```
load_scenario(path) -> dict
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
```

### `thesis_figures.py` — Plotting module

Reads `comparison_results.json` and the original NPZ archives.
One function per figure, each returns a `matplotlib.Figure`.

Planned figures:

1. **4-panel thickness PDF overlay** — all scenarios, CBE/median/1-sigma annotated
2. **Quantile shift plot** — P5/P50/P95 vs enhancement factor with regression lines
3. **Parameter ranking heatmap** — |Spearman rho| for top-5 params x 4 scenarios
4. **Shell structure comparison** — D_cond vs D_conv scatter, 4 panels
5. **Bootstrap convergence** — median +/- CI vs sample size
6. **Enhancement trend summary** — bar chart of median thickness with CI whiskers,
   Jonckheere p annotated

All figures: 300 DPI PNG + PDF, colorblind-safe palette (IBM or Tol),
AGU single/double column widths.

### Dependencies

scipy, numpy, matplotlib, statsmodels (QuantReg). No new dependencies.

## Test Strategy

### Unit tests (`tests/test_thesis_stats.py`)

- **Descriptive summary:** known array, assert percentiles match scipy
- **KS/MW wrappers:** identical arrays -> D=0, p=1; separated normals -> D~1, p~0
- **FDR correction:** known p-values, assert matches statsmodels multipletests
- **Jonckheere-Terpstra:** monotonically shifted samples -> p < 0.05; identical -> p > 0.05
- **Quantile regression:** y = 2x + noise, assert slope CI contains 2.0
- **Kendall's W:** identical rankings -> W=1.0; random rankings -> W near 0

### Smoke test

`run_all()` on the four real archives completes without error and produces
non-empty JSON/CSV.

## Audience and Reporting Style

- Planetary science thesis committee
- AGU-style table formatting
- Statistical choices justified in physical terms (e.g., "KS because
  distributions are non-normal and we care about shape, not just location")
- No over-explanation of standard methods
