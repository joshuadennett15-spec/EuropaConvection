# Bayesian Refit, Covariance, and Posterior Sensitivity Roadmap

Compiled 2026-03-25 to turn recent supervisory feedback into a concrete, statistically defensible workflow for the EuropaConvection 1D Bayesian refit and Sobol analyses.

## Executive Summary

Your meeting feedback can be tightened into three separate questions:

1. Which parameters show real prior-to-posterior tightening under the Juno `D_cond` constraint?
2. Which correlated parameter combinations are informed by the data, even when marginal shrinkage is weak?
3. How does parameter importance change after calibration, without misusing classical Sobol indices?

Those are not the same question, and they should not be answered with one method.

The current repo mostly answers question 1, weakly addresses question 2, and does not yet answer question 3 in a statistically correct way once posterior dependence appears.

The main recommendation is:

- keep the existing Sobol workflow as a prior global sensitivity analysis;
- add covariance-aware posterior diagnostics for the Bayesian refit;
- run a restricted-update ladder where only selected parameter subsets are allowed to tighten between rounds;
- if you want posterior sensitivity, use weighted screening or dependent-input methods, not standard independent-input Sobol indices.

## What The Current Repo Already Does

### Equatorial refit

`EuropaProjectDJ/scripts/bayesian_refit_equatorial.py` already computes:

- Gaussian importance weights from `D_cond`
- ESS and a PSIS-style tail diagnostic
- marginal posterior summaries for `Q_v`, `d_grain`, `q_basal`, `epsilon_0`, and `T_surf`
- marginal shrinkage ratios
- posterior-resampled NPZ archives for downstream use

What it does not compute:

- weighted covariance matrices
- posterior correlation matrices
- pairwise trade-off plots
- principal posterior directions
- any posterior sensitivity metric that respects dependence among parameters

### Mid-latitude refit

`EuropaProjectDJ/scripts/run_midlat_juno_refit.py` currently:

- reweights against `D_cond = 29 +/- 10 km`
- computes marginal posterior summaries for `D_cond`, `H_total`, `q_basal`, and `d_grain`
- tightens priors only for `q_basal` and `d_grain`
- keeps `T_surf` and `epsilon_0` fixed by latitude

This is already close to the "only let one parameter move, then another" idea, but only in a hard-coded two-parameter form.

### Sobol workflow

`EuropaProjectDJ/src/sobol_workflow.py` builds a standard Saltelli/Sobol design on independent prior inputs and then calls SALib for classical Sobol indices.

That is valid for prior global sensitivity.

It is not automatically valid for posterior sensitivity after Bayesian reweighting, because posterior calibration induces dependence between parameters and classical Sobol decompositions assume an input measure that is known and handled consistently in the estimator.

## Tightened Interpretation Of The Meeting Feedback

The phrase "hold only one parameter for reweighting, then add another" should be reframed.

The likelihood weights are generated from model-data mismatch in `D_cond`. They are not parameter-specific weights. So you should not describe this as "reweight parameter A" or "reweight parameter B" in the thesis.

The statistically correct framing is:

- restricted-update calibration experiment
- nested posterior-constrained prior experiment
- subset identifiability ladder

Meaning:

1. Compute posterior weights from the same likelihood.
2. Choose a subset `S` of parameters that are allowed to update from prior to posterior.
3. Keep all other priors fixed.
4. Rerun the forward model with only `S` tightened.
5. Compare posterior predictive behavior and posterior dependence as `S` grows.

This is not Sobol analysis.
This is a calibration-ablation / identifiability experiment.

## What The Literature Says

### 1. Importance reweighting needs stability diagnostics

Vehtari et al. show that raw importance weights can become unstable when the tail is heavy, and recommend Pareto-smoothed importance sampling diagnostics such as ESS and Pareto `k`.

Why this matters here:

- your reweighting is already importance-sampling based;
- any stronger posterior claim should be conditioned on stable weights;
- the current equatorial script is already on the right track by reporting ESS and `k_hat`.

Source:
- Vehtari, Simpson, Gelman, Yao, Gabry (2024), "Pareto Smoothed Importance Sampling", JMLR 25(72). https://jmlr.org/papers/v25/19-556.html

### 2. Posterior covariance is essential when parameters compensate for each other

Raue et al. emphasize that likelihood curvature or profile likelihood can reveal practical non-identifiability and flat trade-off directions that marginal intervals alone can miss.

Why this matters here:

- one Juno-like `D_cond` target can constrain combinations of parameters without strongly shrinking each marginal;
- weak marginal shrinkage does not imply no information;
- the missing object in your current analysis is the joint posterior structure.

Source:
- Raue et al. (2009), "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood". https://academic.oup.com/bioinformatics/article/25/15/1923/213246

### 3. Posterior dependence should be analyzed directly, not inferred from marginals

Albert, Callies, and von Toussaint explicitly argue for representing posterior parameter dependence after calibration and show that step-by-step fixing of parameters can reveal how admissible ranges of other parameters change.

Why this matters here:

- this is very close to what your supervisors asked for;
- it supports pairwise / network / covariance analysis after calibration;
- it also supports a sequential fixing or restricted subset strategy.

Source:
- Albert, Callies, von Toussaint (2022), "A Bayesian Approach to the Estimation of Parameters and Their Interdependencies in Environmental Modeling". https://www.mdpi.com/1099-4300/24/2/231

### 4. Data often inform a low-dimensional combination of parameters, not every parameter separately

Cui, Marzouk, and Willcox develop the idea of a likelihood-informed parameter subspace, where the main prior-to-posterior change happens along a few combinations of parameters.

Why this matters here:

- it gives a principled language for "the data constrain combinations";
- even in a low-dimensional problem, an eigenanalysis of posterior covariance is a cheap version of this idea;
- it is a strong justification for plotting posterior principal directions.

Source:
- Cui, Marzouk, Willcox (2016), "Scalable posterior approximations for large-scale Bayesian inverse problems via likelihood-informed parameter and state reduction". https://doi.org/10.1016/j.jcp.2016.03.055

### 5. Classical Sobol indices are not the right posterior sensitivity tool once inputs are dependent

Kucherenko, Tarantola, and Annoni derive variance-based sensitivity indices for dependent inputs and show that correlations change the behavior of first-order and total sensitivity indices.

Why this matters here:

- posterior calibration induces dependence;
- if you want "Sobol after Bayesian refit", you need a dependent-input formulation or a different metric;
- reusing classical independent-input Sobol logic on posterior samples is not clean.

Source:
- Kucherenko, Tarantola, Annoni (2012), "Estimation of global sensitivity indices for models with dependent variables". https://doi.org/10.1016/j.cpc.2011.12.020

### 6. Shapley effects are one of the cleanest ways to do global sensitivity under dependence

Song, Nelson, and Staum show that Shapley effects remain interpretable under dependence and can avoid the misleading behavior of first-order and total indices under correlated inputs.

Why this matters here:

- if you eventually want a posterior global sensitivity analysis, Shapley effects are a strong candidate;
- they are more expensive than screening methods, but conceptually much cleaner for posterior dependence.

Source:
- Song, Nelson, Staum (2016), "Shapley effects for global sensitivity analysis: Theory and computation". https://doi.org/10.1137/15M1048070

### 7. One-at-a-time parameter moves are often a bad substitute for global sensitivity

Saltelli et al. warn against replacing proper input-space exploration with one-dimensional sweeps.

Why this matters here:

- the proposed one-parameter then two-parameter exercise should not be presented as global sensitivity analysis;
- it is acceptable as an identifiability or calibration-ablation experiment;
- it is not a replacement for Sobol.

Source:
- Saltelli et al. (2019), "Why so many published sensitivity analyses are false: A systematic review of sensitivity analysis practices". https://doi.org/10.1016/j.envsoft.2019.01.012

### 8. There is also a modern "intervention posterior" interpretation

Recent calibration literature uses posterior samples to measure how predictions respond when one parameter is intervened on while accounting for posterior structure.

Why this matters here:

- it provides a literature-backed interpretation for "fix one parameter and see what changes";
- this is likely the cleanest conceptual bridge between your supervisors' wording and a publishable methodology section.

Source:
- Hwang et al. (2025), "Bayesian Model Calibration and Sensitivity Analysis for Oscillating Biological Experiments". https://doi.org/10.1080/00401706.2024.2444310

## Recommended Methodology For This Repo

### A. Add covariance-aware posterior diagnostics first

This should be the immediate next step.

For each refit case, compute on transformed parameters:

- weighted mean
- weighted covariance matrix
- weighted Pearson correlation matrix
- weighted Spearman correlation matrix
- posterior principal components / eigenvectors

Use transformed parameters where appropriate:

- `log10(d_grain)`
- `log10(q_basal)`
- `log10(epsilon_0)`
- `Q_v` in physical units
- `T_surf` in Kelvin

Reason:

- most of these parameters are positive and skewed;
- covariance in raw space will be harder to interpret.

Plots to add:

- posterior correlation heatmap
- pair plot with prior cloud in gray and posterior-resampled cloud in color
- 50% and 90% credible ellipses for the most important pairs
- scree plot of posterior covariance eigenvalues
- loading plot for the leading posterior eigenvector

Interpretation goal:

- identify whether Juno mostly constrains a single parameter, a two-parameter trade-off, or a broader low-dimensional combination.

### B. Replace vague "one parameter reweighting" with a restricted-update ladder

For the iterative refit, define a sequence of allowed update sets.

### Mid-latitude 35 deg

Because the current mid-latitude setup already fixes `T_surf` and `epsilon_0`, start with:

- `S1 = {q_basal}`
- `S2 = {d_grain}`
- `S3 = {q_basal, d_grain}`

Optional stress test:

- `S4 = {q_basal, d_grain, T_surf}`
- `S5 = {q_basal, d_grain, epsilon_0}`

Only include `S4` and `S5` if you are willing to reinterpret latitude priors as uncertain rather than fixed.

### Equatorial refit

Use a richer ladder because the code already treats five target parameters as calibratable:

- singletons: `q_basal`, `d_grain`, `Q_v`, `epsilon_0`, `T_surf`
- pairs: start with `q_basal + d_grain`, then add the best next parameter
- greedy ladder: best singleton -> best pair -> best triple -> full set

Selection criterion for "best":

- posterior predictive improvement in `D_cond`
- plus stable ESS / `k_hat`
- plus meaningful reduction in posterior uncertainty

Do not choose subsets only by visual appeal.

### C. For each subset, report the same scorecard

For every restricted-update run, record:

- posterior `D_cond` median and 68% interval
- posterior `H_total` median and 68% interval
- ESS and ESS fraction
- PSIS `k_hat`
- weighted covariance determinant or generalized variance
- leading covariance eigenvalue fraction
- marginal shrinkage per parameter
- pairwise posterior correlations for the main parameter pairs

This gives you a clean table for the thesis:

- "what changed in the prediction?"
- "what changed in identifiability?"
- "what new covariance appeared or disappeared?"

### D. Keep prior Sobol and posterior sensitivity separate

Recommended language:

- prior Sobol: "global sensitivity of the unconstrained model"
- posterior weighted screening: "data-informed sensitivity after calibration"
- restricted-update ladder: "subset identifiability experiment"

Do not call the restricted-update ladder a Sobol analysis.

### E. Use a two-tier posterior sensitivity strategy

#### Tier 1: cheap and practical

Starting from posterior-resampled draws, compute:

- weighted PRCC
- weighted Spearman
- weighted mutual information
- random-forest permutation importance on posterior-resampled draws

This is fast and enough for a first paper/thesis chapter.

#### Tier 2: fully defensible dependent-input GSA

If you want a stronger posterior analogue to Sobol, use one of:

- dependent-input Sobol estimators
- Shapley effects

This likely requires:

- either a posterior copula fit or direct posterior resampling
- and probably a lightweight surrogate if you need many model evaluations

Given your current workflow, Shapley effects are the clearest eventual upgrade path.

## How To Apply This Specifically Here

### Mid-latitude runs

The current `midlat35_*_constrained.npz` outputs suggest that the three ocean scenarios are nearly identical at 35 deg because `q_mult` is almost unity across scenarios.

That means:

- the scientifically interesting variation is likely not between the three mid-latitude transport scenarios;
- the interesting question is whether `q_basal` and `d_grain` trade off to hit the same `D_cond`.

So the mid-latitude analysis should focus on:

- posterior covariance of `q_basal` and `d_grain`
- how much of the posterior contraction is explained by each parameter alone
- whether allowing both to move materially changes `H_total` relative to one-parameter updates

### Equatorial runs

Your equatorial refit already has saved posterior products in `EuropaProjectDJ/results/bayesian_refit/`.

That is the better place to pilot covariance and posterior sensitivity because:

- more parameters are already in play
- posterior NPZ archives already exist
- PSIS diagnostics are already implemented

This is where you can test:

- whether `q_basal` and `d_grain` dominate
- whether `Q_v` contributes once those are free
- whether `epsilon_0` or `T_surf` only matter through compensating trade-offs

## Concrete Next Implementation Steps

### 1. Add posterior dependence outputs

Recommended new script:

- `EuropaProjectDJ/scripts/posterior_dependence_analysis.py`

Inputs:

- `EuropaProjectDJ/results/bayesian_refit/posterior_*.npz`
- `EuropaProjectDJ/results/midlat_juno/midlat35_*_constrained.npz`

Outputs:

- weighted covariance CSV
- weighted correlation CSV
- posterior pair plots
- covariance heatmap
- principal-direction summary table

### 2. Make the mid-latitude refit accept an allowed-update set

Modify:

- `EuropaProjectDJ/scripts/run_midlat_juno_refit.py`

Change:

- let `derive_constrained_priors(...)` accept something like `allowed_params`
- only tighten priors for parameters in that set
- save subset label into `summary.json`

### 3. Add a subset experiment driver

Recommended script:

- `EuropaProjectDJ/scripts/run_subset_refit_ladder.py`

Responsibilities:

- define subset ladder
- call refit for each subset
- collect scorecard metrics
- write one summary CSV
- generate a "what changes when I let more parameters move?" figure

### 4. Add posterior-weighted sensitivity

Recommended script:

- `EuropaProjectDJ/scripts/posterior_sensitivity_analysis.py`

Start with:

- weighted PRCC
- weighted Spearman
- weighted MI

Only add Shapley after the screening layer is working.

## Suggested Thesis Language

Use wording like:

"We distinguish three related but different tasks: prior global sensitivity analysis, Bayesian calibration, and posterior dependence analysis. Prior Sobol indices quantify which uncertain inputs dominate model variance before conditioning on data. Bayesian refitting against the Juno conductive-lid constraint yields a posterior over parameters that may exhibit strong dependence and trade-offs. We therefore complement marginal prior-to-posterior shrinkage with weighted covariance analysis and subset-restricted calibration experiments, in which only selected parameters are allowed to update between rounds. Because posterior calibration induces dependence among inputs, posterior sensitivity is assessed using weighted screening metrics and, where needed, dependence-aware global sensitivity measures rather than standard independent-input Sobol indices."

## What Will Actually Improve Your Statistics

This workflow will improve the rigor of the analysis, but it will not create information that is not present in the data.

The honest expectation is:

- one `D_cond` constraint will not uniquely pin down many thermal parameters;
- it will often identify a narrow manifold or trade-off direction instead;
- better statistics here means better diagnosis of identifiability and compensation, not necessarily dramatic marginal shrinkage for every parameter.

That is still a stronger and more publishable result.

## Short Recommended Order Of Work

1. Add weighted covariance and posterior pair diagnostics.
2. Run the restricted-update ladder for equatorial refit first.
3. Run the same ladder for mid-latitude `q_basal` and `d_grain`.
4. Compare prior Sobol vs posterior weighted screening.
5. Only then decide whether a dependent-input Shapley analysis is worth the extra work.

## Source List

- Vehtari et al. (2024), Pareto Smoothed Importance Sampling. https://jmlr.org/papers/v25/19-556.html
- Raue et al. (2009), profile-likelihood identifiability. https://academic.oup.com/bioinformatics/article/25/15/1923/213246
- Albert et al. (2022), posterior parameter interdependencies. https://www.mdpi.com/1099-4300/24/2/231
- Cui et al. (2016), likelihood-informed parameter subspaces. https://doi.org/10.1016/j.jcp.2016.03.055
- Kucherenko et al. (2012), dependent-input Sobol-style indices. https://doi.org/10.1016/j.cpc.2011.12.020
- Song et al. (2016), Shapley effects under dependence. https://doi.org/10.1137/15M1048070
- Saltelli et al. (2019), why many sensitivity analyses fail. https://doi.org/10.1016/j.envsoft.2019.01.012
- Hwang et al. (2025), intervention-posterior sensitivity after calibration. https://doi.org/10.1080/00401706.2024.2444310
