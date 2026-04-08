# Refined Framework: Bayesian Calibration, Identifiability, and Posterior Sensitivity for Europa's Ice Shell

**Compiled 2026-03-25. Supersedes the original roadmap with literature-grounded methodology.**

---

## The Single Question This Analysis Must Answer

> **What does the Juno MWR conductive-lid constraint (D_cond = 29 +/- 10 km) actually tell us about the thermal parameters of Europa's ice shell?**

This decomposes into three testable sub-questions, each requiring a different method:

| Sub-question | Method | Key output |
|---|---|---|
| Q1: Which parameters (or combinations) are constrained by the data? | Prior-normalized eigenanalysis of posterior covariance | Identifiability classification per direction |
| Q2: How does information accumulate as we allow more parameters to update? | Restricted-update ladder scored by KL divergence | Information accumulation curve |
| Q3: Which parameters drive remaining prediction uncertainty after calibration? | Shapley effects on posterior-predictive variance | Variance attribution under dependence |

These are **not the same question** and must not be conflated.

---

## Competing Hypotheses

The analysis should be designed to discriminate four mutually exclusive outcomes:

### H1: Single-Parameter Dominance

D_cond is primarily controlled by one parameter (most likely q_basal, given Howell 2021 found rho = -0.984 between surface heat flux and D_cond). The Juno constraint effectively pins that parameter while others remain unconstrained.

**Signature:**
- One parameter has marginal shrinkage > 50%; all others < 10%
- Posterior correlation matrix is nearly diagonal
- Information ladder saturates at |S| = 1
- Leading eigenvalue of R captures < 30% of total prior-normalized variance

**Falsified if:** No single parameter has shrinkage > 40%, or posterior correlations have |r| > 0.5.

### H2: Trade-Off Manifold

D_cond constrains a low-dimensional combination of parameters (e.g., a q_basal-d_grain trade-off surface). No single parameter is pinned, but a direction in parameter space is.

**Signature:**
- Weak marginal shrinkage (< 30% each), but strong posterior correlation (|r| > 0.6)
- Prior-normalized eigenanalysis: leading eigenvector has rho_1 < 0.3 (> 70% variance reduction) but remaining rho_i > 0.7
- Information ladder saturates at |S| = 2
- Posterior pair plot shows an elongated ellipse rotated off-axis

**Falsified if:** Scree plot shows no dominant eigenvalue (top eigenvalue captures < 50% of variance reduction), or all posterior correlations are weak (|r| < 0.3).

### H3: Weak Information

The Juno window (29 +/- 10 km) is too broad relative to the prior predictive spread to meaningfully constrain any parameter or combination.

**Signature:**
- All marginal shrinkage ratios < 10%
- KL(posterior || prior) < 0.1 nats
- ESS/N > 90% (weights are nearly uniform)
- Prior-normalized eigenvalues all rho_i > 0.9

**Falsified if:** Any parameter has shrinkage > 15%, or KL divergence > 0.1 nats, or any rho_i < 0.8.

### H4: Hierarchical Identifiability

Parameters form a natural ordering: some directly constrained, others constrained only through trade-offs, others effectively unidentifiable. This is the "sloppy model" outcome (Brown & Sethna 2003; Gutenkunst et al. 2007).

**Signature:**
- Monotonically decreasing shrinkage waterfall
- Prior-normalized eigenvalues span > 1 order of magnitude with no clear gap
- Information ladder rises monotonically but with diminishing returns
- Eigenvalue spectrum is approximately log-uniform (the sloppiness signature)

**Falsified if:** Information ladder is non-monotonic (adding parameters decreases total information), or there is a sharp gap in the eigenvalue spectrum separating identifiable from non-identifiable directions.

### Prior Expectation

Based on Howell (2021) correlations and the physics (D_cond ~ k * Delta_T / q_total, where q_total depends on q_basal, d_grain via viscosity, and epsilon_0 via tidal heating), the most likely outcome is **H4 with H2-like structure in the leading directions**: a hierarchical identifiability structure where q_basal and d_grain form the dominant trade-off, Q_v contributes weakly, and epsilon_0 and T_surf are effectively prior-dominated from D_cond alone.

---

## The Unifying Mathematical Object

The entire identifiability analysis is organized around one matrix:

```
R = Sigma_prior^{-1/2} * Sigma_post * Sigma_prior^{-1/2}
```

This is the **prior-normalized posterior covariance**. Its eigendecomposition answers all three sub-questions simultaneously.

### Eigenvalue interpretation

| rho_i | Variance reduction (1 - rho_i) | Interpretation |
|---|---|---|
| < 0.1 | > 90% | Strongly data-informed direction |
| 0.1 - 0.5 | 50-90% | Moderately informed |
| 0.5 - 0.9 | 10-50% | Weakly informed |
| > 0.9 | < 10% | Effectively uninformed (non-identifiable) |

### Connection to the Hessian framework

The eigenvalues relate to the prior-preconditioned Hessian eigenvalues lambda_i (Cui, Marzouk, Willcox 2016) by:

```
rho_i = 1 / (1 + lambda_i)
```

where lambda_i are eigenvalues of Sigma_prior^{1/2} * H_misfit * Sigma_prior^{1/2}, and H_misfit is the Hessian of the negative log-likelihood. When lambda_i >> 1, the data dominate the prior; when lambda_i << 1, the prior dominates.

### Why this matrix and not just marginal shrinkage?

Marginal shrinkage (posterior variance / prior variance for each parameter independently) misses trade-offs. Two parameters can each have marginal shrinkage of only 15%, yet their joint posterior can be concentrated on a narrow ridge (high correlation). R captures this through its off-diagonal structure. The eigenvectors of R reveal the **directions** that are constrained, which may not align with any individual parameter axis.

---

## The Information-Theoretic Backbone: KL Divergence

### Total information gained

```
KL(posterior || prior) = log(N) - log(ESS_entropy)
```

where ESS_entropy = exp(-sum_i w_i * log(w_i)) and w_i are normalized importance weights. This is:

- **Parameterization-invariant** for the joint KL (the same number regardless of whether you work in log-space or linear space)
- **Directly computable** from existing importance weights with zero additional model evaluations
- **Interpretable**: KL = 0 means "learned nothing"; KL = log(N) means "collapsed to a single sample"

(Lindley 1956; Martino, Elvira, Louzada 2017)

### Marginal KL per parameter

```
KL_i = KL(p(theta_i | y) || p(theta_i))
```

This measures how much each individual parameter was informed by the data. **Warning**: marginal KLs do not sum to joint KL. The gap is the change in total correlation:

```
KL_joint = sum_i KL_i + (TC_posterior - TC_prior)
```

where TC = KL(joint || product of marginals). If the data introduce correlations between previously independent parameters (common in inverse problems), TC_posterior > TC_prior and the marginal KLs underestimate total information.

(Watanabe 1960; Cover & Thomas 2006)

### Marginal KL is NOT parameterization-invariant

Joint KL is invariant under invertible reparameterization (the Jacobian terms cancel). But **marginal KL changes** under reparameterization because marginalization and transformation do not commute.

**Rule**: compute marginal KL in the parameterization matching the prior specification:
- log10(d_grain) if the prior is lognormal
- log10(q_basal) if uniformly sampled in log-space
- Q_v in kJ/mol (normal prior, already natural scale)
- T_surf in Kelvin (normal prior)

(Amari & Nagaoka 2000)

---

## Methodology A: Posterior Covariance and Identifiability Diagnostics

### What to compute (from existing posterior NPZ archives — zero new model runs)

**Step 1: Weighted posterior statistics in transformed space**

Transform parameters to their "natural" scale (matching prior specification):
- theta_1 = log10(d_grain)     [lognormal prior]
- theta_2 = log10(q_basal)     [if uniform in log-space, else linear]
- theta_3 = Q_v                [normal prior, kJ/mol]
- theta_4 = log10(epsilon_0)   [lognormal prior]
- theta_5 = T_surf             [normal prior, K]

Compute weighted mean and weighted covariance:
```
mu_post = sum_k w_k * theta^(k)
Sigma_post = sum_k w_k * (theta^(k) - mu_post)(theta^(k) - mu_post)^T
```

Also compute weighted Spearman correlation matrix (rank-transform before weighting, or use SIR resampling then standard Spearman — see Methodology C note on SIR).

**Step 2: Prior covariance matrix**

Sigma_prior is known analytically from the prior specification. For independent priors, it is diagonal.

**Step 3: Prior-normalized eigenanalysis**

```
R = Sigma_prior^{-1/2} * Sigma_post * Sigma_prior^{-1/2}
eigendecompose: R * u_i = rho_i * u_i
v_i = Sigma_prior^{1/2} * u_i   (back to parameter space)
```

**Step 4: Bootstrap confidence intervals**

Resample with replacement from weighted samples (SIR first), recompute Sigma_post, R, and eigenvalues. Report 95% CI on each rho_i and on each correlation coefficient. This is essential because weighted covariance estimation requires more effective samples than mean estimation — correlations from ~10K ESS will have non-trivial uncertainty.

### Decision thresholds (stated a priori)

| Metric | Negligible | Weak | Moderate | Strong |
|---|---|---|---|---|
| Marginal shrinkage (1 - Var_post/Var_prior) | < 10% | 10-30% | 30-60% | > 60% |
| Posterior correlation |r| | < 0.2 | 0.2-0.5 | 0.5-0.7 | > 0.7 |
| Prior-normalized eigenvalue rho_i | > 0.9 | 0.5-0.9 | 0.1-0.5 | < 0.1 |
| KL divergence (nats) | < 0.05 | 0.05-0.2 | 0.2-1.0 | > 1.0 |

These must be declared before looking at results. Adjusting thresholds post hoc to make results "interesting" is a form of HARKing.

### Plots

1. **Posterior correlation heatmap** — lower-triangle, weighted Spearman, RdBu_r centered at 0, values annotated
2. **Prior-posterior pair plot** — gray scatter (prior), colored 50%/90% credible ellipses (posterior), for top 3 parameter pairs by |r|
3. **Scree plot** — eigenvalues rho_i of R, with horizontal dashed lines at 0.1, 0.5, 0.9 thresholds
4. **Loading plot** — eigenvector components for the leading 1-2 directions, showing which parameter combinations are constrained
5. **Shrinkage waterfall** — parameters sorted by marginal shrinkage, bar chart with bootstrap 95% CI, horizontal lines at 10% and 30%

---

## Methodology B: Restricted-Update Ladder (Calibration-Ablation Experiment)

### Formal framing

This methodology draws on three established traditions:

1. **Projection predictive forward search** (Piironen & Vehtari 2017; Lindley 1968): greedy forward selection of variables by minimizing KL divergence from the reference (full) model's predictive distribution to the submodel's.

2. **Ablation studies** (ML tradition): systematic component removal/addition with performance scoring.

3. **Step-by-step parameter fixing** (Albert, Callies, von Toussaint 2022): conditioning on parameter subsets to reveal posterior interdependencies.

The synthesis — a scored sequence of nested importance-reweighted calibrations with greedy subset selection — is, to our knowledge, novel in the geophysical calibration literature.

**This is NOT sensitivity analysis.** The Saltelli et al. (2019) critique of one-at-a-time methods applies to sensitivity analysis (estimating variance attributions), not to calibration identifiability experiments (asking which parameters are informed by data). This distinction must be made explicit in the thesis.

### Procedure

1. Compute importance weights from the full likelihood (Gaussian in D_cond).
2. Define subset S of parameters **allowed to update** from prior to posterior.
3. For parameters NOT in S, resample them from the prior (breaking any posterior dependence with parameters in S).
4. For parameters IN S, apply the full importance weights.
5. Compute the restricted posterior and its diagnostics.

### Selection criterion

Use **KL(posterior_S || prior)** as the primary selection criterion:

```
KL_S = log(N) - log(ESS_entropy_S)
```

At each step of the greedy ladder:
- Start with S = {} (KL = 0)
- For each candidate parameter j not in S, compute KL_{S union {j}}
- Add the parameter j* that maximizes KL_{S union {j}} - KL_S (marginal information gain)
- Record the full scorecard at each step
- Stop when adding any parameter increases KL by less than 0.01 nats (diminishing returns threshold)

### Scorecard (per subset S)

| Metric | Purpose |
|---|---|
| KL(posterior_S \|\| prior) | Total information gained |
| D_cond posterior median and 68% CI | Posterior predictive quality |
| H_total posterior median and 68% CI | Secondary prediction |
| ESS and ESS/N fraction | Importance sampling stability |
| PSIS k_hat | Tail diagnostic (k < 0.7 required) |
| Marginal shrinkage per parameter in S | Per-parameter identifiability |
| Pairwise posterior correlations within S | Trade-off detection |
| det(Sigma_post_S) / det(Sigma_prior_S) | Generalized variance reduction |

### Subset design

**Equatorial refit** (5 calibratable parameters):
- Greedy ladder starting from {} up to {q_basal, d_grain, Q_v, epsilon_0, T_surf}
- Expected ordering based on physics: q_basal first, then d_grain, then Q_v, then epsilon_0/T_surf

**Mid-latitude refit** (2 calibratable parameters, T_surf and epsilon_0 fixed by latitude):
- S1 = {q_basal}
- S2 = {d_grain}
- S3 = {q_basal, d_grain}

### The information accumulation curve

The headline figure: plot KL_S vs |S| for the greedy ladder.

- **If curve saturates at |S| = 1**: H1 (single-parameter dominance)
- **If curve saturates at |S| = 2**: H2 (trade-off manifold)
- **If curve never rises above 0.1 nats**: H3 (weak information)
- **If curve rises monotonically with diminishing returns**: H4 (hierarchical identifiability)

This single figure discriminates all four hypotheses.

---

## Methodology C: Posterior Sensitivity Analysis (Shapley Effects)

### Why Shapley effects and not Sobol indices

Classical Sobol indices require the ANOVA decomposition, which requires a **product measure** (independent inputs). After Bayesian calibration, posterior parameters are correlated. Using standard Sobol on posterior samples:

- Produces indices that do not sum to 1
- Can yield total-order indices smaller than first-order indices
- Has no clean variance-attribution interpretation

(Chastaing, Gamboa, Prieur 2012; Kucherenko, Tarantola, Annoni 2012)

**Shapley effects** (Owen 2014; Song, Nelson, Staum 2016) avoid all these problems:
- Always partition total variance exactly (sum_i Sh_i = Var(Y))
- Always non-negative
- Well-defined under any input distribution, including correlated posteriors
- For d = 5 parameters, require only 2^5 = 32 coalition evaluations — trivially cheap

### The two questions posterior sensitivity must answer

These are **different questions** with potentially **opposite answers**:

| Question | Method | A parameter scores high if... |
|---|---|---|
| "Which parameters were most informed by the data?" | Marginal KL: KL(p(theta_i \| y) \|\| p(theta_i)) | Its posterior is far from its prior |
| "Which parameters drive remaining prediction uncertainty?" | Shapley effects on Var_posterior(D_cond) | Its posterior uncertainty still matters for D_cond |

A parameter can be well-informed (large KL) but unimportant for predictions (small Shapley) — the data pinned it, so it no longer contributes uncertainty. Conversely, a parameter can be uninformed (small KL) but important for predictions (large Shapley) — the data didn't constrain it, so it still dominates.

**Both must be reported.** They tell complementary stories: KL says what the data taught us; Shapley says what still matters.

### Procedure

1. **SIR resampling**: Convert importance-weighted samples to ~N_eff unweighted samples using Sampling-Importance Resampling. This avoids all complications with weighted statistics and is valid when ESS > 1000. (Weighted PRCC has no established statistical foundation — avoid it.)

2. **Compute Shapley effects**: For each coalition J subset {1,...,5}, estimate c(J) = Var(E[D_cond | theta_J]) from the unweighted posterior samples using conditional expectation estimation (binning or nearest-neighbor regression).

3. **Report**: Shapley values Sh_i as percentage of Var_posterior(D_cond), with bootstrap 95% CIs.

4. **Compare with prior Sobol**: Present prior Sobol indices (from the existing independent-input Sobol workflow) side-by-side with posterior Shapley effects. The **change** reveals how calibration reshuffles parameter importance.

### Complementary: Mara-Tarantola triple decomposition

If posterior correlations are strong, decompose each parameter's posterior influence into:

- **Structural contribution** S_i^ind: influence through model physics alone (after decorrelation)
- **Correlative contribution** S_i^corr: influence arising from posterior dependence with other parameters
- **Full contribution** S_i^full = S_i^ind + S_i^corr

This separates "the parameter matters because the model is sensitive to it" from "the parameter matters because it's correlated with something the model is sensitive to." (Mara & Tarantola 2012)

### What NOT to do

- **Do not compute standard Sobol indices on posterior samples.** The ANOVA decomposition is invalid under dependence.
- **Do not use weighted PRCC.** No established statistical theory supports rank-correlating weighted samples.
- **Do not call the restricted-update ladder a Sobol analysis.** It is a calibration identifiability experiment.

### Recommended language

| Analysis | Correct term | Incorrect term |
|---|---|---|
| Saltelli design on independent priors | "Prior global sensitivity analysis" | "Sensitivity analysis" (ambiguous) |
| Shapley effects on posterior samples | "Posterior sensitivity analysis under dependence" | "Posterior Sobol" |
| Restricted-update ladder | "Subset identifiability experiment" | "One-at-a-time sensitivity" |
| Marginal KL per parameter | "Data informativeness ranking" | "Parameter importance" (ambiguous) |

---

## Methodology D: Keep Prior Sobol Separate

The existing Sobol workflow (`sobol_workflow.py`) is valid and should be retained as-is. It answers a different question: "Before seeing data, which parameters dominate model variance?"

**Present prior Sobol and posterior analysis as a before/after pair:**

| | Prior (Sobol) | Posterior (Shapley) |
|---|---|---|
| Input distribution | Independent priors | Correlated posterior |
| Decomposition | ANOVA (valid for product measure) | Shapley (valid for any measure) |
| Question answered | "What drives model uncertainty?" | "What drives remaining uncertainty after calibration?" |
| Indices sum to | Var(Y) [if all interactions included] | Var_post(Y) [always] |

The scientifically interesting comparison is how the ranking changes: does the most important prior parameter remain the most important posterior parameter, or does calibration re-order the rankings?

---

## Physical Context and Novelty

### What makes this work novel

Based on comprehensive literature search (see Source List):

1. **No previous Bayesian inversion against the Juno MWR D_cond constraint** has been published. The Levin et al. (2025) Nature Astronomy result is too recent. This work is among the first to use it as a calibration target.

2. **No published Sobol sensitivity analysis** exists for Europa ice shell thermal models. Howell (2021) computed Spearman correlations but not variance-based sensitivity indices.

3. **No formal identifiability analysis** of Europa thermal models has been published. The prior-normalized eigenanalysis and information ladder are novel for this application domain.

4. **The restricted-update ladder** as a scored, greedy, KL-guided calibration-ablation experiment is a novel synthesis of the projection predictive framework (Piironen & Vehtari 2017), ML ablation studies, and the Albert et al. (2022) posterior interdependency workflow.

### Physical predictions to test

Based on Howell (2021) correlations (surface heat flux rho = -0.984, tidal heating rho = -0.927, viscosity rho = +0.828, grain size rho = +0.752), the expected hierarchy is:

1. **q_basal**: directly controls D_cond through the conductive gradient D_cond ~ k * Delta_T / q_total. Expect strong shrinkage if prior range spans the Juno-consistent region.

2. **d_grain**: controls viscosity (eta ~ d^2 for diffusion creep), which controls tidal heating rate and convective vigor. Expect moderate shrinkage or strong trade-off with q_basal.

3. **Q_v**: controls viscosity temperature-dependence. Expect weak to negligible shrinkage from D_cond alone (narrow prior, limited leverage on lid thickness).

4. **epsilon_0**: controls tidal strain heating amplitude. Expect weak shrinkage (partially degenerate with q_basal through total heating budget).

5. **T_surf**: directly affects conductive gradient but prior is already narrow (~7 K sigma). Expect negligible marginal shrinkage but possible correlation with q_basal.

### Caution: mid-latitude scenario degeneracy

The three ocean transport scenarios (uniform, Soderlund, Lemasquerier) produce nearly identical q_tidal_mult ~ 1.0 at 35 degrees latitude. Presenting three identical results does not strengthen the analysis — it shows the experiment cannot discriminate ocean transport at this latitude. Be honest about this. The mid-latitude analysis should focus on the q_basal-d_grain trade-off structure, not on ocean scenario comparison.

### Tension with impact basin constraints

The impact basin analysis (Science Advances, 2024) yields D_cond ~ 6-8 km for Tyre/Callanish, significantly less than the Juno MWR constraint of 29 +/- 10 km. This tension may reflect:
- Regional variation in ice shell structure
- Temporal evolution (impact basins record the ice shell at time of impact)
- Methodological differences (mechanical response vs. microwave thermal gradient)

This tension is a potential thesis discussion point, not a problem with the methodology.

---

## Suggested Thesis Language

> We test four competing hypotheses about the information content of the Juno MWR conductive-lid constraint for Europa's thermal parameters. We frame the analysis around the prior-normalized posterior covariance matrix R = Sigma_prior^{-1/2} * Sigma_post * Sigma_prior^{-1/2}, whose eigenspectrum directly classifies each parameter direction as data-informed (rho < 0.5) or prior-dominated (rho > 0.9). We complement this eigenanalysis with a restricted-update calibration experiment: a greedy forward-selection ladder scored by Kullback-Leibler divergence from prior to posterior. The shape of the information accumulation curve — whether it saturates at one parameter, two parameters, or rises monotonically — discriminates single-parameter dominance, trade-off manifold, weak information, and hierarchical identifiability hypotheses. Because Bayesian calibration induces posterior dependence among parameters, posterior sensitivity is assessed using Shapley effects (Owen 2014; Song, Nelson, Staum 2016), which correctly partition output variance under arbitrary input distributions. Prior sensitivity uses classical Sobol indices on independent priors. The comparison reveals how calibration reshuffles parameter importance and which uncertainties persist despite the observational constraint.

---

## Implementation Order

### Phase 1: Covariance diagnostics (zero new model runs)

**Script**: `EuropaProjectDJ/scripts/posterior_dependence_analysis.py`

**Inputs**: existing posterior NPZ archives

**Outputs**:
- Weighted covariance and correlation matrices (CSV)
- Prior-normalized eigenanalysis (eigenvalues, eigenvectors)
- KL divergence from weights
- Marginal KL per parameter
- Bootstrap CIs on all quantities

**Plots**: correlation heatmap, pair plot, scree plot, loading plot, shrinkage waterfall

### Phase 2: Restricted-update ladder (minimal new computation)

**Script**: `EuropaProjectDJ/scripts/run_subset_refit_ladder.py`

**Procedure**: reweight existing MC samples with restricted parameter subsets

**Outputs**:
- Scorecard CSV (one row per subset)
- Information accumulation curve
- Hypothesis discrimination verdict

### Phase 3: Shapley effects (from existing posterior samples)

**Script**: `EuropaProjectDJ/scripts/posterior_sensitivity_shapley.py`

**Procedure**: SIR resample, compute 32 coalition values, bootstrap CIs

**Outputs**:
- Shapley values with CIs
- Prior Sobol vs posterior Shapley comparison table
- Mara-Tarantola structural/correlative decomposition (if correlations > 0.5)

### Phase 4: Synthesis figures for thesis

1. Information accumulation curve (the headline figure)
2. Prior-posterior pair plot with credible ellipses
3. Shrinkage waterfall with bootstrap CIs
4. Prior Sobol vs posterior Shapley comparison bars
5. Posterior predictive D_cond and H_total distributions per subset

---

## Source List

### Information Theory and Bayesian Design
- Lindley, D.V. (1956). On a Measure of the Information Provided by an Experiment. *Ann. Math. Statist.* 27(4):986-1005.
- Chaloner, K. and Verdinelli, I. (1995). Bayesian Experimental Design: A Review. *Statistical Science* 10(3):273-304.
- Ryan, E.G., Drovandi, C.C., McGree, J.M., and Pettitt, A.N. (2016). A Review of Modern Computational Algorithms for Bayesian Optimal Design. *Int. Stat. Rev.* 84(1):128-154.
- Martino, L., Elvira, V., and Louzada, F. (2017). Effective Sample Size for Importance Sampling Based on Discrepancy Measures. *Signal Processing* 131:386-401.
- Cover, T.M. and Thomas, J.A. (2006). *Elements of Information Theory*, 2nd ed. Wiley.
- Amari, S. and Nagaoka, H. (2000). *Methods of Information Geometry*. AMS/Oxford.
- Tarantola, A. (1982). Inverse Problems = Quest for Information. *J. Geophys.* 50:159-170.
- Watanabe, S. (1960). Information Theoretical Analysis of Multivariate Correlation. *IBM J. Res. Dev.* 4(1):66-82.

### Identifiability
- Raue, A. et al. (2009). Structural and practical identifiability analysis. *Bioinformatics* 25(15):1923-1929.
- Raue, A. et al. (2013). Joining forces of Bayesian and frequentist methodology. *Phil. Trans. R. Soc. A* 371:20110544.
- Brown, K.S. and Sethna, J.P. (2003). Statistical mechanical approaches to models with many poorly known parameters. *Phys. Rev. E* 68:021904.
- Gutenkunst, R.N. et al. (2007). Universally sloppy parameter sensitivities in systems biology. *PLOS Comput. Biol.* 3(10):e189.
- Transtrum, M.K. et al. (2015). Perspective: Sloppiness and emergent theories. *J. Chem. Phys.* 143:010901.

### Likelihood-Informed Subspaces
- Cui, T., Marzouk, Y., and Willcox, K. (2016). Scalable posterior approximations via likelihood-informed parameter and state reduction. *J. Comput. Phys.* 315:363-387.
- Spantini, A. et al. (2015). Optimal low-rank approximations of Bayesian linear inverse problems. *SIAM J. Sci. Comput.* 37(6):A2451-A2487.
- Flath, H.P. et al. (2011). Fast algorithms for Bayesian UQ based on low-rank partial Hessian approximations. *SIAM J. Sci. Comput.* 33(1):407-432.

### Posterior Sensitivity Under Dependence
- Owen, A.B. (2014). Sobol' indices and Shapley value. *SIAM/ASA J. Uncertain. Quantif.* 2(1):245-251.
- Song, E., Nelson, B.L., and Staum, J. (2016). Shapley effects for global sensitivity analysis. *SIAM/ASA J. Uncertain. Quantif.* 4(1):1060-1083.
- Kucherenko, S., Tarantola, S., and Annoni, P. (2012). Estimation of global sensitivity indices for models with dependent variables. *Comput. Phys. Commun.* 183:937-946.
- Mara, T.A. and Tarantola, S. (2012). Variance-based sensitivity indices for models with dependent inputs. *Reliab. Eng. Syst. Saf.* 107:115-121.
- Chastaing, G., Gamboa, F., and Prieur, C. (2012). Generalized Hoeffding-Sobol decomposition for dependent variables. *Electron. J. Statist.* 6:2420-2448.
- Plischke, E., Rabitti, G., and Borgonovo, E. (2020). Computing Shapley effects for sensitivity analysis. *SIAM/ASA J. Uncertain. Quantif.* 9:1411-1437.
- Goda, T. (2021). A simple algorithm for Shapley effects. arXiv:2009.00874.
- Borgonovo, E., Plischke, E., and Rabitti, G. (2024). The many Shapley values for explainable AI: A sensitivity analysis perspective. *Eur. J. Oper. Res.* 318(3):911-926.
- Hart, J.L. and Gremaud, P.A. (2019). Robustness of the Sobol' indices to marginal distribution uncertainty. *SIAM/ASA J. Uncertain. Quantif.* 7(4):1224-1244.

### Sensitivity Analysis Critiques
- Saltelli, A. et al. (2019). Why so many published sensitivity analyses are false. *Environ. Model. Softw.* 114:29-39.

### Calibration Methodology
- Piironen, J. and Vehtari, A. (2017). Comparison of Bayesian predictive methods for model selection. *Stat. Comput.* 27(3):711-735.
- Lindley, D.V. (1968). The choice of variables in multiple regression. *J. R. Statist. Soc. B* 30(1):31-66.
- Albert, C., Callies, U., and von Toussaint, U. (2022). A Bayesian approach to the estimation of parameters and their interdependencies in environmental modeling. *Entropy* 24(2):231.
- Kallioinen, N., Paananen, T., Burkner, P.-C., and Vehtari, A. (2023). Detecting and diagnosing prior and likelihood sensitivity with power-scaling. *Stat. Comput.* 34:57.
- Vehtari, A. et al. (2024). Pareto Smoothed Importance Sampling. *JMLR* 25(72).

### Europa Ice Shell
- Levin, S.M. et al. (2025). Europa's ice thickness and subsurface structure characterized by the Juno microwave radiometer. *Nature Astronomy*.
- Howell, S.M. (2021). The likely thickness of Europa's icy shell. *Planet. Sci. J.* 2:129.
- Green, A.P. et al. (2021). The growth of Europa's icy shell. *J. Geophys. Res. Planets* 126:e2020JE006677.
- Deschamps, F. and Vilella, K. (2021). Scaling laws for mixed-heated stagnant-lid convection. *J. Geophys. Res. Planets* 126:e2021JE006963.
- Lemasquerier, D. et al. (2023). Ocean heat transport from Europa's seafloor to ice shell base. *AGU Advances* 4:e2023AV000994.
- Chen, C. et al. (2026). Temporal changes in Europa's ice shell thickness. *J. Geophys. Res. Planets* 131:e2024JE008928.
- Goldsby, D.L. and Kohlstedt, D.L. (2001). Superplastic deformation of ice. *J. Geophys. Res.* 106(B6):11017-11030.
- Barr, A.C. and Showman, A.P. (2009). Heat transfer in Europa's icy shell. In *Europa* (Univ. Arizona Press).
- Ashkenazy, Y. et al. (2019). The surface temperature of Europa. *Heliyon* 5(7):e01908.
