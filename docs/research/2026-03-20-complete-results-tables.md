# Complete Monte Carlo results tables

All production runs for the Europa ice shell convection project.
Juno MWR constraint: equatorial D_cond = 29 +/- 10 km (acceptable: 19--39 km).

---

## Table 1: 1D global baselines (N = 15,000)

Two rheology models with audited 2026 priors, no latitude forcing.

| Run | N | CBE (km) | Median H (km) | 1-sigma H (km) | Median D_cond (km) | 1-sigma D_cond (km) | Median D_conv (km) | 1-sigma D_conv (km) | Conv. frac | Mean lid frac |
|-----|---|----------|---------------|-----------------|--------------------|--------------------|--------------------|--------------------|-----------|--------------|
| Global Howell (Maxwell) | 10,350 | 56.1 | 50.8 | [32.7, 64.3] | 8.9 | [4.8, 25.6] | 35.5 | [19.4, 50.7] | 100.0% | 28% |
| Global audited (Andrade) | 15,000 | 19.1 | 28.9 | [16.3, 56.7] | 15.2 | [6.8, 25.6] | 9.3 | [0.0, 40.5] | 61.5% | 60% |

**Key finding:** Andrade rheology produces thinner shells with a bimodal
convective/conductive split. Maxwell forces all realizations into
convection with thick shells and thin conductive lids that are
incompatible with Juno MWR.

---

## Table 2: 1D equatorial proxy suite (N = 15,000, Andrade, shared seed 10042)

Ocean heat transport scaled as a multiplier on the tidal component of
equatorial basal heat flux. All modes share the same RNG seed for
paired comparison.

| Mode | Enhancement | N | CBE (km) | Median H (km) | 1-sigma H (km) | Median D_cond (km) | 1-sigma D_cond (km) | Median D_conv (km) | 1-sigma D_conv (km) | Conv. frac | Mean lid frac |
|------|------------|---|----------|---------------|-----------------|--------------------|--------------------|--------------------|--------------------|-----------|--------------|
| Depleted strong | 0.55x | 14,995 | 60.0 | 50.9 | [34.3, 67.0] | 15.6 | [8.7, 31.6] | 31.0 | [14.0, 48.2] | 89.3% | 41% |
| Depleted | 0.67x | 14,997 | 32.6 | 46.6 | [27.9, 65.8] | 15.5 | [8.5, 31.8] | 26.5 | [6.4, 46.6] | 84.7% | 45% |
| Baseline | 1.0x | 14,998 | 21.3 | 35.3 | [18.2, 62.8] | 17.0 | [8.2, 29.0] | 14.6 | [0.0, 42.9] | 67.8% | 58% |
| Moderate | 1.2x | 14,998 | 18.9 | 29.8 | [16.7, 61.1] | 17.2 | [8.1, 27.3] | 7.7 | [0.0, 41.1] | 60.5% | 63% |
| Strong | 1.5x | 14,997 | 16.8 | 23.6 | [14.7, 59.0] | 16.2 | [8.4, 25.4] | 2.5 | [0.0, 38.6] | 52.4% | 69% |

**Key findings:**

- Enhancement thins the shell and suppresses convection (Ra ~ d^3).
  At 1.5x, nearly half of realizations are purely conductive.
- Depletion thickens the shell but barely changes the convective
  fraction because the shell was already well above Ra_crit.
- All modes have median D_cond in the 15--17 km range. Baseline (1.0x)
  has the best Juno overlap: its 1-sigma upper bound reaches 29 km.
- Enhancement narrows the D_cond distribution (strong: [8.4, 25.4]);
  depletion widens it (depleted: [8.5, 31.8]).

Literature mapping:

| Factor | Source | Physical interpretation |
|--------|--------|----------------------|
| 0.55x | Lemasquerier et al. (2023), q*=0.91 | Strong polar-focused tidal heating |
| 0.67x | Lemasquerier et al. (2023), 2:1 pole/eq | Conservative polar-focused tidal heating |
| 1.0x | Ashkenazy & Tziperman (2021) | Uniform meridional ocean transport |
| 1.2x | Soderlund (2014) proxy | Moderate equatorial ocean focusing |
| 1.5x | Upper-bound sensitivity | Strong equatorial ocean focusing |

---

## Table 3: 2D latitude-resolved model (N = 500, Andrade, 37 latitude columns)

Full latitude-dependent shell structure from equator (0 deg) to pole
(90 deg) with scenario-specific q(lat) profiles.

### 3a: Global (latitude-averaged) statistics

| Scenario | N valid | Median H (km) | 1-sigma H (km) | Conv. frac |
|----------|---------|---------------|-----------------|-----------|
| Uniform transport | 477 | 28.8 | [18.5, 39.5] | 43.0% |
| Soderlund equatorial (q*=0.4) | 485 | 30.5 | [20.5, 42.3] | 40.4% |
| Lemasquerier polar (q*=0.455) | 486 | 28.0 | [19.3, 40.6] | 40.7% |
| Lemasquerier polar strong (q*=0.819) | 477 | 27.0 | [19.8, 36.8] | 34.6% |

### 3b: Equatorial column (lat = 0 deg)

| Scenario | Median H_eq (km) | 1-sigma H_eq (km) | Median D_cond_eq (km) | 1-sigma D_cond_eq (km) | Median D_conv_eq (km) | 1-sigma D_conv_eq (km) | Lid frac_eq |
|----------|-----------------|-------------------|----------------------|-----------------------|----------------------|-----------------------|------------|
| Uniform transport | 26.4 | [14.5, 38.5] | 21.3 | [9.1, 31.5] | 4.2 | [3.0, 8.9] | 76% |
| Soderlund equatorial (q*=0.4) | 24.3 | [15.0, 35.7] | 20.7 | [9.0, 30.5] | 3.7 | [2.6, 6.4] | 79% |
| Lemasquerier polar (q*=0.455) | 30.7 | [15.4, 43.4] | 22.5 | [8.7, 34.4] | 4.9 | [3.5, 17.4] | 71% |
| Lemasquerier polar strong (q*=0.819) | 33.3 | [16.9, 46.3] | 18.9 | [10.1, 36.4] | 5.4 | [4.1, 18.4] | 69% |

### 3c: Polar column (lat = 90 deg)

| Scenario | Median H_pole (km) | 1-sigma H_pole (km) | Median D_cond_pole (km) | 1-sigma D_cond_pole (km) | Median D_conv_pole (km) | 1-sigma D_conv_pole (km) | Lid frac_pole |
|----------|-------------------|---------------------|------------------------|-----------------------|------------------------|-----------------------|--------------|
| Uniform transport | 45.2 | [34.9, 65.6] | 41.4 | [31.7, 60.4] | 3.6 | [2.7, 5.6] | 92% |
| Soderlund equatorial (q*=0.4) | 53.4 | [40.9, 73.4] | 48.8 | [36.8, 67.6] | 4.5 | [3.5, 6.7] | 91% |
| Lemasquerier polar (q*=0.455) | 40.3 | [29.5, 63.0] | 37.4 | [27.3, 57.5] | 3.1 | [2.1, 5.1] | 92% |
| Lemasquerier polar strong (q*=0.819) | 33.8 | [26.4, 51.4] | 31.4 | [24.4, 47.0] | 2.5 | [1.9, 4.0] | 92% |

### 3d: Equator-to-pole contrast

| Scenario | Delta-H (km) | Delta-D_cond (km) | Pole/Equator H ratio |
|----------|--------------|--------------------|---------------------|
| Uniform transport | +18.8 | +20.1 | 1.71 |
| Soderlund equatorial (q*=0.4) | +29.1 | +28.1 | 2.20 |
| Lemasquerier polar (q*=0.455) | +9.6 | +14.9 | 1.31 |
| Lemasquerier polar strong (q*=0.819) | +0.5 | +12.5 | 1.02 |

**Key findings:**

- Soderlund produces the largest equator-to-pole contrast (2.2x ratio,
  delta-H = +29 km) and the thickest polar shells (53.4 km median).
  This is the scenario most at odds with a geologically active pole.
- Lemasquerier strong nearly eliminates the thickness contrast
  (ratio 1.02), producing a nearly uniform shell -- but at the cost
  of reducing the convective fraction to 35%.
- Conservative Lemasquerier (q*=0.455) gives the best equatorial
  D_cond match to Juno (22.5 km, closest to 29 km) while keeping
  a moderate pole/equator contrast (1.31x).
- Uniform transport is the balanced middle ground: good equatorial
  D_cond (21.3 km), moderate contrast (1.71x), and highest
  convective fraction (43%).

---

## Table 4: Cross-model Juno MWR comparison

Equatorial D_cond median and 1-sigma against the Juno constraint
(29 +/- 10 km, acceptable window 19--39 km).

| Model | Scenario | Median D_cond_eq (km) | 1-sigma upper (km) | Within Juno 1-sigma? |
|-------|----------|-----------------------|--------------------|---------------------|
| 1D global | Howell (Maxwell) | 8.9 | 25.6 | No -- too thin |
| 1D global | Audited (Andrade) | 15.2 | 25.6 | Marginal |
| 1D equatorial | Depleted strong (0.55x) | 15.6 | 31.6 | Partial overlap |
| 1D equatorial | Depleted (0.67x) | 15.5 | 31.8 | Partial overlap |
| 1D equatorial | Baseline (1.0x) | 17.0 | 29.0 | **Best 1D** |
| 1D equatorial | Moderate (1.2x) | 17.2 | 27.3 | Good |
| 1D equatorial | Strong (1.5x) | 16.2 | 25.4 | Poor -- too narrow |
| 2D lat-resolved | Uniform transport | 21.3 | 31.5 | **Excellent** |
| 2D lat-resolved | Soderlund (q*=0.4) | 20.7 | 30.5 | Excellent |
| 2D lat-resolved | Lemasquerier (q*=0.455) | 22.5 | 34.4 | **Best overall** |
| 2D lat-resolved | Lemasquerier strong (q*=0.819) | 18.9 | 36.4 | Marginal -- broad |

---

## Table 5: Bayesian Juno MWR evidence comparison (1D equatorial modes)

Importance-sampling reweighting of each equatorial mode against two Juno
observation models. Bayes factors relative to the Baseline (1.0x) mode.
Model uncertainty sigma_model = 3 km.

### 5a: Marginal likelihoods and Bayes factors

| Mode | log ML (Model A) | log ML (Model B) | log BF_A | BF_A | log BF_B | BF_B | Interpretation |
|------|------------------|------------------|----------|------|----------|------|----------------|
| Depleted strong (0.55x) | -0.839 | -0.568 | -0.096 | 0.91 | -0.069 | 0.93 | Weak disfavour |
| Depleted (0.67x) | -0.827 | -0.572 | -0.084 | 0.92 | -0.073 | 0.93 | Weak disfavour |
| Baseline (1.0x) | -0.743 | -0.499 | 0.000 | 1.00 | 0.000 | 1.00 | Reference |
| Moderate (1.2x) | -0.737 | -0.468 | +0.006 | 1.01 | +0.031 | 1.03 | Negligible |
| Strong (1.5x) | -0.773 | -0.463 | -0.030 | 0.97 | +0.036 | 1.04 | Negligible |

Observation models: A = pure-water ice, D_cond = 29 +/- 10 km;
B = low-salinity ice, D_cond = 24 +/- 10 km.

### 5b: Posterior D_cond under Juno constraint

| Mode | Post. D_cond_A (km) | 1-sigma_A (km) | Post. D_cond_B (km) | 1-sigma_B (km) | ESS_A | ESS_B |
|------|---------------------|----------------|---------------------|----------------|-------|-------|
| Depleted strong (0.55x) | 21.5 | [13.1, 31.4] | 18.1 | [10.9, 27.1] | 9,930 | 11,817 |
| Depleted (0.67x) | 22.2 | [13.1, 32.1] | 18.3 | [10.8, 27.9] | 9,923 | 11,850 |
| Baseline (1.0x) | 22.4 | [14.5, 30.6] | 19.5 | [11.6, 27.2] | 10,278 | 12,053 |
| Moderate (1.2x) | 21.0 | [15.1, 29.5] | 19.0 | [12.3, 26.2] | 10,581 | 12,238 |
| Strong (1.5x) | 19.5 | [14.7, 28.3] | 17.6 | [12.9, 25.1] | 10,793 | 12,505 |

### 5c: Posterior convective fraction under Juno constraint

| Mode | Conv. frac (Model A) | Conv. frac (Model B) |
|------|---------------------|---------------------|
| Depleted strong (0.55x) | 92.5% | 96.1% |
| Depleted (0.67x) | 82.3% | 88.8% |
| Baseline (1.0x) | 55.2% | 61.6% |
| Moderate (1.2x) | 47.6% | 52.0% |
| Strong (1.5x) | 41.8% | 43.8% |

**Key findings:**

- All Bayes factors are close to unity (0.91--1.04): the Juno constraint
  does not strongly discriminate between ocean transport modes. This is
  because the 10 km observational uncertainty dominates the inter-mode
  D_cond differences (15--17 km medians).
- Under the Juno constraint, posterior D_cond medians converge to
  19--22 km across all modes, pulled toward the observation.
- Juno reweighting substantially increases the posterior convective
  fraction for depleted modes (from 89% prior to 93--96% posterior)
  because thicker shells with convective sublayers have D_cond closer
  to the Juno range.
- Model B (low-salinity, 24 km) slightly favours enhanced transport
  modes (BF up to 1.04), while Model A (pure-water, 29 km) slightly
  favours the baseline. Neither preference is statistically meaningful.

---

## Source files

| Dataset | Results file | Script |
|---------|-------------|--------|
| 1D Global Howell | `EuropaProjectDJ/results/mc_15000_howell.npz` | `run_howell_15k.py` |
| 1D Global Andrade | `EuropaProjectDJ/results/mc_15000_optionA_v2_andrade.npz` | `run_andrade_15k.py` |
| 1D Equatorial suite | `EuropaProjectDJ/results/eq_*_andrade.npz` | `run_equatorial_suite.py` |
| 2D Latitude-resolved | `Europa2D/results/mc_2d_*_500.npz` | `run_2d_mc.py` |
| Bayesian Juno comparison | `EuropaProjectDJ/results/eq_*_andrade.npz` | `bayesian_equatorial_juno.py` |
