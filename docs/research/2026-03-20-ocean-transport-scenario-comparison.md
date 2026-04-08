# Ocean heat transport scenario comparison

Cross-model analysis of which ocean circulation regime best matches
observational constraints, using all Monte Carlo results in the repo.

## Observational target

Juno MWR (Levin et al. 2026): equatorial conductive lid D_cond = 29 +/- 10 km.
Acceptable window: **19--39 km**.

## 1D equatorial suite (15,000 samples each, Andrade rheology)

| Mode | Factor | Median D_cond | 1-sigma range | Juno overlap |
|------|--------|--------------|---------------|-------------|
| Depleted strong | 0.55x | 15.6 km | [8.7, 31.8] km | Partial -- tail reaches in |
| Depleted | 0.67x | 15.5 km | [8.5, 31.9] km | Partial -- tail reaches in |
| **Baseline** | **1.0x** | **17.0 km** | **[8.2, 29.1] km** | **Best 1D -- upper 1-sigma hits 29 km** |
| Moderate | 1.2x | 17.2 km | [8.1, 27.4] km | Good but falls short |
| Strong | 1.5x | 16.2 km | [8.3, 25.5] km | Poor -- too thin |

Enhancement (1.2x, 1.5x) thins the equatorial shell into the conductive regime
(48% conductive at 1.5x), which lowers D_cond because those thin shells are
entirely conductive lid but very thin. Depletion (0.67x, 0.55x) thickens
everything but pushes D_cond past the Juno window.

Source: `EuropaProjectDJ/scripts/run_equatorial_suite.py`,
results in `EuropaProjectDJ/results/eq_*_andrade.npz`.
Figures: `EuropaProjectDJ/figures/pub/fig_eq_*.png`.

## 2D latitude-resolved model (500 samples each)

| Scenario | q* | Equator D_cond | 1-sigma range | Juno overlap |
|----------|-----|---------------|---------------|-------------|
| **Lemasquerier polar** | **0.455** | **22.5 km** | **[8.7, 34.4] km** | **Best overall -- closest to 29 km** |
| Uniform transport | 0 | 21.3 km | [9.0, 31.6] km | Excellent |
| Soderlund equatorial | 0.4 | 20.7 km | [9.0, 30.6] km | Excellent |
| Lemasquerier polar strong | 0.819 | 18.9 km | [10.1, 36.4] km | Marginal -- too broad |

The 2D model consistently predicts thicker equatorial conductive lids than the
1D proxy because the latitude solver resolves the meridional structure rather
than approximating it with a scalar multiplier.

Source: `Europa2D/scripts/run_2d_mc.py`,
results in `Europa2D/results/mc_2d_*.npz`.
Scenarios defined in `Europa2D/src/literature_scenarios.py`.

## 1D vs 2D model agreement

Uniform transport is the only scenario testable in both model classes.
The 1D equatorial proxy predicts a median D_cond of 17.0 km; the 2D
latitude-resolved model predicts 21.3 km at the equator. The ~4 km offset
reflects the 2D model's ability to resolve meridional heat redistribution
that the 1D proxy collapses into a single scalar.

The Lemasquerier scenarios illustrate why the 2D model matters: the 1D proxy
can only represent polar-enhanced transport as equatorial depletion (0.67x,
0.55x), which overshoots the Juno window. The 2D model correctly captures the
global energy balance -- less heat at the equator is compensated by more at
the poles -- and produces the best single D_cond prediction (22.5 km).

## Synthesis

**Uniform transport (Ashkenazy & Tziperman 2021) is the most consistently
supported scenario across both model classes.** It is the best 1D performer,
second-best in 2D, and does not predict anything pathological at the poles.

**Conservative Lemasquerier (q*=0.455) produces the single best equatorial
D_cond prediction** -- 22.5 km, only 6.5 km off Juno's central estimate --
but only in the 2D model where the latitude redistribution is properly
resolved. In the 1D equatorial proxy, that same physics shows up as depletion
(0.67x) which undershoots Juno. The 1D proxy captures only the local
equatorial effect (less heat -> too thick), while the 2D model captures the
global energy balance correctly.

### Not supported

- **Strong Soderlund (1.5x):** thins the equatorial shell too aggressively
  and produces unrealistic polar thickening (>70 km) in 2D.
- **Strong Lemasquerier (q*=0.819):** too extreme, broadens the distribution
  without improving the central estimate.
- **Howell/Maxwell rheology:** conductive lids too thin (<10 km),
  incompatible with Juno.

### Thesis recommendation

Uniform transport is the safe default. Conservative Lemasquerier is the more
physically interesting result -- it is the only scenario where a specific
ocean circulation mechanism improves the Juno match beyond what uniform
transport gives. That claim requires the 2D model to be credible, since the
1D proxy cannot represent it faithfully.
