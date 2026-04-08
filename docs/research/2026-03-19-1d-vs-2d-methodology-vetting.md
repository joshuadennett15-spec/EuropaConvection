# Vetting 1D vs 2D Model Methodology: Why the Results Diverge

Based on the detailed documentation and design notes in the repository (specifically
`EuropaProjectDJ/docs/poster/POSTER_RESULTS_DISCUSSION_CONCLUSION_DRAFT.md`,
`EuropaProjectDJ/docs/research/PARAMETER_PRIOR_AUDIT_2026.md`, the 2D modeling
prospectus, and the 2026-03-22 temperature-field alignment guide),
here is a precise breakdown of what is happening between the 1D and 2D models.

The short answer is: **The core mathematical approach for 2D is completely rigorous,
but the current implementation introduces several 2D-only assumptions that change
the forcing and closure relative to 1D, creating real (not spurious) but
potentially misleading discrepancies.**

Here is a detailed vetting of the methodology, why the results diverge, and what
has been done (and what remains) to make the work defensively rigorous.

---

## 1. Is the 2D column-array methodology scientifically rigorous?

**Yes, absolutely.**

Building a 2D spatial model by stitching together an array of independent 1D
column solvers is exactly the right approach for Europa. Because the ice shell
is extremely thin relative to the radius of the moon
($\Delta z \ll R_{\text{Europa}}$), horizontal heat conduction is mathematically
negligible compared to vertical conduction. Treating adjacent columns as thermally
independent (while varying their top and bottom boundary conditions) is a standard,
highly defensible practice in planetary geophysics (used by Ojakangas & Stevenson
1989, Ashkenazy et al. 2018, and others).

If you are seeing different results between a global 1D model and your 2D model,
**it is not an error in the PDE solver.** It is a consequence of how the inputs,
closures, and filters are handed to the different solvers.

---

## 2. Why are the 1D and 2D results different?

The 1D and 2D branches share the same decomposition logic (`Convection.py`,
`Solver.py`) but differ in forcing and closure. The differences fall into two
categories: **previously identified issues that have been fixed**, and
**intentional 2D assumptions that remain and must be defended or removed**.

### Previously identified issues (now fixed in the current codebase)

#### A. Asymmetric grain size overrides -- FIXED

The old `regional_samplers.py` overrode `d_grain` toward smaller values at the
equator, building the regional difference into the model before it ran.

**Current state:** The `Europa2D` branch uses `LatitudeParameterSampler`, which
draws `d_grain` once per MC realization and applies it identically to all latitude
columns via `SHARED_AUDITED_KEYS`. This is correct. Grain size, viscosity
prefactors, activation energies, and all other rheological parameters are held
constant across latitude within each realization.

**Verified in:** `Europa2D/src/latitude_sampler.py`, lines 35-39.

#### B. Asymmetric subcritical rejection -- FIXED

The old drivers used `reject_subcritical=True` for equatorial runs but `False`
for polar runs, artificially truncating the equatorial thickness distribution.

**Current state:** The `Europa2D` branch has no `reject_subcritical` flag anywhere
in its codebase. All valid samples are kept regardless of convective state.

**Verified by:** `grep -r "reject_subcritical" Europa2D/` returns no matches.

#### D. Global energy conservation -- FIXED

The old concern was that the 2D $q_{\text{ocean}}(\phi)$ profile might integrate
to a different total power than the 1D global mean.

**Current state:** The `ocean_heat_flux()` method in `latitude_profile.py`
explicitly normalizes all patterns (uniform, polar_enhanced, equator_enhanced) to
preserve the area-weighted global mean:

$$\frac{\int_0^{\pi/2} q(\phi) \cos(\phi)\, d\phi}{\int_0^{\pi/2} \cos(\phi)\, d\phi} = q_{\text{mean}}$$

This is tested in `test_latitude_profile.py::test_normalization_preserves_global_mean`.

### Remaining 2D-specific assumptions (intentional but load-bearing)

These are the differences that currently make 1D and 2D results non-comparable.
Each one is a deliberate modeling choice, not a bug, but they change the answer
and must be either defended in the thesis or removed for parity testing.

#### C. Surface temperature and $k(T)$ feedback -- REAL PHYSICS, CALIBRATED

A global 1D model uses a single surface temperature per run. The 2D model applies
a latitude-dependent surface boundary:

$$T_s(\phi) = \left((T_{\text{eq}}^4 - T_{\text{floor}}^4) \cdot \cos^p(\phi) + T_{\text{floor}}^4\right)^{1/4}$$

with $p = 1.25$ calibrated to Ashkenazy (2019) Figure 2d at $Q = 0.05$ W/m$^2$
(RMS error 0.67 K vs 4.27 K for the classic $p = 1$).

**This is real physics, not an artifact.** The cold polar surface drives thicker
conductive lids via two reinforcing mechanisms: higher $k(T) = 567/T$ conductivity
and larger $\Delta T$ across the lid. This is the dominant control on
shell-thickness latitude structure (see `2026-03-20-surface-temperature-dominance.md`).

**Status:** Defensible. The $p = 1.25$ calibration should be cited in the thesis.

#### E. The 20% tidal flux uplift -- AD HOC, NEEDS DEFENSE

`LatitudeParameterSampler` applies `q_tidal_scale = 1.20`, multiplying the
inherited tidal power by 1.2 before computing the ocean heat flux. This has no
1D counterpart.

**Why it exists:** Without it, the entire MC population sits at Ra $\approx$ 20-30
with Nu = 1.00 everywhere -- a trivially conductive regime where the model has no
convective sensitivity.

**Current effect:** Pushes ~40% of MC cells into genuine convection (Nu > 2),
placing the ensemble near the conductive-convective transition.

**Risk:** 1.20 is a round number with no calibration anchor. It changes the global
thermal state before any latitude effects are applied.

**Status:** Useful but ad hoc. The thesis should:
- frame it as a calibration choice, not a physics result,
- show results at both 1.0x and 1.2x for at least one scenario,
- or motivate it as "the factor needed to place the median Ra near critical."

#### F. The cold-column convection ramp -- MOST SUSPICIOUS

`AxialSolver2D._convection_ramp_factor()` smoothly kills convection for columns
with $T_{\text{surf}} \leq 60$ K and blends between 60-80 K. This has **no
counterpart in the 1D solver**.

**What it does:** Every column poleward of ~60 deg latitude has its convection
artificially damped or killed. The ramp multiplies the effective Nusselt
enhancement, making cold columns more lid-dominated even when
Ra > Ra$_{\text{crit}}$.

**Why it exists:** Numerical convenience -- very cold columns can produce
oscillatory or unphysical convection states in the 1D solver when the surface
temperature is far below the rheological transition.

**Why it is suspicious:** The 1D solver has no such ramp. If you feed the same
parameters to the 1D solver with $T_{\text{surf}} = 55$ K, it will attempt full
convection if Ra > Ra$_{\text{crit}}$. The 2D solver will suppress it. This means
the 2D polar shell is artificially more conductive than the shared physics
predicts.

**Status:** Needs validation. Run one 20-iteration MC with
`convection_ramp = 1.0` everywhere and compare:
- If the polar shell barely changes, the ramp is harmless and can stay.
- If it changes significantly, the ramp is load-bearing and needs a literature
  defense or removal.

#### G. Lateral diffusion coupling -- NEGLIGIBLE

The 2D solver applies explicit lateral heat diffusion between columns. The
characteristic timescale is ~200 Gyr, far longer than the convergence time. This
is included for completeness but has negligible effect.

**Status:** Not a concern.

#### H. Ensemble summary definition mismatch -- CONFUSION SOURCE

The 1D and 2D branches use different definitions for "convective fraction":

| Branch | Metric | Definition |
|--------|--------|------------|
| 1D | Conv. frac | Fraction of realizations with Ra $\geq$ Ra$_{\text{crit}}$ |
| 2D Table 3a | Conv. frac | Fraction of realizations where $\geq 50\%$ of columns have Ra $\geq$ Ra$_{\text{crit}}$ |
| 2D diagnostic | Nu > 1.01 | Fraction of (sample $\times$ latitude) cells with Nu > 1.01 |
| 2D diagnostic | D_conv/H | Shell-thickness convective fraction |

These are four different quantities with similar names. The 2D "majority of
columns" metric is much stricter than the 1D per-realization metric.

**Status:** Must be reconciled. The thesis should either:
- compare the same metric (equatorial column Ra $\geq$ Ra$_{\text{crit}}$) in both,
- or explicitly define each metric where it appears.

---

## 3. Current codebase status vs. original concerns

| Original concern | Status | Evidence |
|-----------------|--------|----------|
| A. Asymmetric grain size | **Fixed** | `SHARED_AUDITED_KEYS` includes `d_grain` |
| B. Asymmetric subcritical rejection | **Fixed** | No `reject_subcritical` in Europa2D |
| C. Surface temperature $k(T)$ | **Real physics, calibrated** | $p = 1.25$, RMS = 0.67 K |
| D. Energy conservation | **Fixed** | Area-weighted normalization, tested |
| E. 20% tidal uplift | **Ad hoc, needs defense** | `q_tidal_scale = 1.20` |
| F. Cold-column convection ramp | **Suspicious, needs validation** | No 1D counterpart |
| G. Lateral diffusion | **Negligible** | $\tau \sim 200$ Gyr |
| H. Metric definition mismatch | **Confusion source, needs reconciliation** | Four different "conv frac" |

---

## 4. Diagnostic evidence from the 20-iteration verification run

The following results are from 20 MC iterations with the current codebase
(calibrated $\cos^{1.25}(\phi)$ surface temperature, $q_{\text{tidal\_scale}} = 1.20$,
all other defaults).

### Shell structure by scenario (band means, area-weighted)

| Diagnostic | uniform | soderlund_eq | lemasquerier_polar |
|---|---|---|---|
| Median H_low (0-10 deg) | 24.33 km | 22.11 km | 27.56 km |
| Median H_high (80-90 deg) | 39.05 km | 46.60 km | 32.72 km |
| Median D_cond_low | 15.09 km | 13.45 km | 17.45 km |
| Median D_cond_high | 33.62 km | 40.31 km | 28.08 km |
| Median D_conv_low | 4.45 km | 4.03 km | 4.85 km |
| Median D_conv_high | 4.19 km | 4.99 km | 3.39 km |
| $\Delta H$ (pole $-$ eq) | +14.72 km | +24.49 km | +5.16 km |
| D_cond share of $\Delta H$ | 126% | 110% | 206% |
| Cells Nu > 2.0 | 38.7% | 39.7% | 37.1% |

### Key observations

1. **D_cond dominates the latitude structure.** The convective sublayer is nearly
   flat (4-5 km) across latitude. All of the pole-equator thickness contrast comes
   from the conductive lid, driven by the surface temperature boundary condition.

2. **~40% of cells are genuinely convecting** (Nu > 2), but the median Nu is 1.00
   at most latitudes. The MC ensemble straddles the conductive-convective boundary.

3. **The uniform scenario has $\Delta H = +14.72$ km** despite zero ocean heat
   flux contrast. This is the tidal + surface-temperature baseline. The ocean
   transport signal should be measured as
   $\Delta H_{\text{ocean}} = \Delta H_{\text{scenario}} - \Delta H_{\text{uniform}}$.

4. **The polar-enhanced scenario gives $\Delta H = +5.16$ km**, meaning the polar
   ocean heating partially cancels the cold-pole lid thickening but does not
   fully overcome it.

---

## 5. Validation sequence (from the alignment guide)

The recommended validation sequence, in order, is:

### Stage 1: Exact single-column parity (CRITICAL, NOT YET DONE)

Run `Europa2D` with:
- `n_lat = 1`, `ocean_pattern = "uniform"`, `q_tidal_scale = 1.0`
- Feed the exact same parameter draw to both 1D and 2D solvers
- Same `T_surf`, `epsilon_0`, basal flux, `nx`, `dt`, rheology

Expected: H, D_cond, D_conv, Ra, Nu should match 1D within solver tolerance.

**If this fails, there is a real implementation mismatch.**

### Stage 2: Many columns, all identical (CODE CONSISTENCY TEST)

Run with `n_lat = 37` but:
- uniform surface temperature (same T_surf everywhere)
- uniform strain (epsilon_eq = epsilon_pole)
- uniform ocean heat flux
- convection ramp disabled

Expected: every column produces the same answer.

### Stage 3: Turn on one effect at a time (ATTRIBUTION)

Order:
1. Surface-temperature gradient only
2. Ocean-flux redistribution only
3. Strain gradient only
4. Convection ramp only

Expected: you can attribute exactly which assumption moves D_cond and D_conv.

### Stage 4: Convection ramp sensitivity (RAMP VALIDATION)

Run one 20-iteration MC with `convection_ramp = 1.0` everywhere. Compare against
the production run. If the polar shell changes by more than ~2 km, the ramp is
load-bearing and needs defense.

---

## 6. Recommended interpretation of the current results

The safest interpretation of the current repo state is:

- The 1D and 2D branches are **identical in decomposition method** (same
  `Convection.py`, same `Solver.py`).
- They are **different in forcing and closure** (surface temperature, tidal
  pattern, ocean redistribution, convection ramp, tidal uplift).
- The current 2D production results are therefore **not directly comparable to
  1D** without the validation sequence above.

The current mismatch does **not** mean:
- the decomposition is broken,
- or that you are missing something simple.

It means the 2D model is answering a different question:

> "What happens when the shell sees latitude-dependent forcing, a calibrated
> surface boundary, and cold-column convection damping?"

rather than:

> "What happens if I solve the same 1D thermal problem with a 2D wrapper?"

---

## 7. Summary: what to do before the thesis

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| **1** | Run Stage 1 parity test (n_lat=1, matched params) | 2-3 hours | Confirms or kills decomposition concern |
| **2** | Run Stage 4 ramp sensitivity test | 1 hour | Determines if convection ramp is load-bearing |
| **3** | Frame 2D results honestly in thesis | Writing only | "Same decomposition, different forcing/closure" |
| **4** | Show 1.0x vs 1.2x tidal results | 1 hour | Defends or removes the ad hoc uplift |
| **5** | Reconcile "convective fraction" definitions | Writing only | Prevents reviewer confusion |
| **6** | Run Stage 3 attribution sequence | 4-6 hours | Full scientific attribution (ideal but optional) |

The PDE solver is right. The 2D array architecture is right. The rheology is
shared correctly. The energy conservation is verified. The remaining work is
validating the 2D-specific assumptions (surface boundary, convection ramp, tidal
uplift) and framing the results so a thesis reviewer can follow exactly which
differences are real physics and which are modeling choices.
