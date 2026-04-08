# Europa Ocean Heat Transport Benchmarks for Europa2D

## Purpose

This note translates the Europa ocean heat transport literature into a set of benchmark forcing families for `Europa2D`.

The goal is not to claim that the current latitude-only shell model reproduces any full 3D ocean simulation. The goal is to define a defensible set of test cases:

- which published ocean regimes are worth comparing against,
- which existing model variables are physically justified to vary,
- which current variables should not be used as latitude-by-latitude tuning knobs,
- and which new variables would make the benchmark suite much cleaner.

## Scope and limitations

- `Europa2D` is axisymmetric in latitude. It cannot represent longitude-dependent heat flux, transient eddies, or explicit ocean circulation.
- Published ocean studies are 2D or 3D fluid-dynamical models. In `Europa2D`, they can only be represented as proxy basal heat-flux patterns plus a small number of matched shell-heating assumptions.
- Therefore, the correct use of the literature is "test against benchmark forcing families", not "claim one-to-one reproduction of the published ocean flow."
- The exact `90 deg` node should not be over-interpreted. It is a boundary node, not an interior sample, and it is the point where symmetry, geometry, and the surface-temperature floor meet.

## Bottom-line literature synthesis

There is no single literature consensus that Europa's ocean must deliver more heat to the poles or more heat to the equator.

The literature supports at least four distinct benchmark families:

1. Equator-enhanced ocean heat delivery under homogeneous bottom heating.
2. Polar-enhanced ocean heat delivery under other rotating-convection regimes.
3. Polar-enhanced heat delivery caused by heterogeneous mantle tidal heating.
4. Very efficient meridional redistribution that makes the ice shell nearly uniform.

That means the correct scientific question for `Europa2D` is not "which single pattern is right?" but "which shell-thickness responses are robust across the physically plausible forcing families?"

## Primary literature anchors

### 1. Soderlund et al. 2014: low-latitude heating

Source:
- Soderlund, K. M., Schmidt, B. E., Wicht, J., and Blankenship, D. D. (2014), *Ocean-driven heating of Europa's icy shell at low latitudes*, Nature Geoscience, DOI: https://doi.org/10.1038/ngeo2021
- Open PDF used here: https://website.whoi.edu/gfd/wp-content/uploads/sites/14/2018/10/Soderlund_Ocean_ngeo_2014-1_233284.pdf

What it supports:
- Europa-like quasi-3D rotating convection with homogeneous bottom heating.
- Stronger outward motions and stronger upward heat delivery at low latitudes.
- Zonal- and time-averaged heat flux that varies by about 40 percent, with a clear equatorial maximum and nearly uniform flux elsewhere.
- A shell tendency toward low-latitude thinning and enhanced low-latitude geologic activity.

What it does not support:
- A polar-enhanced basal flux prescription.
- Using Soderlund as the citation for a `polar_enhanced` pattern.

Implication for `Europa2D`:
- The correct Soderlund-like proxy is `equator_enhanced`, not `polar_enhanced`.

### 2. Amit et al. 2020: two cooling regimes in rotating thin shells

Source:
- Amit, H., et al. (2020), *Cooling patterns in rotating thin spherical shells - Application to Titan's subsurface ocean*, Icarus, DOI: https://doi.org/10.1016/j.icarus.2019.113509

What it supports:
- Two persistent outer-boundary cooling patterns exist in rotating subsurface oceans.
- Larger Rossby number gives polar cooling with more moderate lateral heterogeneity.
- Lower Rossby number gives equatorial cooling with stronger lateral heterogeneity.

Implication for `Europa2D`:
- The sign of the latitude contrast should not be hard-coded as universal.
- A good benchmark suite must include both equator-enhanced and polar-enhanced ocean forcing.

### 3. Bire et al. 2022: circulation regime depends on rotation and geometry

Source:
- Bire, S., Kang, W., Ramadhan, A., Campin, J.-M., and Marshall, J. (2022), *Exploring ocean circulation on icy moons heated from below*, JGR Planets, DOI: https://doi.org/10.1029/2021JE007025
- Abstract/PDF entry used here: https://oceans.mit.edu/JohnMarshall/eapsdb/s-bire20224ac8ed98/

What it supports:
- With homogeneous heating from below, two distinct circulation regimes appear.
- If plumes inside the tangent cylinder dominate, heat is transferred efficiently to the polar ice.
- If those plumes are suppressed, lower-latitude rolls dominate and equatorial cooling appears.
- The two leading control parameters are the natural Rossby number and the ratio of inner to outer radius.

Implication for `Europa2D`:
- Ocean geometry matters.
- `D_H2O` is a physically meaningful global variable to vary, because it changes ocean-shell geometry and the tangent-cylinder geometry proxy.
- Again, both polar and equatorial cooling regimes are defensible benchmark targets.

### 4. Lemasquerier et al. 2023: heterogeneous mantle tidal heating can drive polar flux

Source:
- Lemasquerier, D. G., Bierson, C. J., and Soderlund, K. M. (2023), *Europa's ocean translates interior tidal heating patterns to the ice-ocean boundary*, AGU Advances, DOI: https://doi.org/10.1029/2023AV000994
- Open PDF used here: https://research-repository.st-andrews.ac.uk/bitstream/10023/28900/1/Lemasquerier_2023_AGUA_EuropaOcean_CCBY.pdf

What it supports:
- If mantle heating is spatially heterogeneous, the ocean can preserve and transmit the latitudinal anomaly up to the ice-ocean boundary.
- For Europa's mantle, pure tidal heating is predicted to be stronger at the poles than at the equator.
- The paper defines a useful contrast parameter `q* = Delta q / q0`, where `q0` is the lateral mean and `Delta q` is the peak difference.
- For pure tidal mantle heating, `q*` is of order `0.91`.
- If radiogenic heating dominates, the ocean contribution to shell-thickness variations is small.
- If mantle tidal heating dominates, the ocean can control the pole-to-equator shell-thickness contrast.
- In their tidal-dominant case, the resulting equilibrium can include thinner high latitudes and higher heat flux into the polar ice.

What it does not support:
- Treating every polar-enhanced profile as a Soderlund-like case.
- Ignoring the distinction between radiogenic-dominant and tidal-dominant mantle heating.

Implication for `Europa2D`:
- A Lemasquerier-like benchmark needs a polar-enhanced pattern and a variable contrast amplitude.
- The cleanest new variable is `q_star`, not just a pattern name.

### 5. Ashkenazy and Tziperman 2021: efficient meridional transport can flatten the shell

Source:
- Ashkenazy, Y., and Tziperman, E. (2021), *Dynamic Europa ocean shows transient Taylor columns and convection driven by ice melting and salinity*, Nature Communications, https://www.nature.com/articles/s41467-021-26710-0

What it supports:
- Poleward meridional heat transport can be strong enough to maintain a nearly uniform ice thickness.
- Latent heat and salinity materially change the transport picture.
- Their sensitivity tests vary mean salinity, bottom heating, and ice thickness.
- They report sensitivity experiments using salinity values from effectively fresh to tens of ppt, and bottom heating in the rough range of a few tens to 100 mW/m^2.

Implication for `Europa2D`:
- A near-uniform shell is also a literature-backed benchmark target.
- If a future version of the model adds salinity or phase-change coupling, those become first-order scientific variables.
- In the current code, the closest proxy is `uniform` or only very weak latitude contrast in `q_ocean(phi)`.

### 6. Soderlund et al. 2024 review: multiple drivers exist

Source:
- Soderlund, K. M., et al. (2024), *The Physical Oceanography of Ice-Covered Moons*, Annual Review of Marine Science, DOI: https://doi.org/10.1146/annurev-marine-040323-101355

Why this matters:
- The review makes clear that buoyancy-driven circulation is not the only possible driver.
- Tides, libration, precession, and electromagnetic pumping may also matter for ocean-ice interactions.

Implication for `Europa2D`:
- The present benchmark set is only for thermally driven ocean heat transport.
- Any comparison to the full Europa literature should explicitly say that mechanically driven and electromagnetically driven transport are not represented.

## Model families worth testing in Europa2D

The current code can support four benchmark families without pretending to solve the ocean itself.

| Benchmark family | Literature anchor | Proxy sign of `q_ocean(phi)` | Expected shell response |
| --- | --- | --- | --- |
| `uniform_transport` | Ashkenazy and Tziperman 2021, non-rotating/efficient limits | Nearly uniform | Small pole-to-equator thickness contrast |
| `equatorial_cooling` | Soderlund 2014, low-Rossby Amit-like cases | Higher flux at equator | Thinner equator and low latitudes |
| `polar_cooling` | Bire 2022 polar regime, high-Rossby Amit-like cases | Higher flux at poles | Thinner high latitudes if ocean forcing overcomes cold polar surface |
| `mantle_tidal_polar` | Lemasquerier 2023 | Higher flux at poles with variable `q_star` | Poleward thinning when mantle tidal forcing is strong enough |

## The variables that are scientifically justified to vary

These are the variables that should define the benchmark suite.

### A. Variables to vary directly in ocean-transport tests

#### 1. `ocean_pattern`

Current code already has:
- `uniform`
- `equator_enhanced`
- `polar_enhanced`

This is the correct first-order switch because the literature supports multiple signs of the latitudinal flux contrast.

Recommendation:
- Keep this as the top-level benchmark family selector.
- Fix the paper mapping:
  - `equator_enhanced` -> Soderlund-like
  - `polar_enhanced` -> Lemasquerier-like or Bire polar-cooling regime

#### 2. `q_ocean_mean`

This is physically meaningful and should be varied.

Why:
- It sets the globally integrated ocean-to-ice heat supply.
- Both Ashkenazy-like and Lemasquerier-like interpretations depend on the mean basal heat flux scale.
- In the current sampler, `q_ocean_mean` is already derived from `H_rad` plus `P_tidal`.

Recommendation:
- Keep `q_ocean_mean` as a core benchmark variable.
- Compare patterns at fixed mean heat flux before changing shell rheology.

#### 3. `q_star` or `ocean_contrast` (new variable)

This is the most important missing variable in the current 2D implementation.

Why:
- The literature does not only predict different signs. It also predicts different amplitudes.
- Soderlund 2014 reports about 40 percent zonal-mean top-flux variation.
- Lemasquerier 2023 introduces `q*` explicitly and shows that its physically plausible value spans from `0` to about `0.91` depending on the radiogenic versus tidal partition.

Recommendation:
- Add one scalar contrast-amplitude variable.
- Prefer the name `q_star` because it maps directly to Lemasquerier 2023.

Suggested interpretation:
- `q_star = 0.0`: no lateral contrast
- `q_star ~ 0.1 to 0.4`: weak to moderate contrast
- `q_star ~ 0.4`: Soderlund-like equatorial benchmark amplitude
- `q_star ~ 0.5 to 0.9`: strong Lemasquerier-like tidal mantle benchmark

#### 4. `mantle_tidal_fraction` (new variable)

This is the cleanest way to represent the radiogenic versus tidal mantle-heating partition.

Why:
- Lemasquerier 2023 shows that the effect of ocean heat transport depends strongly on whether mantle heating is radiogenic-dominant or tidal-dominant.

Recommendation:
- Add a variable such as `mantle_tidal_fraction = q_tidal_mean / (q_tidal_mean + q_radiogenic)`.
- Use it to determine `q_star` in Lemasquerier-style cases.

Practical approximation:
- `q_star ~= 0.91 * mantle_tidal_fraction`

That approximation follows the Lemasquerier result that pure tidal mantle heating gives `q*` of about `0.91`, while pure radiogenic heating gives `q* = 0`.

#### 5. `D_H2O`

This is already present in the sampler and is scientifically meaningful.

Why:
- It changes the geometry of the ocean and the shell system.
- Bire 2022 identifies geometry, especially the ratio of inner to outer radius, as one of the leading controls on the cooling regime.

Recommendation:
- Keep varying `D_H2O`, but interpret it as an ocean-regime parameter, not only as a structural parameter.

#### 6. `T_eq`

This should remain a benchmark variable.

Why:
- Surface temperature strongly controls conductive shell thickness.
- It is the background state against which ocean heat-flux variations act.

Recommendation:
- Vary `T_eq` globally.
- Do not replace the literature-driven ocean pattern by ad hoc changes to the latitude dependence of `T_surf`.

### B. Variables to vary only conditionally

#### 7. `epsilon_eq` and `epsilon_pole`

These are conditionally defensible, but not as unrestricted benchmark knobs.

Why:
- Shell tidal heating can be latitude-dependent.
- But if the purpose of a benchmark is to test ocean heat transport, changing the shell-heating pattern at the same time can hide what the ocean forcing is doing.

Recommendation:
- For pure ocean-transport benchmarks, hold the shell tidal-heating pattern fixed across cases.
- If you want coupled ocean-plus-shell tests, use one explicit `epsilon_pattern_mode` linked to a published tidal model, rather than independent equator and pole tuning.

#### 8. `H_rad` and `P_tidal`

These are scientifically meaningful, but they should usually be varied through derived quantities.

Why:
- In the current code they already combine into `q_ocean_mean`.
- Changing them independently is useful only if the benchmark is explicitly about heating partition, not simply shell response.

Recommendation:
- Use them to derive `q_ocean_mean` and `mantle_tidal_fraction`.
- Do not treat `P_tidal` as an extra free knob once `q_ocean_mean` and `mantle_tidal_fraction` are already prescribed.

### C. Variables that should not be latitude-specific ocean benchmark knobs

#### 9. `d_grain`

Do not vary this by latitude in ocean-transport benchmarks.

Why:
- Grain size affects shell rheology, not ocean transport directly.
- Latitude-specific grain-size tuning is currently a calibration move, not a literature-backed ocean forcing.

Recommendation:
- Sample `d_grain` globally if desired.
- Keep it shared across latitude when the experiment is about ocean transport.

#### 10. Convection ramp, pole floor, grid resolution, boundary treatment

These are not science-calibration variables.

Examples:
- `n_lat`
- `nx`
- pole boundary condition
- convection ramp thresholds
- surface-temperature floor near `90 deg`

Recommendation:
- Treat these as numerical controls.
- Converge them, then hold them fixed while testing literature benchmarks.

## Variables that should be added to the current code

The current `LatitudeProfile` interface is close, but not yet ideal for literature tests.

### Recommended additions

#### 1. `q_star`

Add a scalar contrast amplitude to `LatitudeProfile`.

Reason:
- A pattern name alone cannot represent the spread between nearly uniform, Soderlund-like, and Lemasquerier-like cases.

#### 2. `ocean_model_family`

This can simply be a renamed or expanded version of `ocean_pattern`.

Suggested values:
- `uniform_transport`
- `soderlund2014_low_lat`
- `bire2022_polar`
- `lemasquerier2023_mantle_tidal`

Reason:
- A named model family is clearer than a purely geometric shape label.

#### 3. `mantle_tidal_fraction`

Reason:
- This is the simplest way to map the Lemasquerier physics into a scalar benchmark axis.

#### 4. `epsilon_pattern_mode`

Suggested values:
- `fixed_shell_tides`
- `uniform_shell_tides`
- `literature_shell_tides`

Reason:
- It prevents ocean-benchmark experiments from quietly changing shell heating at the same time.

#### 5. `ocean_salinity_class` or `salinity_ppt`

This is a future extension, not a required immediate change.

Reason:
- Ashkenazy and Tziperman 2021 shows salinity and phase change can be first-order.
- The present model does not include them, so mark them explicitly as unsupported rather than smuggling them into another parameter.

## Recommended benchmark matrix

This is the benchmark set most worth running.

### Benchmark 0: numerical control

Purpose:
- Confirm that the shell response is not dominated by the exact pole node or by resolution.

Settings:
- Run `uniform` ocean forcing.
- Compare `n_lat = 37` and a finer run if practical.
- Report high-latitude mean thickness over `80 to 90 deg`, not only the single `90 deg` node.

Expected outcome:
- A smooth polar tail with no boundary artifact.

### Benchmark 1: Soderlund-like equatorial cooling

Anchor:
- Soderlund et al. 2014

Settings:
- `ocean_pattern = equator_enhanced`
- `q_star ~ 0.4`
- fixed shell tidal-heating pattern
- fixed rheology across latitude

Expected outcome:
- Minimum thickness at low latitudes or near the equator.
- No literature reason for a polar-enhanced basal flux in this case.

### Benchmark 2: weak-contrast efficient transport

Anchor:
- Ashkenazy and Tziperman 2021

Settings:
- `ocean_pattern = uniform` or a very small `q_star`
- same `q_ocean_mean` as Benchmark 1
- fixed shell tidal-heating pattern

Expected outcome:
- Nearly uniform shell, or at least much weaker pole-to-equator contrast than in the Soderlund-like and Lemasquerier-like cases.

### Benchmark 3: Bire/Amit regime sweep

Anchor:
- Amit et al. 2020
- Bire et al. 2022

Settings:
- run both `equator_enhanced` and `polar_enhanced`
- vary `q_star`
- vary `D_H2O`

Expected outcome:
- A regime map rather than one preferred solution.
- The sign of the thickness contrast should be allowed to flip across the sweep.

### Benchmark 4: Lemasquerier radiogenic-dominant mantle

Anchor:
- Lemasquerier et al. 2023

Settings:
- `ocean_pattern = polar_enhanced`
- small `mantle_tidal_fraction`
- therefore small `q_star`
- fixed shell tidal-heating pattern

Expected outcome:
- Oceanic contribution to shell-thickness variations should be weak.
- Surface temperature and shell tidal heating may dominate the shell pattern.

### Benchmark 5: Lemasquerier tidal-dominant mantle

Anchor:
- Lemasquerier et al. 2023

Settings:
- `ocean_pattern = polar_enhanced`
- `mantle_tidal_fraction` large
- `q_star` in the strong-contrast range
- fixed shell tidal-heating pattern unless the experiment explicitly studies the competition between oceanic and shell tidal forcing

Expected outcome:
- High-latitude thinning becomes possible.
- This is the correct literature family for a polar-enhanced ocean benchmark.

## Diagnostics that should be reported

For literature comparisons, do not rely only on the single equator value and the single pole value.

Report these diagnostics for every benchmark:

- `H_low`: area-weighted mean thickness over low latitudes, for example `0 to 10 deg`
- `H_high`: area-weighted mean thickness over high latitudes, for example `80 to 90 deg`
- `Delta_H = H_high - H_low`
- latitude of minimum thickness
- normalized shell-thickness contrast `(H_max - H_min) / H_mean`
- same diagnostics for `q_ocean(phi)` if using proxy forcing families

This matches the way several ocean papers compare "high latitude" and "low latitude" bands rather than over-interpreting a single boundary node.

## Recommended interpretation rules

Use these rules when reading the `Europa2D` results.

1. If only the exact `90 deg` node turns upward, but `80 to 87.5 deg` continues the opposite trend, treat it as a boundary-sensitive endpoint until proven otherwise.
2. If the model shell is conductive everywhere, do not attribute the thickness pattern to stagnant-lid convection transitions.
3. If `q_ocean(phi)` and `epsilon(phi)` are both varied, state explicitly whether the test is an ocean benchmark or a coupled ocean-plus-shell benchmark.
4. Do not claim "consistent with Soderlund" for any polar-enhanced ocean forcing.
5. Do not use latitude-specific `d_grain` changes to rescue a target thickness pattern.

## Immediate code implications

These are the logic changes most strongly justified by the literature review.

1. Reassign the citation for `equator_enhanced` to Soderlund 2014.
2. Reassign the citation for `polar_enhanced` to Lemasquerier 2023 and, more loosely, to the polar-cooling family discussed by Amit 2020 and Bire 2022.
3. Add `q_star` so that contrast amplitude becomes a real science variable instead of being buried in one fixed shape.
4. Keep `d_grain` global, not latitude-specific, for ocean benchmark suites.
5. Keep shell tidal-heating pattern fixed across ocean benchmark families unless the experiment explicitly studies coupling between shell tides and ocean transport.

## Suggested source list

- Soderlund, K. M., Schmidt, B. E., Wicht, J., and Blankenship, D. D. (2014), *Ocean-driven heating of Europa's icy shell at low latitudes*, Nature Geoscience. https://doi.org/10.1038/ngeo2021
- Amit, H., et al. (2020), *Cooling patterns in rotating thin spherical shells - Application to Titan's subsurface ocean*, Icarus. https://doi.org/10.1016/j.icarus.2019.113509
- Bire, S., Kang, W., Ramadhan, A., Campin, J.-M., and Marshall, J. (2022), *Exploring ocean circulation on icy moons heated from below*, JGR Planets. https://doi.org/10.1029/2021JE007025
- Lemasquerier, D. G., Bierson, C. J., and Soderlund, K. M. (2023), *Europa's ocean translates interior tidal heating patterns to the ice-ocean boundary*, AGU Advances. https://doi.org/10.1029/2023AV000994
- Ashkenazy, Y., and Tziperman, E. (2021), *Dynamic Europa ocean shows transient Taylor columns and convection driven by ice melting and salinity*, Nature Communications. https://www.nature.com/articles/s41467-021-26710-0
- Soderlund, K. M., et al. (2024), *The Physical Oceanography of Ice-Covered Moons*, Annual Review of Marine Science. https://doi.org/10.1146/annurev-marine-040323-101355
