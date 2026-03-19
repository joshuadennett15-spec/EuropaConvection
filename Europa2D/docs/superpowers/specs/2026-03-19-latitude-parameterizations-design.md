# Design Spec: Improved Latitude Parameterizations for Europa2D

## Overview

Upgrade the three latitude-dependent forcing parameterizations in `LatitudeProfile` to be physically grounded in the literature, replacing ad hoc approximations with closed-form expressions that map directly to published results.

**Scope:** Changes are confined to `latitude_profile.py`, `literature_scenarios.py`, `latitude_sampler.py`, and their tests. No changes to the solver, MC framework, or EuropaProjectDJ.

**Approach:** Additive parameters on the existing frozen dataclass. Existing callers that don't pass new params get improved defaults automatically.

## Change 1: Surface Temperature — Energy Balance Floor

### Current

```python
T_s(phi) = T_eq * max(cos(phi), cos(89.5 deg))^(1/4)
```

The `cos(89.5 deg)` clamp is ad hoc. It prevents the singularity at the pole but has no physical basis for its specific value.

### Proposed

```python
T_s(phi) = (T_eq**4 * cos(phi) + T_floor**4)**(1/4)
```

**New field:** `T_floor: float = 52.0` (K)

### Physics basis

This is the exact solution to the surface energy balance:

```
(1 - A) * S * cos(phi) + q_effective = epsilon * sigma * T_s**4
```

where `q_effective` represents all non-solar heat sources at the pole: endogenic flux, obliquity-driven seasonal insolation, thermal inertia, and Jupiter longwave radiation.

Since `T_eq**4 = (1-A)*S / (epsilon*sigma)` and `T_floor**4 = q_effective / (epsilon*sigma)`, dividing through and taking the fourth root gives the proposed formula.

The default `T_floor = 52 K` comes from Ashkenazy (2019), who solved the full seasonal energy balance for Europa including:
- 3.09 deg obliquity (provides polar summer illumination)
- Surface thermal inertia (kappa_s = 7.7e-10 m^2/s)
- Jupiter longwave radiation (J_0 = 0.176 W/m^2)
- Eclipse effects

At zero internal heating, Ashkenazy found T_pole = 51-52 K. With 50 mW/m^2 endogenic flux, T_pole rises to ~63 K. The sensitivity is approximately:

```
T_floor = 52 + 240 * q_endogenic  (K, with q in W/m^2)
```

### Behavior

- At equator (phi=0): T_s = (T_eq^4 + T_floor^4)^(1/4). With defaults (110, 52 K), this is 111.3 K — a 1.2% uplift over pure T_eq due to the floor term. This is physically correct: even the equator receives endogenic heat.
- At pole (phi=pi/2): T_s = T_floor (cos term vanishes)
- Smooth, monotonically decreasing for T_floor < T_eq, C-infinity
- No clamp discontinuity
- `_PHI_FLOOR` and `_COS_FLOOR` constants become unnecessary
- **Guard:** T_floor must be less than T_eq. If T_floor >= T_eq, the profile is no longer monotonically decreasing, which is non-physical for Europa. Raise ValueError at construction.

### MC sampling

**Primary approach (recommended):** Derive T_floor from the sampled q_ocean_mean using Ashkenazy's sensitivity:

```python
T_floor = 52.0 + 240.0 * q_ocean_mean  # K, with q in W/m^2
```

This couples the polar temperature to the ocean heat budget, which is physically correct — a hotter ocean warms the polar surface. With q_ocean_mean in [10, 30] mW/m^2, T_floor falls in [54, 59] K.

**Alternative (for sensitivity studies):** Sample independently as `Normal(52, 5)` clipped to `[40, 70]` to explore the effect of decoupling T_floor from the ocean.

### References

- Ojakangas, G. W., and Stevenson, D. J. (1989), Thermal state of an ice shell on Europa, Icarus 81, 220-241.
- Ashkenazy, Y. (2019), The surface temperature of Europa, Planetary and Space Science 173, 20-30. DOI: 10.1016/j.pss.2019.06.002

## Change 2: Tidal Strain — Beuthe (2013) Square-Root Form

### Current

```python
epsilon_0(phi) = epsilon_eq + (epsilon_pole - epsilon_eq) * sin^2(phi)
```

Linear interpolation in strain amplitude. Gives correct 4:1 heating ratio at the endpoints but ~10% error at mid-latitudes.

### Proposed

```python
epsilon_0(phi) = epsilon_eq * sqrt(1 + c * sin^2(phi))
where c = (epsilon_pole / epsilon_eq)**2 - 1
```

**No new fields.** Uses existing `epsilon_eq` and `epsilon_pole`.

### Physics basis

Beuthe (2013) showed that zonally-averaged eccentricity-tide dissipation in the whole-shell regime follows:

```
q_tidal(phi) proportional to 1 + 3 * sin^2(phi)
```

This comes from the degree-2 tidal potential decomposition: the e_{2,0} radial-tide component (axisymmetric, polar-enhanced) dominates the zonal average, while the e_{2,2} librational component averages down over longitude.

Since tidal heating is proportional to epsilon_0^2, the strain amplitude must go as sqrt(1 + 3*sin^2(phi)). The coefficient `c = (epsilon_pole/epsilon_eq)^2 - 1` generalizes this so the endpoints are preserved exactly:

- At equator: epsilon_0 = epsilon_eq * sqrt(1) = epsilon_eq
- At pole: epsilon_0 = epsilon_eq * sqrt(1 + c) = epsilon_pole
- With defaults (6e-6, 1.2e-5): c = 4 - 1 = 3, recovering the exact Beuthe pattern

### Where the improvement matters

At 45 deg latitude:
- Old (linear): epsilon_0^2 / epsilon_eq^2 = 2.25
- New (sqrt): epsilon_0^2 / epsilon_eq^2 = 2.50
- Theoretical (Beuthe 2013): 2.50

The old form systematically underestimates tidal heating at mid-latitudes by ~10%.

### Backward compatibility

Same field names, same endpoint values, same defaults. Only the interpolation between endpoints changes. The `tidal_strain` method signature is unchanged.

### References

- Beuthe, M. (2013), Spatial patterns of tidal heating, Icarus 223, 308-329. DOI: 10.1016/j.icarus.2012.11.020
- Tobie, G., Choblet, G., and Sotin, C. (2003), Tidally heated convection: Constraints on Europa's ice shell thickness, JGR 108, 5124. DOI: 10.1029/2003JE002099
- Ojakangas, G. W., and Stevenson, D. J. (1989), Thermal state of an ice shell on Europa, Icarus 81, 220-241.

## Change 3: Ocean Heat Flux — q_star and mantle_tidal_fraction

### Current

`ocean_amplitude: Optional[float]` with pattern-specific defaults (0.0, 0.4, 1.0). The amplitude is an internal shape parameter with no direct literature mapping.

### Proposed

Two new science-facing fields:

```python
q_star: Optional[float] = None
mantle_tidal_fraction: float = 0.5
```

### Field definitions

**`q_star`** is the Lemasquerier (2023) contrast parameter: |Delta_q| / q_0, where q_0 is the lateral mean and Delta_q is the peak-minus-trough flux difference. Always non-negative; the sign/direction of the contrast is determined by `ocean_pattern`. Range [0, ~0.91]. Values above ~0.91 are non-physical for the Lemasquerier framework; the implementation must validate `q_star < 1.5` for equator_enhanced and `q_star < 3.0` for polar_enhanced to avoid singularities in the amplitude inversion.

**`mantle_tidal_fraction`** is q_tidal_mean / (q_tidal_mean + q_radiogenic). When `q_star` is None, it auto-derives:

```python
q_star = 0.91 * mantle_tidal_fraction
```

This follows Lemasquerier (2023): pure tidal mantle heating gives q* = 0.91; pure radiogenic gives q* = 0.

### Mapping q_star to shape amplitude a

The existing shape functions are kept:
- polar_enhanced: `1 + a * sin^2(phi)`, normalized by `1 + a/3`
- equator_enhanced: `1 + a * cos^2(phi)`, normalized by `1 + 2a/3`
- uniform: constant

For polar_enhanced, q_star relates to endpoint contrast:

```
q_pole / q_mean = (1 + a) / (1 + a/3)
q_eq / q_mean = 1 / (1 + a/3)
```

The contrast `q_star = (q_pole - q_eq) / q_mean = a / (1 + a/3)`. Inverting:

```
a = 3 * q_star / (3 - q_star)    [for polar_enhanced]
```

For equator_enhanced, analogously:

```
a = 3 * q_star / (3 - 2 * q_star)    [for equator_enhanced]
```

### Resolution order for ocean_amplitude

1. If `ocean_amplitude` is explicitly set (not None), use it directly (backward-compat override)
2. If `q_star` is explicitly set (not None), derive `a` from it using the pattern-specific formula
3. Otherwise, derive `q_star = 0.91 * mantle_tidal_fraction`, then `a` from that
4. For uniform pattern: `a = 0` regardless

### Updated literature scenarios

| Scenario | Pattern | q_star | mantle_tidal_fraction | Resulting a | Endpoint ratio |
| --- | --- | --- | --- | --- | --- |
| uniform_transport | uniform | 0.0 | any | 0.0 | 1.00 |
| soderlund2014_equator | equator_enhanced | 0.4 | N/A | 0.545 (= 3×0.4/(3−0.8)) | q_eq/q_pole = 1.55 |
| lemasquerier2023_polar | polar_enhanced | 0.455 | 0.5 | 0.536 (= 3×0.455/(3−0.455)) | q_pole/q_eq = 1.54 |
| lemasquerier2023_polar_strong | polar_enhanced | 0.819 | 0.9 | 1.127 (= 3×0.819/(3−0.819)) | q_pole/q_eq = 2.13 |

### MC sampling

Sample `mantle_tidal_fraction` from `Uniform(0.1, 0.9)` or `Beta(2, 2)` to reflect genuine uncertainty about the radiogenic/tidal partition. `q_star` and `a` follow automatically through the derivation chain.

For equator_enhanced scenarios, `q_star` is sampled directly (no mantle_tidal_fraction link): `Normal(0.4, 0.1)` clipped to `[0.1, 0.8]`.

### References

- Lemasquerier, D. G., Bierson, C. J., and Soderlund, K. M. (2023), Europa's ocean translates interior tidal heating patterns to the ice-ocean boundary, AGU Advances. DOI: 10.1029/2023AV000994
- Soderlund, K. M., Schmidt, B. E., Wicht, J., and Blankenship, D. D. (2014), Ocean-driven heating of Europa's icy shell at low latitudes, Nature Geoscience. DOI: 10.1038/ngeo2021
- Ashkenazy, Y., and Tziperman, E. (2021), Dynamic Europa ocean shows transient Taylor columns and convection driven by ice melting and salinity, Nature Communications. DOI: 10.1038/s41467-021-26710-0

## Impact on Existing Code

### Files modified

| File | Change |
| --- | --- |
| `latitude_profile.py` | New fields, updated methods, new `resolved_q_star()` and `_q_star_to_amplitude()` helpers |
| `literature_scenarios.py` | Scenarios use `q_star` instead of `ocean_amplitude` |
| `latitude_sampler.py` | Sample `T_floor`, `mantle_tidal_fraction`; derive `q_star` |
| `profile_diagnostics.py` | Add `q_star` and `mantle_tidal_fraction` to diagnostic output |
| `test_latitude_profile.py` | New tests for energy balance floor, sqrt strain, q_star derivation |
| `test_literature_scenarios.py` | Update expected amplitudes |
| `test_latitude_sampler.py` | Test new sampled fields |

### Files NOT modified

- `axial_solver.py` — consumes LatitudeProfile via `evaluate_at()`, no change needed
- `monte_carlo_2d.py` — consumes sampler output, no change needed
- `EuropaProjectDJ/*` — no modifications

### Backward compatibility

**API compatibility (preserved):**
- All existing field names and method signatures are unchanged
- `ocean_amplitude` is preserved as an explicit override (highest priority in resolution order)
- `LatitudeProfile()` with no arguments produces a valid profile
- `evaluate_at()` return dict keys are unchanged

**Numerical changes (deliberate):**
- Surface temperature: the energy balance floor replaces the cosine clamp, changing T_s at all latitudes by 1-3 K. At the equator, T_s increases by ~1.2% due to the floor term. At the pole, T_s changes from ~33.6 K (old clamp) to 52 K (Ashkenazy floor).
- Tidal strain: mid-latitude values change by up to ~5% in strain (~10% in heating). Endpoint values are unchanged.
- Ocean heat flux defaults: the default polar_enhanced amplitude changes from `a=1.0` (2:1 ratio) to `a~0.54` (1.5:1 ratio) when no `ocean_amplitude` is explicitly set. This is because the new derivation chain (`mantle_tidal_fraction=0.5` -> `q_star=0.455` -> `a=0.536`) produces a more conservative contrast than the old hardcoded default. **Existing code that explicitly sets `ocean_amplitude=1.0` (including the current `literature_scenarios.py`) is unaffected.** The `literature_scenarios.py` will be updated to use `q_star` with values that produce the desired endpoint ratios.

## Removed Code

- `_PHI_FLOOR` constant
- `_COS_FLOOR` constant
- The `np.maximum(np.cos(phi_arr), _COS_FLOOR)` clamp in `surface_temperature()`

## Validation Criteria

1. `surface_temperature(0.0)` returns `(T_eq**4 + T_floor**4)**(1/4)` — approximately 111.3 K with defaults, NOT exactly T_eq
2. `surface_temperature(pi/2)` returns T_floor exactly
3. `surface_temperature` is monotonically decreasing from equator to pole for T_floor < T_eq
4. `LatitudeProfile(T_floor=T_eq)` raises ValueError (non-physical)
5. `tidal_strain(0.0)` returns epsilon_eq; `tidal_strain(pi/2)` returns epsilon_pole
6. `tidal_strain(pi/4)**2 / tidal_strain(0.0)**2` equals `(1 + c*0.5)` exactly
7. Ocean heat flux normalization still preserves global mean for all patterns
8. `resolved_q_star()` returns `0.91 * mantle_tidal_fraction` when q_star is None and ocean_amplitude is None
9. `_q_star_to_amplitude()` raises ValueError for q_star >= 3.0 (polar) or q_star >= 1.5 (equator)
10. Literature scenarios produce the expected endpoint ratios from the corrected scenario table
11. 2D single-column validation test still passes (equator column matches 1D within 2%, accounting for the ~1.2% T_s uplift)
