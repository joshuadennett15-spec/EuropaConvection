# Design Spec: Latitude-Dependent Convection Hypothesis Testing

## Problem Statement

The 2D axisymmetric Europa ice shell model produces a convective sublayer
(D_conv) that is nearly uniform across all latitudes and ocean transport
scenarios. The latitude mode score is -5.20 with JS divergence
discriminability of only 0.0013, meaning the model cannot distinguish
between uniform, polar-enhanced, and equator-enhanced ocean heat flux
patterns.

### Root Cause

The stagnant-lid convection parameterization in `EuropaProjectDJ/src/Convection.py`
computes the rheological transition temperature T_c from activation energies
and T_melt alone:

```
T_c = Ti - theta_lid * DTv

where:
  Ti   = Deschamps Eq.18 (depends on Q_v, T_melt, T_surface)
  DTv  = -1 / (d ln eta / dT)   [pure rheology]
  theta_lid = 2.24               [fixed constant]
```

When `use_composite_transition_closure=True` (already enabled for 2D), T_c
does incorporate d_grain through the composite viscosity derivative. However,
T_c still does not depend on:
- Ocean heat flux q_ocean(phi)
- Volumetric tidal heating rate
- Any heat-budget constraint

The result: D_cond/H varies only ~6% from equator (~0.70H) to pole (~0.76H).
The convective layer ratio is essentially constant regardless of ocean forcing.

### Scientific Context

In real stagnant-lid convection:
1. Convective vigor adjusts to transport the imposed basal + internal heat
2. Higher heat flux requires higher Ra, which reorganizes the convective layer
3. Tidal strain softens ice, changing the effective viscosity and transition depth
4. Grain size evolves with local stress, feeding back on viscosity

The parameterization decouples the heat budget from the convective partitioning.

## Architecture

### Design Constraint

The 1D solver (`EuropaProjectDJ/`) gets one minimal hook: an optional
`convection_adjuster` callback inside `build_conductivity_profile()`. All
hypothesis logic lives in `Europa2D/src/convection_2d.py`.

### The Hook: `convection_adjuster` Callback

**Insertion point:** `Convection.py:build_conductivity_profile()`, after
`scan_temperature_profile()` returns the default `ConvectionState`, before
the Nu enhancement is applied to k_profile.

Current flow (Convection.py ~lines 1070-1151):
```
state = scan_temperature_profile(...)           # line ~1072
k_profile = Thermal.conductivity(T).copy()      # line ~1133
# ... porosity, salt corrections ...            # lines ~1136-1143
k_profile[state.idx_c:] *= Nu_eff              # line ~1149
return k_profile, state
```

New flow with hook:
```
state = scan_temperature_profile(...)           # unchanged
k_profile = Thermal.conductivity(T).copy()      # unchanged
# ... porosity, salt corrections ...            # unchanged

# HOOK: let adjuster modify state before k_profile is built
if convection_adjuster is not None:
    convection_adjuster(state, T_profile, z_grid, total_thickness, q_ocean)

# Apply Nu from (possibly adjusted) state
if state.is_convecting:
    Nu_eff = 1 + nu_ramp * (state.Nu - 1)
    k_profile[state.idx_c:] *= Nu_eff

return k_profile, state
```

**Why this works:** The adjuster can modify any field on the mutable
`ConvectionState` (Nu, Ra, idx_c, D_conv, D_cond, is_convecting, T_c).
The modified state is then used to build k_profile, so the change propagates
to the next timestep's thermal solve via the effective conductivity.

### Adjuster Callback Signature

```python
Callable[[ConvectionState, ndarray, ndarray, float, float], None]
#         state          T_prof  z_grid  H       q_ocean
```

Mutates `state` in place, returns None.

### Threading q_ocean to the Hook

q_ocean must be available inside `build_conductivity_profile()` so the
heat-balance hypothesis can use it. Threading path:

1. `Thermal_Solver.__init__()`: Accept `convection_adjuster` parameter,
   store as `self._convection_adjuster`
2. `Thermal_Solver.solve_step(q_ocean)`: Store `self._current_q_ocean = q_ocean`
   before calling `_assemble_system()`
3. `_assemble_system()`: Pass `self._convection_adjuster` and
   `self._current_q_ocean` to `build_conductivity_profile()`
4. `build_conductivity_profile()`: Accept two new optional keyword args:
   `convection_adjuster=None`, `q_ocean=0.0`. Call adjuster after scan,
   before Nu application.

### 1D Code Changes (Exact)

**Solver.py:**
- `__init__()`: Add `convection_adjuster: Optional[Callable] = None` param,
  store as `self._convection_adjuster`. Add `self._current_q_ocean = 0.0`.
- `solve_step()`: Add `self._current_q_ocean = q_ocean` before Picard loop.
- `_assemble_system()`: Pass `convection_adjuster=self._convection_adjuster`
  and `q_ocean=self._current_q_ocean` to the `build_conductivity_profile()`
  call at line ~229.

**Convection.py:**
- `build_conductivity_profile()`: Add `convection_adjuster=None` and
  `q_ocean=0.0` keyword args. After `state = scan_temperature_profile(...)`,
  insert 2-line adjuster call. Restructure Nu application to use
  `state.idx_c` after potential adjuster modification.

Total: ~15 lines changed in 1D code. No changes to function return types,
no changes to scan_temperature_profile, no changes to ConvectionState.

### Acceptance Test: hypothesis=None Parity

When `convection_adjuster=None` (default), the code path is identical to
the current behavior: the adjuster call is skipped, Nu is applied from
the unmodified state. This must be verified by running the existing test
suite and confirming zero regression in:
- `test_validation.py` (1D/2D consistency)
- `test_monte_carlo_2d.py` (MC results)
- All existing autoresearch baselines

### Acceptance Test: Changes Solver Evolution

When a non-trivial adjuster is provided (e.g., one that doubles Nu), the
k_profile must change, which must change the next-step temperature profile,
which must change the equilibrium thickness. This is verified by running a
single deterministic 2D solve with and without the adjuster and confirming
the H profiles differ.

## Hypothesis Implementations

### Hypothesis 1: Grain-Strain Coupling (CONFIG-ONLY)

**No new code.** The 2D path already enables
`use_composite_transition_closure=True` and passes latitude-scaled d_grain
into `compute_transition_temperature()` via `green_cond_base_temp()`.

Experiments 1-2 simply set `grain_latitude_mode="strain"` with different
`grain_strain_exponent` values in the LatitudeProfile constructor.

**Parameters:** `grain_strain_exponent` (test 0.5 and 1.0)

**What to measure:** Whether the existing grain-T_c coupling creates
meaningful D_conv latitude variation. If D_conv contrast improves but
JS divergence does not, this confirms the grain effect is scenario-invariant.

### Hypothesis 2: Heat-Balance D_conv (`heat_balance`)

The adjuster iterates D_conv until the convective heat transport matches
the local heat budget.

At each column, after the 1D solver produces the default ConvectionState:

```
q_conv_required = q_ocean + q_tidal_integrated - q_conducted_lid

where:
  q_conducted_lid = k(T_c) * (T_c - T_surface) / D_cond
  q_tidal_integrated = integral of q_tidal(z) over convective layer
                       (from column's tidal heating profile)
```

The iteration:
1. Start from D_conv_0 (1D solver default)
2. Compute Ra(D_conv) = rho*g*alpha*DT*D_conv^3 / (kappa*eta)
3. Compute Nu(Ra) = C * Ra^(1/3)
4. Compute q_transported = Nu * k * DT / D_conv
5. If q_transported < q_conv_required: increase D_conv
   If q_transported > q_conv_required: decrease D_conv
6. Update state: Nu, Ra, D_conv, D_cond, idx_c, is_convecting
7. Repeat until |q_transported - q_conv_required| < tolerance or max_iter

**Bounds:** D_conv is clamped to [0, 0.9*H] to avoid unphysical states.
If iteration does not converge, fall back to 1D default.

**Parameters:**
- `include_tidal` (bool): whether q_tidal enters the budget
- `max_iterations` (int, default 5)
- `tolerance` (float, default 1e-4 W/m^2)

**Expected effect:** D_conv responds directly to q_ocean(phi). Different
ocean scenarios produce different D_conv distributions, which should improve
JS divergence. This is the primary candidate for discriminability improvement.

### Hypothesis 3: Ra-Onset Override (`ra_onset`)

The adjuster overrides `is_convecting` using a custom Ra_crit threshold.

```python
def adjuster(state, T_profile, z_grid, H, q_ocean):
    state.is_convecting = state.Ra >= ra_crit_override
    if not state.is_convecting:
        state.Nu = 1.0
        # D_conv, D_cond preserved for diagnostics but Nu=1 means
        # no conductivity enhancement -> effectively conductive
```

This is clean because Ra_crit does not need to be injected into the 1D
constant system. The adjuster simply re-evaluates the onset criterion
after the default state is computed.

**Parameters:** `ra_crit_override` (float, test 800 and 1200)

**Expected effect:** Binary latitude switch. Ra_crit=1200 pushes more
polar columns into conductive regime, creating latitude contrast.

### Hypothesis 4: Tidal Viscosity Feedback (`tidal_viscosity`)

The adjuster recalculates Nu using a tidal-softened viscosity:

```
eta_eff = eta_default / (1 + (epsilon_0(phi) / epsilon_ref)^n)
Ra_adj = Ra_default * (eta_default / eta_eff)
Nu_adj = C * Ra_adj^(1/3)
```

The adjuster modifies state.Ra and state.Nu based on the local tidal
strain rate, leaving D_conv and D_cond unchanged.

**Parameters:**
- `epsilon_ref` (float, default 6e-6 = equatorial strain)
- `softening_exponent` n (test 1.0 and 2.0)

**Expected effect:** Poles get higher effective Ra and Nu, increasing
convective heat transport. Like grain-strain, this creates latitude
variation but the same variation regardless of ocean scenario.

## Experiment Plan

### Campaign Structure

10 experiments, each running 3 scenarios (uniform, polar_enhanced q*=0.455,
equator_enhanced q*=0.4) x 150 MC samples.

| Exp | Name              | Type           | Key Parameters               | New Code? |
|-----|-------------------|----------------|------------------------------|-----------|
| 0   | baseline_150      | Control        | Current defaults             | No        |
| 1   | grain_alpha05     | Config-only    | grain_latitude_mode="strain", exp=0.5 | No |
| 2   | grain_alpha10     | Config-only    | grain_latitude_mode="strain", exp=1.0 | No |
| 3   | heatbal_ocean     | Adjuster       | include_tidal=False          | Yes       |
| 4   | heatbal_total     | Adjuster       | include_tidal=True           | Yes       |
| 5   | ra_crit_800       | Adjuster       | ra_crit_override=800         | Yes       |
| 6   | ra_crit_1200      | Adjuster       | ra_crit_override=1200        | Yes       |
| 7   | tidal_visc_n1     | Adjuster       | n=1.0, epsilon_ref=6e-6      | Yes       |
| 8   | winner_validate   | Re-run best    | TBD                          | No        |
| 9   | combo_top2        | Combination    | TBD (top-2 combined)         | Maybe     |

### Decision Logic

- After experiment 7: rank all by primary score (latitude_score). Best
  single mechanism becomes experiment 8.
- After experiment 8: if score improvement > 20% over baseline, the top-2
  combination goes to experiment 9.
- If no mechanism improves JS divergence above 0.01 on D_cond, check whether
  D_conv JS improved. If yes, the mechanism changes internal structure but
  that change does not propagate to the observable. Document as a finding.

### Scoring: Primary and Secondary

**Primary (optimized):** Existing `compute_latitude_score()` from
`autoresearch/objectives.py`, which uses D_cond JS divergence at 35 deg.
This is the Juno-facing observable.

**Secondary (tracked, tie-breaker):** New D_conv JS divergence metric,
computed identically but on D_conv_profiles instead of D_cond_profiles.
Added to the metrics dict as `JS_discriminability_Dconv`. Not included in
the scalar score. Used to interpret whether internal structure changes are
propagating to the observable.

This requires a small addition to `compute_latitude_score()`:
```python
# After existing D_cond JS computation:
d_conv_js_values = []
for i in range(len(scenario_names)):
    for j in range(i + 1, len(scenario_names)):
        d_a = np.asarray(scenarios[scenario_names[i]]['D_conv_profiles'])[:, idx_35]
        d_b = np.asarray(scenarios[scenario_names[j]]['D_conv_profiles'])[:, idx_35]
        d_conv_js_values.append(_js_divergence(d_a, d_b))
# Add to metrics:
metrics['JS_discriminability_Dconv'] = min(d_conv_js_values) if d_conv_js_values else 0.0
```

### Success Metrics

| Metric              | Baseline | Target    | Role        |
|---------------------|----------|-----------|-------------|
| JS div (D_cond)     | 0.0013   | >0.01     | Primary     |
| JS div (D_conv)     | ~0.001   | >0.01     | Secondary   |
| D_conv contrast     | ~4 km    | >8 km     | Secondary   |
| D_cond @ 35 median  | ~28 km   | 19-39 km  | Constraint  |
| Latitude score      | -5.20    | < -10     | Primary     |

## File Changes

### New Files

```
Europa2D/src/convection_2d.py              # Adjuster factory + hypothesis logic
autoresearch/experiments/
    hypothesis_config.json                  # Experiment definitions
    run_hypothesis_campaign.py              # Campaign runner script
```

### Modified Files: 1D Solver (Minimal Hook)

```
EuropaProjectDJ/src/Solver.py (~10 lines)
  - __init__(): accept convection_adjuster param
  - solve_step(): store q_ocean before Picard loop
  - _assemble_system(): pass adjuster + q_ocean to build_conductivity_profile

EuropaProjectDJ/src/Convection.py (~8 lines)
  - build_conductivity_profile(): accept convection_adjuster + q_ocean kwargs
  - Insert 2-line adjuster call after scan, before Nu application
```

### Modified Files: Europa2D (Threading)

```
Europa2D/src/axial_solver.py
  - __init__(): accept hypothesis, create per-column adjusters
  - solve_step(): unchanged (adjusters are per-column, set at init)

Europa2D/src/monte_carlo_2d.py
  - _run_single_2d_sample(): accept and pass hypothesis
  - MonteCarloRunner2D: accept hypothesis in constructor and run()

autoresearch/harness.py
  - _run_latitude_experiment(): accept optional hypothesis

autoresearch/objectives.py
  - compute_latitude_score(): add D_conv JS to metrics dict (not score)
```

### Unchanged

```
Europa2D/src/latitude_profile.py
Europa2D/src/profile_diagnostics.py
Europa2D/src/latitude_sampler.py
EuropaProjectDJ/src/Convection.py (scan_temperature_profile, ConvectionState)
```

## `convection_2d.py` Module Design

```python
"""
Experimental convection hypothesis adjusters for Europa2D.

Each hypothesis is implemented as a factory that returns a
convection_adjuster callback. The callback mutates a ConvectionState
in place during build_conductivity_profile(), before Nu is applied
to k_profile. This is the only coupling point with the 1D solver.
"""

@dataclass(frozen=True)
class ConvectionHypothesis:
    mechanism: str   # "grain_strain" | "heat_balance" | "ra_onset" | "tidal_viscosity"
    params: dict     # mechanism-specific parameters

def make_adjuster(hypothesis, phi, profile):
    """Create a convection_adjuster closure for one latitude column.

    Args:
        hypothesis: ConvectionHypothesis with mechanism and params
        phi: geographic latitude in radians for this column
        profile: LatitudeProfile instance (for epsilon_0, q_ocean, etc.)

    Returns:
        Callable matching the convection_adjuster signature, or None
        if hypothesis is None.
    """
    ...

# --- Hypothesis implementations (private) ---

def _heat_balance_adjuster(state, T_profile, z_grid, H, q_ocean,
                           include_tidal, max_iterations, tolerance):
    """Iterate D_conv to match local heat budget."""
    ...

def _ra_onset_adjuster(state, T_profile, z_grid, H, q_ocean,
                       ra_crit_override):
    """Override is_convecting with custom Ra_crit."""
    ...

def _tidal_viscosity_adjuster(state, T_profile, z_grid, H, q_ocean,
                              epsilon_0_local, epsilon_ref, n):
    """Rescale Ra and Nu using tidal-softened viscosity."""
    ...
```

Note: grain_strain has no adjuster. It is config-only via
`grain_latitude_mode="strain"` in LatitudeProfile.

## Backward Compatibility

- `convection_adjuster=None` (default in Thermal_Solver): adjuster call
  is skipped, identical code path to current behavior
- All existing tests pass without modification
- hypothesis=None in AxialSolver2D: no adjusters created
- Existing autoresearch baselines are preserved

## Runtime Estimate

Per experiment: 3 scenarios x 150 samples x ~2.8s/sample / 7 workers
= ~3 minutes. Heat-balance experiments may take ~4.5 minutes due to
inner iteration (~5 steps per column per timestep).

Total campaign: ~30-45 minutes.

## Priority Prediction

Hypothesis 2 (heat-balance) is most likely to improve D_cond JS divergence
because it is the only mechanism that directly couples D_conv to q_ocean(phi),
and changes in D_conv propagate to D_cond through the conductive lid
thickness constraint (D_cond = H - D_conv).

Hypotheses 1 and 4 create latitude variation in D_conv but the same
variation regardless of ocean scenario. Hypothesis 3 creates binary
contrast. These may improve D_conv contrast without improving discriminability.

The combination of heat-balance + grain-strain (experiment 9) should capture
both effects: grain-strain provides baseline latitude variation in convective
vigor, while heat-balance makes that variation respond differently to
different ocean patterns.
