# How To Make The 2D Temperature Field Behave Like The 1D Temperature Field

## Why this note exists

The confusion here is reasonable. The current `Europa2D` production branch is
not just "the 1D solver with latitude added". It reuses the same underlying
`Thermal_Solver` and the same `D_cond`/`D_conv` decomposition logic, but it
changes several inputs and closures at the same time:

- latitude-dependent surface temperature,
- latitude-dependent tidal strain,
- latitude-dependent ocean heat flux,
- a 2D-only cold-column convection ramp,
- and different ensemble summaries.

That means the current 1D and 2D production tables are not a strict
apples-to-apples comparison.

## Bottom line

The shell decomposition itself is mostly the same in both branches.

Both branches ultimately define the shell split from the same logic in
`EuropaProjectDJ/src/Convection.py`:

1. compute the rheological transition temperature `T_c`,
2. scan the temperature profile `T(z)` for the first depth where `T >= T_c`,
3. set `D_cond = z_c`,
4. set `D_conv = H - z_c`,
5. compute `Ra` and `Nu` for the convective sublayer.

The big difference is not "a different definition of `D_cond`". The big
difference is that the 2D branch produces a different temperature field before
that decomposition is applied.

## What is already shared between 1D and 2D

- Each 2D latitude column is a `Thermal_Solver` from `EuropaProjectDJ/src/Solver.py`.
- The conductive/convective split is computed by the same convection code in
  `EuropaProjectDJ/src/Convection.py`.
- `RA_CRIT` comes from the same constants file.
- A single-column conductive validation already exists in
  `Europa2D/tests/test_validation.py`.

So the right way to think about the mismatch is:

`same decomposition algorithm + different forcing/closure choices = different shell splits`

## What changes in 2D today

| Item | 1D equatorial / global | 2D current production branch | Why it changes `D_cond` and `D_conv` |
|---|---|---|---|
| Surface temperature | One scalar `T_surf` per run | `T_surf(phi)` varies with latitude in `Europa2D/src/latitude_profile.py` | Colder columns keep more of the shell below `T_c`, so `z_c` moves deeper and `D_cond` increases |
| Tidal strain | One scalar `epsilon_0` per run | `epsilon_0(phi)` varies with latitude | Changes internal tidal heating and therefore the temperature profile |
| Basal/ocean forcing | One scalar basal flux per run | `q_ocean(phi)` is redistributed with latitude | Local columns no longer see the same heat flux as the 1D proxy |
| Tidal flux scaling | No extra 2D uplift | `LatitudeParameterSampler` uses `q_tidal_scale = 1.20` | Changes the global mean thermal state before latitude redistribution |
| Convection ramp | No latitude-based cold-column ramp | `AxialSolver2D` damps convection below `80 K` and turns it off below `60 K` | Makes cold columns more lid-dominated even if a geometric warm sublayer exists |
| Lateral coupling | None | Explicit lateral diffusion step between columns | Small effect in practice, but still not identical to 1D |
| Ensemble summary | One column per realization | 37 columns per realization in the production runs | Global 2D summaries mix warm equatorial and cold polar columns |

## The specific 2D differences that matter most

### 1. The 2D surface boundary condition is not the 1D surface boundary condition

The 1D equatorial suite samples a single warm surface temperature:

- `EuropaProjectDJ/src/audited_equatorial_sampler.py`

The 2D branch samples an equatorial anchor and a polar floor, then applies a
latitude law:

- `Europa2D/src/latitude_sampler.py`
- `Europa2D/src/latitude_profile.py`

That means the 2D run is not using one shell-wide surface temperature. It is
using a latitude-dependent outer boundary.

Practical effect:

- equator stays relatively warm,
- poles become much colder,
- cold columns get much thicker conductive lids,
- global 2D medians shift away from the 1D single-column medians.

### 2. The 2D ocean forcing is redistributed, not just scaled

The 1D equatorial suite is a scalar proxy:

- take the 1D basal forcing,
- scale the tidal part up or down,
- solve one column.

The 2D branch does something different:

- build a global-mean heat flux,
- optionally apply a 2D-only tidal uplift,
- redistribute that flux with latitude while preserving the global mean.

Files:

- `EuropaProjectDJ/src/audited_equatorial_sampler.py`
- `Europa2D/src/latitude_sampler.py`
- `Europa2D/src/latitude_profile.py`

Practical effect:

- in 1D, "less equatorial heat" just means the single column gets colder,
- in 2D, "less equatorial heat" also means some other latitude gets more heat,
- so the global shell structure responds differently.

### 3. The 2D branch has a cold-column convection ramp

`Europa2D/src/axial_solver.py` sets a per-column ramp:

- `0` for `T_surf <= 60 K`
- `1` for `T_surf >= 80 K`
- smooth blend in between

That ramp is passed into `IceConvection.build_conductivity_profile()` in
`EuropaProjectDJ/src/Convection.py`, where it modifies the effective Nusselt
enhancement in the convective layer.

Practical effect:

- a 2D cold column can still have a warm sublayer geometrically,
- but its transport can be damped toward conduction,
- which thickens the lid relative to a pure 1D treatment.

For the equator this usually matters less, because `T_surf` is warm enough that
the ramp is close to `1`. For the full 2D global statistics it matters a lot.

### 4. The reported "convective fraction" is not one consistent quantity

This caused a lot of the confusion.

In the current table scripts:

- 1D `Conv. frac` means `fraction of realizations with Ra >= Ra_crit`
- 2D Table 3a `Conv. frac` means `fraction of realizations where at least half
  of latitude columns have Ra >= Ra_crit`

There are also other quantities in the repo labeled similarly:

- share of columns with `Nu > 1.01`
- shell-thickness convective fraction `D_conv / H`

So the table label hides multiple different metrics.

## Are the current 1D and 2D results actually wildly inconsistent?

Not really. They are different, but not absurdly different once you compare the
same quantity.

Archive-level checks from the current saved runs give:

- 1D equatorial 15k runs: `52.4%` to `89.3%` supercritical realizations
- 1D equatorial mean `D_conv / H`: `31.2%` to `59.4%`
- 2D production Table 3a: `34.6%` to `43.0%` for the much stricter
  "majority of columns convect" metric
- 2D equatorial-column supercritical fraction: about `35.9%` to `58.9%`
- 2D mean equatorial `D_conv / H`: about `21.3%` to `30.7%`

So the current branches are not giving an order-of-magnitude contradiction.
They are giving moderately different answers from non-identical models.

## What "make 2D work like 1D" can mean

There are three different goals you might mean.

### Goal A: strict solver parity

Question:

"If I remove the extra latitude physics, does one 2D column reproduce the 1D
column?"

This is the cleanest and most important validation target.

### Goal B: bookkeeping parity with `n_lat > 1`

Question:

"Can I run the 2D machinery with many columns, but force every column to see
the same temperature boundary, same strain, and same basal heat flux?"

This is mainly a code-consistency test, not a science model.

### Goal C: physics parity for Monte Carlo comparison

Question:

"Can I compare 1D and 2D ensembles without changing the priors and closures at
the same time?"

This is what you want for a clean interpretation paper or thesis chapter.

## Recommended strict recipe: make 2D match 1D as closely as possible

If the goal is "I want the 2D solver to behave exactly like the 1D solver",
the best recipe is:

1. Run `Europa2D` with `n_lat = 1`.
2. Use `ocean_pattern = "uniform"`.
3. Set `q_tidal_scale = 1.0` in `LatitudeParameterSampler`.
4. Feed the same basal flux to the single 2D column that the 1D run uses.
5. Set `epsilon_eq = epsilon_pole = epsilon_0_1d`.
6. Set `T_eq = T_surf_1d`.
7. Use the same rheology, `nx`, `dt`, `max_steps`, and initial thickness.
8. Compare the single 2D column only, not any latitude-averaged statistic.
9. Use the same convection diagnostic in both branches, preferably:
   `Ra >= Ra_crit` and `D_conv / H`.

Important note:

With `n_lat = 1`, the 2D surface law is evaluated only at the equator. That
means `T_floor` does not affect the solved column, as long as it stays below
`T_eq`.

## Why `n_lat = 1` is the cleanest parity target

Because it removes almost every 2D-specific change in one step:

- no cold off-equator columns,
- no mixed global statistic,
- no latitude redistribution effect at the point of comparison,
- no polar damping dominating the summary,
- no ambiguity about what "convective fraction" means.

If `n_lat = 1` still disagrees strongly with 1D after all inputs are matched,
then the problem is in solver implementation.

If `n_lat = 1` agrees, then the later mismatch is coming from the added 2D
physics assumptions, not a broken decomposition.

## If you want `n_lat > 1` but still want 1D-style physics

That requires code changes, because the current 2D defaults intentionally add
latitude structure.

### Minimal code changes I recommend

#### 1. Add a compatibility mode to `LatitudeParameterSampler`

File:

- `Europa2D/src/latitude_sampler.py`

Add something like:

- `compatibility_mode = "default_2d" | "match_1d_global" | "match_1d_equator"`

Behavior for compatibility mode:

- sample `T_surf`, `epsilon_0`, and basal flux from the matching 1D sampler,
- set `q_tidal_scale = 1.0`,
- do not introduce extra 2D-only prior changes.

#### 2. Add a uniform surface-temperature option

File:

- `Europa2D/src/latitude_profile.py`

Add something like:

- `surface_pattern = "ashkenazy" | "uniform"`

For `uniform`:

- return `T_surf(phi) = T_eq` for every latitude,
- ignore `T_floor` in that mode.

That is the cleanest way to make the 2D outer boundary identical to 1D across
all columns.

#### 3. Add a uniform strain option

File:

- `Europa2D/src/latitude_profile.py`

Add something like:

- `tidal_pattern = "mantle_core" | "uniform" | ...`

For `uniform`:

- return `epsilon_0(phi) = epsilon_eq` for every latitude.

#### 4. Add a switch to disable the cold-column convection ramp

File:

- `Europa2D/src/axial_solver.py`

Add something like:

- `convection_ramp_mode = "surface_temp" | "none"`

For parity mode:

- use `"none"` so `convection_ramp = 1.0` for all columns.

#### 5. Add a switch to force uniform ocean heat flux

Files:

- `Europa2D/src/latitude_profile.py`
- `Europa2D/scripts/run_2d_mc.py`

The current `uniform` ocean pattern already keeps the heat flux constant with
latitude, which is good. The remaining parity issue is making sure the mean
value is the exact 1D value and is not modified by `q_tidal_scale`.

#### 6. Add a 2D Monte Carlo reporting mode that matches 1D diagnostics

File:

- `Europa2D/src/monte_carlo_2d.py`

Report at least:

- equatorial `Ra >= Ra_crit` fraction,
- equatorial `D_conv / H`,
- equatorial `D_cond`,
- equatorial `D_conv`,
- and only then the global 2D summaries.

That gives you one directly comparable 1D-vs-2D column before the global
latitude effects are mixed in.

## Recommended validation sequence

Do not try to debug everything at once. Reintroduce one difference at a time.

### Stage 1: exact single-column parity

Target:

- `n_lat = 1`
- `uniform` ocean forcing
- same `T_surf`
- same `epsilon_0`
- same basal flux
- `q_tidal_scale = 1.0`
- same `nx`, `dt`, and rheology

Expected outcome:

- `H`, `D_cond`, `D_conv`, `Ra`, and `Nu` should match 1D closely.

### Stage 2: many columns, but all columns identical

Target:

- `n_lat = 37`
- uniform surface temperature
- uniform strain
- uniform ocean heat flux
- no convection ramp

Expected outcome:

- every latitude column should collapse to the same 1D-like answer,
- any remaining spread means the 2D bookkeeping is changing the result.

### Stage 3: turn on one latitude effect at a time

Order:

1. surface-temperature gradient only
2. ocean-flux redistribution only
3. strain gradient only
4. convection ramp only
5. lateral diffusion only

Expected outcome:

- you can then see exactly which assumption is moving `D_cond` and `D_conv`.

## What I would change first

If the goal is to reduce confusion quickly, I would do these first:

1. Add a documented `match_1d_equator` compatibility mode in
   `Europa2D/src/latitude_sampler.py`.
2. Add `surface_pattern = "uniform"` in `Europa2D/src/latitude_profile.py`.
3. Add `convection_ramp_mode = "none"` in `Europa2D/src/axial_solver.py`.
4. Add a strict convective single-column validation test in
   `Europa2D/tests/test_validation.py`.
5. Rename the reported 2D metrics so `convective_fraction` is never ambiguous.

## Recommended interpretation of the current results

The safest interpretation of the current repo state is:

- the 1D and 2D branches are similar in decomposition method,
- different in forcing and closure assumptions,
- and therefore not directly interchangeable.

So the current mismatch does **not** mean:

- the decomposition is obviously broken,
- or that you are missing something simple.

It means the current 2D production setup is answering a different question:

"What happens when the shell sees latitude-dependent forcing and cold-column
convection damping?"

rather than:

"What happens if I solve the same 1D thermal problem with a 2D wrapper?"

## Concrete next step

If you want a clean scientific answer, the next run to do is:

1. `Europa2D` with `n_lat = 1`
2. same priors as the 1D equatorial baseline
3. `q_tidal_scale = 1.0`
4. `uniform` forcing
5. compare `H`, `D_cond`, `D_conv`, `Ra`, `Nu`

If that matches, the decomposition is fine and the production-run differences
come from the added 2D assumptions.

If that does not match, then there is a real implementation mismatch to fix.
