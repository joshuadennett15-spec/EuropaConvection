# Surface Temperature Dominates Shell-Thickness Latitude Structure

Date: 2026-03-20

## Summary

The uniform_transport benchmark (uniform ocean heat flux, default tidal strain
gradient) reveals that Europa's shell-thickness latitude structure is controlled
primarily by the surface temperature boundary condition, not by tidal heating or
ocean heat transport. Despite 4x stronger tidal heating at the pole, the polar
shell is 29% thicker than the equatorial shell. The conductive lid accounts for
101.5% of the pole-equator thickness contrast.

Note: this diagnostic used the colder `T_eq = 96 K`, `T_floor = 46 K`
Ashkenazy-style surface boundary and should be interpreted as a cold-surface
benchmark, not as the later `T_eq = 110 K` default used to align the 2D equator
with the 1D equatorial workflow.

## Diagnostic Table

All runs use: T_eq = 96 K, T_floor = 46 K, epsilon_eq = 6e-6, epsilon_pole = 1.2e-5,
q_ocean = 20 mW/m^2 (uniform), n_lat = 37, nx = 31.

| Latitude | T_surface | q_tidal/q_tidal_eq | H_total | D_cond | D_conv | lid_fraction | Ra   | Nu   |
|----------|-----------|-------------------|---------|--------|--------|-------------|------|------|
| 0.0 deg  | 96.0 K    | 1.00              | 28.23   | 26.02  | 2.21   | 0.922       | 30.5 | 1.00 |
| 30.0 deg | 92.8 K    | 1.75              | 28.43   | 26.24  | 2.19   | 0.923       | 29.8 | 1.00 |
| 45.0 deg | 88.5 K    | 2.50              | 28.98   | 26.80  | 2.18   | 0.925       | 29.0 | 1.00 |
| 60.0 deg | 81.8 K    | 3.25              | 30.29   | 28.14  | 2.15   | 0.929       | 27.6 | 1.00 |
| 75.0 deg | 70.9 K    | 3.80              | 33.26   | 31.14  | 2.11   | 0.936       | 25.5 | 1.00 |
| 90.0 deg | 46.0 K    | 4.00              | 42.85   | 40.83  | 2.02   | 0.953       | 20.3 | 1.00 |

Band means (area-weighted):

| Band         | T_surface | H_total | D_cond | D_conv | lid_fraction | Ra   |
|--------------|-----------|---------|--------|--------|-------------|------|
| Low (0-10)   | 95.9 K    | 28.23   | 26.02  | 2.21   | 0.922       | 30.4 |
| High (80-90) | 61.8 K    | 36.44   | 34.35  | 2.08   | 0.943       | 23.7 |
| Ratio (H/L)  | 0.64      | 1.29    | 1.32   | 0.94   | 1.02        | 0.78 |

## Key Finding: The Conductive Lid Controls Everything

The pole-equator thickness difference is DeltaH = 8.21 km (band means). Of this:

- D_cond contributes 8.33 km (101.5%)
- D_conv contributes -0.13 km (-1.5%)

The convective sublayer is nearly uniform across latitude (2.08 - 2.21 km).
The entire latitude structure comes from the conductive lid.

## Physical Mechanism

The conductive lid thickness scales as:

    D_cond ~ k(T) * (T_interior - T_surface) / q_conducted

where k(T) = 567/T (W/m-K) for ice Ih (Howell 2021).

At the pole:
- T_surface = 46 K (cold)
- k(46 K) = 12.3 W/m-K (high conductivity)
- Delta_T = T_interior - 46 = ~214 K (large gradient across lid)

At the equator:
- T_surface = 96 K (warm)
- k(96 K) = 5.9 W/m-K (lower conductivity)
- Delta_T = T_interior - 96 = ~164 K (smaller gradient)

Both effects reinforce: colder ice has higher conductivity AND a larger temperature
drop across the lid. The product k * Delta_T is substantially larger at the pole,
requiring a thicker conductive lid to transport the same heat flux.

## Why Tidal Heating Does Not Offset This

The tidal heating is 4x stronger at the pole than the equator (Beuthe 2013
whole-shell pattern with epsilon_pole/epsilon_eq = 2). Naively, more internal
heating should thin the shell. Two reasons it does not:

1. The extra tidal heat is generated within the shell, not at the base. In the
   Deschamps and Vilella (2021) mixed-heating framework, internal heating can
   stiffen the stagnant lid rather than thin it, because it reduces the
   temperature contrast across the convective sublayer without reducing the
   conductive lid's temperature gradient to the surface.

2. The Rayleigh numbers are low (20-30) and the Nusselt number is 1.00 at all
   latitudes. This means the shell is in a weakly convective or barely
   supercritical regime. Convection is not vigorous enough to significantly
   modify the thermal structure. The shell is overwhelmingly conductive
   (lid_fraction > 0.92 everywhere), so the conductive scaling dominates.

At higher basal heat fluxes or in a more vigorously convecting regime (Ra >> 100,
Nu >> 1), the tidal heating pattern would have a stronger effect because
convection could transport the extra heat more efficiently. But for the
current parameter regime, the surface boundary condition wins.

## Implications for the Thesis

### 1. The cold-pole boundary condition is not cosmetic

Ashkenazy (2019) showed that Europa's annual-mean polar temperature is
significantly affected by internal heating. The benchmark confirms that this
boundary condition is the dominant control on shell-thickness latitude structure.
Any sensitivity analysis of the 2D model must include surface temperature
uncertainty as a first-order parameter.

### 2. "Uniform ocean" does not mean "uniform shell"

The uniform_transport scenario produces DeltaH = 8.21 km despite zero ocean
heat-flux contrast. This baseline tidal-driven thickness contrast must be
subtracted when interpreting the ocean-transport scenarios. The relevant
quantity for isolating ocean effects is:

    DeltaH_ocean = DeltaH_scenario - DeltaH_uniform

For the three benchmarks:
- uniform_transport: DeltaH = +8.21 km (tidal + surface baseline)
- soderlund2014_equator: DeltaH = +19.82 km, so DeltaH_ocean = +11.61 km
- lemasquerier2023_polar: DeltaH = -2.54 km, so DeltaH_ocean = -10.75 km

The ocean transport effect is roughly symmetric (+/-11 km) between the
equator-enhanced and polar-enhanced scenarios, which is a useful consistency check.

### 3. D_cond is the right observable for Juno comparison

The Juno MWR constraint of 29 +/- 10 km (Levin et al. 2026) maps to D_cond,
not H_total. In the uniform_transport benchmark:

- H_total at the equator: 28.23 km
- D_cond at the equator: 26.02 km

The difference (2.21 km) is small for this parameter set but would grow in a
more vigorously convecting regime. Reporting D_cond separately from H_total
prevents conflation of the Juno-facing quantity with the geodynamically
relevant total thickness.

### 4. The model is in a weakly convective regime

With Nu = 1.00 everywhere, the current parameter set produces a shell that is
effectively conductive with a thin vestigial convective sublayer. This is not
necessarily wrong for Europa (some authors argue for a conductive shell), but
it means:

- The 2D model's sensitivity to ocean transport patterns comes almost entirely
  through the basal boundary condition, not through convective heat transport.
- Stronger conclusions about convective transport effects would require
  parameter sets that push the shell into Nu > 1 territory (higher q_ocean,
  lower viscosity, or finer grains).
- The mixed-heating diagnostic (Deschamps and Vilella 2021) would be most
  informative precisely at the boundary between conductive and convective
  regimes — flagging which MC samples cross that threshold.
