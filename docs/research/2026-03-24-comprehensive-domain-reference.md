# Comprehensive Domain Reference: Europa Ice Shell Thermal Modeling

> Compiled 2026-03-24. Research synthesis covering geophysical modeling, heat transfer physics, ice rheology, convection mathematics, Monte Carlo uncertainty quantification, scientific visualization, and publication standards for the EuropaConvection project.

---

## Table of Contents

1. [Europa Ice Shell Geophysics](#1-europa-ice-shell-geophysics)
2. [Heat Transfer Numerical Methods](#2-heat-transfer-numerical-methods)
3. [Ice Ih Material Properties & Rheology](#3-ice-ih-material-properties--rheology)
4. [Convection Scaling Laws Mathematics](#4-convection-scaling-laws-mathematics)
5. [Monte Carlo Uncertainty Quantification](#5-monte-carlo-uncertainty-quantification)
6. [Scientific Visualization & Publication Figures](#6-scientific-visualization--publication-figures)
7. [Academic Paper Writing Standards](#7-academic-paper-writing-standards)
8. [Master Reference List](#8-master-reference-list)

---

## 1. Europa Ice Shell Geophysics

### 1.1 Europa's Ice Shell: What We Know

Europa's ice shell sits atop a global subsurface ocean, with the ice-ocean system representing one of the most astrobiologically significant environments in the solar system. The total H2O layer (ice + ocean) is approximately 127 +/- 21 km thick.

**Thickness estimates from observations and models:**

| Source | Method | Thickness (km) | Notes |
|--------|--------|-----------------|-------|
| Billings & Kattenhorn (2005) | Review of geological evidence | 3-30+ | Compiled thin vs thick shell evidence |
| Hussmann et al. (2002) | Thermal-orbital evolution | 10-30 | Coupled tidal-thermal models |
| Nimmo et al. (2003) | Tidal dissipation constraints | 15-25 | Dissipation in ice shell |
| Howell (2021) | 1D MC thermal model | 24.3 (-1.5/+22.8) | 10^7 MC samples, asymmetric PDF |
| Green et al. (2021) | 1D convection growth model | 5-30 | Depends on ocean heat flux |
| Impact crater analysis (2024) | Multiring basin formation | >20 | 6-8 km conductive lid required |
| Juno MWR (2025) | Microwave radiometry | Surface porosity profile | Exponential density increase to 2m |

**The thin-shell vs thick-shell debate:**
- **Thin shell (<10 km)**: Supported by surface geological features (chaos terrain, ridges) suggesting active ice-ocean interaction. Requires high ocean heat flux (>30 mW/m^2).
- **Thick shell (>20 km)**: Supported by crater morphology, thermal models with parameterized convection. More thermally stable, allows stagnant-lid convection.
- **Current consensus**: ~20-30 km total with ~5-10 km conductive lid overlying a convecting sublayer. Howell (2021) CBE of 24.3 km is widely cited.

### 1.2 Tidal Dissipation in Ice Shells

Tidal heating is the dominant internal heat source in Europa's ice shell. Jupiter's gravitational field raises periodic tides as Europa orbits with eccentricity e = 0.0101.

**Maxwell viscoelastic dissipation:**
```
q_vol = epsilon_0^2 * omega^2 * eta / [2 * (1 + omega^2 * eta^2 / mu^2)]
```

where:
- epsilon_0 = tidal strain amplitude (~10^-5)
- omega = orbital frequency = 2.047 x 10^-5 rad/s (3.55-day period)
- eta = dynamic viscosity (Pa s)
- mu = shear modulus = 3.3 x 10^9 Pa

**Peak dissipation** occurs when omega * eta / mu = 1, i.e., eta_peak = mu/omega ~ 1.6 x 10^14 Pa s. This corresponds to ice at ~250-260 K, near the base of the conductive lid.

**Andrade rheology** (preferred for tidal frequencies):
```
J*(omega) = J_U + beta * Gamma(1+alpha) * (i*omega)^(-alpha) - i/(omega*eta)
```

Separating real and imaginary parts:
```
J_1 = J_U + beta * Gamma(1+alpha) * omega^(-alpha) * cos(alpha*pi/2)
J_2 = 1/(omega*eta) + beta * Gamma(1+alpha) * omega^(-alpha) * sin(alpha*pi/2)
```

Dissipative shear modulus: `Im(G*) = J_2 / (J_1^2 + J_2^2)`

Volumetric heating: `q = 0.5 * omega * epsilon^2 * Im(G*)`

**Why Andrade over Maxwell**: Maxwell predicts a single sharp dissipation peak at omega*tau_M = 1 and dramatically underestimates high-frequency dissipation. Andrade captures the broadband anelastic response observed in laboratory experiments on polycrystalline materials. Bierson (2024) showed different rheology models cause orders-of-magnitude differences in predicted tidal heating rates.

### 1.3 Tidal Strain Latitude Dependence

The tidal strain field on Europa varies with latitude. The simplified parameterization (Tobie et al. 2003 inspired):
```
epsilon_0(phi) = epsilon_eq + (epsilon_pole - epsilon_eq) * sin^2(phi)
```

where epsilon_eq ~ 6 x 10^-6, epsilon_pole ~ 1.2 x 10^-5.

**Note**: This is a first-order approximation. The full spherical harmonic tidal strain pattern involves e_{2,0} and e_{2,2} components with non-monotonic latitude dependence. The simplification is acceptable for parameterized convection models where the exact spatial pattern is less important than the latitude-averaged heating rate.

### 1.4 Ocean Heat Transport Patterns

The distribution of heat from Europa's ocean to the ice shell base depends on ocean circulation patterns:

**Soderlund et al. (2014)**: 3D ocean circulation simulations showed that Europa's ocean may preferentially deliver heat to the poles due to Rossby-regime convection:
```
q_ocean(phi) = q_mean * [1 + alpha_pattern * sin^2(phi)] / normalization
```
where normalization ensures the global integral equals q_mean * A_surface.

**Lemasquerier et al. (2023)**: Updated ocean dynamics modeling with polar-enhanced transport patterns.

**Three scenarios implemented in the project:**
- `uniform`: q_ocean = constant (global Howell 2021 baseline)
- `polar_enhanced`: q proportional to 1 + alpha*sin^2(phi) (Soderlund 2014)
- `equator_enhanced`: q proportional to 1 + alpha*cos^2(phi) (alternative)

All patterns integrate to the same global total: integral from 0 to pi/2 of q(phi) * cos(phi) dphi = q_global_mean.

### 1.5 Europa Clipper Predictions

Europa Clipper (arriving ~2030-2031) will carry the REASON ice-penetrating radar instrument capable of detecting the ice-ocean interface.

**REASON specifications:**
- Dual-frequency radar sounder (HF: 9 MHz, VHF: 60 MHz)
- Penetration depth: up to ~30 km in pure ice (frequency-dependent)
- Vertical resolution: ~10 m
- Sensitive to ice purity, temperature, and structure

**Testable predictions from this model:**
1. Latitude-dependent ice shell thickness profile H(phi) with uncertainty bounds
2. Conductive lid thickness D_cond(phi) — detectable as a dielectric boundary
3. Thinning toward equator (if polar-enhanced ocean transport) or toward poles (if equator-enhanced)
4. Thickness variance: regions with larger uncertainty may show more geological complexity

### 1.6 Surface Temperature

Ojakangas & Stevenson (1989) equilibrium temperature:
```
T_s(phi) = T_eq * max(cos(phi), cos(85deg))^(1/4)
```

| Latitude | T_s (K) | Notes |
|----------|---------|-------|
| 0deg (equator) | ~110 | Maximum solar input |
| 30deg | ~105 | |
| 60deg | ~85 | |
| 85deg+ (near pole) | ~50 | Floor at 85deg prevents singularity |

The 1/4 power law derives from radiative equilibrium: epsilon * sigma * T_s^4 = S * (1-A) * cos(phi), so T_s proportional to cos(phi)^(1/4).

---

## 2. Heat Transfer Numerical Methods

### 2.1 The Governing PDE

The 2D axisymmetric heat equation in (z, phi) coordinates:
```
rho(T) * cp(T) * dT/dt = (1/G) * d/dz [G * k_eff(T,z) * dT/dz]
                        + (k / R^2 cos(phi)) * d/dphi [cos(phi) * dT/dphi]
                        + q_tidal(T, phi)
```

where G = r^2 (spherical) or 1 (Cartesian), R = Europa's radius = 1,561 km.

### 2.2 Crank-Nicolson Time Discretization

The Crank-Nicolson method (theta = 0.5) provides second-order accuracy in time:
```
[I + theta * dt * A^{n+1}] T^{n+1} = [I - (1-theta) * dt * A^n] T^n + dt * S
```

where A is the spatial discretization operator and S is the source term.

**Advantages:**
- O(dt^2) accuracy (vs O(dt) for Backward Euler)
- Unconditionally stable for linear problems (any dt)
- Time-centered: no numerical dissipation of smooth modes

**Challenges with nonlinear problems:**
- Material properties k(T), rho(T), cp(T) depend on the unknown T^{n+1}
- Requires iteration (Picard or Newton) within each timestep
- The project uses 3 Picard iterations with 0.01 K convergence tolerance

**Picard iteration convergence**: For the ice shell problem, 3 iterations typically suffice because material property variations are smooth and the temperature change per timestep is small (~0.01-1 K over dt = 10^12 s ~ 31,700 years).

### 2.3 Rannacher Startup

**Problem**: Crank-Nicolson produces 2-1 node oscillations when applied to discontinuous initial data or sharp gradients (Gibbs-like phenomenon in the temporal discretization).

**Solution**: Rannacher (1984) showed that starting with a few half-timesteps of Backward Euler (theta = 1.0) smooths the initial data, after which CN can proceed without oscillations.

**Implementation in the project** (4 Rannacher steps):
```
Steps 0-3: theta = 1.0 (Backward Euler), dt_eff = dt/2
Steps 4+:  theta = 0.5 (Crank-Nicolson), dt_eff = dt
```

Each Rannacher step consists of two BE half-steps, providing O(dt) smoothing of the initial profile. After 4 steps (8 half-steps), the solution is smooth enough for CN to maintain its O(dt^2) accuracy.

**Why it matters for ice shells**: The initial linear temperature profile T(z) = T_s + (T_m - T_s) * z/H has a sharp kink at the surface (where T transitions from the imposed BC to the geotherm), and the tidal heating profile q_tidal(T(z)) can have steep gradients near the base of the conductive lid. Without Rannacher startup, these features would generate persistent oscillations.

### 2.4 Flux-Conservative Differencing

The spatial discretization uses flux-conservative differencing for variable conductivity:
```
d/dz [k(z) dT/dz] -> [k_{i+1/2} (T_{i+1} - T_i) - k_{i-1/2} (T_i - T_{i-1})] / dz^2
```

**Half-node conductivity evaluation:**

For non-convective cases, arithmetic mean:
```
k_{i+1/2} = (k_i + k_{i+1}) / 2
```

For convective cases with k-discontinuity at the lid-convection interface, **harmonic mean**:
```
k_{i+1/2} = 2 * k_i * k_{i+1} / (k_i + k_{i+1})
```

**Why harmonic mean**: When k jumps by orders of magnitude at the conductive-convective interface (k -> Nu*k, with Nu ~ 5-10), arithmetic averaging creates numerical instability. Harmonic mean ensures flux conservation across the interface:
- If k1 >> k2, harmonic mean -> 2*k2 (limited by the smaller value)
- Physically: heat flux through two resistors in series is controlled by the higher resistance
- Guarantees continuous heat flux q = k * dT/dz across the interface

**Comparison of averaging methods:**

| Method | Formula | Best for |
|--------|---------|----------|
| Arithmetic | (k1+k2)/2 | Smooth, slowly varying k |
| Harmonic | 2k1k2/(k1+k2) | Sharp interfaces, flux conservation |
| Geometric | sqrt(k1*k2) | Log-spaced variations |

### 2.5 Stefan Condition and Moving Boundary

The ice-ocean interface is tracked via the Stefan condition:
```
v_freeze = (q_conducted - q_ocean) / (rho_ice * L)
```

where:
- q_conducted = k_basal * dT/dz|_base (heat flux conducted into ice from base)
- q_ocean = ocean heat flux (W/m^2)
- L = 334,000 J/kg (latent heat of fusion)
- v_freeze > 0: freezing (shell thickens)
- v_freeze < 0: melting (shell thins)

**Basal temperature gradient** (2nd-order one-sided finite difference):
```
dT/dz = (3*T[-1] - 4*T[-2] + T[-3]) / (2*dz)
```

This is O(dz^2) accurate and uses only interior points (no ghost nodes needed).

**Enthalpy method alternative**: Some models use the enthalpy method, which embeds the latent heat into the energy equation via H(T) = integral of rho*cp dT + rho*L * Heaviside(T - T_m). This avoids explicit interface tracking but requires fine grid resolution near the phase boundary. The Stefan condition approach used here is more efficient for 1D column models.

**Thickness update:**
```
H^{n+1} = H^n + v_freeze * dt
H = max(H, 500 m)   # Floor: minimum thickness prevents numerical collapse
```

### 2.6 Operator Splitting for 2D Extension

The 2D model uses Lie-Trotter operator splitting:
1. **Radial step (implicit)**: Each latitude column solves the 1D heat equation independently using CN
2. **Lateral step (explicit)**: Diffuse heat between adjacent columns

**Lateral diffusion operator** in geographic latitude coordinates:
```
(k / R^2 cos(phi)) * d/dphi [cos(phi) * dT/dphi]
```

**Splitting error**: Lie-Trotter splitting is O(dt) accurate. Strang splitting (half lateral, full radial, half lateral) would give O(dt^2), but is unnecessary here because:

**CFL analysis for explicit lateral step:**
```
dt_max = 0.5 * R^2 * dphi^2 / kappa
```

With kappa ~ 10^-6 m^2/s, R ~ 1.56 x 10^6 m, dphi ~ 5deg ~ 0.087 rad:
```
dt_max = 0.5 * (1.56e6)^2 * (0.087)^2 / 1e-6 = 9.2 x 10^15 s
```

This is ~10,000x larger than the radial timestep (dt = 10^12 s), so the explicit lateral step is unconditionally stable with no sub-cycling needed.

**Physical justification**: Lateral thermal diffusion timescale across 90deg of latitude (~2400 km):
```
tau_lateral ~ L^2 / kappa ~ (2.4e6)^2 / 1e-6 ~ 6 x 10^18 s ~ 200 billion years
```

This far exceeds Europa's age, so lateral coupling is genuinely weak and operator splitting is physically well-justified.

### 2.7 Spherical Coordinate Geometry

For thick shells (H >= 30 km), the model uses spherical coordinates where the geometric factor G = r^2 enters the divergence operator:
```
(1/r^2) * d/dr [r^2 * k * dT/dr]
```

The half-node geometric factors:
```
G_plus = (G_i + G_{i+1}) / 2
G_minus = (G_i + G_{i-1}) / 2
```

**When spherical matters**: For Europa with R = 1561 km and H ~ 25 km, the curvature correction is:
```
(R - H)^2 / R^2 = (1536/1561)^2 = 0.968
```

This is a ~3% correction — small but non-negligible for precise thickness predictions. The model automatically selects spherical geometry when H >= 30 km.

### 2.8 Boundary Conditions

**Surface (z = 0)**: Fixed temperature (Dirichlet):
```
T(z=0) = T_s(phi)
```
where T_s depends on latitude via the Ojakangas & Stevenson (1989) formula.

**Base (z = H)**: Pressure-dependent melting point:
```
T(z=H) = T_melt + dT_m/dP * rho * g * H
```
where dT_m/dP = -7.4 x 10^-8 K/Pa (Clausius-Clapeyron for ice Ih).

At 25 km depth: T_base = 273 + (-7.4e-8)(917)(1.315)(25000) = 273 - 2.2 = 270.8 K.

---

## 3. Ice Ih Material Properties & Rheology

### 3.1 Thermal Conductivity k(T)

Ice Ih thermal conductivity follows an inverse temperature relationship from phonon-phonon (Umklapp) scattering theory.

**Models compared:**

| Model | Formula | Coefficient | Source |
|-------|---------|-------------|--------|
| Carnahan et al. (2021) | k = 612/T | 612 W K/m | Comprehensive dataset, EPSL |
| Howell (2021) | k = 567/T | 567 W K/m | JPL reference code |
| Klinger (1975) | k ~ 630/T | ~630 W K/m | Monocrystalline, J. Glaciology |
| Slack (1980) | Wurtzite theory | — | Phys. Rev. B |

**Values at Europa conditions:**

| T (K) | k (567/T) | k (612/T) | Difference |
|-------|-----------|-----------|------------|
| 100 (surface) | 5.67 | 6.12 | 7.9% |
| 150 | 3.78 | 4.08 | 7.9% |
| 200 | 2.84 | 3.06 | 7.7% |
| 250 | 2.27 | 2.45 | 7.9% |
| 270 (base) | 2.10 | 2.27 | 8.1% |

The 567/T coefficient used in this project (following Howell 2021) is ~8% lower than Carnahan's 612/T. The difference stems from which experimental datasets are weighted. The Hobbs (1974) model used by some older codes included low-conductivity data from Dillard & Timmerhaus (1966) later deemed inaccurate. Both 567/T and 612/T are within the uncertainty range.

**Temperature range validity**: The 1/T form is valid from approximately 30 K to the melting point (~273 K). Below ~30 K, conductivity peaks and then decreases as phonon mean free path becomes boundary-limited. For Europa's ice shell (100-273 K), both models are within their regime of validity.

### 3.2 Specific Heat Capacity cp(T)

```
cp(T) = 7.49 * T + 90.0   [J/(kg K)]
```

Linear fit to ice Ih data (Giauque & Stout 1936, Feistel & Wagner 2006):

| T (K) | cp [J/(kg K)] |
|-------|---------------|
| 100 | 839 |
| 150 | 1214 |
| 200 | 1588 |
| 250 | 1963 |
| 273 | 2135 |

### 3.3 Density rho(T)

```
rho(T) = rho_0 * (1 + alpha * (T_m - T))
```

where rho_0 = 917 kg/m^3, alpha = 1.6 x 10^-4 K^-1, T_m = 273 K.

| T (K) | rho [kg/m^3] |
|-------|-------------|
| 100 | 942 |
| 150 | 935 |
| 200 | 928 |
| 250 | 920 |
| 273 | 917 |

### 3.4 Thermal Diffusivity

```
kappa(T) = k(T) / [rho(T) * cp(T)]
```

At T = 200 K: kappa = 2.84 / (928 * 1588) = 1.93 x 10^-6 m^2/s
At T = 250 K: kappa = 2.27 / (920 * 1963) = 1.26 x 10^-6 m^2/s

Representative value: kappa ~ 10^-6 m^2/s.

### 3.5 Ice Creep Mechanisms

Four distinct creep mechanisms identified by Goldsby & Kohlstedt (2001, JGR):

**(a) Dislocation creep (climb-limited):**
- Stress exponent n = 4.0
- Grain-size exponent p = 0 (grain-size independent)
- Activation energy Q = 60 kJ/mol (low T), 181 kJ/mol (high T, above ~255 K)
- Dominates at high stresses (> 1 MPa)

**(b) Grain-boundary sliding (GBS) accommodated by basal slip:**
- n = 1.8, p = 1.4
- Q = 49 kJ/mol (below ~255 K), 192 kJ/mol (above ~255 K)
- "Superplastic" regime; dominates at intermediate stresses for fine-grained ice

**(c) Basal slip (rate-limited by non-basal slip):**
- n = 2.4, p = 0 (grain-size independent)
- Q = 60 kJ/mol
- Bridges GBS and dislocation creep

**(d) Diffusion creep (Nabarro-Herring + Coble):**
- n = 1 (Newtonian), p = 2 (Nabarro-Herring) or p = 3 (Coble)
- Theoretically dominant at very low stresses
- Experimentally inaccessible even at finest grain sizes (~3 micron)

**2025 review update** (PMC 11981940): Recommended three-component model from 70 years of lab data:

| Component | n | p | Q (kJ/mol) |
|-----------|---|---|------------|
| GSI (dislocation) | 3.6 | 0 | 62 |
| GSS1 (disGBS, low T) | 1.9 | 1.2 | 52 |
| GSS2 (disGBS, high T) | 2.5 | 1.9 | 182 |

### 3.6 Diffusion Creep Viscosity (Howell 2021 Formulation)

The composite Nabarro-Herring + Coble viscosity:
```
eta = (1/2) * [42 * Omega_m / (R * T * d^2) * (Dv + pi * delta / d * Db)]^(-1)
```

where:
- Omega_m = 1.97 x 10^-5 m^3/mol (molar volume, Fletcher 1970)
- d = grain size [m]
- delta = grain boundary width [m]
- Dv = D0v * exp(-Qv/(RT)) — volume diffusion coefficient
- Db = D0b * exp(-Qb/(RT)) — boundary diffusion coefficient
- Factor 42: geometric factor for combined N-H and Coble creep in polycrystalline aggregates
- Factor 1/2: converts uniaxial to shear viscosity

**Diffusion parameters** (Ramseier 1967b):

| Parameter | Symbol | Value | Uncertainty |
|-----------|--------|-------|-------------|
| Volume diffusion pre-factor | D0v | 9.1 x 10^-4 m^2/s | 3.3% |
| Volume diffusion activation energy | Qv | 59.4 kJ/mol | 5% |
| Boundary diffusion pre-factor | D0b | 8.4 x 10^-4 m^2/s | 3.3% |
| Boundary diffusion activation energy | Qb | 49.0 kJ/mol | 5% |
| Grain boundary width | delta | 7.13 x 10^-10 m | range: 5.22-9.04 x 10^-10 m |

Grain boundary width sources:
- Frost & Ashby (1982): 9.04 x 10^-10 m
- Hondoh (2019), 2x Burgers vector: 5.22 x 10^-10 m

**Frank-Kamenetskii approximation** (for simpler viscosity model):
```
eta(T) = eta_m * exp[(Qv/R) * (1/T - 1/T_m)]
```

This exponential approximation linearizes the Arrhenius law around T_m. Green et al. (2021) uses eta_ref = 5 x 10^13 Pa s; Howell (2021) uses 10^14.7 Pa s.

**FK vs full Arrhenius**: The FK approximation overestimates surface heat flux by approximately 30% (Stein et al. 2013), because it underestimates viscosity in the cold lid, making it slightly thinner and more thermally transmissive. This is a known, accepted limitation that should be documented.

### 3.7 Which Mechanism Dominates at Europa?

At Europa tidal stresses (~0.01-0.1 MPa) and convecting layer temperatures (~250-270 K), the GBS mechanism (n = 1.8) is theoretically rate-limiting. However, the project uses diffusion creep (n = 1, Newtonian) because:
1. Conservative (high) viscosity estimate
2. Most tractable for parameterized convection models
3. Theoretical diffusion creep regime not experimentally confirmed at relevant conditions
4. Composite law behavior at very low stresses remains uncertain

### 3.8 Grain Size in Ice Shells

**Barr & McKinnon (2007)** grain growth model:
```
d^n = d_0^n + k_gg * t * exp(-Q_gg / RT)
```

where n = 2 for normal grain growth, Q_gg ~ 48-52 kJ/mol.

**Equilibrium grain sizes for Europa:**

| Condition | Grain size | Notes |
|-----------|-----------|-------|
| Pure ice, no impurities | 1-100 mm | Grows unchecked |
| With Zener pinning (impurities) | 0.1-1 mm | Second-phase particles limit growth |
| During active tidal deformation | 0.1-0.5 mm | Dynamic recrystallization |
| Quiescent periods | ~10 mm | Growth between deformation events |

**2022 finding**: Grain growth is inhibited during grain-size-sensitive creep itself — an energy dissipation feedback that keeps grains smaller than static growth models predict.

The project's prior range of 0.05-3.0 mm (audited) is well-justified, covering the physically plausible range for tidally deformed ice with impurities.

### 3.9 Viscoelastic Rheology Models

#### Maxwell

**Complex compliance:**
```
J*(omega) = J_U - i/(omega * eta)
```

where J_U = 1/mu (unrelaxed compliance).

Single relaxation timescale tau_M = eta/mu. Peak dissipation at omega * tau_M = 1. Underestimates high-frequency dissipation.

#### Andrade

**Creep compliance (time domain):**
```
J(t) = J_U + beta * t^alpha + t/eta
```

**Complex compliance (frequency domain):**
```
J*(omega) = J_U + beta*Gamma(1+alpha)*(i*omega)^(-alpha) - i/(omega*eta)
```

Real and imaginary parts:
```
J_1 = J_U + beta*Gamma(1+alpha)*omega^(-alpha)*cos(alpha*pi/2)
J_2 = 1/(omega*eta) + beta*Gamma(1+alpha)*omega^(-alpha)*sin(alpha*pi/2)
```

**Parameters used in project:**
- alpha = 0.2 (Andrade exponent; literature range: 0.15-0.4)
- zeta = 1.0 (tau_Andrade / tau_Maxwell ratio)
- beta = J_U * tau_M^(-alpha) * zeta^(-alpha) (derived)

#### Burgers

Sum of Maxwell and Voigt-Kelvin elements:
```
J*(omega) = J_Maxwell + J_Voigt
```

Exhibits a secondary dissipation peak. Used less commonly than Andrade for tidal problems.

#### Extended Burgers / Sundberg-Cooper

Distribution of Voigt elements approximating Andrade power-law response. Equivalent to Andrade in the limit of many elements but allows time-domain ODE integration.

### 3.10 Porosity Effects

**Surface porosity**: Europa's near-surface ice is porous from impact gardening, sublimation/redeposition, and thermal stress fracturing. Juno MWR (2025) found exponential density increase from ~200 kg/m^3 at surface to 934 kg/m^3 at >2 m depth.

**Conductivity correction:**
```
k_eff = k_ice * (1 - phi)    for T < T_phi (porous zone)
k_eff = k_ice                for T >= T_phi (pores closed)
```

where phi = porosity fraction (0-0.3), T_phi = porosity curing temperature (~150 +/- 20 K).

**Porous fraction of conductive lid:**
```
f_phi = ln(T_phi/T_s) / ln(T_cond_base/T_s)
```

### 3.11 Salt Effects

**Melting point depression** (NaCl eutectic model):
```
T_m = T_m0 - dT_m * f_s / f_thresh
```
where T_m0 = 273 K, dT_m = 21 K (NaCl eutectic depression), f_thresh = 0.22.

**Conductivity modification:**
```
k_total = (1 - f_s) * k_ice + f_s * B_k * k_ice
```
where B_k is sampled log-uniformly from [0.1, 10] (Howell 2021).

**Salt concentrations on Europa**: Likely dominated by MgSO4 and NaCl, with total salt mass fractions of a few percent. Gonzalez Diaz et al. (2022, MNRAS) measured frozen salt solution conductivities and found consistent reductions.

### 3.12 Pressure-Dependent Melting

**Clausius-Clapeyron for ice Ih:**
```
dT_m/dP = T * Delta_V / L = -7.4 x 10^-8 K/Pa = -0.074 K/MPa
```

The negative slope (unique to water ice) means melting point decreases with pressure because ice is less dense than liquid water.

**Pressure at depth z on Europa:**
```
P(z) = rho * g * z = 917 * 1.315 * z = 1206 * z  [Pa]
```

| Depth (km) | Pressure (MPa) | T_m depression (K) | T_melt (K) |
|-----------|----------------|---------------------|------------|
| 10 | 12.1 | -0.9 | 272.1 |
| 20 | 24.1 | -1.8 | 271.2 |
| 30 | 36.2 | -2.7 | 270.3 |

**High-pressure ice phases**: Ice Ih stable to ~210 MPa (depth ~174 km). Since Europa's ice shell is ~20-25 km, high-pressure phases are not relevant.

### 3.13 Complete Parameter Table

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Thermal conductivity | k(T) | 567/T | W/(m K) | Howell 2021 |
| Specific heat | cp(T) | 7.49T + 90 | J/(kg K) | Giauque & Stout 1936 |
| Reference density | rho_0 | 917 | kg/m^3 | Standard |
| Thermal expansion | alpha | 1.6 x 10^-4 | K^-1 | Howell 2021 |
| Latent heat | L | 334,000 | J/kg | Standard |
| Molar volume | Omega_m | 1.97 x 10^-5 | m^3/mol | Fletcher 1970 |
| D0v | — | 9.1 x 10^-4 | m^2/s | Ramseier 1967 |
| D0b | — | 8.4 x 10^-4 | m^2/s | Ramseier 1967 |
| Qv | — | 59.4 | kJ/mol | Ramseier 1967 |
| Qb | — | 49.0 | kJ/mol | Ramseier 1967 |
| Grain boundary width | delta | 7.13 x 10^-10 | m | Frost & Ashby / Hondoh |
| Shear modulus | mu | 3.3 x 10^9 | Pa | Green et al. 2021 |
| Orbital frequency | omega | 2.047 x 10^-5 | rad/s | Standard |
| Eccentricity | e | 0.0101 | — | Standard |
| Surface gravity | g | 1.315 | m/s^2 | Standard |
| Europa radius | R | 1,561,000 | m | Standard |
| Clausius-Clapeyron | dTm/dP | -7.4 x 10^-8 | K/Pa | Standard |
| Stefan-Boltzmann | sigma | 5.67 x 10^-8 | W/(m^2 K^4) | Standard |
| Emissivity | epsilon | 0.94 | — | Standard |
| Andrade alpha | alpha_A | 0.2 | — | Project default |
| Andrade zeta | zeta | 1.0 | — | Project default |

---

## 4. Convection Scaling Laws Mathematics

### 4.1 The Rayleigh Number — First Principles Derivation

Starting from the Boussinesq equations:

**Momentum (Stokes):** `nabla . sigma_bar - nabla(P) = -alpha * rho * g * T * z_hat`

**Mass (incompressibility):** `nabla . v = 0`

**Energy:** `rho * cp * (dT/dt + v . nabla(T)) = k * nabla^2(T) + rho * H`

Non-dimensionalizing with length d, time d^2/kappa, temperature Delta_T yields the single controlling parameter:

```
Ra = (rho * g * alpha * Delta_T * d^3) / (kappa * eta)
```

**Physical meaning**: Ra measures the ratio of buoyancy-driven destabilizing forces to viscous and thermal-diffusive stabilizing forces. When Ra > Ra_crit, buoyancy overcomes dissipation and convection initiates.

### 4.2 Computing Ra for the Convecting Sublayer

In stagnant-lid convection, only the warm ductile sublayer participates:
- **Layer thickness**: d = D_conv = H_total - D_cond
- **Temperature contrast**: Delta_T = T_melt - T_c (where T_c = lid base temperature)
- **Viscosity**: eta evaluated at T_mean = (T_melt + T_c) / 2

The critical Ra = 1000 (Green et al. 2021 methodology). At Ra_crit, convection jumps to finite amplitude (Barr & Showman 2009) — there is no gradual onset.

### 4.3 Nusselt-Rayleigh Scaling — Derivation

The classical 1/3 scaling derives from the **marginal stability hypothesis** (Malkus 1954, Howard 1966):

**Step 1**: Define local Ra for thermal boundary layer of thickness delta:
```
Ra_delta = (rho * g * alpha * Delta_T_bl * delta^3) / (kappa * eta) = Ra_c  (critical)
```

**Step 2**: The boundary layer controls thermal resistance: Nu = d/delta

**Step 3**: Solving for delta and substituting:
```
delta = d * (Ra_c / Ra)^(1/3)
Nu = d/delta = (Ra / Ra_c)^(1/3) = C * Ra^(1/3)
```

**Key physical insight**: Nu ~ Ra^(1/3) means heat flux q = k * Delta_T * Nu / d becomes **independent of layer thickness d**. The d cancels because heat transport is controlled by boundary layer instabilities, not by the distance between boundaries.

### 4.4 Stagnant-Lid Nu-Ra Scaling (Solomatov & Moresi 2000)

For strongly temperature-dependent viscosity (stagnant lid regime):

```
Nu = C * Ra^(xi) * theta^(-zeta)
```

where:
- C = 0.3446 (pre-factor calibrated from numerical experiments)
- xi = 1/3 (Ra exponent)
- zeta = 4/3 (internal heating correction exponent)
- theta = E * Delta_T / (R * T_i^2) — Frank-Kamenetskii parameter

**The project implements this as:**
```
Nu = 0.3446 * Ra^(1/3) * ((T_i - T_c) / Delta_T)^(4/3)
```

The ((T_i - T_c) / Delta_T)^(4/3) factor is the **internal heating correction**. With volumetric tidal heating, the interior temperature T_i is elevated above what pure bottom heating produces. The 4/3 exponent comes from boundary layer stability analysis generalized to internally heated fluids.

### 4.5 Comparison of Scaling Laws

| Source | Nu formula | Regime |
|--------|-----------|--------|
| Solomatov & Moresi (2000) | `0.3446 * Ra^(1/3) * theta^(-4/3)` | Stagnant lid, internal heating, 2D Cartesian |
| Grasset & Parmentier (1998) | `Nu ~ Ra^(1/3)` with theta dependence | Stagnant lid, volumetric heating |
| Deschamps & Sotin (2001) | Modified with ice-specific parameters | Icy satellite shells |
| Deschamps & Vilella (2021) | Extended mixed-heating scaling | Mixed basal + internal heating |

### 4.6 The Viscous Temperature Scale

```
Delta_T_v = R * T_i^2 / E
```

This is the temperature interval over which viscosity changes by a factor of e (~2.72).

For Europa (E = 59.4 kJ/mol, T_i ~ 260 K, R = 8.314 J/(mol K)):
```
Delta_T_v = (8.314 * 260^2) / 59400 = 9.5 K
```

Over ~10 K, viscosity changes by a factor of e. Over ~25 K, it changes by ~15x, effectively shutting down convective flow. This is why the conductive lid is so thick relative to the total shell — only the warmest ~10 K of ice can participate in convection.

### 4.7 The Rheological Temperature T_c (Lid Base)

```
T_c = T_i - theta_lid * Delta_T_v = T_i - 2.24 * R * T_i^2 / E
```

where theta_lid = 2.24 is numerically calibrated (number of e-folding viscous temperature scales separating interior from lid base).

### 4.8 Interior Temperature T_i (Deschamps Eq. 18)

```
T_i = B * [sqrt(1 + (2/B)(T_m - c2 * DT_s)) - 1]
```

where:
- B = E / (2 * R * c1) with c1 = 1.43
- c2 = -0.03
- DT_s = T_m - T_s (total temperature drop across the shell)

The quadratic form arises from balancing tidal heat production (temperature-dependent through viscosity) against boundary-layer heat loss (dependent on T_i through Ra-Nu scaling).

**Physical constraints on T_i:**
- T_i > T_c (interior warmer than lid base)
- T_i <= T_m (cannot exceed melting point)
- T_i is typically within 10-20 K of T_m for Europa

### 4.9 The Green et al. (2021) Temperature Profile Scanning

Rather than using a fixed lid fraction, the algorithm scans the evolving temperature profile:

1. Compute T_c from rheology (Deschamps Eq. 18 for T_i, then T_c = T_i - 2.24 * Delta_T_v)
2. Scan temperature profile: find shallowest depth where T(z) >= T_c
3. Linearly interpolate between grid nodes for precise interface location
4. Compute D_conv = H - z_c and sublayer Ra, Nu

This adapts self-consistently as the shell evolves.

### 4.10 Effective Conductivity Approach

In the convective sublayer:
```
k_eff(z) = Nu * k_molecular(T(z))    for z >= z_c  (convective sublayer)
k_eff(z) = k_molecular(T(z))          for z < z_c   (conductive lid)
```

**Why this works:**
1. Correct integrated heat flux through the convecting layer
2. Produces nearly isothermal interior (large k -> flat T profile), matching physics
3. No boundary layer resolution needed — the integrated effect is captured

**Limitations:**
1. No lateral structure (inherently 1D)
2. Instantaneous mixing assumption
3. Sharp transition at z_c (reality has a gradual rheological sublayer)
4. FK overestimate (~30%)

### 4.11 Lid Fraction

For conductive temperature profile:
```
D_cond / H = (T_c - T_s) / (T_m - T_s)
```

With T_s = 104 K, T_m = 273 K, T_c ~ 250 K:
```
D_cond / H = (250 - 104) / (273 - 104) = 146/169 = 0.86
```

Approximately 86% of the shell is conductive, only 14% convects. The conductive lid dominates thermal resistance.

### 4.12 Thermal Equilibrium — Stefan Condition

**Energy balance at the ice-ocean interface:**
```
rho * L * dH/dt = q_conducted - q_ocean
```

**Steady-state conductive thickness** (no internal heating, constant k):
```
H_eq = k * (T_m - T_s) / q_ocean
```

For Europa (k ~ 3 W/(m K), Delta_T = 169 K, q_ocean ~ 11 mW/m^2):
```
H_eq = 3 * 169 / 0.011 = 46 km
```

**With temperature-dependent conductivity** k(T) = 567/T:
```
q = (567/H) * ln(T_m/T_s)
H_eq = 567 * ln(273/104) / q_ocean = 567 * 0.965 / 0.011 = 49.7 km
```

**With convection** (Nu ~ 5): Equilibrium shell thickness is substantially thinner because convective enhancement increases heat transport. Green et al. (2021) find 5-30 km depending on parameters.

### 4.13 The Complete Equation Chain

```
E, R, T_m, T_s  -->  T_i          [Deschamps Eq. 18]
                 -->  Delta_T_v    [= R*T_i^2/E]
                 -->  T_c          [= T_i - 2.24*Delta_T_v]
                 -->  D_cond       [from T profile scan at T_c]
                 -->  D_conv       [= H - D_cond]
                 -->  Delta_T      [= T_m - T_c]
                 -->  Ra           [= rho*g*alpha*Delta_T*D_conv^3/(kappa*eta)]
                 -->  Nu           [= 0.3446*Ra^(1/3)*((T_i-T_c)/Delta_T)^(4/3)]
                 -->  k_eff        [= Nu*k in convective sublayer]
                 -->  q_base       [= k_eff * dT/dz at base]
                 -->  dH/dt        [= (q_base - q_ocean)/(rho*L)]
                 -->  H(t) evolves, feeding back to all above
```

This chain shows how the activation energy E (a microscopic parameter of ice crystal diffusion) ultimately controls the macroscopic ice shell thickness through a cascade of scaling relations spanning 20+ orders of magnitude in spatial scale.

### 4.14 Key Dimensionless Groups

| Group | Definition | Typical value | Controls |
|-------|-----------|---------------|----------|
| Ra | rho*g*alpha*DT*d^3/(kappa*eta) | 10^4 - 10^8 | Convective vigor |
| Nu | q_total / q_cond | 1 - 10 | Heat transfer enhancement |
| theta | E*DT/(R*Ti^2) | 10 - 30 | Viscosity contrast |
| D_cond/H | Lid fraction | 0.80 - 0.95 | Shell structure |
| Stefan number | cp*DT/L | ~1 | Latent heat importance |
| Prandtl number | eta*cp/k | ~10^20 | Effectively infinite |

Infinite Prandtl number means momentum diffuses infinitely fast relative to heat — the flow field adjusts instantaneously to thermal buoyancy. This justifies the Stokes approximation.

---

## 5. Monte Carlo Uncertainty Quantification

### 5.1 Sampling Strategies

**Simple Random Sampling (SRS):**
- Convergence rate: O(1/sqrt(N))
- To halve error: need 4x more samples
- At N=100: ~11% MC error; N=1000: ~3.5%

**Latin Hypercube Sampling (LHS):**
- Stratified technique: divides each parameter's range into N equal-probability intervals
- Ensures exactly one sample per interval per dimension
- Convergence rate: O((ln N)^2 / N) — substantially faster than SRS
- Needs roughly half as many samples as SRS for equivalent precision
- Particularly effective for monotonic model responses

**Quasi-Monte Carlo (QMC) / Sobol Sequences:**
- Low-discrepancy sequences fill parameter space more uniformly
- Convergence approaches O(1/N) for well-behaved functions
- Best performance in higher dimensions

**Recommendation for this project**: LHS with N=250-1000 for distribution estimation. This is adequate for mean, standard deviation, and moderate percentiles (5th-95th).

### 5.2 How Many Samples Are Needed?

| Quantity | Minimum N | Recommended N |
|----------|-----------|---------------|
| Mean | 100-250 | 500+ |
| Standard deviation | 250-500 | 1000+ |
| 5th/95th percentiles | 500-1000 | 2000+ |
| 1st/99th percentiles | 2000-5000 | 10,000+ |
| Full PDF shape | 1000+ | 5000+ |
| Sobol indices | N*(d+2), N >= 2^12 | N >= 2^14 for d=7 |

**Howell & Pappalardo (2021) benchmark**: 10^7 samples (feasible because their forward model is algebraic, not PDE-based). For PDE-based solvers, N=250-1000 with LHS is the practical sweet spot.

### 5.3 Convergence Assessment

From EPA Guiding Principles for Monte Carlo Analysis:
- **Running-mean stability**: Plot running mean, std, and key percentiles vs N. Converged when changes < 0.1% of current estimate.
- **Replicate stability**: Run multiple independent MC batches and compare statistics.
- **Record random seeds** for reproducibility.
- **Tail sensitivity**: If output doesn't stabilize, scrutinize input distribution tails.

### 5.4 Sensitivity Analysis — Sobol Indices

**Variance decomposition (ANOVA-HDMR):**
```
Var(Y) = Sum_i V_i + Sum_{i<j} V_ij + ... + V_{1,2,...,d}
```

**First-order Sobol index:**
```
S_i = Var_{Xi}[E_{X~i}(Y | Xi)] / Var(Y)
```
Measures fraction of output variance from Xi alone.

**Total-order Sobol index:**
```
ST_i = E_{X~i}[Var_{Xi}(Y | X~i)] / Var(Y)
```
Captures all variance involving Xi, including interactions.

**Interpretation:**
- S_i approximately equal to ST_i: parameter acts mainly through main effect
- ST_i >> S_i: influence primarily through interactions
- Sum S_i approximately equal to 1: model is approximately additive
- Sum S_i << 1: strong interaction effects dominate

**Computational cost** (Saltelli 2010): N * (d + 2) evaluations.
For d=7 parameters with N=4096: 4096 * 9 = 36,864 evaluations.

### 5.5 Morris Screening Method

Elementary effect for parameter k:
```
EE_k = [f(x1,...,xk+Delta,...,xd) - f(x1,...,xk,...,xd)] / Delta
```

Summary statistics over r trajectories:
- mu_k = mean elementary effect (can cancel for non-monotonic models)
- mu*_k = mean absolute elementary effect (overall influence)
- sigma_k = standard deviation (interaction/nonlinearity indicator)

**Interpretation (mu* vs sigma plot):**
- Near origin: negligible influence — screen out
- High mu*, low sigma: important linear/additive effect
- High mu*, high sigma: important with interactions/nonlinearity
- Low mu*, high sigma: weak main effect but involved in interactions

**Cost**: r * (d + 1) evaluations. For d=7, r=20: 160 evaluations.

**Recommendation**: Use Morris first as pre-screening (~160 evaluations), then Sobol indices for the top 3-4 parameters if budget allows.

### 5.6 Prior Distribution Strategies

Following Howell & Pappalardo (2021):

| Parameter type | Recommended prior | Example |
|----------------|-------------------|---------|
| Well-constrained | Normal/truncated normal | Surface temperature (104 +/- 7 K) |
| Order-of-magnitude uncertainty | Lognormal or loguniform | Grain size (10^-3 m, +/- 1 order) |
| Unknown over bounded range | Uniform | Porosity (0-0.3) |
| Positive with large spread | Lognormal | Salt fraction |
| Compositional fractions | Beta or truncated normal | Bounded on [0,1] |

**Important**: Uninformative priors are not "safe" — uniform on linear scale vs log scale implies very different beliefs. For parameters spanning orders of magnitude, loguniform (Jeffreys) prior assigns equal probability to each decade.

### 5.7 Bayesian Model Comparison

**Bayes factors** for comparing model structures:
```
BF_12 = P(data | Model 1) / P(data | Model 2) = Z_1 / Z_2
```

**Jeffreys' scale:**
- BF < 1: evidence favors Model 2
- 1-3: anecdotal
- 3-10: moderate
- 10-30: strong
- 30-100: very strong
- >100: decisive

Bayes factors naturally penalize complexity (Occam's razor). Useful for comparing pure-ice vs salty-ice models, or conduction-only vs convection models.

### 5.8 Uncertainty Propagation Through Nonlinear Models

**Nonlinearity effects:**
- Symmetric inputs -> asymmetric, skewed, or multimodal outputs
- Viscosity depends exponentially on T (Arrhenius)
- Equilibrium thickness determined by nonlinear heat balance

**Howell (2021) result**: Thickness PDF has CBE = 24.3 km with asymmetric uncertainties -1.5/+22.8 km — strongly right-skewed from the lognormal grain size input.

**Reporting for non-Gaussian outputs:**
- **Mode** (peak of PDF): current best estimate (CBE)
- **Median** (50th percentile): robust central tendency
- **Asymmetric credible intervals**: [5th, 95th] percentiles, not symmetric +/- sigma
- **Skewness**: gamma_1 = E[(X-mu)^3] / sigma^3

**Credible intervals (Bayesian)**: "90% probability that truth lies in [X, Y] given model and priors."
- Different from confidence intervals (frequentist procedure statement)
- **Highest Density Interval (HDI)**: shortest interval containing specified probability mass — preferred for skewed distributions

### 5.9 Parameter Correlation

**Default**: Independent sampling when correlation structure is unknown (conservative — produces wider uncertainty bounds). This is what Howell (2021) does, and it is defensible.

**Methods if correlations are known:**
- **Cholesky decomposition**: For multivariate normal; x = mu + L * z where Sigma = L * L^T
- **Iman-Conover method**: Induces rank-correlation in existing LHS without changing marginals
- **Copula methods**: Separate dependence structure from marginals (Sklar's theorem)

**Practical recommendation**: Sample independently, document the assumption, test sensitivity to plausible correlations (rho = +/-0.5 between related parameters).

### 5.10 Publication Standards for MC Results

**Mandatory reporting:**
1. All input parameter distributions (type, parameters, bounds, justification)
2. Sampling method and sample size
3. Convergence evidence
4. Summary statistics: mean, median, mode, std, skewness
5. Credible intervals: [5th, 50th, 95th] at minimum; ideally [2.5th, 16th, 50th, 84th, 97.5th]
6. Random seed / reproducibility information
7. Parameter correlation assumptions

**Desirable:**
8. Sensitivity analysis results
9. Full PDF or CDF
10. Convergence diagnostic plots

---

## 6. Scientific Visualization & Publication Figures

### 6.1 Resolution and Format Requirements

| Content type | Resolution | Preferred format |
|-------------|-----------|-----------------|
| Line art (graphs, plots) | 600-1200 DPI or vector | PDF, EPS, SVG |
| Photos / raster images | 300-600 DPI | TIFF, PNG |
| Combination figures | 300+ DPI | PDF |

**Never use JPEG** for scientific plots (creates compression artifacts).

### 6.2 Color Palettes

**Recommended for this project:**

| Data type | Colormap | Package |
|----------|---------|---------|
| Temperature profiles | `cmocean.thermal` | cmocean |
| General sequential data | `viridis`, `cividis` | matplotlib |
| Diverging data (anomalies) | `cmocean.balance`, `RdBu_r` | cmocean/mpl |
| Categorical comparisons | Okabe-Ito palette | Manual |

**Okabe-Ito colorblind-safe palette:**
```python
okabe_ito = ['#E69F00', '#56B4E9', '#009E73', '#F0E442',
             '#0072B2', '#D55E00', '#CC79A7', '#000000']
```

**Never use**: jet, rainbow, red-green color schemes.

**Always test** figures in grayscale for interpretability.

### 6.3 Typography

- **Font**: Sans-serif (Arial, Helvetica, Calibri)
- **Minimum sizes at final print size**:
  - Axis labels: 7-9 pt
  - Tick labels: 6-8 pt
  - Panel labels: 8-12 pt bold
- **Sentence case** for labels: "Temperature (K)" not "TEMPERATURE (K)"
- **Always include units** in parentheses

### 6.4 Figure Dimensions

| Journal | Single column | Double column |
|---------|--------------|---------------|
| Nature/Nature Geo | 89 mm | 183 mm |
| Science | 55 mm | 175 mm |
| Cell | 85 mm | 178 mm |
| JGR/AGU | Flexible | — |

### 6.5 Multi-Panel Figures

- Label panels with **bold lowercase letters**: **(a)**, **(b)**, **(c)** (Nature convention) or uppercase **A**, **B**, **C** (most other journals)
- Labels in upper-left corner of each panel
- Consistent styling across all panels
- Shared colorbars at bottom for related panels
- Adequate white space (hspace=0.4, wspace=0.4 in GridSpec)

### 6.6 Depth Profile Conventions

- **Y-axis inverted** (depth increasing downward) — standard geophysics convention
- Temperature on x-axis, depth on y-axis
- Conductive-convective boundary marked with horizontal dashed line
- Surface temperature and melting point labeled

### 6.7 Uncertainty Visualization

**For MC distributions:**

| Plot type | Shows | Best for |
|-----------|-------|----------|
| Violin plot | Full distribution shape + summary stats | Comparing scenarios |
| Histogram + KDE | Raw distribution with smooth estimate | Single distribution |
| CDF (empirical) | Cumulative probability | Reading percentiles |
| Corner plot | Pairwise parameter correlations | Joint posterior structure |
| Tornado / bar chart | Sensitivity ranking | Sobol/Morris results |
| Running-mean plot | Convergence | Supplementary material |

**For thickness profiles with uncertainty:**
```python
ax.fill_between(latitude, H_5th, H_95th, alpha=0.3, label='90% CI')
ax.fill_between(latitude, H_25th, H_75th, alpha=0.5, label='50% CI')
ax.plot(latitude, H_median, 'k-', linewidth=2, label='Median')
```

### 6.8 Statistical Rigor in Figures

Always include:
- Error bars (SD, SEM, or CI — specify which in caption)
- Sample size (n) in figure or caption
- Significance markers (*, **, ***) if applicable
- Individual data points when possible (not just summary statistics)

### 6.9 Matplotlib Style Configuration

```python
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.format'] = 'pdf'
```

### 6.10 Publication Checklist

- [ ] Resolution >= 300 DPI (600+ for line art)
- [ ] Vector format for plots (PDF/EPS)
- [ ] Figure size matches journal specs
- [ ] All text readable at final size (>= 6 pt)
- [ ] Colorblind-friendly palette
- [ ] Works in grayscale
- [ ] All axes labeled with units
- [ ] Error bars present with definition in caption
- [ ] Panel labels consistent
- [ ] No chart junk / 3D effects
- [ ] Fonts consistent across all figures
- [ ] Legend clear and complete

---

## 7. Academic Paper Writing Standards

### 7.1 Journal Requirements

**JGR: Planets (AGU):**
- Abstract: single paragraph, < 250 words
- Key Points: up to 3 bullet points, <= 140 characters each
- Plain Language Summary: required, single paragraph, max 200 words, no jargon
- Length: up to 25 publication units (1 PU = 500 words OR 1 figure OR 1 table)
- Open Research section required (data availability)
- Citation format: Author-date inline

**Icarus (Elsevier):**
- Abstract: concise, single paragraph, typically 150-250 words
- Highlights: mandatory, 3-5 bullet points, max 85 characters each
- Letters: limited to 3,500 words, up to 3 figures
- LaTeX: Elsevier article class, BibTeX

**Planetary Science Journal (PSJ, AAS):**
- Abstract: 250-word limit
- Line numbers required
- Dual Anonymous Review (DAR) by default
- Gold Open Access (CC-BY)
- LaTeX: AASTeX macro package

### 7.2 Standard Paper Structure

Based on Deschamps & Vilella (2021), Howell (2021), Green et al. (2021):

1. **Introduction** (1.5-3 pages)
   - Broad scientific context (Europa, icy moons, convection)
   - Literature review (chronological or thematic)
   - Gap statement identifying what has not been addressed
   - Brief statement of approach and outline

2. **Model Description / Numerical Methods** (2-4 pages)
   - Governing equations
   - Rheological formulation (viscosity law)
   - Nondimensionalization (if applicable)
   - Boundary conditions
   - Grid resolution and geometry
   - Numerical solver details
   - Monte Carlo sampling procedure

3. **Results** (often subdivided)
   - Thermal Structure / Temperature Profiles
   - Thickness Predictions
   - Parameter Sensitivity
   - Comparison of Scenarios

4. **Application to Europa** (or combined with Discussion)
   - Physical parameter values
   - Model predictions
   - Comparison with prior estimates

5. **Discussion**
   - Interpretation
   - Comparison with other models
   - Limitations and caveats

6. **Conclusions** (0.5-1 page)
   - Summary of main findings
   - Implications for Europa Clipper
   - Future directions (brief)

7. **Open Research / Data Availability**
8. **Acknowledgments**
9. **References** (typically 50-60 for a full modeling paper)

### 7.3 Mathematical Notation Conventions

**Standard symbols in geophysical heat transfer papers:**

| Symbol | Meaning | Notes |
|--------|---------|-------|
| T | Temperature | Dimensional |
| T-tilde | Nondimensional temperature | T-tilde = (T - T_s) / Delta_T |
| P | Non-hydrostatic pressure | |
| v | Velocity vector | |
| eta | Dynamic viscosity | Pa s |
| rho | Density | kg/m^3 |
| k | Thermal conductivity | W/(m K) |
| cp or C_p | Specific heat | J/(kg K) |
| alpha | Thermal expansion | K^-1 |
| kappa | Thermal diffusivity | m^2/s |
| g | Gravity | m/s^2 |
| D | Layer thickness | m |
| Delta_T | Temperature contrast | K |
| H | Internal heating rate | W/kg or W/m^3 |
| Ra | Rayleigh number | |
| Nu | Nusselt number | |
| Pr | Prandtl number | |
| Delta_eta | Viscosity ratio | top-to-bottom |

**Nondimensionalization convention** (Deschamps & Vilella 2021):
- Tildes for nondimensional: `T-tilde`, `Phi-tilde`
- Length scale: D (layer thickness)
- Time scale: D^2/kappa
- Temperature scale: Delta_T

### 7.4 How to Introduce Equations

Standard patterns from published papers:

1. **Physical context first**, then equation:
   > "The conservation equations of momentum, mass, and energy are then [Eqs. 1-3]"

2. **Define all symbols immediately after**:
   > "where sigma_bar is the deviatoric stress tensor, P is the non-hydrostatic pressure, v is the velocity, T is the temperature..."

3. **Explain physical meaning of dimensionless numbers**:
   > "This number measures the ratio between buoyancy and viscous forces"

4. **Reference equations by number** in subsequent text:
   > "Following Equations 6 and 8, Ra_eff is given by [Eq. 10]"

5. **Motivate before showing**: Never present an equation cold.

### 7.5 Writing Style

**Active voice is now standard in AGU journals:**
- "We performed numerical simulations..."
- "We modeled this dependency using the Frank-Kamenetskii approximation."
- "Our calculations suggest a shell thickness in the range 20-80 km."

**Mixed voice for methods:**
- Active for choices: "We used the Howell (2021) conductivity model."
- Passive for procedures: "Conservation equations are nondimensionalized with..."

**Citation density:**
- Introduction: highest (nearly every claim cited, multiple citations per sentence common)
- Methods: moderate (code, rheological laws, benchmarks)
- Results: lower (mostly self-referential to equations/figures)
- Discussion: moderate to high (comparisons with published estimates)

### 7.6 Discussing Limitations

- Position at the beginning of Discussion (to frame interpretation) or near the end
- **Acknowledge explicitly**: "This approximation overestimates the surface heat flux by up to 30%"
- **Balance with justification**: "Nevertheless, it facilitates the calculations and allows capturing the role of one given specific parameter"
- **Quantify where possible**: state the magnitude of the effect
- **Connect to future work**: each limitation points to a natural next step

### 7.7 Presenting MC Results (Howell 2021 Standard)

**Methodology presentation:**
- Parameter table listing: name, CBE, distribution type, distribution parameters
- Sampling procedure clearly described
- Number of iterations stated

**Results presentation:**
- **Asymmetric uncertainties**: CBE values as `24.3_{-1.5}^{+22.8} km`
- **Probability density distributions**: histograms with CDF overlay
- **Conditional distributions**: separate panels for sub-components
- **Sensitivity analysis**: Spearman rank correlation coefficients

**Key insight from Howell (2021)**: "CBE layer thicknesses do not sum to CBE total thickness" because full models exhibiting both CBE values simultaneously are statistically rare. Always show the full distribution, not just point estimates.

### 7.8 Model Validation Strategies

**Analytical benchmarks:**
- Conductive steady-state: T(z) linear for constant k, logarithmic for k(T) = a/T
- Critical Rayleigh number comparison
- With internal heating: quadratic correction to conductive profile

**Convergence testing:**
- Multiple grid resolutions (demonstrate key outputs don't change)
- Time-averaging over oscillation periods
- Grid refinement at boundaries for boundary layer resolution

**Comparison with previous models:**
- Nu-Ra scaling exponents against published values
- Cross-code benchmarking (Green et al. 2021 validated against ASPECT)
- Chi-square goodness-of-fit for scaling law fits

**Observational validation:**
- Impact crater morphology constrains H > 20 km with 6-8 km conductive lid
- Magnetic field constraints on ocean existence
- Surface geology (tidal flexure, ridges, chaos terrain)
- Juno MWR data on surface porosity structure

### 7.9 Template Papers

| Paper | Journal | Use as template for |
|-------|---------|---------------------|
| Deschamps & Vilella (2021) | JGR:Planets | Equation presentation, scaling laws, application to Europa |
| Howell (2021) | PSJ | MC uncertainty, parameter tables, probability distributions |
| Green et al. (2021) | JGR:Planets | Convection + validation approach |
| Shibley & Goodman (2024) | Icarus | Concise modeling paper (16pp/4 figs) |
| Chen & Deschamps (2026) | JGR:Planets | Most recent, parameterized convection |
| Harel et al. (2020) | Icarus | FK vs Arrhenius comparison |

---

## 8. Master Reference List

### Europa Ice Shell & Geophysics
- Billings & Kattenhorn (2005). "The great thickness debate: Ice shell thickness models for Europa and comparisons with estimates based on flexure at ridges." Icarus 177.
- Hussmann, Spohn & Wieczerkowski (2002). "Thermal equilibrium of Europa's ice shell." Icarus 156.
- Nimmo, Giese & Pappalardo (2003). "Estimates of Europa's ice shell thickness from elastically-supported topography." GRL 30.
- Howell (2021). "The Likely Thickness of Europa's Icy Shell." PSJ 2:129.
- Howell & Pappalardo (2021). Companion to Howell 2021.
- Ojakangas & Stevenson (1989). "Thermal state of an ice shell on Europa." Icarus 81.

### Tidal Dissipation & Rheology
- Tobie, Choblet & Sotin (2003). "Tidally heated convection: Constraints on Europa's ice shell thickness." JGR 108.
- McCarthy, Takei & Hiraga (2011). "Experimental study of attenuation and dispersion over a broad frequency range." JGR 116.
- Renaud & Henning (2018). "Increased tidal dissipation using advanced rheological models." ApJ 857.
- Bierson (2024). "Impact of rheology model choices on tidal heating." Icarus.
- Andrade (1910). "On the viscous flow in metals." Proc. R. Soc. London A 84.

### Convection Scaling Laws
- Solomatov & Moresi (2000). "Scaling of time-dependent stagnant lid convection." JGR 105.
- Grasset & Parmentier (1998). "Thermal convection in a volumetrically heated, infinite Prandtl number fluid." JGR 103.
- Deschamps & Sotin (2001). "Thermal convection in the outer shell of large icy satellites." JGR 106.
- Deschamps & Vilella (2021). "Scaling Laws for Mixed-Heated Stagnant-Lid Convection and Application to Europa." JGR:Planets 126.
- Green et al. (2021). "The Growth of Europa's Icy Shell: Convection and Crystallization." JGR:Planets 126.
- Solomatov (1995). "Scaling of temperature- and stress-dependent viscosity convection." Phys. Fluids 7.
- Malkus (1954). "The heat transport and spectrum of thermal turbulence." Proc. R. Soc. London A 225.
- Howard (1966). "Convection at high Rayleigh number." Applied Mechanics, Proc. 11th Congress.
- Barr & Showman (2009). "Heat transfer in Europa's icy shell." In Europa (UAP).
- Stein, Lowman & Hansen (2013). "The influence of mantle internal heating on lithospheric mobility." EPSL 361.
- Chen & Deschamps (2026). "Temporal Changes in Europa's Ice Shell." JGR:Planets.
- Harel, Olson & Finkel (2020). "Scaling of heat transfer in stagnant lid convection for application to icy moons." Icarus 338.

### Ice Material Properties
- Carnahan, Wolfenbarger, Jordan & Hesse (2021). "New insights into temperature-dependent ice properties." EPSL 563.
- Goldsby & Kohlstedt (2001). "Superplastic deformation of ice." JGR 106.
- Barr & McKinnon (2007). "Convection in ice I shells and mantles with self-consistent grain size." JGR 112.
- Klinger (1975). "Low-temperature heat conduction in pure monocrystalline ice." J. Glaciology 14.
- Slack (1980). "Thermal conductivity of ice." Phys. Rev. B 22.
- Ramseier (1967). "Self-diffusion of tritiated water in natural and synthetic ice monocrystals." J. Applied Physics 38.
- Giauque & Stout (1936). "The entropy of water." JACS 58.
- Feistel & Wagner (2006). "A new equation of state for H2O Ice Ih." J. Phys. Chem. Ref. Data 35.
- Fletcher (1970). "The Chemical Physics of Ice." Cambridge University Press.
- Frost & Ashby (1982). "Deformation-Mechanism Maps." Pergamon Press.
- Gonzalez Diaz et al. (2022). "Thermal conductivity of frozen salt solutions." MNRAS 510.
- Johnson et al. (2017). "Porosity and salt content determine if subduction can occur." JGR:Planets 122.
- Wagner et al. (2011). "New equations for sublimation and melting pressure of H2O Ice Ih." J. Phys. Chem. Ref. Data 40.

### Ocean Transport
- Soderlund et al. (2014). "Ocean-driven heating of Europa's icy shell at low latitudes." Nature Geoscience 7.
- Lemasquerier et al. (2023). Ocean dynamics modeling with polar-enhanced patterns.

### Numerical Methods
- Rannacher (1984). "Finite element solution of diffusion problems with irregular data." Numer. Math. 43.
- Crank & Nicolson (1947). "A practical method for numerical evaluation of solutions of partial differential equations." Math. Proc. Cambridge Phil. Soc. 43.

### Monte Carlo & Sensitivity Analysis
- Sambridge & Mosegaard (2002). "Monte Carlo Methods in Geophysical Inverse Problems." Reviews of Geophysics 40.
- Sobol (1993). "Sensitivity estimates for nonlinear mathematical models." Math. Model. Comput. Exp. 1.
- Saltelli et al. (2010). "Variance based sensitivity analysis of model output." Computer Physics Comm. 181.
- Morris (1991). "Factorial Sampling Plans for Preliminary Computational Experiments." Technometrics 33.
- McKay, Beckman & Conover (1979). "A comparison of three methods for selecting values of input variables." Technometrics 21.
- Iman & Conover (1982). "A distribution-free approach to inducing rank correlation among input variables." Comm. Statistics B 11.
- EPA (1997). "Guiding Principles for Monte Carlo Analysis." EPA/630/R-97/001.
- Trotta (2008). "Bayes in the sky: Bayesian inference and model selection in cosmology." Contemporary Physics 49.
- Foreman-Mackey (2016). "corner.py: Scatterplot matrices in Python." JOSS 1.
- Urrego-Blanco et al. (2016). "UQ and Global Sensitivity of the LANL Sea Ice Model." JGR:Oceans 121.

### Publication Standards
- AGU Text & Graphics Requirements: https://www.agu.org/publications/authors/journals/text-graphics-requirements
- Icarus Guide for Authors: https://www.sciencedirect.com/journal/icarus/publish/guide-for-authors
- PSJ Policies: https://journals.aas.org/psj-policies/
- cmocean colormaps: https://matplotlib.org/cmocean/
- SciencePlots: https://github.com/garrettj403/SciencePlots

### Europa Clipper
- REASON (Radar for Europa Assessment and Sounding: Ocean to Near-surface) instrument documentation
- Arrival ~2030-2031; dual-frequency radar (9 MHz HF, 60 MHz VHF)
- Penetration up to ~30 km in pure ice

---

*This document is a living reference for the EuropaConvection project. It should be updated as new literature, mission data, or methodological improvements become available.*
