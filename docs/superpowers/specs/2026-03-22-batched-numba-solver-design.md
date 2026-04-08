# Batched Numba 1D Ice Shell Solver — Design Spec

**Date**: 2026-03-22
**Status**: Draft
**Scope**: Green-path parity — `USE_GREEN_METHOD=True` convection, Andrade/Maxwell tidal heating, fixed-temperature Dirichlet BC

---

## 1. Purpose

Refactor the existing `Solver.py` 1D transient finite-difference ice-shell solver into a
latitude-batched solver that advances `n_lat` independent vertical columns simultaneously
using NumPy arrays and Numba `@njit`.

There is no lateral conduction. Each latitude column is independent. The solver preserves
the exact numerical structure of the current `Solver.py`, including:
- Both `use_convection=True` and `use_convection=False` paths
- Both Maxwell and Andrade tidal heating
- Composite and simple viscosity models
- Phase 2 temperature-profile based convection (Green et al. 2021)

**Only fixed-temperature surface BC** is supported in the JIT kernel.
StefanBoltzmann radiative BC is deferred to a future extension.

---

## 2. Dependency Map

### 2.1 JIT Helper Functions — Thermophysical Properties

| Original | JIT replacement | Formula | Source |
|---|---|---|---|
| `Thermal.conductivity(T)` | `_conductivity(T)` | `567.0 / T` | constants.py:55 |
| `Thermal.specific_heat(T)` | `_specific_heat(T)` | `7.49 * T + 90.0` | constants.py:70 |
| `Thermal.density_ice(T)` | `_density_ice(T)` | `917.0 * (1 + 1.6e-4 * (273.0 - T))` | constants.py:85 |
| `IcePhysics.effective_conductivity(...)` | `_effective_conductivity(T, por, salt, B_k, T_phi)` | k=567/T with porosity/salt corrections | Physics.py:109 |
| `IcePhysics.basal_melting_point(H)` | `_basal_melting_point(H, g)` | `273.0 + (-7.4e-8) * 917.0 * g * H` | Physics.py:295 |
| `IcePhysics.stefan_velocity(...)` | `_stefan_velocity(...)` | 2nd-order one-sided gradient, `rho(T_base)` T-dependent. `k_basal = _effective_conductivity(T[-1])` with **default** (zero) porosity/salt corrections — i.e. bare `567/T`. The original Solver.py:441 calls `IcePhysics.effective_conductivity(self.T[-1])` without passing the column's porosity/salt params. | Physics.py:331 |

### 2.2 JIT Helper Functions — Viscosity & Tidal Heating

| Original | JIT replacement | Formula | Source |
|---|---|---|---|
| `IcePhysics.viscosity_simple(T, eta_ref)` | `_viscosity_simple(T, eta_ref, Q_v, R, T_melt)` | `eta_ref * exp((Q_v/R)*(1/T - 1/T_melt))` | constants.py:118 |
| `IcePhysics.composite_viscosity(...)` | `_composite_viscosity(T, d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R)` | Howell (2021) diffusion creep. `eta = 0.5 * (42*d_m/(R*T*d^2) * (Dv + pi*delta/d * Db))^(-1)`, clipped [1e12, 1e25]. `Dv = D0v*exp(-Qv/RT)`, `Db = D0b*exp(-Qb/RT)` | Physics.py:36 |
| `IcePhysics.tidal_heating(...)` Maxwell | `_tidal_heating_maxwell(eps0, omega, eta, mu)` | `(eps0^2 * omega^2 * eta) / (2 * (1 + (omega*eta/mu)^2))` | Physics.py:282 |
| `IcePhysics.tidal_heating(...)` Andrade | `_tidal_heating_andrade(eps0, omega, eta, mu, alpha, zeta, gamma_val)` | Complex compliance J*(w). `gamma_val = gamma(1+alpha)` pre-computed by wrapper. | Physics.py:254 |

**Andrade SciPy dependency**: `scipy.special.gamma(1 + alpha)` is called once in the Python
wrapper with the Andrade alpha parameter (a constant, typically 0.2). The resulting scalar
`gamma_val` is passed into the JIT kernel. No SciPy is used inside `@njit`.

### 2.3 JIT Helper Functions — Convection (Phase 2)

| Original | JIT replacement | Notes |
|---|---|---|
| `IceConvection.deschamps_interior_temp(T_melt, T_surf, Q_v)` | `_deschamps_Ti(T_melt, T_surf, Q_v, R, c1, c2)` | Pure math, Eq. 18 |
| `IceConvection.green_cond_base_temp(T_melt, T_surf, Q_v, eta_ref)` | `_green_Tc_Ti(T_melt, T_surf, Q_v, R, eta_ref, theta_lid)` | Returns (Tc, Ti). Includes Howell fallback. |
| `IceConvection.howell_cond_base_temp(T_melt, Q_v)` | `_howell_Tc(T_melt, Q_v, R)` | Fallback for Green overflow |
| `IceConvection.scan_temperature_profile(...)` | `_scan_profile(T_col, nz, H, T_melt, T_surf, Q_v, Q_b, d_grain, ...)` | Returns scalars: (idx_c, z_c, D_cond, D_conv, T_c, Ra, Nu, is_convecting) |
| `IceConvection.rayleigh_number(...)` | `_rayleigh_number(DT, d, T_mean, eta, g, alpha_exp)` | `Ra = rho*g*alpha*DT*d^3 / (kappa*eta)` |
| `IceConvection.nusselt_number_green(Ra, Ti, Tc, DT)` | `_nusselt_green(Ra, Ti, Tc, DT, Ra_crit, C, xi, zeta)` | Returns max(Nu, 1.0) |
| `IceConvection.nusselt_number(Ra)` | `_nusselt_simple(Ra, Ra_crit, C, xi)` | Solomatov & Moresi |
| `IceConvection.build_conductivity_profile(...)` | `_build_k_profile(T_col, nz, H, ...)` | Fills k_impl in-place, returns convection scalars |
| `IceConvection.harmonic_mean_vectorized(k)` | `_harmonic_mean(k, k_half, nz)` | `k_half[j] = 2*k[j]*k[j+1] / (k[j]+k[j+1]+1e-30)` |

**ConvectionState replacement**: The dataclass `ConvectionState` is replaced with 8 scalar
return values from `_scan_profile`. These are stored into per-column output arrays by the
batched driver.

### 2.4 Three Temperature Roles (must not be conflated)

- `T_melt_const = 273.0` — used **only** by `_viscosity_simple` as the reference melt
  point for the Frank-Kamenetskii viscosity law. Never used for convection or boundaries.
- `T_base_dirichlet = _basal_melting_point(H, g)` — pressure-dependent, used for:
  1. The basal Dirichlet boundary row (`rhs[nz-1]`)
  2. The convection scan: passed to `_scan_profile`, `_green_Tc_Ti`, `_deschamps_Ti`,
     and `_rayleigh_number` as the basal melt temperature. This matches Solver.py:225
     where `IcePhysics.basal_melting_point(self.H)` is passed to the convection code.
  Recomputed per half-step (when H changes), NOT per Picard iteration.
- Within `_green_Tc_Ti`: the viscosity temperature scale uses `Q_v / (N * R * T_melt)`
  where `T_melt` is `T_base_dirichlet` (pressure-dependent), matching Convection.py:163.

### 2.5 Solver path switch

Each column has a `use_convection` boolean flag:

**When `use_convection=False`**:
- Conductivity: `_effective_conductivity()` (no Nu enhancement)
- Half-node: arithmetic mean `k_half[j] = 0.5*(k[j]+k[j+1])`
- Tidal heating viscosity: `_viscosity_simple()` (Frank-Kamenetskii)
- No profile scanning, no convection diagnostics

**When `use_convection=True`**:
- Conductivity: `_build_k_profile()` which calls `_scan_profile()`, applies Nu enhancement
  below z_c, then applies porosity/salt corrections
- Half-node: harmonic mean `k_half[j] = 2*k[j]*k[j+1]/(k[j]+k[j+1]+1e-30)`
- Tidal heating viscosity: `_composite_viscosity()` (Howell diffusion creep)
- Convection diagnostics returned as arrays

This matches the original Solver.py:218-283 branch.

### 2.6 Tidal heating rheology dispatch

Each column has a `rheology_model` int flag:
- `0` = Maxwell: `_tidal_heating_maxwell(eps0, omega, eta, mu)`
- `1` = Andrade: `_tidal_heating_andrade(eps0, omega, eta, mu, alpha, zeta, gamma_val)`

The original dispatches on `Rheology.MODEL` at Physics.py:252. The batched kernel accepts
this as a per-column flag so mixed-rheology ensembles are possible.

### 2.7 What is NOT ported

- StefanBoltzmann radiative surface BC — requires Newton-Raphson iteration per step
- Non-Green convection methods (`USE_GREEN_METHOD=False` Howell path) — historical fallback, not actively used
- `find_convective_boundary()` search — replaced by Phase 2 profile scanning

---

## 3. Indexing Conventions

### 3.1 Node indexing

- `i` is always a physical node index: `i = 0..nz-1`
- Interior nodes: `i = 1..nz-2`
- Surface: `i = 0`, Base: `i = nz-1`

### 3.2 Half-node indexing

- `j` is a half-node index: `j = 0..nz-2`
- `j` sits between nodes `j` and `j+1`
- Arithmetic mean: `k_half[j] = 0.5 * (k[j] + k[j+1])` (non-convective)
- Harmonic mean: `k_half[j] = 2*k[j]*k[j+1] / (k[j]+k[j+1]+1e-30)` (convective)
- `G_half[j] = 0.5 * (G[j] + G[j+1])` (always arithmetic for geometry)

### 3.3 Stencil for interior node `i`

```
factor_plus  = k_half[i]   * G_half[i]   / (G[i] * dz^2)
factor_minus = k_half[i-1] * G_half[i-1] / (G[i] * dz^2)
```

This matches Solver.py:291 where `factor_plus = k_plus[1:] * G_plus[1:]` uses the
half-node array shifted by 1 relative to the interior slice.

### 3.4 Tridiagonal storage contract

All arrays are length `nz`:

- `diag_lower[k]` couples row `k+1` to row `k`, for `k = 0..nz-2`.
  `diag_lower[nz-1]` is unused sentinel (always 0).
- `diag_main[k]` is the diagonal entry for row `k`, for `k = 0..nz-1`.
- `diag_upper[k]` couples row `k` to row `k+1`, for `k = 0..nz-2`.
  `diag_upper[nz-1]` is unused sentinel (always 0).

Assembly writes `diag_lower[i-1]` for interior node `i` (filling `diag_lower[0..nz-3]`).
The basal boundary row sets `diag_lower[nz-2] = 0.0`.

Compatible with scipy banded packing: `ab[2, :-1] = diag_lower`, `ab[0, 1:] = diag_upper`.

---

## 4. Preconditions

- `nz >= 3` — enforced by the Python wrapper. Required for 2nd-order Stefan gradient.

### 4.1 Default Parameter Values

The Python wrapper must enforce these defaults when the caller does not provide overrides.

```
R_planet      = 1_561_000.0    # Planetary.RADIUS (m)
g             = 1.315          # Planetary.GRAVITY (m/s^2)
T_melt_const  = 273.0          # Thermal.MELT_TEMP (K)
R_gas         = 8.314          # Rheology.GAS_CONSTANT (J/mol·K)
omega         = 2.047e-5       # Planetary.ORBITAL_FREQ (rad/s)
eta_ref       = 5.0e13         # Rheology.VISCOSITY_REF (Pa·s)
Q_v           = 59_400.0       # Rheology.ACTIVATION_ENERGY_V (J/mol)
Q_b           = 49_000.0       # Rheology.ACTIVATION_ENERGY_B (J/mol)
mu_ice        = 3.3e9          # Rheology.RIGIDITY_ICE (Pa)
eps0          = 1.0e-5         # HeatFlux.TIDAL_STRAIN
d_grain       = 1.0e-3         # Rheology.GRAIN_SIZE (m)
d_del         = 7.13e-10       # Rheology.GRAIN_WIDTH (m)
D0v           = 9.1e-4         # Rheology.D0V_MEAN (m^2/s)
D0b           = 8.4e-4         # Rheology.D0B_MEAN (m^2/s)
d_molar       = 1.97e-5        # Rheology.MOLAR_VOLUME (m^3/mol)
porosity      = 0.0            # default no porosity
salt_frac     = 0.0            # default no salt
salt_scale    = 1.0            # Porosity.B_K_DEFAULT
por_cure_temp = 150.0          # Thermal.POR_CUR_TEMP_MEAN (K)
alpha_exp     = 1.6e-4         # Convection.ALPHA_EXPANSION (K^-1)
Ra_crit       = 1000.0         # Convection.RA_CRIT
Nu_C          = 0.3446         # Convection.NU_PREFACTOR
Nu_xi         = 0.333          # Convection.BETA_NU (1/3)
Nu_zeta       = 1.333          # Convection.ZETA_NU (4/3)
theta_lid     = 2.24           # Convection.THETA_LID
c1_deschamps  = 1.43           # Convection.C1_DESCHAMPS
c2_deschamps  = -0.03          # Convection.C2_DESCHAMPS
andrade_alpha = 0.2            # Rheology.ANDRADE_ALPHA
andrade_zeta  = 1.0            # Rheology.ANDRADE_ZETA
```

The wrapper pre-computes `gamma_val = scipy.special.gamma(1 + andrade_alpha)` and passes
the scalar result into the JIT kernel.

---

## 5. Scratch Arrays

All `float64`, allocated once per `batched_step` call, reused across all `n_lat` columns.

```
# Per-column work arrays (size nz)
T_guess        (nz,)      Picard iterate
T_work         (nz,)      Thomas solve output
diag_lower     (nz,)      sub-diagonal
diag_main      (nz,)      main diagonal
diag_upper     (nz,)      super-diagonal
rhs            (nz,)      right-hand side
k_impl         (nz,)      conductivity at T_guess (may include Nu enhancement)
rho_impl       (nz,)      density at T_guess
cp_impl        (nz,)      specific heat at T_guess
q_tidal_impl   (nz,)      tidal heating at T_guess
k_half_impl    (nz-1,)    half-node conductivities (arithmetic or harmonic)
G              (nz,)      geometric factors
G_half         (nz-1,)    half-node geometric factors

# Explicit-side (only populated when theta < 1.0)
k_exp          (nz,)
rho_exp        (nz,)
cp_exp         (nz,)
k_half_exp     (nz-1,)
flux_exp       (nz,)      cached explicit flux for interior nodes
```

Performance note: a `SolverWorkspace` pre-allocation wrapper that persists across
`batched_step` calls is a natural optimisation but not required for correctness.

---

## 6. Dual Property Evaluation (CN Detail)

The assembly evaluates material properties at two temperature states:

1. **Implicit side** (`T_guess` = current Picard iterate):
   - Conductivity (with optional convection), density, cp for LHS matrix coefficients
   - Tidal heating source term (using implicit-side viscosity and `rho*cp`)
   - Recomputed at each Picard iteration
   - When `use_convection=True`: calls `_build_k_profile` which scans T_guess for z_c,
     applies Nu enhancement, uses harmonic mean for half-nodes

2. **Explicit side** (`T_col` = T at time n):
   - Conductivity (with optional convection), density, cp for RHS flux term
   - Computed once before the Picard loop (per half-step)
   - Only computed when `theta < 1.0` (CN mode)
   - During BE (Rannacher), `flux_exp[:] = 0.0` and explicit-side arrays are not populated
   - When `use_convection=True`: explicit-side also uses `_build_k_profile` and harmonic
     mean. This matches Solver.py:325-337 where the CN cache builds k_exp from T_explicit
     through the same convection path.

The explicit-side flux is pre-multiplied by `(1-theta) * alpha_exp_i` so the Picard
loop simply adds `flux_exp[i]` to the RHS without recomputation.

---

## 7. _do_half_step Control Flow

```python
@njit
def _do_half_step(
    T_col,             # float64[:] (nz,) — modified in-place as output
    H,                 # float64 — fixed during this half-step
    q_ocean,           # float64
    T_surf,            # float64
    theta,             # float64 — 1.0 (BE) or 0.5 (CN)
    dt_eff,            # float64 — dt/2 (BE) or dt (CN)
    nz,                # int
    R_planet, g,       # float64 scalars
    is_spherical,      # bool
    use_convection,    # bool
    rheology_model,    # int (0=Maxwell, 1=Andrade)
    nu_ramp,           # float64 (convection ramp factor, 0..1)
    # --- material params ---
    porosity, salt_frac, salt_scale, por_cure_temp,
    eps0, mu_ice, eta_ref, Q_v, Q_b,
    d_grain, d_del, D0v, D0b, d_molar,
    # --- constants ---
    T_melt_const, R_gas, omega,
    alpha_exp_coeff, Ra_crit,
    Nu_C, Nu_xi, Nu_zeta, theta_lid,
    c1_deschamps, c2_deschamps,
    andrade_alpha, andrade_zeta, gamma_val,
    # --- scratch arrays ---
    <all scratch arrays>,
) -> Tuple[float64, float64, int64, float64, float64, float64, float64, float64, float64, bool]:
    # Returns: (H_new, dbdt, idx_c, z_c, D_cond, D_conv, Ra, Nu, T_c, is_convecting)
```

### Conductivity construction (step 6b, implicit side):

```
IF use_convection:
    # Phase 2: scan T_guess to find conductive/convective interface
    # 1. Compute transition temperature Tc via Green/Deschamps method
    # 2. Find first i where T_guess[i] >= Tc, interpolate z_c
    # 3. Compute Ra for convective sublayer (D_conv = H - z_c)
    # 4. Compute Nu (Green or simple, based on USE_GREEN_METHOD)
    # 5. Build k_impl: base conductivity + porosity/salt + Nu enhancement below z_c
    #    Nu_eff = 1.0 + nu_ramp * (Nu - 1.0)
    #    k_impl[idx_c:] *= Nu_eff
    # 6. Half-node: harmonic mean
    #    k_half_impl[j] = 2*k_impl[j]*k_impl[j+1] / (k_impl[j]+k_impl[j+1]+1e-30)

    # Viscosity for tidal heating: composite
    eta_i = _composite_viscosity(T_guess[i], d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R_gas)
ELSE:
    # Non-convective: effective_conductivity with porosity/salt, no Nu
    k_impl[i] = _effective_conductivity(T_guess[i], por, salt, B_k, T_phi)
    # Half-node: arithmetic mean
    k_half_impl[j] = 0.5 * (k_impl[j] + k_impl[j+1])

    # Viscosity for tidal heating: simple (Frank-Kamenetskii)
    eta_i = _viscosity_simple(T_guess[i], eta_ref, Q_v, R_gas, T_melt_const)

# Tidal heating (both paths):
IF rheology_model == 0:  # Maxwell
    q_tidal_impl[i] = _tidal_heating_maxwell(eps0, omega, eta_i, mu_ice)
ELSE:  # Andrade
    q_tidal_impl[i] = _tidal_heating_andrade(eps0, omega, eta_i, mu_ice,
                                              andrade_alpha, andrade_zeta, gamma_val)
```

### Full control flow:

```
1.  dz = H / (nz - 1)
2.  T_base_dirichlet = _basal_melting_point(H, g)

3.  Compute G[i] and G_half[j]:
      if spherical: G[i] = (R_planet - (i/(nz-1)) * H)^2
      else:         G[i] = 1.0
      G_half[j] = 0.5 * (G[j] + G[j+1])

4.  T_guess[:] = T_col[:]

5.  IF theta < 1.0:
      Compute explicit-side properties from T_col:
        IF use_convection:
          Build k_exp via _build_k_profile(T_col, ...) + harmonic mean
        ELSE:
          k_exp[i] = _effective_conductivity(T_col[i], ...) + arithmetic mean
        rho_exp[i] = _density_ice(T_col[i])
        cp_exp[i] = _specific_heat(T_col[i])

      For interior i = 1..nz-2:
        alpha_exp_i = dt_eff / (rho_exp[i] * cp_exp[i])
        fp_exp = k_half_exp[i] * G_half[i] / (G[i] * dz^2)
        fm_exp = k_half_exp[i-1] * G_half[i-1] / (G[i] * dz^2)
        T_dp = T_col[i+1] - T_col[i]
        T_dm = T_col[i] - T_col[i-1]
        flux_exp[i] = (1.0 - theta) * alpha_exp_i * (fp_exp * T_dp - fm_exp * T_dm)
    ELSE:
      flux_exp[:] = 0.0

6.  PICARD LOOP (max_iter = 3, tol = 0.01 K):

    a.  Reset arrays:
          diag_main[:] = 0.0
          diag_lower[:] = 0.0
          diag_upper[:] = 0.0
          rhs[:] = 0.0

    b.  Compute implicit-side properties from T_guess:
          (conductivity construction as described above)
          rho_impl[i] = _density_ice(T_guess[i])
          cp_impl[i] = _specific_heat(T_guess[i])
          (tidal heating as described above)

    c.  Assemble interior (i = 1..nz-2):
          alpha_i = dt_eff / (rho_impl[i] * cp_impl[i])
          fp = k_half_impl[i] * G_half[i] / (G[i] * dz^2)
          fm = k_half_impl[i-1] * G_half[i-1] / (G[i] * dz^2)
          diag_main[i]   = 1.0 + theta * alpha_i * (fp + fm)
          diag_upper[i]  = -theta * alpha_i * fp
          diag_lower[i-1] = -theta * alpha_i * fm
          source_i = dt_eff * q_tidal_impl[i] / (rho_impl[i] * cp_impl[i])
          rhs[i] = T_col[i] + flux_exp[i] + source_i

    d.  Boundary rows:
          diag_main[0]    = 1.0
          diag_upper[0]   = 0.0
          rhs[0]          = T_surf

          diag_main[nz-1]   = 1.0
          diag_lower[nz-2]  = 0.0
          rhs[nz-1]         = T_base_dirichlet

    e.  Thomas solve:
          _thomas_solve(diag_lower, diag_main, diag_upper, rhs, T_work, nz)

    f.  Convergence check:
          max_diff = max|T_work[i] - T_guess[i]| for all i
          T_guess[:] = T_work[:]
          if max_diff < 0.01: break

7.  T_col[:] = T_guess[:]    # in-place output

8.  Stefan update:
      k_basal = 567.0 / T_col[nz-1]    # bare conductivity, no porosity/salt (matches Solver.py:441)
      dTdz = (3*T_col[nz-1] - 4*T_col[nz-2] + T_col[nz-3]) / (2*dz)
      q_cond = k_basal * dTdz
      rho_base = _density_ice(T_col[nz-1])
      dbdt = (q_cond - q_ocean) / (rho_base * 334000.0)
      H_new = max(H + dbdt * dt_eff, 500.0)

9.  Return (H_new, dbdt, idx_c, z_c, D_cond, D_conv, Ra, Nu, T_c, is_convecting)
```

---

## 8. batched_step Control Flow

```python
@njit
def batched_step(
    T_grid,                # float64[:, :] (nz, n_lat) — modified in-place
    H_array,               # float64[:] (n_lat,) — modified in-place
    q_ocean_array,         # float64[:] (n_lat,)
    T_surf_array,          # float64[:] (n_lat,)
    dt,                    # float64
    current_step,          # int64
    rannacher_steps,       # int64
    nz,                    # int64
    R_planet, g,           # float64 scalars
    # --- per-column flags ---
    is_spherical_array,    # bool[:] (n_lat,)
    use_convection_array,  # bool[:] (n_lat,)
    rheology_model_array,  # int64[:] (n_lat,) — 0=Maxwell, 1=Andrade
    nu_ramp_array,         # float64[:] (n_lat,)
    # --- per-column material params (all float64[:] shape n_lat) ---
    porosity_array, salt_frac_array, salt_scale_array, por_cure_array,
    eps0_array, mu_ice_array, eta_ref_array, Q_v_array, Q_b_array,
    d_grain_array, d_del_array, D0v_array, D0b_array, d_molar_array,
    # --- scalar constants ---
    T_melt_const, R_gas, omega,
    alpha_exp_coeff, Ra_crit,
    Nu_C, Nu_xi, Nu_zeta, theta_lid,
    c1_deschamps, c2_deschamps,
    andrade_alpha, andrade_zeta, gamma_val,
    # --- diagnostic output arrays (written by driver) ---
    Ra_out,                # float64[:] (n_lat,)
    Nu_out,                # float64[:] (n_lat,)
    D_cond_out,            # float64[:] (n_lat,)
    D_conv_out,            # float64[:] (n_lat,)
    T_c_out,               # float64[:] (n_lat,)
    is_convecting_out,     # bool[:] (n_lat,)
) -> float64[:]:           # dbdt_array (n_lat,)
```

### Control flow:

```
1.  n_lat = H_array.shape[0]
2.  Allocate scratch arrays (once, reused across columns)
3.  Allocate dbdt_array = np.empty(n_lat)

4.  Determine stepping mode:
      if current_step < rannacher_steps:
        theta = 1.0;  dt_eff = dt / 2.0
      else:
        theta = 0.5;  dt_eff = dt

5.  for lat in range(n_lat):
      T_col = T_grid[:, lat]

      # --- Half-step 1 (or only step for CN) ---
      H_new, dbdt1, idx_c, z_c, D_cond, D_conv, Ra, Nu, T_c, is_conv = _do_half_step(
          T_col, H_array[lat], q_ocean_array[lat], T_surf_array[lat],
          theta, dt_eff, nz, R_planet, g, is_spherical_array[lat],
          use_convection_array[lat], rheology_model_array[lat], nu_ramp_array[lat],
          porosity_array[lat], salt_frac_array[lat], salt_scale_array[lat],
          por_cure_array[lat], eps0_array[lat], mu_ice_array[lat],
          eta_ref_array[lat], Q_v_array[lat], Q_b_array[lat],
          d_grain_array[lat], d_del_array[lat], D0v_array[lat], D0b_array[lat],
          d_molar_array[lat],
          T_melt_const, R_gas, omega,
          alpha_exp_coeff, Ra_crit, Nu_C, Nu_xi, Nu_zeta, theta_lid,
          c1_deschamps, c2_deschamps, andrade_alpha, andrade_zeta, gamma_val,
          <scratch arrays>
      )
      H_array[lat] = H_new

      if current_step < rannacher_steps:
        # --- Rannacher half-step 2 ---
        H_new2, dbdt2, idx_c, z_c, D_cond, D_conv, Ra, Nu, T_c, is_conv = _do_half_step(
            T_col, H_array[lat], q_ocean_array[lat], T_surf_array[lat],
            theta, dt_eff, nz, R_planet, g, is_spherical_array[lat],
            use_convection_array[lat], rheology_model_array[lat], nu_ramp_array[lat],
            porosity_array[lat], ...same params...,
            <scratch arrays>
        )
        H_array[lat] = H_new2
        dbdt_array[lat] = (dbdt1 + dbdt2) / 2.0
      else:
        dbdt_array[lat] = dbdt1

      # Store diagnostics from final half-step
      Ra_out[lat] = Ra
      Nu_out[lat] = Nu
      D_cond_out[lat] = D_cond
      D_conv_out[lat] = D_conv
      T_c_out[lat] = T_c
      is_convecting_out[lat] = is_conv

6.  Return dbdt_array
```

### Key behaviors:
- `T_grid` and `H_array` modified in-place
- `current_step` managed by Python caller, not inside JIT
- Rannacher half-step 2 sees updated T and H from half-step 1
- Explicit-side properties recomputed fresh for half-step 2
- Basal Dirichlet recomputed per half-step (H changes)
- Convection diagnostics stored from the final half-step of each column

---

## 9. Convection Detail: _scan_profile

This is the Phase 2 core algorithm, ported from `IceConvection.scan_temperature_profile()`.

```
Input: T_col (nz,), H, T_melt_basal (pressure-dependent), T_surf, Q_v, Q_b, d_grain,
       eta_ref, R_gas, theta_lid, c1_deschamps, c2_deschamps, Ra_crit, Nu_C, Nu_xi, Nu_zeta,
       alpha_exp_coeff, g

1.  Compute transition temperature via Green method (calls Deschamps internally):
      (Tc, Ti) = _green_Tc_Ti(T_melt_basal, T_surf, Q_v, R_gas, eta_ref, theta_lid)
      # _green_Tc_Ti internally calls _deschamps_Ti to get Ti,
      # then computes DTv and Tc = Ti - theta_lid * DTv.
      # Includes Howell fallback if dni/dTi overflows.
      # T_melt_basal is pressure-dependent, matching Solver.py:225.

2.  Scan T_col for first index where T >= Tc:
      idx_c = nz-1  (default: no convection)
      for i in 0..nz-1:
        if T_col[i] >= Tc:
          idx_c = i
          break

3.  If no warm index found (loop completed without break):
      Return (nz-1, H, H, 0.0, Tc, 0.0, 1.0, False)
    # Note: idx_c == 0 is NOT treated as non-convecting. It means T_col[0] >= Tc,
    # so z_c = 0, D_cond = 0, D_conv = H. The Ra/Nu calculation proceeds normally.
    # This matches Convection.py:756 which only returns non-convecting for empty warm_indices.

4.  Linear interpolation for exact z_c:
      if 0 < idx_c < nz:
        T_above = T_col[idx_c-1]
        T_below = T_col[idx_c]
        z_above = (idx_c-1) / (nz-1) * H
        z_below = idx_c / (nz-1) * H
        if T_below > T_above:
          frac = (Tc - T_above) / (T_below - T_above)
          z_c = z_above + frac * (z_below - z_above)
        else:
          z_c = z_below
      else:
        z_c = idx_c / (nz-1) * H

5.  D_cond = z_c; D_conv = H - z_c
    if D_conv <= 0: return non-convecting state

6.  Compute Ra:
      DT = T_melt_basal - Tc
      T_mean = (T_melt_basal + Tc) / 2
      eta_mean = _composite_viscosity(T_mean, d_grain, ...) or _viscosity_simple(T_mean, ...)
      rho_mean = _density_ice(T_mean)
      k_mean = 567.0 / T_mean
      cp_mean = _specific_heat(T_mean)
      kappa = k_mean / (rho_mean * cp_mean)
      Ra = rho_mean * g * alpha_exp * DT * D_conv^3 / (kappa * eta_mean)

7.  Compute Nu:
      if Ra < Ra_crit: Nu = 1.0
      else:
        temp_ratio = max((Ti - Tc) / DT, 0.01)
        Nu = max(Nu_C * Ra^Nu_xi * temp_ratio^Nu_zeta, 1.0)

8.  is_convecting = (Ra >= Ra_crit)

9.  Return (idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_convecting)
```

---

## 10. Andrade Tidal Heating Detail

Ported from Physics.py:254-280. The only SciPy dependency (`gamma(1+alpha)`) is
pre-computed in the Python wrapper.

```
@njit
def _tidal_heating_andrade(eps0, omega, eta, mu, alpha, zeta, gamma_val):
    J_elastic = 1.0 / mu
    tau = eta / mu
    andrade_term = max(omega * tau * zeta, 1e-100)

    const_term = J_elastic * (andrade_term ** -alpha) * gamma_val

    J_real = J_elastic + const_term * cos(alpha * pi / 2.0)
    J_imag = J_elastic * (omega * tau) ** -1 + const_term * sin(alpha * pi / 2.0)

    G_imag = J_imag / (J_real**2 + J_imag**2)

    return 0.5 * omega * (eps0 ** 2) * G_imag
```

---

## 11. Validation Checklist

### 11.1 Single-column parity (`n_lat = 1`)

Run the original `Thermal_Solver` and the batched solver with `n_lat=1` using identical
parameters. Test both `use_convection=False` and `use_convection=True`:
- Same `nx`, `dt`, `thickness`, `T_surf`, `q_ocean`, physics params
- Compare T profiles at each step: max absolute difference < 1e-6 K
- Compare H trajectory: max absolute difference < 1e-4 m
- Compare `dbdt` values: relative difference < 1e-6
- Note: Thomas solver and `scipy.linalg.solve_banded` may produce O(1e-12) differences
  due to different pivoting/accumulation order. These are acceptable.

### 11.2 Identical columns remain identical

Set `n_lat = 4` with identical parameters for all columns.
- All 4 T profiles must be identical to machine epsilon after N steps (max diff < 1e-14 K).
- All 4 H values must be identical to machine epsilon.
- Note: since all columns use the same scratch arrays sequentially, and the same arithmetic
  operations, these should be exactly identical. Bitwise identity is expected but not
  required if FP non-determinism arises from compiler reordering.

### 11.3 q_ocean isolation

Set `n_lat = 3` with identical parameters except `q_ocean_array = [0.01, 0.02, 0.03]`.
- T profiles at step 1 must be identical (q_ocean only affects Stefan update).
- H values must differ after step 1.
- T profiles diverge at step 2 (because H changed dz and T_base).

### 11.4 Spherical vs Cartesian parity

Run original solver with `coordinate_system='spherical'` and batched solver with
`is_spherical_array=[True]`. Compare as in 11.1.
Repeat with `coordinate_system='cartesian'` / `is_spherical_array=[False]`.

### 11.5 Rannacher startup

Run both solvers for `rannacher_steps + 2` steps.
- During Rannacher: verify theta=1.0, dt_eff=dt/2, two half-steps per call.
- After Rannacher: verify theta=0.5, dt_eff=dt, single step per call.
- Compare T and H at each step.

### 11.6 Convection parity

Run original solver with `use_convection=True` and batched solver with
`use_convection_array=[True]`:
- Compare D_cond, D_conv, Ra, Nu at each step
- Compare k profiles (conductivity with Nu enhancement)
- Verify harmonic mean half-node conductivities match

### 11.7 Andrade vs Maxwell parity

Run original solver with `Rheology.MODEL="Andrade"` and batched solver with
`rheology_model_array=[1]`:
- Compare tidal heating profiles at each step
- Compare final H and T

### 11.8 Mixed ensemble

Set `n_lat = 2` with column 0 using `use_convection=False` and column 1 using
`use_convection=True`. Verify each column matches its corresponding single-column run.

---

## 12. File Location

`EuropaProjectDJ/src/batched_solver.py`

Python wrapper class in same file provides:
- Parameter validation (`nz >= 3`)
- Pre-computation of `gamma_val` from `andrade_alpha`
- Scratch workspace management
- `current_step` tracking
- Default parameter population from constants module
- Convenience methods mapping to `batched_step`

---

## 13. Future Extensions (not in scope)

- StefanBoltzmann radiative surface BC (Newton-Raphson inside JIT)
- `prange` parallelism over the latitude loop
- `SolverWorkspace` persistent scratch allocation
- Non-Green convection paths (`USE_GREEN_METHOD=False`)
