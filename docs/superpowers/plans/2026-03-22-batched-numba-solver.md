# Batched Numba 1D Ice Shell Solver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the existing `Solver.py` 1D thermal solver into a Numba `@njit` batched kernel that advances `n_lat` independent vertical columns per call, with Green-path convection, Andrade/Maxwell tidal heating, and exact baseline parity.

**Architecture:** Composable `@njit` helper functions (thermophysical, viscosity, convection, tidal heating, Thomas solver) called from a `_do_half_step` kernel, which is called from a `batched_step` driver looping over latitude columns. A Python wrapper class manages `current_step`, defaults, and `gamma_val` pre-computation.

**Tech Stack:** Python 3.14, NumPy, Numba 0.63 (`@njit`), pytest. Source code: `EuropaProjectDJ/src/`. Tests: `EuropaProjectDJ/tests/`.

**Spec:** `docs/superpowers/specs/2026-03-22-batched-numba-solver-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `EuropaProjectDJ/src/batched_solver.py` | All `@njit` helpers + `batched_step` driver + `BatchedSolverWrapper` Python class |
| `EuropaProjectDJ/tests/test_batched_helpers.py` | Unit tests for individual `@njit` helper functions vs original implementations |
| `EuropaProjectDJ/tests/test_batched_solver.py` | Integration/parity tests: batched solver vs original `Thermal_Solver` |

The single-file approach keeps all JIT functions in one compilation unit (Numba benefits from this) and avoids circular import issues with the existing `src/` modules.

---

## Task 1: Thermophysical Property Helpers

**Files:**
- Create: `EuropaProjectDJ/src/batched_solver.py`
- Create: `EuropaProjectDJ/tests/test_batched_helpers.py`

These are the leaf-level functions with no dependencies on each other.

- [ ] **Step 1: Write failing tests for all 5 thermophysical helpers**

```python
# test_batched_helpers.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from constants import Thermal, Planetary
from Physics import IcePhysics


class TestThermophysicalHelpers:
    """Test JIT helpers against original implementations."""

    TEMPS = np.array([100.0, 150.0, 200.0, 250.0, 273.0])

    def test_conductivity(self):
        from batched_solver import _conductivity
        for T in self.TEMPS:
            expected = float(Thermal.conductivity(T))
            result = _conductivity(T)
            assert abs(result - expected) < 1e-12, f"k({T}): {result} != {expected}"

    def test_specific_heat(self):
        from batched_solver import _specific_heat
        for T in self.TEMPS:
            expected = float(Thermal.specific_heat(T))
            result = _specific_heat(T)
            assert abs(result - expected) < 1e-12

    def test_density_ice(self):
        from batched_solver import _density_ice
        for T in self.TEMPS:
            expected = float(Thermal.density_ice(T))
            result = _density_ice(T)
            assert abs(result - expected) < 1e-10

    def test_basal_melting_point(self):
        from batched_solver import _basal_melting_point
        for H in [5e3, 20e3, 50e3, 100e3]:
            expected = float(IcePhysics.basal_melting_point(H))
            result = _basal_melting_point(H, Planetary.GRAVITY)
            assert abs(result - expected) < 1e-10, f"T_melt({H}): {result} != {expected}"

    def test_effective_conductivity_bare(self):
        """No porosity/salt — should equal bare 567/T."""
        from batched_solver import _effective_conductivity
        for T in self.TEMPS:
            expected = 567.0 / T
            result = _effective_conductivity(T, 0.0, 0.0, 1.0, 150.0)
            assert abs(result - expected) < 1e-12

    def test_effective_conductivity_porous(self):
        """Porosity correction at T < T_phi."""
        from batched_solver import _effective_conductivity
        T, por, T_phi = 120.0, 0.1, 150.0
        expected = float(IcePhysics.effective_conductivity(T, porosity=por, porosity_cure_temp=T_phi))
        result = _effective_conductivity(T, por, 0.0, 1.0, T_phi)
        assert abs(result - expected) < 1e-12

    def test_effective_conductivity_salt(self):
        """Salt scaling."""
        from batched_solver import _effective_conductivity
        T, f_s, B_k = 200.0, 0.03, 2.0
        expected = float(IcePhysics.effective_conductivity(T, salt_fraction=f_s, salt_scaling_factor=B_k))
        result = _effective_conductivity(T, 0.0, f_s, B_k, 150.0)
        assert abs(result - expected) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestThermophysicalHelpers -v
```

Expected: `ModuleNotFoundError: No module named 'batched_solver'`

- [ ] **Step 3: Implement the 5 thermophysical helpers**

```python
# batched_solver.py
"""
Batched Numba 1D Ice Shell Solver.

Green-path parity port of Solver.py for latitude-batched execution.
See docs/superpowers/specs/2026-03-22-batched-numba-solver-design.md
"""
import numpy as np
from numba import njit


# =============================================================================
# 1. THERMOPHYSICAL PROPERTY HELPERS
# =============================================================================

@njit
def _conductivity(T):
    """Howell (2021): k(T) = 567 / T  [W/m·K]"""
    return 567.0 / T


@njit
def _specific_heat(T):
    """cp(T) = 7.49*T + 90  [J/kg·K]"""
    return 7.49 * T + 90.0


@njit
def _density_ice(T):
    """rho(T) = 917 * (1 + 1.6e-4 * (273 - T))  [kg/m³]"""
    return 917.0 * (1.0 + 1.6e-4 * (273.0 - T))


@njit
def _basal_melting_point(H, g):
    """Pressure-dependent melting: T_m = 273 + CC * rho * g * H  [K]"""
    return 273.0 + (-7.4e-8) * 917.0 * g * H


@njit
def _effective_conductivity(T, porosity, salt_frac, salt_scale, por_cure_temp):
    """k_eff with porosity and salt corrections."""
    k = 567.0 / T
    if porosity > 0.0 and T < por_cure_temp:
        k = k * (1.0 - porosity)
    if salt_frac > 0.0:
        k = k * (1.0 + salt_frac * (salt_scale - 1.0))
    return k
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestThermophysicalHelpers -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_helpers.py
git commit -m "feat: add thermophysical JIT helpers for batched solver"
```

---

## Task 2: Viscosity & Tidal Heating Helpers

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`
- Modify: `EuropaProjectDJ/tests/test_batched_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
class TestViscosityTidalHelpers:

    def test_viscosity_simple(self):
        from batched_solver import _viscosity_simple
        from constants import Rheology
        T = 250.0
        eta_ref = 5.0e13
        Q_v = Rheology.ACTIVATION_ENERGY_V
        R = Rheology.GAS_CONSTANT
        T_melt = 273.0
        expected = float(IcePhysics.viscosity_simple(T, eta_ref, T_melt))
        result = _viscosity_simple(T, eta_ref, Q_v, R, T_melt)
        assert abs(result - expected) / expected < 1e-12

    def test_composite_viscosity(self):
        from batched_solver import _composite_viscosity
        from constants import Rheology
        T = 250.0
        expected = float(IcePhysics.composite_viscosity(T))
        result = _composite_viscosity(
            T, Rheology.GRAIN_SIZE, Rheology.GRAIN_WIDTH,
            Rheology.D0V_MEAN, Rheology.D0B_MEAN,
            Rheology.ACTIVATION_ENERGY_V, Rheology.ACTIVATION_ENERGY_B,
            Rheology.MOLAR_VOLUME, Rheology.GAS_CONSTANT,
        )
        assert abs(result - expected) / expected < 1e-10

    def test_composite_viscosity_clipping(self):
        """Low T should clip to 1e25, high T should clip to 1e12."""
        from batched_solver import _composite_viscosity
        from constants import Rheology
        R = Rheology.GAS_CONSTANT
        result_cold = _composite_viscosity(
            50.0, 1e-3, 7.13e-10, 9.1e-4, 8.4e-4, 59400.0, 49000.0, 1.97e-5, R)
        result_hot = _composite_viscosity(
            273.0, 1e-3, 7.13e-10, 9.1e-4, 8.4e-4, 59400.0, 49000.0, 1.97e-5, R)
        assert result_cold <= 1e25
        assert result_hot >= 1e12

    def test_tidal_maxwell(self):
        from batched_solver import _tidal_heating_maxwell, _viscosity_simple
        from constants import Rheology, Planetary, HeatFlux
        T = 250.0
        eta = _viscosity_simple(T, 5e13, Rheology.ACTIVATION_ENERGY_V,
                                Rheology.GAS_CONSTANT, 273.0)
        result = _tidal_heating_maxwell(HeatFlux.TIDAL_STRAIN, Planetary.ORBITAL_FREQ,
                                        eta, Rheology.RIGIDITY_ICE)
        expected = float(IcePhysics.tidal_heating(
            T, epsilon_0=HeatFlux.TIDAL_STRAIN, use_composite_viscosity=False))
        assert abs(result - expected) / max(expected, 1e-30) < 1e-10

    def test_tidal_andrade(self):
        """Compare Andrade JIT helper against manual Andrade computation.
        Note: IcePhysics.tidal_heating dispatches on Rheology.MODEL config.
        Current config is 'Andrade' so the comparison is valid. If config
        changes, this test must be updated or compute expected from first principles.
        """
        from batched_solver import _tidal_heating_andrade, _viscosity_simple
        from scipy.special import gamma
        from constants import Rheology, Planetary, HeatFlux
        assert Rheology.MODEL == "Andrade", "Test requires Andrade config"
        T = 250.0
        eta = _viscosity_simple(T, 5e13, Rheology.ACTIVATION_ENERGY_V,
                                Rheology.GAS_CONSTANT, 273.0)
        gamma_val = gamma(1 + Rheology.ANDRADE_ALPHA)
        result = _tidal_heating_andrade(
            HeatFlux.TIDAL_STRAIN, Planetary.ORBITAL_FREQ, eta,
            Rheology.RIGIDITY_ICE, Rheology.ANDRADE_ALPHA,
            Rheology.ANDRADE_ZETA, gamma_val)
        # Compare against original (which uses Andrade since MODEL="Andrade")
        expected = float(IcePhysics.tidal_heating(
            T, epsilon_0=HeatFlux.TIDAL_STRAIN, use_composite_viscosity=False))
        assert abs(result - expected) / max(expected, 1e-30) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestViscosityTidalHelpers -v
```

- [ ] **Step 3: Implement viscosity and tidal heating helpers**

```python
# Add to batched_solver.py

# =============================================================================
# 2. VISCOSITY HELPERS
# =============================================================================

@njit
def _viscosity_simple(T, eta_ref, Q_v, R, T_melt):
    """Frank-Kamenetskii: eta = eta_ref * exp((Q_v/R)*(1/T - 1/T_melt))"""
    T_safe = max(T, 50.0)
    return eta_ref * np.exp((Q_v / R) * (1.0 / T_safe - 1.0 / T_melt))


@njit
def _composite_viscosity(T, d_grain, d_del, D0v, D0b, Q_v, Q_b, d_molar, R):
    """Howell (2021) diffusion creep viscosity, clipped [1e12, 1e25]."""
    T_safe = max(T, 50.0)
    Dv = D0v * np.exp(-Q_v / (R * T_safe))
    Db = D0b * np.exp(-Q_b / (R * T_safe))
    prefactor = (42.0 * d_molar) / (R * T_safe * d_grain * d_grain)
    diff_term = Dv + (np.pi * d_del / d_grain) * Db
    eta = 0.5 / (prefactor * diff_term)
    if eta < 1e12:
        return 1e12
    if eta > 1e25:
        return 1e25
    return eta


# =============================================================================
# 3. TIDAL HEATING HELPERS
# =============================================================================

@njit
def _tidal_heating_maxwell(eps0, omega, eta, mu):
    """Maxwell viscoelastic dissipation [W/m³]."""
    num = eps0 * eps0 * omega * omega * eta
    den = 2.0 * (1.0 + (omega * omega * eta * eta / (mu * mu)))
    return num / den


@njit
def _tidal_heating_andrade(eps0, omega, eta, mu, alpha, zeta, gamma_val):
    """Andrade transient creep dissipation [W/m³]."""
    J_elastic = 1.0 / mu
    tau = eta / mu
    andrade_term = omega * tau * zeta
    if andrade_term < 1e-100:
        andrade_term = 1e-100
    const_term = J_elastic * (andrade_term ** (-alpha)) * gamma_val
    J_real = J_elastic + const_term * np.cos(alpha * np.pi / 2.0)
    J_imag = J_elastic / (omega * tau) + const_term * np.sin(alpha * np.pi / 2.0)
    G_imag = J_imag / (J_real * J_real + J_imag * J_imag)
    return 0.5 * omega * eps0 * eps0 * G_imag
```

- [ ] **Step 4: Run tests**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestViscosityTidalHelpers -v
```

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_helpers.py
git commit -m "feat: add viscosity and tidal heating JIT helpers"
```

---

## Task 3: Convection Helpers (Transition Temperature, Ra, Nu, Profile Scan)

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`
- Modify: `EuropaProjectDJ/tests/test_batched_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
class TestConvectionHelpers:

    def test_howell_Tc(self):
        from batched_solver import _howell_Tc
        from Convection import IceConvection
        from constants import Rheology
        Q_v = 59400.0
        T_melt = 272.0  # pressure-dependent example
        expected = IceConvection.howell_cond_base_temp(T_melt, Q_v)
        result = _howell_Tc(T_melt, Q_v, Rheology.GAS_CONSTANT)
        assert abs(result - expected) < 1e-10

    def test_deschamps_Ti(self):
        from batched_solver import _deschamps_Ti
        from Convection import IceConvection
        from constants import Rheology
        T_melt, T_surf, Q_v = 272.0, 104.0, 59400.0
        expected = IceConvection.deschamps_interior_temp(T_melt, T_surf, Q_v)
        result = _deschamps_Ti(T_melt, T_surf, Q_v, Rheology.GAS_CONSTANT, 1.43, -0.03)
        assert abs(result - expected) < 1e-10

    def test_green_Tc_Ti(self):
        from batched_solver import _green_Tc_Ti
        from Convection import IceConvection
        from constants import Rheology
        T_melt, T_surf, Q_v = 272.0, 104.0, 59400.0
        eta_ref = 5e13
        expected_Tc, expected_Ti = IceConvection.green_cond_base_temp(
            T_melt, T_surf, Q_v, eta_ref)
        Tc, Ti = _green_Tc_Ti(T_melt, T_surf, Q_v, Rheology.GAS_CONSTANT, eta_ref, 2.24)
        assert abs(Tc - expected_Tc) < 1e-8
        assert abs(Ti - expected_Ti) < 1e-8

    def test_rayleigh_number(self):
        from batched_solver import _rayleigh_number
        from Convection import IceConvection
        from constants import Planetary
        DT, d, T_mean = 20.0, 10e3, 260.0
        eta = 1e14
        expected = IceConvection.rayleigh_number(
            DT, d, T_mean, use_composite_viscosity=False, eta_ref=eta)
        result = _rayleigh_number(DT, d, T_mean, eta, Planetary.GRAVITY, 1.6e-4)
        assert abs(result - expected) / expected < 1e-6

    def test_nusselt_green_subcritical(self):
        from batched_solver import _nusselt_green
        result = _nusselt_green(500.0, 260.0, 240.0, 30.0, 1000.0, 0.3446, 0.333, 1.333)
        assert result == 1.0

    def test_nusselt_green_supercritical(self):
        from batched_solver import _nusselt_green
        from Convection import IceConvection
        Ra, Ti, Tc, DT = 1e6, 260.0, 240.0, 30.0
        expected = IceConvection.nusselt_number_green(Ra, Ti, Tc, DT)
        result = _nusselt_green(Ra, Ti, Tc, DT, 1000.0, 0.3446, 0.333, 1.333)
        assert abs(result - expected) / expected < 1e-10

    def test_nusselt_simple_subcritical(self):
        from batched_solver import _nusselt_simple
        assert _nusselt_simple(500.0, 1000.0, 0.3446, 0.333) == 1.0

    def test_nusselt_simple_supercritical(self):
        from batched_solver import _nusselt_simple
        from Convection import IceConvection
        Ra = 1e6
        expected = float(IceConvection.nusselt_number(Ra))
        result = _nusselt_simple(Ra, 1000.0, 0.3446, 0.333)
        assert abs(result - expected) / expected < 1e-10

    def test_harmonic_mean(self):
        from batched_solver import _harmonic_mean
        from Convection import IceConvection
        k = np.array([5.0, 3.0, 1.0, 2.0, 4.0])
        k_half = np.empty(4)
        _harmonic_mean(k, k_half, 5)
        expected = IceConvection.harmonic_mean_vectorized(k)
        np.testing.assert_allclose(k_half, expected, atol=1e-14)

    def test_scan_profile_idx_c_zero(self):
        """Entire shell warm (T[0] >= Tc) should give idx_c=0, D_cond=0."""
        from batched_solver import _scan_profile
        nz = 31
        T_col = np.linspace(270.0, 272.0, nz)  # all above any reasonable Tc
        H = 30e3
        T_melt = 272.0
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 270.0, 59400.0, 49000.0, 1e-3,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315)
        assert idx_c == 0
        assert D_cond == 0.0 or D_cond < 1.0  # z_c ~ 0
        assert D_conv > 0.0

    def test_scan_profile_no_convection(self):
        """Cold linear profile below Tc should return non-convecting."""
        from batched_solver import _scan_profile
        nz = 31
        T_col = np.linspace(104.0, 200.0, nz)  # too cold
        H = 10e3
        T_melt = 272.0
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 104.0, 59400.0, 49000.0, 1e-3,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315)
        assert not is_conv
        assert D_conv == 0.0

    def test_scan_profile_vs_original(self):
        """Warm profile should match IceConvection.scan_temperature_profile."""
        from batched_solver import _scan_profile
        from Convection import IceConvection
        from Physics import IcePhysics
        nz = 31
        H = 30e3
        T_melt = float(IcePhysics.basal_melting_point(H))
        T_col = np.linspace(104.0, T_melt, nz)
        z_grid = np.linspace(0, H, nz)
        state = IceConvection.scan_temperature_profile(
            T_col, z_grid, H, T_melt, Q_v=59400.0, Q_b=49000.0,
            d_grain=1e-3, use_composite_viscosity=True, eta_ref=5e13)
        idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
            T_col, nz, H, T_melt, 104.0, 59400.0, 49000.0, 1e-3,
            5e13, 8.314, 2.24, 1.43, -0.03, 1000.0, 0.3446, 0.333, 1.333,
            1.6e-4, 1.315)
        assert abs(z_c - state.z_c) < 1.0  # within 1 m
        assert abs(D_cond - state.D_cond) < 1.0
        if state.Ra > 0:
            assert abs(Ra - state.Ra) / state.Ra < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestConvectionHelpers -v
```

- [ ] **Step 3: Implement convection helpers**

```python
# Add to batched_solver.py

# =============================================================================
# 4. CONVECTION HELPERS
# =============================================================================

@njit
def _howell_Tc(T_melt, Q_v, R):
    """Howell (2021) conductive-base temperature."""
    ratio = R / Q_v
    T_c = (np.sqrt(4.0 * T_melt * ratio + 1.0) - 1.0) / (2.0 * ratio)
    return 2.0 * T_c - T_melt


@njit
def _deschamps_Ti(T_melt, T_surf, Q_v, R, c1, c2):
    """Deschamps & Vilella (2021) Eq. 18: interior temperature."""
    DTs = T_melt - T_surf
    B = Q_v / (2.0 * R * c1)
    return B * (np.sqrt(1.0 + (2.0 / B) * (T_melt - c2 * DTs)) - 1.0)


@njit
def _green_Tc_Ti(T_melt, T_surf, Q_v, R, eta_ref, theta_lid):
    """Green et al. (2021) lid base temperature. Returns (Tc, Ti)."""
    Ti = _deschamps_Ti(T_melt, T_surf, Q_v, R, 1.43, -0.03)
    if Ti < T_surf + 1.0:
        Ti = T_surf + 1.0
    if Ti > T_melt - 1.0:
        Ti = T_melt - 1.0

    A = Q_v / (R * T_melt)
    exponent = A * ((T_melt / Ti) - 1.0)
    if exponent > 500.0:
        exponent = 500.0
    if exponent < -500.0:
        exponent = -500.0

    exp_term = np.exp(exponent)
    dni_dTi = -eta_ref * (A * T_melt / (Ti * Ti)) * exp_term

    if abs(dni_dTi) < 1e-100 or not np.isfinite(dni_dTi):
        Tc = _howell_Tc(T_melt, Q_v, R)
        if Tc < T_surf + 1.0:
            Tc = T_surf + 1.0
        if Tc > T_melt - 1.0:
            Tc = T_melt - 1.0
        return Tc, Ti

    ni = eta_ref * exp_term
    DTv = -ni / dni_dTi
    DTe = theta_lid * DTv
    Tc = Ti - DTe

    if Tc < T_surf + 1.0:
        Tc = T_surf + 1.0
    if Tc > T_melt - 1.0:
        Tc = T_melt - 1.0
    return Tc, Ti


@njit
def _rayleigh_number(DT, d, T_mean, eta, g, alpha_exp):
    """Ra = rho*g*alpha*DT*d^3 / (kappa*eta)"""
    rho = _density_ice(T_mean)
    k = 567.0 / T_mean
    cp = _specific_heat(T_mean)
    kappa = k / (rho * cp)
    return rho * g * alpha_exp * DT * d * d * d / (kappa * eta)


@njit
def _nusselt_green(Ra, Ti, Tc, DT, Ra_crit, C, xi, zeta):
    """Green et al. (2021) Nu with internal heating correction."""
    if Ra < Ra_crit or DT <= 0.0:
        return 1.0
    temp_ratio = (Ti - Tc) / DT
    if temp_ratio < 0.01:
        temp_ratio = 0.01
    Nu = C * (Ra ** xi) * (temp_ratio ** zeta)
    if Nu < 1.0:
        return 1.0
    return Nu


@njit
def _nusselt_simple(Ra, Ra_crit, C, xi):
    """Solomatov & Moresi (2000): Nu = C * Ra^xi if Ra >= Ra_crit, else 1."""
    if Ra < Ra_crit:
        return 1.0
    Nu = C * (Ra ** xi)
    if Nu < 1.0:
        return 1.0
    return Nu


@njit
def _harmonic_mean(k, k_half, nz):
    """Fill k_half with harmonic mean half-node conductivities."""
    for j in range(nz - 1):
        k_half[j] = 2.0 * k[j] * k[j + 1] / (k[j] + k[j + 1] + 1e-30)


@njit
def _build_k_profile(T_col, k_impl, nz, H, T_melt_basal, T_surf,
                     Q_v, Q_b, d_grain, d_del, D0v, D0b, d_molar,
                     eta_ref, R_gas, theta_lid, c1, c2,
                     Ra_crit, Nu_C, Nu_xi, Nu_zeta, alpha_exp, g,
                     porosity, salt_frac, salt_scale, por_cure_temp,
                     nu_ramp):
    """Build conductivity profile with convection enhancement.
    Fills k_impl in-place. Returns convection diagnostics as 8-tuple."""
    # Step 1: scan for convection interface
    idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv = _scan_profile(
        T_col, nz, H, T_melt_basal, T_surf,
        Q_v, Q_b, d_grain, eta_ref, R_gas,
        theta_lid, c1, c2, Ra_crit, Nu_C, Nu_xi, Nu_zeta, alpha_exp, g)

    # Step 2: base conductivity + porosity/salt
    for i in range(nz):
        k_impl[i] = _effective_conductivity(T_col[i], porosity, salt_frac,
                                             salt_scale, por_cure_temp)

    # Step 3: Nu enhancement below z_c
    if idx_c < nz:
        Nu_eff = 1.0 + nu_ramp * (Nu - 1.0)
        for i in range(idx_c, nz):
            k_impl[i] = k_impl[i] * Nu_eff

    return idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_conv


@njit
def _scan_profile(T_col, nz, H, T_melt_basal, T_surf,
                  Q_v, Q_b, d_grain, eta_ref, R_gas,
                  theta_lid, c1, c2,
                  Ra_crit, Nu_C, Nu_xi, Nu_zeta,
                  alpha_exp, g):
    """Phase 2 profile scan. Returns 8 scalars matching ConvectionState fields."""
    Tc, Ti = _green_Tc_Ti(T_melt_basal, T_surf, Q_v, R_gas, eta_ref, theta_lid)

    # Scan for first warm index
    idx_c = nz - 1
    found = False
    for i in range(nz):
        if T_col[i] >= Tc:
            idx_c = i
            found = True
            break

    if not found:
        return idx_c, H, H, 0.0, Tc, 0.0, 1.0, False

    # Interpolate z_c
    if 0 < idx_c < nz:
        T_above = T_col[idx_c - 1]
        T_below = T_col[idx_c]
        z_above = (idx_c - 1) / (nz - 1) * H
        z_below = idx_c / (nz - 1) * H
        if T_below > T_above:
            frac = (Tc - T_above) / (T_below - T_above)
            z_c = z_above + frac * (z_below - z_above)
        else:
            z_c = z_below
    else:
        z_c = idx_c / (nz - 1) * H

    D_cond = z_c
    D_conv = H - z_c
    if D_conv <= 0.0:
        return idx_c, z_c, D_cond, 0.0, Tc, 0.0, 1.0, False

    # Ra
    DT = T_melt_basal - Tc
    T_mean = (T_melt_basal + Tc) / 2.0
    eta_mean = _composite_viscosity(T_mean, d_grain, 7.13e-10,
                                     9.1e-4, 8.4e-4, Q_v, Q_b,
                                     1.97e-5, R_gas)
    Ra = _rayleigh_number(DT, D_conv, T_mean, eta_mean, g, alpha_exp)

    # Nu
    Nu = _nusselt_green(Ra, Ti, Tc, DT, Ra_crit, Nu_C, Nu_xi, Nu_zeta)

    is_convecting = Ra >= Ra_crit
    return idx_c, z_c, D_cond, D_conv, Tc, Ra, Nu, is_convecting
```

**Note on `_scan_profile`**: The composite viscosity params (`d_del`, `D0v`, `D0b`, `d_molar`)
are hardcoded to defaults here. Task 5 will refactor the signature to pass these through
from the column's per-column arrays when wiring into `_do_half_step`.

- [ ] **Step 4: Run tests**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestConvectionHelpers -v
```

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_helpers.py
git commit -m "feat: add convection JIT helpers (Green/Deschamps, Ra, Nu, profile scan)"
```

---

## Task 4: Thomas Solver & Stefan Velocity

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`
- Modify: `EuropaProjectDJ/tests/test_batched_helpers.py`

- [ ] **Step 1: Write failing tests**

```python
class TestThomasAndStefan:

    def test_thomas_solve_identity(self):
        """Thomas solver on a trivial system: I*x = b → x = b."""
        from batched_solver import _thomas_solve
        n = 5
        a = np.zeros(n)  # lower
        b = np.ones(n)   # main
        c = np.zeros(n)  # upper
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = np.empty(n)
        _thomas_solve(a, b, c, d, x, n)
        np.testing.assert_allclose(x, d, atol=1e-14)

    def test_thomas_solve_tridiagonal(self):
        """Compare Thomas vs scipy.linalg.solve_banded on a known system."""
        from batched_solver import _thomas_solve
        from scipy.linalg import solve_banded
        n = 10
        np.random.seed(42)
        a_lower = np.random.randn(n)
        a_main = np.random.randn(n) + 5  # diagonally dominant
        a_upper = np.random.randn(n)
        rhs = np.random.randn(n)
        a_lower[n - 1] = 0.0  # sentinel
        a_upper[n - 1] = 0.0

        # scipy
        ab = np.zeros((3, n))
        ab[0, 1:] = a_upper[:n - 1]
        ab[1, :] = a_main
        ab[2, :-1] = a_lower[:n - 1]
        x_scipy = solve_banded((1, 1), ab, rhs.copy())

        # Thomas
        x_thomas = np.empty(n)
        _thomas_solve(a_lower.copy(), a_main.copy(), a_upper.copy(), rhs.copy(), x_thomas, n)
        np.testing.assert_allclose(x_thomas, x_scipy, atol=1e-10)

    def test_stefan_velocity(self):
        """Stefan velocity vs original."""
        from batched_solver import _stefan_velocity
        nz = 31
        H = 20e3
        dz = H / (nz - 1)
        T_melt = float(IcePhysics.basal_melting_point(H))
        T_col = np.linspace(104.0, T_melt, nz)
        q_ocean = 0.02
        k_basal = float(IcePhysics.effective_conductivity(T_col[-1]))
        expected = float(IcePhysics.stefan_velocity(T_col, dz, q_ocean, k_basal))
        result = _stefan_velocity(T_col, nz, dz, q_ocean)
        assert abs(result - expected) / abs(expected) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement Thomas solver and Stefan velocity**

```python
# Add to batched_solver.py

# =============================================================================
# 5. TRIDIAGONAL SOLVER
# =============================================================================

@njit
def _thomas_solve(a, b, c, d, x, n):
    """
    Thomas algorithm for tridiagonal system.
    a: lower diagonal (length n, a[n-1] unused sentinel)
    b: main diagonal (length n)
    c: upper diagonal (length n, c[n-1] unused sentinel)
    d: right-hand side (length n) — MODIFIED in-place as workspace
    x: solution output (length n)
    """
    # Forward elimination
    for i in range(1, n):
        if b[i - 1] == 0.0:
            continue
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Back substitution
    x[n - 1] = d[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]


# =============================================================================
# 6. STEFAN VELOCITY
# =============================================================================

@njit
def _stefan_velocity(T_col, nz, dz, q_ocean):
    """Stefan condition: db/dt = (k*dT/dz - q_ocean) / (rho*L)."""
    T_base = T_col[nz - 1]
    k_basal = 567.0 / T_base  # bare conductivity, no porosity/salt
    # 2nd order one-sided gradient
    dTdz = (3.0 * T_col[nz - 1] - 4.0 * T_col[nz - 2] + T_col[nz - 3]) / (2.0 * dz)
    q_cond = k_basal * dTdz
    rho_base = _density_ice(T_base)
    return (q_cond - q_ocean) / (rho_base * 334000.0)
```

- [ ] **Step 4: Run tests**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py::TestThomasAndStefan -v
```

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_helpers.py
git commit -m "feat: add Thomas solver and Stefan velocity JIT helpers"
```

---

## Task 5: _do_half_step Kernel

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`
- Create: `EuropaProjectDJ/tests/test_batched_solver.py`

This is the core kernel. Test it with `n_lat=1` against the original solver.

- [ ] **Step 1: Write failing single-step parity test (non-convective)**

```python
# test_batched_solver.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from constants import Thermal, Planetary, Rheology, HeatFlux
from Physics import IcePhysics
from Boundary_Conditions import FixedTemperature
from Solver import Thermal_Solver


class TestSingleStepParity:
    """Compare one step of batched solver vs original Thermal_Solver."""

    NX = 31
    H0 = 20e3
    DT = 1e11
    T_SURF = 104.0
    Q_OCEAN = 0.02

    def _run_original_step(self, use_convection=False):
        solver = Thermal_Solver(
            nx=self.NX, thickness=self.H0, dt=self.DT,
            surface_bc=FixedTemperature(self.T_SURF),
            use_convection=use_convection, rannacher_steps=4,
            coordinate_system='cartesian',
        )
        T_before = solver.T.copy()
        H_before = solver.H
        dbdt = solver.solve_step(self.Q_OCEAN)
        return T_before, H_before, solver.T.copy(), solver.H, dbdt

    def test_one_step_nonconvective(self):
        from batched_solver import batched_step_single_column
        T_before, H_before, T_after_orig, H_after_orig, dbdt_orig = \
            self._run_original_step(use_convection=False)

        T_col = T_before.copy()
        H_arr = np.array([H_before])
        T_grid = T_col.reshape(-1, 1)

        # Call batched solver for 1 step
        dbdt_result = batched_step_single_column(
            T_grid, H_arr, self.Q_OCEAN, self.T_SURF,
            self.DT, current_step=0, rannacher_steps=4,
            use_convection=False,
        )

        np.testing.assert_allclose(T_grid[:, 0], T_after_orig, atol=1e-6,
                                   err_msg="T profiles diverge after 1 step")
        assert abs(H_arr[0] - H_after_orig) < 1e-4, \
            f"H diverges: {H_arr[0]} vs {H_after_orig}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_solver.py::TestSingleStepParity::test_one_step_nonconvective -v
```

- [ ] **Step 3: Implement `_do_half_step` and a thin `batched_step_single_column` convenience wrapper**

Implement the full `_do_half_step` following spec Section 7 exactly:
- Geometry computation
- Explicit-side property evaluation (when theta < 1.0)
- Picard loop with array reset, implicit properties, assembly, Thomas solve, convergence
- Stefan update
- Return 10-tuple

Then implement `batched_step` following spec Section 8, and a convenience wrapper
`batched_step_single_column` that handles default parameter population from the
`constants` module for single-column testing.

This is the largest implementation step (~200 lines). The code follows the spec control
flow exactly — no creative interpretation.

- [ ] **Step 4: Run test**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_solver.py::TestSingleStepParity::test_one_step_nonconvective -v
```

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_solver.py
git commit -m "feat: implement _do_half_step and batched_step kernel"
```

---

## Task 6: Multi-Step & Convection Parity Tests

**Files:**
- Modify: `EuropaProjectDJ/tests/test_batched_solver.py`

- [ ] **Step 1: Write parity tests for remaining validation cases**

```python
class TestMultiStepParity:

    NX = 31
    H0 = 20e3
    DT = 1e11
    T_SURF = 104.0
    Q_OCEAN = 0.02
    N_STEPS = 10

    def test_rannacher_then_cn(self):
        """Run 6 steps (4 Rannacher + 2 CN), compare T and H."""
        # ... run original for 6 steps, run batched for 6 steps, compare

    def test_convective_parity(self):
        """use_convection=True, compare D_cond, Ra, Nu, T, H."""
        # ... run original with use_convection=True for 10 steps
        # ... run batched with use_convection_array=[True] for 10 steps
        # ... compare T, H, and convection diagnostics

    def test_spherical_parity(self):
        """Spherical geometry matches original."""
        # ... coordinate_system='spherical' vs is_spherical_array=[True]

    def test_identical_columns(self):
        """4 identical columns produce identical results."""
        # ... n_lat=4, same params, verify max diff < 1e-14

    def test_q_ocean_isolation(self):
        """Different q_ocean only affects Stefan, not T at step 1."""
        # ... n_lat=3, different q_ocean, T identical at step 1

    def test_mixed_ensemble(self):
        """Column 0 non-convective, column 1 convective, each matches solo run."""
        # ... n_lat=2, use_convection_array=[False, True]

    def test_andrade_parity(self):
        """Andrade tidal heating parity: batched vs original with Rheology.MODEL='Andrade'."""
        # ... run original with Andrade config for 10 steps
        # ... run batched with rheology_model_array=[1] for 10 steps
        # ... compare T, H, and tidal heating profiles (spec 11.7)

    def test_nu_ramp(self):
        """nu_ramp < 1 should reduce effective Nu toward 1."""
        # ... run batched with nu_ramp=0.5, compare against nu_ramp=1.0
        # ... verify D_cond and H differ, and Nu_eff = 1 + 0.5*(Nu-1)
```

- [ ] **Step 2: Run tests, debug any failures**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_solver.py -v
```

- [ ] **Step 3: Fix any parity issues found**

- [ ] **Step 4: Commit**

```bash
git add EuropaProjectDJ/tests/test_batched_solver.py EuropaProjectDJ/src/batched_solver.py
git commit -m "test: add multi-step and convection parity tests for batched solver"
```

---

## Task 7: Python Wrapper Class

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`

- [ ] **Step 1: Implement `BatchedSolverWrapper`**

```python
class BatchedSolverWrapper:
    """
    Python-side wrapper for the batched JIT solver.

    Manages: current_step tracking, default parameter population,
    gamma_val pre-computation, nz >= 3 validation, output array allocation.
    """
    def __init__(self, nz, n_lat, dt, rannacher_steps=4, physics_params=None):
        if nz < 3:
            raise ValueError(f"nz must be >= 3, got {nz}")
        self.nz = nz
        self.n_lat = n_lat
        self.dt = dt
        self.rannacher_steps = rannacher_steps
        self.current_step = 0
        # ... populate per-column arrays from physics_params or defaults
        # ... pre-compute gamma_val
        # ... allocate diagnostic output arrays

    def step(self, T_grid, H_array, q_ocean_array, T_surf_array):
        """Advance all columns by one logical step. Returns dbdt_array."""
        dbdt = batched_step(T_grid, H_array, ...)
        self.current_step += 1
        return dbdt

    def run_to_equilibrium(self, T_grid, H_array, q_ocean_array, T_surf_array,
                           threshold=1e-15, max_steps=1500):
        """Run until max|dbdt| < threshold."""
        ...
```

- [ ] **Step 2: Test wrapper drives the kernel correctly**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_solver.py -v
```

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py
git commit -m "feat: add BatchedSolverWrapper Python class"
```

---

## Task 8: Refactor _scan_profile to Accept Per-Column Viscosity Params

**Files:**
- Modify: `EuropaProjectDJ/src/batched_solver.py`
- Modify: `EuropaProjectDJ/tests/test_batched_helpers.py`

Task 3 hardcoded `d_del`, `D0v`, `D0b`, `d_molar` in `_scan_profile`. Now pass them
through from per-column arrays for Monte Carlo support.

- [ ] **Step 1: Update `_scan_profile` signature to accept all composite viscosity params**

- [ ] **Step 2: Update `_do_half_step` to thread the params through**

- [ ] **Step 3: Run full test suite**

```bash
cd EuropaProjectDJ && python -m pytest tests/test_batched_helpers.py tests/test_batched_solver.py -v
```

- [ ] **Step 4: Commit**

```bash
git add EuropaProjectDJ/src/batched_solver.py EuropaProjectDJ/tests/test_batched_helpers.py
git commit -m "refactor: pass per-column viscosity params through _scan_profile"
```

---

## Dependency Graph

```
Task 1 (thermo helpers)
  └─→ Task 2 (viscosity + tidal)
       └─→ Task 3 (convection helpers)
            └─→ Task 4 (Thomas + Stefan)
                 └─→ Task 5 (_do_half_step + batched_step)
                      ├─→ Task 7 (wrapper class)
                      └─→ Task 8 (refactor scan_profile params)
                           └─→ Task 6 (parity tests)
```

Tasks 7 and 8 can run in parallel after Task 5.
Task 6 (convection parity tests) depends on Task 8 (per-column viscosity params).
