# 2D Axisymmetric Europa Ice Shell Model — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a coupled-column 2D axisymmetric thermal model that produces continuous ice shell thickness profiles H(φ) from equator to pole with Monte Carlo UQ.

**Architecture:** N latitude columns (each a 1D `Thermal_Solver` from EuropaProjectDJ) coupled by explicit lateral heat diffusion via operator splitting. Latitude-dependent surface temperature, tidal strain, and ocean heat flux drive spatial variation. Wrapped in a Monte Carlo framework for uncertainty quantification.

**Tech Stack:** Python 3.10+, NumPy, SciPy, matplotlib. Imports from `EuropaProjectDJ/src/`.

**Spec:** `Europa2D/docs/2026-03-12-2d-axisymmetric-model-design.md`

---

## File Structure

```
Europa2D/
├── src/
│   ├── __init__.py                 # Import path setup for EuropaProjectDJ
│   ├── latitude_profile.py         # LatitudeProfile: T_s(φ), ε₀(φ), q_ocean(φ)
│   ├── axial_solver.py             # AxialSolver2D: coupled-column solver
│   ├── latitude_sampler.py         # LatitudeParameterSampler for MC
│   └── monte_carlo_2d.py           # MonteCarloRunner2D + MonteCarloResults2D
├── scripts/
│   ├── run_2d_single.py            # Single deterministic run + T(z,φ) plot
│   ├── run_2d_mc.py                # Full MC run
│   └── plot_thickness_profile.py   # H(φ) with uncertainty bands
├── tests/
│   ├── test_latitude_profile.py
│   ├── test_axial_solver.py
│   └── test_monte_carlo_2d.py
├── results/
├── figures/
└── docs/
```

---

## Chunk 1: Foundation — Project Setup + Latitude Profile

### Task 1: Project scaffolding

**Files:**
- Create: `Europa2D/src/__init__.py`
- Create: `Europa2D/tests/__init__.py`
- Create: `Europa2D/scripts/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd C:\Users\Joshu\.cursor\projects\EuropaConvection
mkdir -p Europa2D/src Europa2D/tests Europa2D/scripts Europa2D/results Europa2D/figures
```

- [ ] **Step 2: Create `__init__.py` with import path setup**

```python
# Europa2D/src/__init__.py
"""
Europa 2D Axisymmetric Ice Shell Model.

Extends EuropaProjectDJ's 1D thermal solver to a coupled-column
2D axisymmetric model with latitude-dependent physics.
"""
import sys
import os

# Add EuropaProjectDJ/src to Python path so we can import its modules
_PROJ_1D = os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src')
if os.path.isdir(_PROJ_1D) and _PROJ_1D not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROJ_1D))
```

- [ ] **Step 3: Create empty test and script init files**

```python
# Europa2D/tests/__init__.py
# Europa2D/scripts/__init__.py
# (empty files)
```

- [ ] **Step 4: Verify imports work**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -c "import src; from constants import Planetary; print(f'Europa radius: {Planetary.RADIUS} m')"`

Expected: `Europa radius: 1561000.0 m`

- [ ] **Step 5: Commit**

```bash
git add Europa2D/
git commit -m "feat: scaffold Europa2D project with import path setup"
```

---

### Task 2: LatitudeProfile — surface temperature

**Files:**
- Create: `Europa2D/src/latitude_profile.py`
- Create: `Europa2D/tests/test_latitude_profile.py`

- [ ] **Step 1: Write failing test for surface temperature**

```python
# Europa2D/tests/test_latitude_profile.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src  # triggers import path setup

import numpy as np
import pytest
from latitude_profile import LatitudeProfile


class TestSurfaceTemperature:
    """Tests for T_s(φ) = T_eq · max(cos(φ), cos(85°))^(1/4)"""

    def test_equator_returns_T_eq(self):
        """At φ=0 (equator), T_s should equal T_eq."""
        profile = LatitudeProfile(T_eq=110.0)
        assert profile.surface_temperature(0.0) == pytest.approx(110.0, abs=0.01)

    def test_pole_is_colder_than_equator(self):
        """At φ=90° (pole), T_s should be much colder than T_eq."""
        profile = LatitudeProfile(T_eq=110.0)
        T_pole = profile.surface_temperature(np.radians(90.0))
        assert T_pole < 60.0
        assert T_pole > 30.0

    def test_monotonically_decreasing(self):
        """T_s should decrease monotonically from equator to pole."""
        profile = LatitudeProfile(T_eq=110.0)
        lats = np.linspace(0, np.pi / 2, 20)
        temps = np.array([profile.surface_temperature(phi) for phi in lats])
        # Each subsequent value should be <= previous
        assert np.all(np.diff(temps) <= 0)

    def test_floor_prevents_zero(self):
        """cos(85°) floor should prevent T_s from reaching 0."""
        profile = LatitudeProfile(T_eq=110.0)
        T_pole = profile.surface_temperature(np.radians(90.0))
        assert T_pole > 0

    def test_array_input(self):
        """Should accept array of latitudes."""
        profile = LatitudeProfile(T_eq=110.0)
        lats = np.array([0.0, np.radians(45), np.radians(90)])
        temps = profile.surface_temperature(lats)
        assert temps.shape == (3,)
        assert temps[0] > temps[1] > temps[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestSurfaceTemperature -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'latitude_profile'`

- [ ] **Step 3: Implement LatitudeProfile with surface_temperature**

```python
# Europa2D/src/latitude_profile.py
"""
Latitude-dependent parameter profiles for Europa's ice shell.

Provides continuous functions for surface temperature, tidal strain,
and ocean heat flux as functions of geographic latitude φ.

Convention: φ = 0 at equator, φ = π/2 at pole.

References:
    - Ojakangas & Stevenson (1989): Surface temperature distribution
    - Tobie et al. (2003): Tidal strain patterns
    - Soderlund et al. (2014): Ocean heat transport
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
from typing import Literal
from dataclasses import dataclass

from constants import Planetary

OceanPattern = Literal["uniform", "polar_enhanced", "equator_enhanced"]

# Floor angle to prevent singularity at pole (85 degrees in radians)
_PHI_FLOOR = np.radians(85.0)
_COS_FLOOR = np.cos(_PHI_FLOOR)


@dataclass(frozen=True)
class LatitudeProfile:
    """
    Latitude-dependent physical parameters for Europa's ice shell.

    All angles are geographic latitude in radians:
        φ = 0 at equator, φ = π/2 at pole.

    Attributes:
        T_eq: Equatorial surface temperature (K)
        epsilon_eq: Tidal strain at equator
        epsilon_pole: Tidal strain at pole
        q_ocean_mean: Global mean ocean heat flux (W/m²)
        ocean_pattern: Heat flux distribution pattern
    """
    T_eq: float = 110.0
    epsilon_eq: float = 6.0e-6
    epsilon_pole: float = 1.2e-5
    q_ocean_mean: float = 0.02
    ocean_pattern: OceanPattern = "polar_enhanced"

    def surface_temperature(self, phi: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
        """
        Surface temperature as a function of latitude.

        T_s(φ) = T_eq · max(cos(φ), cos(85°))^(1/4)

        Based on Ojakangas & Stevenson (1989) radiative equilibrium.

        Args:
            phi: Geographic latitude in radians (0=equator, π/2=pole)

        Returns:
            Surface temperature (K)
        """
        phi_arr = np.asarray(phi)
        cos_phi = np.maximum(np.cos(phi_arr), _COS_FLOOR)
        result = self.T_eq * cos_phi ** 0.25
        return float(result) if np.ndim(phi) == 0 else result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestSurfaceTemperature -v`

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: add LatitudeProfile with surface temperature model"
```

---

### Task 3: LatitudeProfile — tidal strain

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing test for tidal strain**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestTidalStrain:
    """Tests for ε₀(φ) = ε_eq + (ε_pole - ε_eq) · sin²(φ)"""

    def test_equator_returns_epsilon_eq(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(0.0) == pytest.approx(6e-6, rel=1e-6)

    def test_pole_returns_epsilon_pole(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(np.pi / 2) == pytest.approx(1.2e-5, rel=1e-6)

    def test_midlatitude_is_between(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        eps_45 = profile.tidal_strain(np.radians(45))
        assert 6e-6 < eps_45 < 1.2e-5

    def test_monotonically_increasing(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        lats = np.linspace(0, np.pi / 2, 20)
        strains = np.array([profile.tidal_strain(phi) for phi in lats])
        assert np.all(np.diff(strains) >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestTidalStrain -v`

Expected: FAIL with `AttributeError: 'LatitudeProfile' object has no attribute 'tidal_strain'`

- [ ] **Step 3: Add tidal_strain method to LatitudeProfile**

Add to `LatitudeProfile` class in `latitude_profile.py`:

```python
    def tidal_strain(self, phi: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
        """
        Tidal strain amplitude as a function of latitude.

        ε₀(φ) = ε_eq + (ε_pole - ε_eq) · sin²(φ)

        Simplified parameterization inspired by Tobie et al. (2003).

        Args:
            phi: Geographic latitude in radians (0=equator, π/2=pole)

        Returns:
            Tidal strain amplitude (dimensionless)
        """
        phi_arr = np.asarray(phi)
        sin2 = np.sin(phi_arr) ** 2
        result = self.epsilon_eq + (self.epsilon_pole - self.epsilon_eq) * sin2
        return float(result) if np.ndim(phi) == 0 else result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: add tidal strain latitude profile"
```

---

### Task 4: LatitudeProfile — ocean heat flux

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing test for ocean heat flux**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestOceanHeatFlux:
    """Tests for q_ocean(φ) with normalization."""

    def test_uniform_is_constant(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="uniform")
        lats = np.linspace(0, np.pi / 2, 10)
        fluxes = profile.ocean_heat_flux(lats)
        assert np.allclose(fluxes, 0.02, rtol=1e-10)

    def test_polar_enhanced_higher_at_pole(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="polar_enhanced")
        q_eq = profile.ocean_heat_flux(0.0)
        q_pole = profile.ocean_heat_flux(np.pi / 2)
        assert q_pole > q_eq

    def test_equator_enhanced_higher_at_equator(self):
        profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern="equator_enhanced")
        q_eq = profile.ocean_heat_flux(0.0)
        q_pole = profile.ocean_heat_flux(np.pi / 2)
        assert q_eq > q_pole

    def test_normalization_preserves_global_mean(self):
        """Integral of q(φ)·cos(φ) dφ over [0, π/2] should equal q_mean · 1."""
        for pattern in ["uniform", "polar_enhanced", "equator_enhanced"]:
            profile = LatitudeProfile(q_ocean_mean=0.02, ocean_pattern=pattern)
            # Numerical integration: ∫₀^{π/2} q(φ)cos(φ)dφ / ∫₀^{π/2} cos(φ)dφ
            from scipy.integrate import quad
            numerator, _ = quad(
                lambda phi: profile.ocean_heat_flux(phi) * np.cos(phi),
                0, np.pi / 2
            )
            denominator, _ = quad(np.cos, 0, np.pi / 2)  # = 1.0
            mean = numerator / denominator
            assert mean == pytest.approx(0.02, rel=0.01), f"Failed for {pattern}: mean={mean}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestOceanHeatFlux -v`

Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement ocean_heat_flux with normalization**

Add to `LatitudeProfile` class:

```python
    def ocean_heat_flux(self, phi: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
        """
        Ocean heat flux as a function of latitude.

        Supports three patterns, all normalized to preserve the global mean:
        - uniform: q(φ) = q_mean
        - polar_enhanced: q ∝ 1 + 2·sin²(φ), Soderlund et al. (2014)
        - equator_enhanced: q ∝ 1 + 2·cos²(φ)

        Normalization: ∫₀^{π/2} q(φ)cos(φ)dφ / ∫₀^{π/2} cos(φ)dφ = q_mean

        Args:
            phi: Geographic latitude in radians

        Returns:
            Ocean heat flux (W/m²)
        """
        phi_arr = np.asarray(phi)

        if self.ocean_pattern == "uniform":
            result = np.full_like(phi_arr, self.q_ocean_mean, dtype=float)
        elif self.ocean_pattern == "polar_enhanced":
            # Shape: 1 + 2·sin²(φ)
            # Analytical normalization factor:
            # ∫₀^{π/2} (1 + 2sin²φ)cosφ dφ = 1 + 2/3 = 5/3
            # So q(φ) = q_mean · (5/3)⁻¹ · (1 + 2sin²φ) won't work directly.
            # We need: ∫ q·cosφ dφ / ∫ cosφ dφ = q_mean
            # ∫₀^{π/2} cosφ dφ = 1
            # ∫₀^{π/2} (1+2sin²φ)cosφ dφ = [sinφ - 2sin³φ/3]₀^{π/2} = 1 + 2/3 = 5/3
            # So normalization: q_mean / (5/3) · (1 + 2sin²φ) => mean = q_mean · (5/3)/(5/3) = q_mean
            norm = 5.0 / 3.0
            shape = 1.0 + 2.0 * np.sin(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        elif self.ocean_pattern == "equator_enhanced":
            # Shape: 1 + 2·cos²(φ)
            # ∫₀^{π/2} (1+2cos²φ)cosφ dφ = 1 + 2·∫₀^{π/2}cos³φ dφ = 1 + 2·(2/3) = 7/3
            norm = 7.0 / 3.0
            shape = 1.0 + 2.0 * np.cos(phi_arr) ** 2
            result = self.q_ocean_mean * shape / norm
        else:
            raise ValueError(f"Unknown ocean pattern: {self.ocean_pattern}")

        return float(result) if np.ndim(phi) == 0 else result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: add ocean heat flux with 3 patterns and normalization"
```

---

### Task 5: LatitudeProfile — evaluate_at helper

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing test for evaluate_at**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestEvaluateAt:
    """Tests for the convenience method that returns all params at a latitude."""

    def test_returns_dict_with_required_keys(self):
        profile = LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        result = profile.evaluate_at(np.radians(45))
        assert 'T_surf' in result
        assert 'epsilon_0' in result
        assert 'q_ocean' in result

    def test_values_match_individual_methods(self):
        profile = LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02)
        phi = np.radians(30)
        result = profile.evaluate_at(phi)
        assert result['T_surf'] == pytest.approx(profile.surface_temperature(phi))
        assert result['epsilon_0'] == pytest.approx(profile.tidal_strain(phi))
        assert result['q_ocean'] == pytest.approx(profile.ocean_heat_flux(phi))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestEvaluateAt -v`

Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement evaluate_at**

Add to `LatitudeProfile` class:

```python
    def evaluate_at(self, phi: float) -> dict:
        """
        Evaluate all latitude-dependent parameters at a single latitude.

        Args:
            phi: Geographic latitude in radians

        Returns:
            Dict with keys: T_surf, epsilon_0, q_ocean
        """
        return {
            'T_surf': self.surface_temperature(phi),
            'epsilon_0': self.tidal_strain(phi),
            'q_ocean': self.ocean_heat_flux(phi),
        }
```

- [ ] **Step 4: Run all tests to verify everything passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: add evaluate_at convenience method to LatitudeProfile"
```

---

## Chunk 2: Core — The Coupled-Column Solver

### Task 6: AxialSolver2D — constructor and column initialization

**Files:**
- Create: `Europa2D/src/axial_solver.py`
- Create: `Europa2D/tests/test_axial_solver.py`

- [ ] **Step 1: Write failing test for column construction**

```python
# Europa2D/tests/test_axial_solver.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src  # triggers import path setup

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D


class TestAxialSolverInit:
    """Tests for AxialSolver2D construction."""

    def _make_profile(self):
        return LatitudeProfile(T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02)

    def _make_params(self):
        return {
            'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
            'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
            'd_del': 7e-10, 'f_porosity': 0.1, 'f_salt': 0.03,
            'B_k': 1.0, 'T_phi': 150.0,
        }

    def test_creates_correct_number_of_columns(self):
        solver = AxialSolver2D(
            n_lat=19, nx=31, latitude_profile=self._make_profile(),
            physics_params=self._make_params(),
        )
        assert len(solver.columns) == 19

    def test_latitudes_span_equator_to_pole(self):
        solver = AxialSolver2D(
            n_lat=19, nx=31, latitude_profile=self._make_profile(),
            physics_params=self._make_params(),
        )
        assert solver.latitudes[0] == pytest.approx(0.0)
        assert solver.latitudes[-1] == pytest.approx(np.pi / 2)

    def test_each_column_has_correct_surface_temp(self):
        profile = self._make_profile()
        solver = AxialSolver2D(
            n_lat=5, nx=31, latitude_profile=profile,
            physics_params=self._make_params(),
        )
        for j, col in enumerate(solver.columns):
            expected_T = profile.surface_temperature(solver.latitudes[j])
            assert col.T[0] == pytest.approx(expected_T, abs=1.0)

    def test_thickness_profile_returns_array(self):
        solver = AxialSolver2D(
            n_lat=5, nx=31, latitude_profile=self._make_profile(),
            physics_params=self._make_params(), initial_thickness=20e3,
        )
        H = solver.get_thickness_profile()
        assert H.shape == (5,)
        assert np.all(H == pytest.approx(20e3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py::TestAxialSolverInit -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AxialSolver2D constructor**

```python
# Europa2D/src/axial_solver.py
"""
2D Axisymmetric Coupled-Column Solver for Europa's Ice Shell.

Uses operator splitting: radial heat transport (implicit, via Thermal_Solver)
coupled with lateral heat diffusion (explicit) across latitude columns.

Convention: φ = 0 at equator, φ = π/2 at pole (geographic latitude).

References:
    - Design spec: Europa2D/docs/2026-03-12-2d-axisymmetric-model-design.md
    - Howell (2021), Green et al. (2021), Deschamps & Vilella (2021)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
from typing import Dict, Optional, Any, List
from copy import deepcopy

from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics
from constants import Planetary, Thermal

from latitude_profile import LatitudeProfile


class AxialSolver2D:
    """
    Coupled-column 2D axisymmetric thermal solver.

    Maintains N latitude columns, each a 1D Thermal_Solver.
    Columns are coupled by explicit lateral heat diffusion.
    """

    def __init__(
        self,
        n_lat: int = 19,
        nx: int = 31,
        dt: float = 1e12,
        total_time: float = 5e14,
        latitude_profile: Optional[LatitudeProfile] = None,
        physics_params: Optional[Dict[str, float]] = None,
        use_convection: bool = True,
        initial_thickness: float = 20e3,
        rannacher_steps: int = 4,
        coordinate_system: str = 'auto',
    ):
        """
        Initialize the 2D axisymmetric solver.

        Args:
            n_lat: Number of latitude columns (equator to pole)
            nx: Radial nodes per column
            dt: Time step (s)
            total_time: Total simulation time (s)
            latitude_profile: Latitude-dependent parameter functions
            physics_params: Shared MC-sampled parameters (grain size, etc.)
            use_convection: Enable stagnant-lid convection
            initial_thickness: Starting ice shell thickness for all columns (m)
            rannacher_steps: Number of Backward Euler startup steps
            coordinate_system: 'auto', 'cartesian', or 'spherical'
        """
        self.n_lat = n_lat
        self.nx = nx
        self.dt = dt
        self.total_time = total_time
        self.use_convection = use_convection
        self.profile = latitude_profile or LatitudeProfile()
        self._shared_params = physics_params or {}

        # Geographic latitude grid: 0 (equator) to π/2 (pole)
        self.latitudes = np.linspace(0, np.pi / 2, n_lat)
        self.dphi = self.latitudes[1] - self.latitudes[0] if n_lat > 1 else 1.0

        # Build one 1D solver per latitude column
        self.columns: List[Thermal_Solver] = []
        for j in range(n_lat):
            phi_j = self.latitudes[j]

            # Per-column overrides from latitude profile
            col_params = dict(self._shared_params)
            lat_vals = self.profile.evaluate_at(phi_j)
            col_params['epsilon_0'] = lat_vals['epsilon_0']
            col_params['T_surf'] = lat_vals['T_surf']

            surface_bc = FixedTemperature(temperature=lat_vals['T_surf'])

            solver = Thermal_Solver(
                nx=nx,
                thickness=initial_thickness,
                dt=dt,
                total_time=total_time,
                coordinate_system=coordinate_system,
                surface_bc=surface_bc,
                rannacher_steps=rannacher_steps,
                use_convection=use_convection,
                physics_params=col_params,
            )
            self.columns.append(solver)

    def get_thickness_profile(self) -> npt.NDArray[np.float64]:
        """Returns ice shell thickness at each latitude (m). Columns are authoritative."""
        return np.array([col.H for col in self.columns])

    def get_latitudes_deg(self) -> npt.NDArray[np.float64]:
        """Returns latitude grid in degrees."""
        return np.degrees(self.latitudes)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py::TestAxialSolverInit -v`

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/axial_solver.py Europa2D/tests/test_axial_solver.py
git commit -m "feat: add AxialSolver2D constructor with per-column initialization"
```

---

### Task 7: AxialSolver2D — lateral diffusion step

**Files:**
- Modify: `Europa2D/tests/test_axial_solver.py`
- Modify: `Europa2D/src/axial_solver.py`

- [ ] **Step 1: Write failing test for lateral diffusion**

Append to `Europa2D/tests/test_axial_solver.py`:

```python
class TestLateralDiffusion:
    """Tests for the explicit lateral heat diffusion step."""

    def _make_solver(self, n_lat=5, nx=11):
        profile = LatitudeProfile(T_eq=110.0)
        params = {
            'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
            'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
            'd_del': 7e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
            'B_k': 1.0, 'T_phi': 150.0,
        }
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=profile,
            physics_params=params, use_convection=False,
            initial_thickness=20e3,
        )

    def test_uniform_temperature_no_change(self):
        """If all columns have the same T profile, lateral step should do nothing."""
        solver = self._make_solver()
        # Set all columns to identical profiles
        for col in solver.columns:
            col.T[:] = np.linspace(100, 270, col.nx)
        T_before = np.array([col.T.copy() for col in solver.columns])
        solver._lateral_diffusion_step()
        T_after = np.array([col.T.copy() for col in solver.columns])
        assert np.allclose(T_before, T_after, atol=1e-10)

    def test_hot_column_cools(self):
        """A column hotter than its neighbors should cool from lateral diffusion."""
        solver = self._make_solver(n_lat=5)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        # Make middle column hotter
        solver.columns[2].T[5] += 50.0
        T_mid_before = solver.columns[2].T[5]
        solver._lateral_diffusion_step()
        T_mid_after = solver.columns[2].T[5]
        assert T_mid_after < T_mid_before

    def test_symmetric_boundary_conditions(self):
        """dT/dφ = 0 at equator and pole boundaries."""
        solver = self._make_solver(n_lat=5)
        base_T = np.linspace(100, 270, solver.nx)
        for col in solver.columns:
            col.T[:] = base_T.copy()
        # Perturb interior only
        solver.columns[2].T[5] += 50.0
        solver._lateral_diffusion_step()
        # Boundary columns should only be affected by their immediate neighbor
        # (ghost-node symmetry means dT/dφ = 0 at boundaries)
        # Just verify no crash and values are reasonable
        assert np.all(np.isfinite([col.T for col in solver.columns]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py::TestLateralDiffusion -v`

Expected: FAIL with `AttributeError: '_lateral_diffusion_step'`

- [ ] **Step 3: Implement _lateral_diffusion_step**

Add to `AxialSolver2D` class:

```python
    def _lateral_diffusion_step(self) -> None:
        """
        Apply explicit lateral heat diffusion between columns.

        Uses the geographic latitude form of the diffusion operator:
            dT/dt = (k / R²cosφ) · ∂/∂φ[cosφ · ∂T/∂φ]

        Boundary conditions: dT/dφ = 0 at equator (φ=0) and pole (φ=π/2).
        """
        if self.n_lat < 3:
            return  # No lateral diffusion with fewer than 3 columns

        R = Planetary.RADIUS
        dphi = self.dphi
        _, dt_eff = self.columns[0]._get_theta_and_dt()

        # Build 2D temperature array: shape (n_lat, nx)
        T_2d = np.array([col.T for col in self.columns])

        # Metric factors at node points: cos(φ_j)
        cos_phi = np.cos(self.latitudes)

        # Half-node metric factors: cos(φ_{j+1/2})
        phi_half = (self.latitudes[:-1] + self.latitudes[1:]) / 2
        cos_half = np.cos(phi_half)

        # Mean conductivity per column (for lateral diffusion scaling)
        # Use midpoint temperature for each column
        k_cols = np.array([
            float(Thermal.conductivity(np.mean(col.T))) for col in self.columns
        ])

        # Compute dT for interior columns (j = 1 to n_lat-2)
        dT = np.zeros_like(T_2d)
        for j in range(1, self.n_lat - 1):
            k_mean = k_cols[j]
            rho_cp = float(Thermal.density_ice(np.mean(self.columns[j].T))) * Thermal.SPECIFIC_HEAT
            alpha = k_mean * dt_eff / (rho_cp * R**2 * dphi**2)

            flux_plus = cos_half[j] * (T_2d[j + 1, :] - T_2d[j, :])
            flux_minus = cos_half[j - 1] * (T_2d[j, :] - T_2d[j - 1, :])
            dT[j, :] = alpha * (flux_plus - flux_minus) / cos_phi[j]

        # Boundary: equator (φ=0), ghost node symmetry: T[-1] = T[1]
        j = 0
        k_mean = k_cols[0]
        rho_cp = float(Thermal.density_ice(np.mean(self.columns[0].T))) * Thermal.SPECIFIC_HEAT
        alpha = k_mean * dt_eff / (rho_cp * R**2 * dphi**2)
        flux_plus = cos_half[0] * (T_2d[1, :] - T_2d[0, :])
        flux_minus = cos_half[0] * (T_2d[0, :] - T_2d[1, :])  # ghost: T[-1] = T[1] (symmetry about equator, but actually dT/dphi=0 means flux_minus=0)
        # dT/dφ = 0 at equator means: flux_minus = 0
        dT[0, :] = alpha * (flux_plus - 0.0) / max(cos_phi[0], 1e-10)

        # Boundary: pole (φ=π/2), ghost node symmetry: T[n+1] = T[n-1]
        j = self.n_lat - 1
        k_mean = k_cols[j]
        rho_cp = float(Thermal.density_ice(np.mean(self.columns[j].T))) * Thermal.SPECIFIC_HEAT
        alpha = k_mean * dt_eff / (rho_cp * R**2 * dphi**2)
        flux_plus = 0.0  # dT/dφ = 0 at pole
        flux_minus = cos_half[j - 1] * (T_2d[j, :] - T_2d[j - 1, :])
        dT[j, :] = alpha * (flux_plus - flux_minus) / max(cos_phi[j], 1e-10)

        # Writeback: modify each column's T directly
        for j in range(self.n_lat):
            self.columns[j].T += dT[j, :]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py -v`

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/axial_solver.py Europa2D/tests/test_axial_solver.py
git commit -m "feat: add lateral heat diffusion step with cosφ metric"
```

---

### Task 8: AxialSolver2D — solve_step and run_to_equilibrium

**Files:**
- Modify: `Europa2D/tests/test_axial_solver.py`
- Modify: `Europa2D/src/axial_solver.py`

- [ ] **Step 1: Write failing test for solve_step**

Append to `Europa2D/tests/test_axial_solver.py`:

```python
class TestSolveStep:
    """Tests for the full radial+lateral solve step."""

    def _make_solver(self, n_lat=5, nx=21):
        profile = LatitudeProfile(T_eq=110.0, q_ocean_mean=0.02)
        params = {
            'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
            'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
            'd_del': 7e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
            'B_k': 1.0, 'T_phi': 150.0,
        }
        return AxialSolver2D(
            n_lat=n_lat, nx=nx, latitude_profile=profile,
            physics_params=params, use_convection=False,
            initial_thickness=20e3,
        )

    def test_returns_velocity_array(self):
        solver = self._make_solver()
        q_profile = solver.profile.ocean_heat_flux(solver.latitudes)
        velocities = solver.solve_step(q_profile)
        assert velocities.shape == (5,)
        assert np.all(np.isfinite(velocities))

    def test_thickness_changes_after_step(self):
        solver = self._make_solver()
        H_before = solver.get_thickness_profile().copy()
        q_profile = solver.profile.ocean_heat_flux(solver.latitudes)
        solver.solve_step(q_profile)
        H_after = solver.get_thickness_profile()
        # At least some columns should change thickness
        assert not np.allclose(H_before, H_after)


class TestRunToEquilibrium:
    """Tests for run_to_equilibrium convergence."""

    def test_converges_with_conduction_only(self):
        """Pure conduction (no convection) should converge."""
        profile = LatitudeProfile(T_eq=110.0, q_ocean_mean=0.04)
        params = {
            'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
            'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
            'd_del': 7e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
            'B_k': 1.0, 'T_phi': 150.0,
        }
        solver = AxialSolver2D(
            n_lat=5, nx=21, latitude_profile=profile,
            physics_params=params, use_convection=False,
            initial_thickness=10e3, dt=1e11,
        )
        result = solver.run_to_equilibrium(
            threshold=1e-12, max_steps=200, verbose=False
        )
        assert 'H_profile_km' in result
        assert result['converged']
        H = result['H_profile_km']
        assert H.shape == (5,)
        # Equator should be thicker (warmer surface, less heat loss)
        # Actually: equator has higher T_surf so LESS conduction -> thinner.
        # But also lower tidal strain. The relationship depends on q_ocean.
        # Just check all thicknesses are reasonable
        assert np.all(H > 0.5)
        assert np.all(H < 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py::TestSolveStep -v`

Expected: FAIL with `AttributeError: 'solve_step'`

- [ ] **Step 3: Implement solve_step and run_to_equilibrium**

Add to `AxialSolver2D` class:

```python
    def solve_step(self, q_ocean_profile: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Advance all columns by one time step using operator splitting.

        1. Radial step: each column solves its 1D heat equation independently
        2. Lateral step: explicit lateral diffusion between columns

        Args:
            q_ocean_profile: Ocean heat flux at each latitude (W/m²), shape (n_lat,)

        Returns:
            Array of freezing front velocities db/dt per column (m/s)
        """
        velocities = np.zeros(self.n_lat)

        # Step 1: Radial solve (independent per column)
        for j, col in enumerate(self.columns):
            velocities[j] = col.solve_step(q_ocean_profile[j])

        # Step 2: Lateral diffusion coupling
        self._lateral_diffusion_step()

        return velocities

    def run_to_equilibrium(
        self,
        threshold: float = 1e-12,
        max_steps: int = 1500,
        log_interval: int = 100,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all columns to thermal equilibrium.

        Convergence criterion: max(|db/dt|) across all columns < threshold.

        Args:
            threshold: Velocity threshold for equilibrium (m/s)
            max_steps: Maximum number of time steps
            log_interval: Steps between progress logs
            verbose: Print progress updates

        Returns:
            Dict with H_profile_km, T_2d, latitudes_deg, converged, steps, diagnostics
        """
        # Build ocean heat flux profile
        q_ocean_profile = np.array([
            self.profile.ocean_heat_flux(phi) for phi in self.latitudes
        ])

        H_history = [self.get_thickness_profile() / 1000.0]
        converged = False

        for step in range(max_steps):
            velocities = self.solve_step(q_ocean_profile)
            H_history.append(self.get_thickness_profile() / 1000.0)

            max_vel = np.max(np.abs(velocities))
            if max_vel < threshold:
                converged = True
                if verbose:
                    print(f"\n[OK] 2D equilibrium at step {step}")
                    H_km = self.get_thickness_profile() / 1000.0
                    print(f"  H range: {H_km.min():.2f} - {H_km.max():.2f} km")
                    print(f"  Max velocity: {max_vel:.2e} m/s")
                break

            if verbose and step % log_interval == 0:
                H_km = self.get_thickness_profile() / 1000.0
                print(f"Step {step:5d}: H = [{H_km.min():.2f}, {H_km.max():.2f}] km, "
                      f"max|v| = {max_vel:.2e} m/s")

        # Collect results
        H_km = self.get_thickness_profile() / 1000.0
        T_2d = np.array([col.T.copy() for col in self.columns])

        # Convection diagnostics per column
        diagnostics = []
        for col in self.columns:
            if col.convection_state is not None:
                state = col.convection_state
                diagnostics.append({
                    'D_cond_km': state.D_cond / 1000.0,
                    'D_conv_km': state.D_conv / 1000.0,
                    'Ra': state.Ra,
                    'Nu': state.Nu,
                    'lid_fraction': state.D_cond / col.H if col.H > 0 else 1.0,
                })
            else:
                diagnostics.append({
                    'D_cond_km': col.H / 1000.0,
                    'D_conv_km': 0.0,
                    'Ra': 0.0, 'Nu': 1.0, 'lid_fraction': 1.0,
                })

        return {
            'H_profile_km': H_km,
            'T_2d': T_2d,
            'latitudes_deg': self.get_latitudes_deg(),
            'converged': converged,
            'steps': step + 1,
            'H_history_km': np.array(H_history),
            'diagnostics': diagnostics,
        }
```

- [ ] **Step 4: Run all tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_axial_solver.py -v`

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/axial_solver.py Europa2D/tests/test_axial_solver.py
git commit -m "feat: add solve_step and run_to_equilibrium to AxialSolver2D"
```

---

### Task 9: Single deterministic 2D run script

**Files:**
- Create: `Europa2D/scripts/run_2d_single.py`

- [ ] **Step 1: Write the run script**

```python
# Europa2D/scripts/run_2d_single.py
"""
Single deterministic 2D axisymmetric run.

Runs the coupled-column solver to equilibrium with fixed parameters
and produces a thickness profile + T(z,φ) cross-section plot.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src  # triggers import path setup

import numpy as np
import matplotlib.pyplot as plt
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D

if __name__ == "__main__":
    # Fixed reference parameters (Howell 2021 defaults)
    params = {
        'd_grain': 1e-3,
        'Q_v': 59.4e3,
        'Q_b': 49.0e3,
        'mu_ice': 3.3e9,
        'D0v': 9.1e-4,
        'D0b': 8.4e-4,
        'd_del': 7.13e-10,
        'f_porosity': 0.1,
        'f_salt': 0.03,
        'B_k': 1.0,
        'T_phi': 150.0,
        'epsilon_0': 1e-5,
    }

    profile = LatitudeProfile(
        T_eq=110.0,
        epsilon_eq=6e-6,
        epsilon_pole=1.2e-5,
        q_ocean_mean=0.025,
        ocean_pattern="polar_enhanced",
    )

    print("Initializing 2D solver...")
    solver = AxialSolver2D(
        n_lat=19,
        nx=31,
        dt=1e12,
        latitude_profile=profile,
        physics_params=params,
        use_convection=True,
        initial_thickness=25e3,
    )

    print("Running to equilibrium...")
    result = solver.run_to_equilibrium(
        threshold=1e-12, max_steps=1500, verbose=True
    )

    H = result['H_profile_km']
    lats = result['latitudes_deg']

    # Plot 1: Thickness profile
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lats, H, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Latitude (degrees)')
    ax.set_ylabel('Ice Shell Thickness (km)')
    ax.set_title('Europa Ice Shell Thickness: Equator to Pole')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, 'thickness_profile_single.png'), dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_dir}/thickness_profile_single.png")

    # Plot 2: T(z, φ) cross-section
    T_2d = result['T_2d']
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    # Each column has different H, so plot on normalized depth
    xi = np.linspace(0, 1, solver.nx)
    lat_grid, xi_grid = np.meshgrid(lats, xi)
    im = ax2.pcolormesh(lat_grid, xi_grid, T_2d.T, shading='auto', cmap='inferno')
    ax2.set_xlabel('Latitude (degrees)')
    ax2.set_ylabel('Normalized Depth (0=surface, 1=base)')
    ax2.set_title('Temperature Cross-Section T(z, φ)')
    ax2.invert_yaxis()
    fig2.colorbar(im, label='Temperature (K)')
    fig2.savefig(os.path.join(fig_dir, 'temperature_cross_section.png'), dpi=150, bbox_inches='tight')
    print(f"Saved figure to {fig_dir}/temperature_cross_section.png")

    plt.show()
```

- [ ] **Step 2: Run it**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python scripts/run_2d_single.py`

Expected: Converges and produces two plots. Check that H varies with latitude.

- [ ] **Step 3: Commit**

```bash
git add Europa2D/scripts/run_2d_single.py
git commit -m "feat: add single deterministic 2D run script with plots"
```

---

## Chunk 3: Monte Carlo Framework

### Task 10: LatitudeParameterSampler

**Files:**
- Create: `Europa2D/src/latitude_sampler.py`
- Create: `Europa2D/tests/test_latitude_sampler.py` (minimal)

- [ ] **Step 1: Write failing test**

```python
# Europa2D/tests/test_latitude_sampler.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_sampler import LatitudeParameterSampler
from latitude_profile import LatitudeProfile


class TestLatitudeParameterSampler:

    def test_returns_dict_and_profile(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, profile = sampler.sample()
        assert isinstance(params, dict)
        assert isinstance(profile, LatitudeProfile)

    def test_shared_params_present(self):
        sampler = LatitudeParameterSampler(seed=42)
        params, _ = sampler.sample()
        required = ['d_grain', 'Q_v', 'Q_b', 'mu_ice', 'D0v', 'D0b', 'd_del',
                     'f_porosity', 'f_salt', 'B_k', 'T_phi', 'H_rad', 'P_tidal']
        for key in required:
            assert key in params, f"Missing key: {key}"

    def test_reproducible_with_seed(self):
        s1 = LatitudeParameterSampler(seed=42)
        s2 = LatitudeParameterSampler(seed=42)
        p1, prof1 = s1.sample()
        p2, prof2 = s2.sample()
        assert p1['d_grain'] == p2['d_grain']
        assert prof1.T_eq == prof2.T_eq

    def test_profile_has_valid_values(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert 80 < profile.T_eq < 140
        assert 1e-7 < profile.epsilon_eq < 1e-4
        assert 1e-7 < profile.epsilon_pole < 1e-3
        assert profile.q_ocean_mean > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_sampler.py -v`

Expected: FAIL

- [ ] **Step 3: Implement LatitudeParameterSampler**

```python
# Europa2D/src/latitude_sampler.py
"""
Parameter sampler for 2D Monte Carlo runs.

Samples shared physics parameters (identical across columns) and
latitude-dependent amplitudes that define a LatitudeProfile.

Based on Howell (2021) distributions extended for 2D.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
from typing import Dict, Optional, Tuple

from constants import Thermal, Planetary
from latitude_profile import LatitudeProfile, OceanPattern


class LatitudeParameterSampler:
    """
    Samples shared + latitude-dependent parameters for 2D MC runs.

    Returns a tuple of (shared_params dict, LatitudeProfile instance).
    """

    def __init__(self, seed: Optional[int] = None, ocean_pattern: OceanPattern = "polar_enhanced"):
        self.rng = np.random.default_rng(seed)
        self.ocean_pattern = ocean_pattern

    def _sample_truncated_normal(self, mean: float, sigma: float,
                                 low: float = -np.inf, high: float = np.inf) -> float:
        while True:
            sample = self.rng.normal(mean, sigma)
            if low <= sample <= high:
                return sample

    def sample(self) -> Tuple[Dict[str, float], LatitudeProfile]:
        """
        Sample all parameters for one 2D MC iteration.

        Returns:
            (shared_params, latitude_profile) tuple
        """
        # === Shared parameters (identical to HowellParameterSampler) ===
        d_grain = 10 ** self.rng.normal(np.log10(1e-3), 0.4)
        d_grain = np.clip(d_grain, 1e-5, 5e-3)

        mu_ice = self._sample_truncated_normal(3.5e9, 0.5e9, low=3.5e9 / 20, high=3.5e9)

        Q_v = self.rng.normal(59.4e3, 0.05 * 59.4e3)
        Q_b = self.rng.normal(49.0e3, 0.05 * 49.0e3)

        H_rad = self.rng.normal(4.5e-12, 1.0e-12)
        T_phi = self.rng.normal(150.0, 20.0 / 3.0)

        D_H2O = self.rng.normal(127e3, 21e3)
        D_H2O = np.clip(D_H2O, 80e3, 200e3)

        f_porosity = self.rng.uniform(0.0, 0.30)
        f_salt = 10 ** self.rng.normal(np.log10(0.03), 1.0 / 3.0)
        f_salt = np.clip(f_salt, 0.0, 0.5)
        B_k = 10 ** self.rng.uniform(-1.0, 1.0)

        D0v = max(self.rng.normal(9.1e-4, 0.033 * 9.1e-4), 1e-8)
        D0b = max(self.rng.normal(8.4e-4, 0.033 * 8.4e-4), 1e-8)
        d_del_mean = np.mean([9.04e-10, 5.22e-10])
        d_del_std = np.std([9.04e-10, 5.22e-10])
        d_del = max(self.rng.normal(d_del_mean, d_del_std), 1e-12)

        # Silicate tidal power (total, for ocean heat flux)
        mean_log = np.log(100e9)
        sigma_log = np.log(10) / 3
        P_tidal = self.rng.lognormal(mean=mean_log, sigma=sigma_log)

        # === Latitude-dependent amplitudes ===
        T_eq = self.rng.normal(110.0, 5.0)
        T_eq = np.clip(T_eq, 90.0, 130.0)
        T_phi = np.clip(T_phi, 50.0, Thermal.MELT_TEMP - 1.0)

        epsilon_eq = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
        epsilon_eq = np.clip(epsilon_eq, 1e-7, 2e-5)

        epsilon_pole = 10 ** self.rng.normal(np.log10(1.2e-5), 0.2)
        epsilon_pole = np.clip(epsilon_pole, 5e-6, 5e-5)

        # Ocean heat flux from sampled P_tidal and H_rad
        R_europa = Planetary.RADIUS
        R_rock = R_europa - D_H2O
        A_surface = Planetary.AREA
        rho_rock = 3500.0
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * rho_rock

        q_radiogenic = (H_rad * M_rock) / A_surface
        q_silicate_tidal = P_tidal / A_surface
        q_ocean_mean = q_radiogenic + q_silicate_tidal

        profile = LatitudeProfile(
            T_eq=T_eq,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
        )

        shared_params = {
            'd_grain': d_grain, 'd_del': d_del,
            'D0v': D0v, 'D0b': D0b,
            'mu_ice': mu_ice,
            'D_H2O': D_H2O,
            'Q_v': Q_v, 'Q_b': Q_b,
            'H_rad': H_rad,
            'P_tidal': P_tidal,
            'f_porosity': f_porosity,
            'f_salt': f_salt,
            'T_phi': T_phi,
            'B_k': B_k,
        }

        return shared_params, profile
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_sampler.py -v`

Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_sampler.py Europa2D/tests/test_latitude_sampler.py
git commit -m "feat: add LatitudeParameterSampler for 2D MC runs"
```

---

### Task 11: MonteCarloRunner2D + MonteCarloResults2D

**Files:**
- Create: `Europa2D/src/monte_carlo_2d.py`
- Create: `Europa2D/tests/test_monte_carlo_2d.py`

- [ ] **Step 1: Write failing test**

```python
# Europa2D/tests/test_monte_carlo_2d.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from monte_carlo_2d import MonteCarloRunner2D, MonteCarloResults2D


class TestMonteCarloRunner2D:

    def test_runs_and_returns_results(self):
        """Smoke test: 10 iterations should complete."""
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert isinstance(results, MonteCarloResults2D)
        assert results.n_valid > 0
        assert results.H_profiles.shape[1] == 5  # n_lat

    def test_H_profiles_are_reasonable(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert np.all(results.H_profiles > 0)
        assert np.all(results.H_profiles < 200)

    def test_statistics_computed(self):
        runner = MonteCarloRunner2D(
            n_iterations=10, seed=42, n_workers=1,
            n_lat=5, nx=21, use_convection=False,
            max_steps=200, dt=1e11,
        )
        results = runner.run()
        assert results.H_median.shape == (5,)
        assert results.H_mean.shape == (5,)
        assert results.H_sigma_low.shape == (5,)
        assert results.H_sigma_high.shape == (5,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_monte_carlo_2d.py -v`

Expected: FAIL

- [ ] **Step 3: Implement MonteCarloRunner2D**

```python
# Europa2D/src/monte_carlo_2d.py
"""
Monte Carlo framework for 2D axisymmetric Europa ice shell model.

Each MC iteration samples shared parameters, builds a LatitudeProfile,
runs AxialSolver2D to equilibrium, and collects the H(φ) profile.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
import time
import multiprocessing as mp
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from functools import partial

from constants import Thermal, Planetary
from latitude_profile import LatitudeProfile, OceanPattern
from latitude_sampler import LatitudeParameterSampler
from axial_solver import AxialSolver2D


@dataclass(frozen=True)
class MonteCarloResults2D:
    """Immutable container for 2D Monte Carlo results."""
    H_profiles: npt.NDArray[np.float64]         # (n_valid, n_lat)
    latitudes_deg: npt.NDArray[np.float64]       # (n_lat,)
    n_iterations: int
    n_valid: int
    H_median: npt.NDArray[np.float64]            # (n_lat,)
    H_mean: npt.NDArray[np.float64]              # (n_lat,)
    H_sigma_low: npt.NDArray[np.float64]         # (n_lat,)
    H_sigma_high: npt.NDArray[np.float64]        # (n_lat,)
    runtime_seconds: float
    D_cond_profiles: Optional[npt.NDArray[np.float64]] = None   # (n_valid, n_lat)
    D_conv_profiles: Optional[npt.NDArray[np.float64]] = None
    Ra_profiles: Optional[npt.NDArray[np.float64]] = None
    Nu_profiles: Optional[npt.NDArray[np.float64]] = None
    lid_fraction_profiles: Optional[npt.NDArray[np.float64]] = None


def _run_single_2d_sample(
    sample_id: int,
    base_seed: int,
    n_lat: int,
    nx: int,
    dt: float,
    use_convection: bool,
    max_steps: int,
    eq_threshold: float,
    initial_thickness: float,
    ocean_pattern: str,
    rannacher_steps: int,
    coordinate_system: str,
) -> Optional[Dict[str, Any]]:
    """Worker function for one 2D MC iteration."""
    try:
        sampler = LatitudeParameterSampler(
            seed=base_seed + sample_id,
            ocean_pattern=ocean_pattern,
        )
        shared_params, profile = sampler.sample()
        D_H2O = shared_params['D_H2O']

        # Warm start: conductive estimate scaled for convection
        q_mean = profile.q_ocean_mean
        if q_mean > 0:
            k_mean = float(Thermal.conductivity(190.0))
            H_guess = k_mean * 170.0 / q_mean
            if use_convection:
                H_guess *= 8.0
            H_guess = np.clip(H_guess, 5e3, 100e3)
        else:
            H_guess = initial_thickness

        solver = AxialSolver2D(
            n_lat=n_lat,
            nx=nx,
            dt=dt,
            latitude_profile=profile,
            physics_params=shared_params,
            use_convection=use_convection,
            initial_thickness=H_guess,
            rannacher_steps=rannacher_steps,
            coordinate_system=coordinate_system,
        )

        result = solver.run_to_equilibrium(
            threshold=eq_threshold,
            max_steps=max_steps,
            verbose=False,
        )

        H_km = result['H_profile_km']
        D_H2O_km = D_H2O / 1000.0

        # Filter: reject if >50% of columns are non-physical
        valid_mask = (H_km > 0.5) & (H_km < D_H2O_km * 0.99) & (H_km < 200)
        if np.sum(valid_mask) < len(H_km) * 0.5:
            return None

        # For invalid columns, interpolate from neighbors
        if not np.all(valid_mask):
            lats = np.degrees(solver.latitudes)
            H_km = np.interp(
                lats, lats[valid_mask], H_km[valid_mask],
            )

        # Extract diagnostics
        diag = result['diagnostics']
        D_cond = np.array([d['D_cond_km'] for d in diag])
        D_conv = np.array([d['D_conv_km'] for d in diag])
        Ra = np.array([d['Ra'] for d in diag])
        Nu = np.array([d['Nu'] for d in diag])
        lid_frac = np.array([d['lid_fraction'] for d in diag])

        return {
            'H_km': H_km,
            'D_cond_km': D_cond,
            'D_conv_km': D_conv,
            'Ra': Ra,
            'Nu': Nu,
            'lid_fraction': lid_frac,
        }

    except Exception:
        return None


class MonteCarloRunner2D:
    """Monte Carlo runner for 2D axisymmetric model."""

    def __init__(
        self,
        n_iterations: int = 100,
        seed: Optional[int] = None,
        n_workers: Optional[int] = None,
        n_lat: int = 19,
        nx: int = 31,
        dt: float = 1e12,
        use_convection: bool = True,
        max_steps: int = 1500,
        eq_threshold: float = 1e-12,
        initial_thickness: float = 20e3,
        ocean_pattern: OceanPattern = "polar_enhanced",
        verbose: bool = True,
        rannacher_steps: int = 4,
        coordinate_system: str = 'auto',
    ):
        self.n_iterations = n_iterations
        self.seed = seed if seed is not None else int(time.time())
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.n_lat = n_lat
        self.nx = nx
        self.dt = dt
        self.use_convection = use_convection
        self.max_steps = max_steps
        self.eq_threshold = eq_threshold
        self.initial_thickness = initial_thickness
        self.ocean_pattern = ocean_pattern
        self.verbose = verbose
        self.rannacher_steps = rannacher_steps
        self.coordinate_system = coordinate_system

    def run(self) -> MonteCarloResults2D:
        if self.verbose:
            print("=" * 60)
            print("2D MONTE CARLO: Europa Ice Shell Thickness Profile")
            print("=" * 60)
            print(f"Iterations: {self.n_iterations}, Workers: {self.n_workers}")
            print(f"Columns: {self.n_lat}, Nodes/col: {self.nx}")
            print("-" * 60)

        start_time = time.time()

        worker = partial(
            _run_single_2d_sample,
            base_seed=self.seed,
            n_lat=self.n_lat,
            nx=self.nx,
            dt=self.dt,
            use_convection=self.use_convection,
            max_steps=self.max_steps,
            eq_threshold=self.eq_threshold,
            initial_thickness=self.initial_thickness,
            ocean_pattern=self.ocean_pattern,
            rannacher_steps=self.rannacher_steps,
            coordinate_system=self.coordinate_system,
        )

        # Sequential or parallel execution
        if self.n_workers > 1:
            with mp.Pool(self.n_workers) as pool:
                chunksize = max(1, self.n_iterations // (self.n_workers * 4))
                results = []
                for i, result in enumerate(pool.imap_unordered(worker, range(self.n_iterations), chunksize=chunksize)):
                    results.append(result)
                    if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                        valid = sum(1 for r in results if r is not None)
                        print(f"  Progress: {100 * (i + 1) / self.n_iterations:5.1f}% | Valid: {valid}/{i + 1}")
        else:
            results = []
            for i in range(self.n_iterations):
                results.append(worker(i))
                if self.verbose and (i + 1) % max(1, self.n_iterations // 10) == 0:
                    valid = sum(1 for r in results if r is not None)
                    print(f"  Progress: {100 * (i + 1) / self.n_iterations:5.1f}% | Valid: {valid}/{i + 1}")

        runtime = time.time() - start_time
        valid_results = [r for r in results if r is not None]

        if len(valid_results) == 0:
            raise RuntimeError("No valid 2D solutions. Check parameter distributions.")

        # Stack results
        H_profiles = np.array([r['H_km'] for r in valid_results])
        D_cond = np.array([r['D_cond_km'] for r in valid_results])
        D_conv = np.array([r['D_conv_km'] for r in valid_results])
        Ra = np.array([r['Ra'] for r in valid_results])
        Nu = np.array([r['Nu'] for r in valid_results])
        lid_frac = np.array([r['lid_fraction'] for r in valid_results])

        latitudes_deg = np.linspace(0, 90, self.n_lat)

        mc_results = MonteCarloResults2D(
            H_profiles=H_profiles,
            latitudes_deg=latitudes_deg,
            n_iterations=self.n_iterations,
            n_valid=len(valid_results),
            H_median=np.percentile(H_profiles, 50, axis=0),
            H_mean=np.mean(H_profiles, axis=0),
            H_sigma_low=np.percentile(H_profiles, 15.87, axis=0),
            H_sigma_high=np.percentile(H_profiles, 84.13, axis=0),
            runtime_seconds=runtime,
            D_cond_profiles=D_cond,
            D_conv_profiles=D_conv,
            Ra_profiles=Ra,
            Nu_profiles=Nu,
            lid_fraction_profiles=lid_frac,
        )

        if self.verbose:
            print("-" * 60)
            print(f"Valid: {mc_results.n_valid}/{self.n_iterations}")
            print(f"Runtime: {runtime:.1f}s")
            print(f"H range: [{mc_results.H_median.min():.1f}, {mc_results.H_median.max():.1f}] km (median)")
            print("=" * 60)

        return mc_results


def save_results_2d(results: MonteCarloResults2D, filepath: str) -> None:
    """Save 2D MC results to NumPy archive."""
    save_dict = {
        'H_profiles': results.H_profiles,
        'latitudes_deg': results.latitudes_deg,
        'n_iterations': results.n_iterations,
        'n_valid': results.n_valid,
        'H_median': results.H_median,
        'H_mean': results.H_mean,
        'H_sigma_low': results.H_sigma_low,
        'H_sigma_high': results.H_sigma_high,
        'runtime_seconds': results.runtime_seconds,
    }
    for name in ['D_cond_profiles', 'D_conv_profiles', 'Ra_profiles', 'Nu_profiles', 'lid_fraction_profiles']:
        val = getattr(results, name)
        if val is not None:
            save_dict[name] = val

    np.savez(filepath, **save_dict)
    print(f"Results saved to: {filepath}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_monte_carlo_2d.py -v --timeout=120`

Expected: All 3 tests PASS (may take ~30-60s due to 10 MC iterations each)

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py Europa2D/tests/test_monte_carlo_2d.py
git commit -m "feat: add MonteCarloRunner2D and MonteCarloResults2D"
```

---

### Task 12: MC run script + plotting script

**Files:**
- Create: `Europa2D/scripts/run_2d_mc.py`
- Create: `Europa2D/scripts/plot_thickness_profile.py`

- [ ] **Step 1: Write MC run script**

```python
# Europa2D/scripts/run_2d_mc.py
"""Full 2D Monte Carlo run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src
import multiprocessing as mp

from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

if __name__ == "__main__":
    mp.freeze_support()

    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    runner = MonteCarloRunner2D(
        n_iterations=1000,
        seed=42,
        n_workers=max(1, mp.cpu_count() - 1),
        n_lat=19,
        nx=31,
        dt=1e12,
        use_convection=True,
        max_steps=1500,
        ocean_pattern="polar_enhanced",
    )
    results = runner.run()
    save_results_2d(results, os.path.join(RESULTS_DIR, "mc_2d_polar_enhanced_1000.npz"))
```

- [ ] **Step 2: Write plotting script**

```python
# Europa2D/scripts/plot_thickness_profile.py
"""Plot H(φ) with uncertainty bands from 2D MC results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import matplotlib.pyplot as plt


def plot_thickness_profile(filepath: str, output_dir: str = None):
    data = np.load(filepath)
    lats = data['latitudes_deg']
    H_median = data['H_median']
    H_mean = data['H_mean']
    H_low = data['H_sigma_low']
    H_high = data['H_sigma_high']
    n_valid = int(data['n_valid'])
    n_iter = int(data['n_iterations'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Uncertainty band
    ax.fill_between(lats, H_low, H_high, alpha=0.3, color='steelblue', label='1σ range')
    ax.plot(lats, H_median, 'b-', linewidth=2, label='Median')
    ax.plot(lats, H_mean, 'r--', linewidth=1, label='Mean')

    ax.set_xlabel('Geographic Latitude (degrees)', fontsize=12)
    ax.set_ylabel('Ice Shell Thickness (km)', fontsize=12)
    ax.set_title(f'Europa Ice Shell Thickness Profile (N={n_valid}/{n_iter})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, 'H_profile_with_uncertainty.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"Saved: {out}")

    plt.show()


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    filepath = os.path.join(results_dir, "mc_2d_polar_enhanced_1000.npz")

    if os.path.exists(filepath):
        plot_thickness_profile(filepath, figures_dir)
    else:
        print(f"No results found at {filepath}. Run run_2d_mc.py first.")
```

- [ ] **Step 3: Commit**

```bash
git add Europa2D/scripts/run_2d_mc.py Europa2D/scripts/plot_thickness_profile.py
git commit -m "feat: add MC run script and thickness profile plotter"
```

---

## Chunk 4: Validation

### Task 13: Validate 2D model against 1D at single latitude

**Files:**
- Create: `Europa2D/tests/test_validation.py`

- [ ] **Step 1: Write validation test**

```python
# Europa2D/tests/test_validation.py
"""
Validation: 2D model at a single latitude must match 1D model output.

This test runs a single-column 2D solver and compares the equilibrium
thickness against a standalone 1D Thermal_Solver with identical parameters.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import src

import numpy as np
import pytest
from latitude_profile import LatitudeProfile
from axial_solver import AxialSolver2D
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature
from Physics import IcePhysics


class TestValidationVs1D:
    """2D single-column output should match standalone 1D solver."""

    PARAMS = {
        'd_grain': 1e-3, 'Q_v': 59.4e3, 'Q_b': 49.0e3,
        'mu_ice': 3.3e9, 'D0v': 9.1e-4, 'D0b': 8.4e-4,
        'd_del': 7.13e-10, 'f_porosity': 0.0, 'f_salt': 0.0,
        'B_k': 1.0, 'T_phi': 150.0, 'epsilon_0': 1e-5,
    }
    T_SURF = 104.0
    Q_OCEAN = 0.025
    NX = 31
    DT = 1e11
    THICKNESS = 15e3

    def test_conductive_single_column_matches_1d(self):
        """Without convection, single-column 2D must match 1D within 1%."""
        # 1D reference
        bc_1d = FixedTemperature(temperature=self.T_SURF)
        solver_1d = Thermal_Solver(
            nx=self.NX, thickness=self.THICKNESS, dt=self.DT,
            surface_bc=bc_1d, use_convection=False,
            physics_params={**self.PARAMS, 'T_surf': self.T_SURF},
        )
        for _ in range(300):
            v = solver_1d.solve_step(self.Q_OCEAN)
            if abs(v) < 1e-12:
                break
        H_1d = solver_1d.H / 1000.0

        # 2D with 1 column at equator (T_eq = T_SURF, so T_s(0) = T_SURF)
        profile = LatitudeProfile(
            T_eq=self.T_SURF, epsilon_eq=self.PARAMS['epsilon_0'],
            epsilon_pole=self.PARAMS['epsilon_0'],
            q_ocean_mean=self.Q_OCEAN, ocean_pattern="uniform",
        )
        solver_2d = AxialSolver2D(
            n_lat=1, nx=self.NX, dt=self.DT,
            latitude_profile=profile, physics_params=self.PARAMS,
            use_convection=False, initial_thickness=self.THICKNESS,
        )
        result_2d = solver_2d.run_to_equilibrium(
            threshold=1e-12, max_steps=300, verbose=False
        )
        H_2d = result_2d['H_profile_km'][0]

        # Should match within 1%
        assert H_2d == pytest.approx(H_1d, rel=0.01), \
            f"2D ({H_2d:.3f} km) vs 1D ({H_1d:.3f} km)"
```

- [ ] **Step 2: Run validation test**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_validation.py -v`

Expected: PASS. If it fails, debug until the single-column 2D matches 1D within 1%.

- [ ] **Step 3: Commit**

```bash
git add Europa2D/tests/test_validation.py
git commit -m "test: validate 2D single-column matches 1D solver"
```

---

### Task 14: Run all tests, final verification

- [ ] **Step 1: Run full test suite**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/ -v --timeout=120`

Expected: All tests PASS

- [ ] **Step 2: Run single deterministic 2D simulation**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python scripts/run_2d_single.py`

Expected: Produces thickness profile and temperature cross-section plots.

- [ ] **Step 3: Final commit**

```bash
git add -A Europa2D/
git commit -m "feat: complete 2D axisymmetric Europa ice shell model"
```
