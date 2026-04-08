# Latitude-Dependent Convection Hypothesis Testing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal convection_adjuster hook to the 1D solver, implement three hypothesis adjusters in Europa2D, and run a 10-experiment campaign to improve latitude discriminability (JS divergence from 0.0013 toward >0.01).

**Architecture:** One `convection_adjuster` callback inserted in `Convection.py:build_conductivity_profile()` after `scan_temperature_profile()` returns the default `ConvectionState`. The adjuster mutates the state in place before Nu is applied to k_profile. All hypothesis logic lives in `Europa2D/src/convection_2d.py`. Grain-strain experiments are config-only (no new code). The campaign runner automates 10 experiments scored by the existing latitude objective.

**Tech Stack:** Python 3.11+, NumPy, SciPy, multiprocessing. No new dependencies.

**Design spec:** `docs/superpowers/specs/2026-04-01-latitude-convection-hypotheses-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `EuropaProjectDJ/src/Convection.py` | Modify (~8 lines) | Add `convection_adjuster` + `q_ocean` kwargs to `build_conductivity_profile()` |
| `EuropaProjectDJ/src/Solver.py` | Modify (~10 lines) | Accept `convection_adjuster`, thread q_ocean to assembly |
| `Europa2D/src/convection_2d.py` | Create | Hypothesis dispatch + 3 adjuster implementations |
| `Europa2D/src/axial_solver.py` | Modify (~15 lines) | Accept `hypothesis`, create per-column adjusters |
| `Europa2D/src/monte_carlo_2d.py` | Modify (~20 lines) | Thread `hypothesis` through MC pipeline |
| `autoresearch/objectives.py` | Modify (~10 lines) | Add D_conv JS secondary metric |
| `autoresearch/harness.py` | Modify (~10 lines) | Thread `hypothesis` to latitude experiment |
| `autoresearch/experiments/hypothesis_config.json` | Create | 10 experiment definitions |
| `autoresearch/experiments/run_hypothesis_campaign.py` | Create | Campaign runner script |
| `Europa2D/tests/test_convection_2d.py` | Create | Tests for all adjusters |
| `Europa2D/tests/test_adjuster_hook.py` | Create | Tests for 1D hook + parity |

---

## Task 1: Add convection_adjuster hook to Convection.py

**Files:**
- Modify: `EuropaProjectDJ/src/Convection.py:1021-1153`
- Test: `Europa2D/tests/test_adjuster_hook.py`

- [ ] **Step 1: Write the failing test for the hook**

Create `Europa2D/tests/test_adjuster_hook.py`:

```python
"""Tests for the convection_adjuster hook in build_conductivity_profile."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from Convection import IceConvection, ConvectionState
from constants import Thermal


def _make_test_profile(nx=31, H=20e3, T_surf=96.0, T_melt=273.0):
    """Create a linear temperature profile for testing."""
    z_grid = np.linspace(0, H, nx)
    T_profile = T_surf + (T_melt - T_surf) * z_grid / H
    return T_profile, z_grid, H, T_melt


class TestConvectionAdjusterHook:
    """Tests that build_conductivity_profile calls the adjuster."""

    def test_no_adjuster_returns_same_as_before(self):
        """Passing convection_adjuster=None must give identical results."""
        T, z, H, T_m = _make_test_profile()
        k_ref, state_ref = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
        )
        k_new, state_new = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=None, q_ocean=0.02,
        )
        np.testing.assert_array_equal(k_ref, k_new)
        assert state_ref.Nu == state_new.Nu
        assert state_ref.Ra == state_new.Ra
        assert state_ref.D_conv == state_new.D_conv

    def test_adjuster_is_called_with_correct_args(self):
        """The adjuster receives (state, T_profile, z_grid, H, q_ocean)."""
        T, z, H, T_m = _make_test_profile()
        call_log = []

        def spy_adjuster(state, T_prof, z_grid, total_thickness, q_ocean):
            call_log.append({
                'state_type': type(state).__name__,
                'T_shape': T_prof.shape,
                'z_shape': z_grid.shape,
                'H': total_thickness,
                'q_ocean': q_ocean,
            })

        IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=spy_adjuster, q_ocean=0.025,
        )
        assert len(call_log) == 1
        assert call_log[0]['state_type'] == 'ConvectionState'
        assert call_log[0]['T_shape'] == (31,)
        assert call_log[0]['H'] == 20e3
        assert call_log[0]['q_ocean'] == 0.025

    def test_adjuster_mutation_affects_k_profile(self):
        """If adjuster doubles Nu, k_profile in convective layer must change."""
        T, z, H, T_m = _make_test_profile()

        k_base, state_base = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
        )

        def double_nu(state, T_prof, z_grid, total_thickness, q_ocean):
            if state.is_convecting:
                state.Nu = state.Nu * 2.0

        k_adj, state_adj = IceConvection.build_conductivity_profile(
            T_profile=T, z_grid=z, total_thickness=H, T_melt=T_m,
            use_composite_viscosity=True,
            convection_adjuster=double_nu, q_ocean=0.02,
        )

        if state_base.is_convecting:
            # Convective region should have higher k
            conv_region = slice(state_base.idx_c, None)
            assert np.all(k_adj[conv_region] > k_base[conv_region])
            # Conductive lid should be unchanged
            cond_region = slice(0, state_base.idx_c)
            np.testing.assert_array_equal(k_adj[cond_region], k_base[cond_region])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_adjuster_hook.py -v`
Expected: FAIL — `build_conductivity_profile() got an unexpected keyword argument 'convection_adjuster'`

- [ ] **Step 3: Add convection_adjuster and q_ocean kwargs to build_conductivity_profile**

In `EuropaProjectDJ/src/Convection.py`, modify the `build_conductivity_profile` signature. Add two new keyword arguments after the existing `use_onset_consistent_partition` parameter (line ~1042):

```python
        use_onset_consistent_partition: bool = False,
        convection_adjuster=None,
        q_ocean: float = 0.0,
) -> Tuple[npt.NDArray[np.float64], ConvectionState]:
```

Then, after the `scan_temperature_profile()` call returns `state` (after line ~1087), insert the adjuster call:

```python
        # Allow external modification of convection state before k_profile is built
        if convection_adjuster is not None:
            convection_adjuster(state, T_profile, z_grid, total_thickness, q_ocean)
```

This must go BEFORE the existing onset-consistent-partition check and Nu enhancement at lines ~1149-1151. The existing code that applies Nu to k_profile already uses `state.is_convecting`, `state.idx_c`, and `state.Nu`, so the adjusted values flow through automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_adjuster_hook.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Verify existing 1D tests still pass**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest EuropaProjectDJ/tests/ -v --tb=short`
Expected: All existing tests PASS (no regression — adjuster defaults to None)

- [ ] **Step 6: Commit**

```bash
git add EuropaProjectDJ/src/Convection.py Europa2D/tests/test_adjuster_hook.py
git commit -m "feat: add convection_adjuster hook to build_conductivity_profile"
```

---

## Task 2: Thread adjuster + q_ocean through Solver.py

**Files:**
- Modify: `EuropaProjectDJ/src/Solver.py:44-79` (init), `~437-441` (solve_step), `~229-249` (assemble)
- Test: `Europa2D/tests/test_adjuster_hook.py` (append)

- [ ] **Step 1: Write failing test for Solver adjuster threading**

Append to `Europa2D/tests/test_adjuster_hook.py`:

```python
from Solver import Thermal_Solver
from Boundary_Conditions import FixedTemperature


class TestSolverAdjusterThreading:
    """Tests that Thermal_Solver threads the adjuster to build_conductivity_profile."""

    def test_solver_accepts_convection_adjuster(self):
        """Solver.__init__ accepts convection_adjuster kwarg."""
        call_count = [0]

        def counting_adjuster(state, T_prof, z_grid, H, q_ocean):
            call_count[0] += 1

        solver = Thermal_Solver(
            nx=31, thickness=20e3, dt=1e12, total_time=5e14,
            use_convection=True,
            convection_adjuster=counting_adjuster,
        )
        solver.solve_step(q_ocean=0.02)
        # Adjuster should have been called at least once during assembly
        assert call_count[0] >= 1

    def test_solver_threads_q_ocean_to_adjuster(self):
        """The adjuster receives the q_ocean value from solve_step."""
        received_q = []

        def capture_q(state, T_prof, z_grid, H, q_ocean):
            received_q.append(q_ocean)

        solver = Thermal_Solver(
            nx=31, thickness=20e3, dt=1e12, total_time=5e14,
            use_convection=True,
            convection_adjuster=capture_q,
        )
        solver.solve_step(q_ocean=0.042)
        assert len(received_q) >= 1
        assert received_q[0] == pytest.approx(0.042)

    def test_solver_without_adjuster_unchanged(self):
        """Solver with no adjuster produces identical results to default."""
        kwargs = dict(nx=31, thickness=20e3, dt=1e12, total_time=5e14,
                      use_convection=True)
        s1 = Thermal_Solver(**kwargs)
        s2 = Thermal_Solver(**kwargs, convection_adjuster=None)

        v1 = s1.solve_step(0.02)
        v2 = s2.solve_step(0.02)
        assert v1 == pytest.approx(v2)
        np.testing.assert_array_almost_equal(s1.T, s2.T)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_adjuster_hook.py::TestSolverAdjusterThreading -v`
Expected: FAIL — `Thermal_Solver.__init__() got an unexpected keyword argument 'convection_adjuster'`

- [ ] **Step 3: Modify Solver.py to accept and thread the adjuster**

Three changes in `EuropaProjectDJ/src/Solver.py`:

**A. `__init__` signature** — add `convection_adjuster=None` after `physics_params`:

```python
def __init__(self, nx: int = 101, thickness: float = 10e3, dt: float = 1e11, total_time: float = 1e15,
             coordinate_system: str = 'auto', surface_bc: Optional[surfacecondition] = None,
             rannacher_steps: int = 4, use_convection: bool = False,
             physics_params: Optional[Dict[str, float]] = None,
             convection_adjuster=None):
```

Store it after `self.use_convection = use_convection` (after line ~75):

```python
        self.use_convection = use_convection
        self._convection_adjuster = convection_adjuster
        self._current_q_ocean = 0.0
```

**B. `solve_step`** — store q_ocean before Picard loop (after line ~435, before `T_old = self.T.copy()`):

```python
        self._current_q_ocean = q_ocean
```

**C. `_assemble_system`** — pass adjuster + q_ocean to `build_conductivity_profile` call. Add two kwargs to the existing call at lines ~229-249:

```python
            nu_ramp_factor=self.convection_ramp,
            use_onset_consistent_partition=self.use_onset_consistent_partition,
            convection_adjuster=self._convection_adjuster,
            q_ocean=self._current_q_ocean,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_adjuster_hook.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Verify all existing tests still pass**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest EuropaProjectDJ/tests/ Europa2D/tests/test_validation.py -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add EuropaProjectDJ/src/Solver.py Europa2D/tests/test_adjuster_hook.py
git commit -m "feat: thread convection_adjuster + q_ocean through Solver"
```

---

## Task 3: Parity Acceptance Test

**Files:**
- Test: `Europa2D/tests/test_adjuster_hook.py` (append)

- [ ] **Step 1: Write full-run parity test**

Append to `Europa2D/tests/test_adjuster_hook.py`:

```python
from axial_solver import AxialSolver2D
from latitude_profile import LatitudeProfile


class TestParityAcceptance:
    """hypothesis=None must produce results identical to pre-hook code."""

    def test_2d_solver_parity(self):
        """AxialSolver2D with no hypothesis gives identical H profile."""
        profile = LatitudeProfile(q_ocean_mean=0.02)
        kwargs = dict(
            n_lat=5, nx=21, dt=1e12, total_time=5e14,
            latitude_profile=profile, use_convection=True,
            initial_thickness=20e3, rannacher_steps=2,
        )
        s1 = AxialSolver2D(**kwargs)
        s2 = AxialSolver2D(**kwargs)

        q_ocean = np.array([profile.ocean_heat_flux(phi) for phi in s1.latitudes])

        # Run 10 steps each
        for _ in range(10):
            s1.solve_step(q_ocean)
            s2.solve_step(q_ocean)

        np.testing.assert_array_almost_equal(
            s1.get_thickness_profile(), s2.get_thickness_profile(),
            decimal=10,
            err_msg="Solver with default adjuster=None must match baseline exactly"
        )
```

- [ ] **Step 2: Run test**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_adjuster_hook.py::TestParityAcceptance -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add Europa2D/tests/test_adjuster_hook.py
git commit -m "test: add parity acceptance test for convection_adjuster=None"
```

---

## Task 4: Create convection_2d.py — Hypothesis Dispatch

**Files:**
- Create: `Europa2D/src/convection_2d.py`
- Test: `Europa2D/tests/test_convection_2d.py`

- [ ] **Step 1: Write failing test for hypothesis dispatch**

Create `Europa2D/tests/test_convection_2d.py`:

```python
"""Tests for convection_2d hypothesis adjusters."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import pytest
from convection_2d import ConvectionHypothesis, make_adjuster
from latitude_profile import LatitudeProfile
from Convection import ConvectionState


def _make_test_state(**overrides):
    """Create a ConvectionState with sensible defaults for testing."""
    defaults = dict(
        idx_c=20, z_c=14000.0, D_cond=14000.0, D_conv=6000.0,
        T_c=230.0, Ti=250.0, Ra=1500.0, Nu=3.5, is_convecting=True,
    )
    defaults.update(overrides)
    return ConvectionState(**defaults)


class TestConvectionHypothesis:
    """Tests for hypothesis dataclass and make_adjuster dispatch."""

    def test_make_adjuster_returns_none_for_none_hypothesis(self):
        profile = LatitudeProfile()
        result = make_adjuster(None, 0.0, profile)
        assert result is None

    def test_make_adjuster_returns_callable_for_heat_balance(self):
        hypothesis = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False, "max_iterations": 5, "tolerance": 1e-4},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adjuster = make_adjuster(hypothesis, 0.5, profile)
        assert callable(adjuster)

    def test_make_adjuster_returns_callable_for_ra_onset(self):
        hypothesis = ConvectionHypothesis(
            mechanism="ra_onset",
            params={"ra_crit_override": 1200},
        )
        profile = LatitudeProfile()
        adjuster = make_adjuster(hypothesis, 0.0, profile)
        assert callable(adjuster)

    def test_make_adjuster_returns_callable_for_tidal_viscosity(self):
        hypothesis = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        profile = LatitudeProfile()
        adjuster = make_adjuster(hypothesis, 0.5, profile)
        assert callable(adjuster)

    def test_make_adjuster_raises_for_unknown_mechanism(self):
        hypothesis = ConvectionHypothesis(mechanism="nonsense", params={})
        profile = LatitudeProfile()
        with pytest.raises(ValueError, match="Unknown mechanism"):
            make_adjuster(hypothesis, 0.0, profile)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'convection_2d'`

- [ ] **Step 3: Create convection_2d.py with dispatch skeleton**

Create `Europa2D/src/convection_2d.py`:

```python
"""
Experimental convection hypothesis adjusters for Europa2D.

Each hypothesis is a factory that returns a convection_adjuster callback.
The callback mutates a ConvectionState in place during
build_conductivity_profile(), before Nu is applied to k_profile.

This is the only coupling point with the 1D solver.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'EuropaProjectDJ', 'src'))

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from Convection import ConvectionState, IceConvection
from constants import Thermal, Planetary, Rheology, ConvectionConstants
from latitude_profile import LatitudeProfile


@dataclass(frozen=True)
class ConvectionHypothesis:
    """Defines which convection adjustment to apply and its parameters."""
    mechanism: str   # "heat_balance" | "ra_onset" | "tidal_viscosity"
    params: dict     # mechanism-specific parameters


def make_adjuster(
    hypothesis: Optional[ConvectionHypothesis],
    phi: float,
    profile: LatitudeProfile,
) -> Optional[Callable]:
    """Create a convection_adjuster closure for one latitude column.

    Args:
        hypothesis: ConvectionHypothesis or None (no adjustment)
        phi: Geographic latitude in radians for this column
        profile: LatitudeProfile instance

    Returns:
        Callable matching the convection_adjuster signature, or None.
    """
    if hypothesis is None:
        return None

    mechanism = hypothesis.mechanism
    params = hypothesis.params

    if mechanism == "heat_balance":
        include_tidal = params.get("include_tidal", False)
        max_iterations = params.get("max_iterations", 5)
        tolerance = params.get("tolerance", 1e-4)
        epsilon_0 = float(profile.tidal_strain(phi))
        mu_ice = 3.3e9  # default shear modulus

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _heat_balance_adjuster(
                state, T_profile, z_grid, total_thickness, q_ocean,
                include_tidal, max_iterations, tolerance, epsilon_0, mu_ice,
            )
        return adjuster

    if mechanism == "ra_onset":
        ra_crit = params["ra_crit_override"]

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _ra_onset_adjuster(state, ra_crit)
        return adjuster

    if mechanism == "tidal_viscosity":
        epsilon_0_local = float(profile.tidal_strain(phi))
        epsilon_ref = params.get("epsilon_ref", 6e-6)
        n = params.get("softening_exponent", 1.0)

        def adjuster(state, T_profile, z_grid, total_thickness, q_ocean):
            _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n)
        return adjuster

    raise ValueError(f"Unknown mechanism: {mechanism!r}")


# --- Hypothesis implementations (private) ---

def _heat_balance_adjuster(state, T_profile, z_grid, H, q_ocean,
                           include_tidal, max_iterations, tolerance,
                           epsilon_0, mu_ice):
    """Adjust D_cond so conductive lid flux matches local q_ocean."""
    pass  # Implemented in Task 5


def _ra_onset_adjuster(state, ra_crit_override):
    """Override is_convecting with custom Ra_crit."""
    pass  # Implemented in Task 6


def _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n):
    """Rescale Ra and Nu using tidal-softened viscosity."""
    pass  # Implemented in Task 7
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/convection_2d.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: add convection_2d module with hypothesis dispatch skeleton"
```

---

## Task 5: Implement Heat-Balance Adjuster

**Files:**
- Modify: `Europa2D/src/convection_2d.py` (replace `_heat_balance_adjuster` stub)
- Test: `Europa2D/tests/test_convection_2d.py` (append)

- [ ] **Step 1: Write failing tests for heat-balance adjuster**

Append to `Europa2D/tests/test_convection_2d.py`:

```python
from Physics import IcePhysics


class TestHeatBalanceAdjuster:
    """Tests for the heat-balance D_cond correction."""

    def test_higher_q_ocean_thins_conductive_lid(self):
        """Higher q_ocean should reduce D_cond (thinner lid to conduct more heat)."""
        state_lo = _make_test_state()
        state_hi = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)

        profile = LatitudeProfile(q_ocean_mean=0.02)
        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False, "max_iterations": 5, "tolerance": 1e-4},
        )
        adj = make_adjuster(hyp, 0.0, profile)

        adj(state_lo, T, z, 20e3, 0.01)   # low q_ocean
        adj_hi = _make_test_state()
        adj_fn_hi = make_adjuster(hyp, 0.0, profile)
        adj_fn_hi(state_hi, T, z, 20e3, 0.05)  # high q_ocean

        assert state_hi.D_cond < state_lo.D_cond, (
            f"D_cond should decrease with higher q_ocean: "
            f"lo={state_lo.D_cond:.0f}, hi={state_hi.D_cond:.0f}"
        )

    def test_d_conv_increases_when_q_ocean_increases(self):
        """Thinner lid means thicker convective layer."""
        state_lo = _make_test_state()
        state_hi = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)

        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj_lo = make_adjuster(hyp, 0.0, profile)
        adj_hi = make_adjuster(hyp, 0.0, profile)

        adj_lo(state_lo, T, z, 20e3, 0.01)
        adj_hi(state_hi, T, z, 20e3, 0.05)

        assert state_hi.D_conv > state_lo.D_conv

    def test_non_convecting_state_unchanged(self):
        """If is_convecting=False, adjuster should not modify state."""
        state = _make_test_state(is_convecting=False, Ra=500.0, Nu=1.0)
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)
        D_cond_before = state.D_cond

        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj = make_adjuster(hyp, 0.0, profile)
        adj(state, T, z, 20e3, 0.02)

        assert state.D_cond == D_cond_before

    def test_d_cond_clamped_to_physical_range(self):
        """D_cond must stay within [0.05*H, 0.95*H]."""
        state = _make_test_state()
        T = np.linspace(96.0, 273.0, 31)
        z = np.linspace(0, 20e3, 31)
        H = 20e3

        hyp = ConvectionHypothesis(
            mechanism="heat_balance",
            params={"include_tidal": False},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        adj = make_adjuster(hyp, 0.0, profile)

        # Extreme q_ocean should hit clamp, not go negative
        adj(state, T, z, H, 10.0)
        assert state.D_cond >= 0.05 * H
        assert state.D_conv <= 0.95 * H
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestHeatBalanceAdjuster -v`
Expected: FAIL — assertions fail because `_heat_balance_adjuster` is a no-op stub

- [ ] **Step 3: Implement _heat_balance_adjuster**

Replace the stub in `Europa2D/src/convection_2d.py`:

```python
def _heat_balance_adjuster(state, T_profile, z_grid, H, q_ocean,
                           include_tidal, max_iterations, tolerance,
                           epsilon_0, mu_ice):
    """Adjust D_cond so conductive lid flux matches local q_ocean.

    At equilibrium the conductive lid must transport q_ocean:
        q_lid = k_lid * (T_c - T_surface) / D_cond = q_ocean

    This gives: D_cond_eq = k_lid * (T_c - T_surface) / q_ocean

    Then D_conv = H - D_cond_eq, and Ra/Nu are recomputed for the
    new convective layer thickness.
    """
    if not state.is_convecting or q_ocean <= 0:
        return

    T_surface = float(T_profile[0])
    T_c = state.T_c
    T_melt = float(T_profile[-1])

    # Mean conductivity in conductive lid
    T_mean_lid = 0.5 * (T_surface + T_c)
    k_lid = float(Thermal.conductivity(T_mean_lid))

    # Tidal heating contribution (optional)
    q_tidal_contrib = 0.0
    if include_tidal and epsilon_0 > 0 and state.D_conv > 0:
        # Approximate tidal heating rate at mean convective temperature
        T_mean_conv = 0.5 * (T_c + T_melt)
        try:
            q_vol = IcePhysics.tidal_heating(
                np.array([T_mean_conv]),
                epsilon_0=epsilon_0,
                mu_ice=mu_ice,
                use_composite_viscosity=True,
            )
            q_tidal_contrib = float(q_vol[0]) * state.D_conv
        except Exception:
            q_tidal_contrib = 0.0

    # Total heat that the lid must conduct
    q_total = q_ocean + q_tidal_contrib

    if q_total <= 0:
        return

    # Equilibrium conductive lid thickness
    dT_lid = T_c - T_surface
    if dT_lid <= 0:
        return

    D_cond_eq = k_lid * dT_lid / q_total

    # Clamp to physical range
    D_cond_min = 0.05 * H
    D_cond_max = 0.95 * H
    D_cond_eq = max(D_cond_min, min(D_cond_max, D_cond_eq))
    D_conv_new = H - D_cond_eq

    # Find new grid index for interface
    idx_new = int(np.searchsorted(z_grid, D_cond_eq))
    idx_new = max(1, min(len(z_grid) - 2, idx_new))

    # Recompute Ra for new D_conv: Ra scales as D_conv^3
    if state.D_conv > 0:
        Ra_new = state.Ra * (D_conv_new / state.D_conv) ** 3
    else:
        Ra_new = 0.0

    # Recompute Nu from Ra
    if Ra_new > 0:
        Nu_new = max(1.0, ConvectionConstants.NU_PREFACTOR * Ra_new ** (1.0 / 3.0))
    else:
        Nu_new = 1.0

    # Update state in place
    state.D_cond = D_cond_eq
    state.D_conv = D_conv_new
    state.z_c = D_cond_eq
    state.idx_c = idx_new
    state.Ra = Ra_new
    state.Nu = Nu_new
    state.is_convecting = Ra_new >= ConvectionConstants.RA_CRIT
```

Also add at the top of the file with the other imports:

```python
from Physics import IcePhysics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestHeatBalanceAdjuster -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/convection_2d.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: implement heat-balance convection adjuster"
```

---

## Task 6: Implement Ra-Onset Adjuster

**Files:**
- Modify: `Europa2D/src/convection_2d.py` (replace `_ra_onset_adjuster` stub)
- Test: `Europa2D/tests/test_convection_2d.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `Europa2D/tests/test_convection_2d.py`:

```python
class TestRaOnsetAdjuster:
    """Tests for the Ra_crit override adjuster."""

    def test_high_ra_crit_shuts_off_weak_convection(self):
        """Ra=1500 with ra_crit=2000 should set is_convecting=False."""
        state = _make_test_state(Ra=1500.0, Nu=3.5, is_convecting=True)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 2000})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)

        assert state.is_convecting is False
        assert state.Nu == 1.0

    def test_low_ra_crit_keeps_convection(self):
        """Ra=1500 with ra_crit=800 should keep is_convecting=True."""
        state = _make_test_state(Ra=1500.0, Nu=3.5, is_convecting=True)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 800})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)

        assert state.is_convecting is True
        assert state.Nu == 3.5  # unchanged

    def test_ra_onset_can_enable_subcritical_column(self):
        """Ra=500 with ra_crit=400 should enable convection."""
        state = _make_test_state(Ra=500.0, Nu=1.0, is_convecting=False)
        hyp = ConvectionHypothesis(mechanism="ra_onset", params={"ra_crit_override": 400})
        adj = make_adjuster(hyp, 0.0, LatitudeProfile())
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)

        assert state.is_convecting is True
        assert state.Nu > 1.0
```

- [ ] **Step 2: Run to verify fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestRaOnsetAdjuster -v`
Expected: FAIL

- [ ] **Step 3: Implement _ra_onset_adjuster**

Replace the stub in `Europa2D/src/convection_2d.py`:

```python
def _ra_onset_adjuster(state, ra_crit_override):
    """Override is_convecting with custom Ra_crit.

    If Ra >= ra_crit_override: enable convection (recompute Nu if needed).
    If Ra < ra_crit_override: disable convection (set Nu=1).
    """
    should_convect = state.Ra >= ra_crit_override

    if should_convect and not state.is_convecting:
        # Enable convection that the default Ra_crit suppressed
        state.is_convecting = True
        if state.Ra > 0:
            state.Nu = max(1.0, ConvectionConstants.NU_PREFACTOR * state.Ra ** (1.0 / 3.0))
        else:
            state.Nu = 1.0
    elif not should_convect and state.is_convecting:
        # Suppress convection that the default Ra_crit allowed
        state.is_convecting = False
        state.Nu = 1.0
```

- [ ] **Step 4: Run test to verify passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestRaOnsetAdjuster -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/convection_2d.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: implement ra-onset convection adjuster"
```

---

## Task 7: Implement Tidal-Viscosity Adjuster

**Files:**
- Modify: `Europa2D/src/convection_2d.py` (replace `_tidal_viscosity_adjuster` stub)
- Test: `Europa2D/tests/test_convection_2d.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `Europa2D/tests/test_convection_2d.py`:

```python
class TestTidalViscosityAdjuster:
    """Tests for the tidal viscosity feedback adjuster."""

    def test_higher_strain_increases_nu(self):
        """Polar column (higher epsilon) should get higher Nu."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )

        state_eq = _make_test_state(Ra=1500.0, Nu=3.5)
        state_pole = _make_test_state(Ra=1500.0, Nu=3.5)

        adj_eq = make_adjuster(hyp, 0.0, profile)        # equator
        adj_pole = make_adjuster(hyp, np.pi / 2, profile)  # pole

        adj_eq(state_eq, np.zeros(31), np.zeros(31), 20e3, 0.02)
        adj_pole(state_pole, np.zeros(31), np.zeros(31), 20e3, 0.02)

        assert state_pole.Nu > state_eq.Nu, (
            f"Polar Nu ({state_pole.Nu:.2f}) should exceed equatorial ({state_eq.Nu:.2f})"
        )

    def test_equatorial_column_unchanged(self):
        """At equator (epsilon = epsilon_ref), no softening should occur."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        state = _make_test_state(Ra=1500.0, Nu=3.5)
        adj = make_adjuster(hyp, 0.0, profile)
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)

        # At equator, epsilon_0 == epsilon_ref, so softening factor = 1/(1+1) = 0.5
        # This means eta is halved, Ra doubles, Nu increases
        # The equator IS softened because epsilon_0/epsilon_ref = 1 → denominator = 2
        # Only epsilon_0 = 0 would give no softening
        assert state.Ra > 1500.0

    def test_non_convecting_unchanged(self):
        """Non-convecting column should not be modified."""
        state = _make_test_state(is_convecting=False, Ra=100.0, Nu=1.0)
        hyp = ConvectionHypothesis(
            mechanism="tidal_viscosity",
            params={"epsilon_ref": 6e-6, "softening_exponent": 1.0},
        )
        adj = make_adjuster(hyp, np.pi / 4, LatitudeProfile())
        Nu_before = state.Nu
        adj(state, np.zeros(31), np.zeros(31), 20e3, 0.02)
        assert state.Nu == Nu_before
```

- [ ] **Step 2: Run to verify fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestTidalViscosityAdjuster -v`
Expected: FAIL

- [ ] **Step 3: Implement _tidal_viscosity_adjuster**

Replace the stub in `Europa2D/src/convection_2d.py`:

```python
def _tidal_viscosity_adjuster(state, epsilon_0_local, epsilon_ref, n):
    """Rescale Ra and Nu using tidal-softened viscosity.

    eta_eff = eta_default / (1 + (epsilon_0 / epsilon_ref)^n)
    Ra_adj = Ra_default * (eta_default / eta_eff)
           = Ra_default * (1 + (epsilon_0 / epsilon_ref)^n)
    Nu_adj = C * Ra_adj^(1/3)
    """
    if not state.is_convecting or epsilon_ref <= 0:
        return

    softening = 1.0 + (epsilon_0_local / epsilon_ref) ** n
    Ra_adj = state.Ra * softening

    Nu_adj = max(1.0, ConvectionConstants.NU_PREFACTOR * Ra_adj ** (1.0 / 3.0))

    state.Ra = Ra_adj
    state.Nu = Nu_adj
```

- [ ] **Step 4: Run test to verify passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestTidalViscosityAdjuster -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/convection_2d.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: implement tidal-viscosity convection adjuster"
```

---

## Task 8: Wire AxialSolver2D to Accept Hypothesis

**Files:**
- Modify: `Europa2D/src/axial_solver.py:38-120`
- Test: `Europa2D/tests/test_convection_2d.py` (append)

- [ ] **Step 1: Write failing test**

Append to `Europa2D/tests/test_convection_2d.py`:

```python
from axial_solver import AxialSolver2D


class TestAxialSolverHypothesis:
    """Tests that AxialSolver2D wires hypotheses to column adjusters."""

    def test_accepts_hypothesis_kwarg(self):
        """AxialSolver2D should accept hypothesis without error."""
        hyp = ConvectionHypothesis(
            mechanism="ra_onset",
            params={"ra_crit_override": 1200},
        )
        profile = LatitudeProfile(q_ocean_mean=0.02)
        solver = AxialSolver2D(
            n_lat=5, nx=21, dt=1e12,
            latitude_profile=profile,
            use_convection=True,
            initial_thickness=20e3,
            hypothesis=hyp,
        )
        assert solver is not None

    def test_hypothesis_changes_equilibrium(self):
        """A non-trivial hypothesis must change the thickness profile."""
        profile = LatitudeProfile(q_ocean_mean=0.02)
        base_kwargs = dict(
            n_lat=5, nx=21, dt=1e12,
            latitude_profile=profile,
            use_convection=True,
            initial_thickness=20e3,
            rannacher_steps=2,
        )
        s_none = AxialSolver2D(**base_kwargs)
        s_hyp = AxialSolver2D(
            **base_kwargs,
            hypothesis=ConvectionHypothesis(
                mechanism="heat_balance",
                params={"include_tidal": False},
            ),
        )

        q_ocean = np.array([profile.ocean_heat_flux(phi) for phi in s_none.latitudes])

        for _ in range(50):
            s_none.solve_step(q_ocean)
            s_hyp.solve_step(q_ocean)

        H_none = s_none.get_thickness_profile()
        H_hyp = s_hyp.get_thickness_profile()

        assert not np.allclose(H_none, H_hyp, atol=1.0), (
            "hypothesis must change solver evolution, not just diagnostics"
        )
```

- [ ] **Step 2: Run to verify fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestAxialSolverHypothesis -v`
Expected: FAIL — `AxialSolver2D.__init__() got an unexpected keyword argument 'hypothesis'`

- [ ] **Step 3: Modify axial_solver.py**

In `Europa2D/src/axial_solver.py`:

**A.** Add import at top (after existing imports):

```python
from convection_2d import ConvectionHypothesis, make_adjuster
```

**B.** Add `hypothesis` parameter to `__init__` signature (after `lateral_method`):

```python
    lateral_method: Literal['implicit', 'explicit'] = 'implicit',
    hypothesis: Optional[ConvectionHypothesis] = None,
):
```

**C.** Store it after `self.lateral_method = lateral_method`:

```python
        self.hypothesis = hypothesis
```

**D.** In the column construction loop, create per-column adjuster and pass to Thermal_Solver. Replace the `solver = Thermal_Solver(...)` block (lines ~108-118):

```python
        # Create per-column convection adjuster from hypothesis
        adjuster = make_adjuster(self.hypothesis, phi_j, self.profile)

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
            convection_adjuster=adjuster,
        )
```

- [ ] **Step 4: Run test to verify passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestAxialSolverHypothesis -v`
Expected: 2 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add Europa2D/src/axial_solver.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: wire hypothesis through AxialSolver2D to column adjusters"
```

---

## Task 9: Thread Hypothesis Through MC Pipeline

**Files:**
- Modify: `Europa2D/src/monte_carlo_2d.py:97-115` (worker sig), `~144-154` (solver construction), `~202-228` (runner init)
- Test: `Europa2D/tests/test_convection_2d.py` (append)

- [ ] **Step 1: Write failing test**

Append to `Europa2D/tests/test_convection_2d.py`:

```python
from monte_carlo_2d import MonteCarloRunner2D


class TestMCHypothesisThreading:
    """Tests that MonteCarloRunner2D passes hypothesis through."""

    def test_runner_accepts_hypothesis(self):
        """MonteCarloRunner2D should accept hypothesis kwarg."""
        hyp = ConvectionHypothesis(
            mechanism="ra_onset",
            params={"ra_crit_override": 1200},
        )
        runner = MonteCarloRunner2D(
            n_iterations=2,
            seed=42,
            n_workers=1,
            n_lat=5,
            nx=21,
            ocean_pattern="uniform",
            verbose=False,
            hypothesis=hyp,
        )
        results = runner.run()
        assert results.n_valid >= 1
```

- [ ] **Step 2: Run to verify fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestMCHypothesisThreading -v`
Expected: FAIL — `MonteCarloRunner2D.__init__() got an unexpected keyword argument 'hypothesis'`

- [ ] **Step 3: Modify monte_carlo_2d.py**

Three changes:

**A. `_run_single_2d_sample` function** — add `hypothesis=None` parameter after `T_floor`:

```python
def _run_single_2d_sample(
    sample_id: int,
    base_seed: int,
    # ... existing params ...
    T_floor: float = 50.0,
    hypothesis_mechanism: Optional[str] = None,
    hypothesis_params: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
```

Inside the function, reconstruct the hypothesis and pass to AxialSolver2D. After the `solver = AxialSolver2D(...)` call, add hypothesis. Replace the AxialSolver2D construction:

```python
        # Reconstruct hypothesis if provided (can't pickle closures)
        hyp = None
        if hypothesis_mechanism is not None:
            from convection_2d import ConvectionHypothesis
            hyp = ConvectionHypothesis(
                mechanism=hypothesis_mechanism,
                params=hypothesis_params or {},
            )

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
            hypothesis=hyp,
        )
```

**B. `MonteCarloRunner2D.__init__`** — add `hypothesis=None` parameter after `q_tidal_scale`:

```python
    def __init__(
        self,
        # ... existing params ...
        q_tidal_scale: float = 1.20,
        hypothesis: Optional['ConvectionHypothesis'] = None,
    ):
```

Store it: `self.hypothesis = hypothesis`

**C. `MonteCarloRunner2D.run`** — pass hypothesis fields through `partial()` call. Add to the `worker = partial(...)` kwargs:

```python
            hypothesis_mechanism=self.hypothesis.mechanism if self.hypothesis else None,
            hypothesis_params=self.hypothesis.params if self.hypothesis else None,
```

Note: We pass `mechanism` and `params` separately instead of the full hypothesis object because `ConvectionHypothesis` creates closures that can't be pickled for multiprocessing. The worker reconstructs the hypothesis from these fields.

- [ ] **Step 4: Run test to verify passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/test_convection_2d.py::TestMCHypothesisThreading -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/ -v --tb=short`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py Europa2D/tests/test_convection_2d.py
git commit -m "feat: thread hypothesis through MonteCarloRunner2D"
```

---

## Task 10: Add D_conv JS Secondary Metric

**Files:**
- Modify: `autoresearch/objectives.py:151-196`
- Test: `autoresearch/tests/test_objectives.py` (append)

- [ ] **Step 1: Write failing test**

Append to `autoresearch/tests/test_objectives.py`:

```python
def test_latitude_score_includes_dconv_js():
    """compute_latitude_score must report JS_discriminability_Dconv in metrics."""
    n_lat = 5
    n_valid = 20
    latitudes = np.linspace(0, 90, n_lat)

    # Create scenarios with DIFFERENT D_conv distributions
    rng = np.random.default_rng(42)
    scenarios = {}
    for name, shift in [('uniform', 0), ('polar', 3), ('equator', -2)]:
        D_cond = rng.normal(20, 3, (n_valid, n_lat)) + shift
        D_conv = rng.normal(5, 1, (n_valid, n_lat)) + shift * 0.5
        scenarios[name] = {
            'latitudes_deg': latitudes,
            'D_cond_profiles': D_cond,
            'D_conv_profiles': D_conv,
            'H_profiles': D_cond + D_conv,
            'Ra_profiles': rng.normal(1500, 200, (n_valid, n_lat)),
        }

    score, metrics = compute_latitude_score(scenarios, consistency_error=0.01)
    assert 'JS_discriminability_Dconv' in metrics
    assert metrics['JS_discriminability_Dconv'] >= 0.0
```

- [ ] **Step 2: Run to verify fails**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest autoresearch/tests/test_objectives.py::test_latitude_score_includes_dconv_js -v`
Expected: FAIL — `KeyError: 'JS_discriminability_Dconv'`

- [ ] **Step 3: Add D_conv JS computation**

In `autoresearch/objectives.py`, inside `compute_latitude_score()`, after line 158 (`min_js = min(js_values) if js_values else 0.0`), add:

```python
    # Secondary: D_conv JS divergence (tracked, not scored)
    d_conv_js_values = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            d_a = np.asarray(scenarios[scenario_names[i]]['D_conv_profiles'])[:, idx_35]
            d_b = np.asarray(scenarios[scenario_names[j]]['D_conv_profiles'])[:, idx_35]
            d_conv_js_values.append(_js_divergence(d_a, d_b))
    min_js_dconv = min(d_conv_js_values) if d_conv_js_values else 0.0
```

Then in the `metrics` dict at the bottom, add:

```python
        'JS_discriminability_Dconv': min_js_dconv,
```

- [ ] **Step 4: Run test to verify passes**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest autoresearch/tests/test_objectives.py -v --tb=short`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add autoresearch/objectives.py autoresearch/tests/test_objectives.py
git commit -m "feat: add D_conv JS divergence as secondary latitude metric"
```

---

## Task 11: Thread Hypothesis Through Harness

**Files:**
- Modify: `autoresearch/harness.py:194-207` (_run_latitude_experiment), `~251-264` (_run_mc_ensemble)

- [ ] **Step 1: Modify _run_mc_ensemble to accept hypothesis**

In `autoresearch/harness.py`, modify `_run_mc_ensemble`:

```python
    def _run_mc_ensemble(self, ocean_pattern: str, n_samples: int, n_workers: int,
                         q_star=None, hypothesis=None,
                         grain_latitude_mode='global', grain_strain_exponent=0.5):
        """Run one MC ensemble via MonteCarloRunner2D."""
        from monte_carlo_2d import MonteCarloRunner2D

        runner = MonteCarloRunner2D(
            n_iterations=n_samples,
            seed=42,
            n_workers=n_workers,
            ocean_pattern=ocean_pattern,
            q_star=q_star,
            verbose=False,
            hypothesis=hypothesis,
            grain_latitude_mode=grain_latitude_mode,
        )
        return runner.run()
```

- [ ] **Step 2: Modify _run_latitude_experiment to accept and pass hypothesis**

```python
    def _run_latitude_experiment(self, n_samples: int, n_workers: int,
                                 hypothesis=None,
                                 grain_latitude_mode='global',
                                 grain_strain_exponent=0.5):
        """Run 3 scenarios + 1D/2D calibration, then score."""
        scenario_configs = [
            ('uniform', 'uniform', None),
            ('polar', 'polar_enhanced', 0.455),
            ('equator', 'equator_enhanced', 0.4),
        ]
        scenarios = {}
        for name, pattern, q_star in scenario_configs:
            mc = self._run_mc_ensemble(
                pattern, n_samples=n_samples, n_workers=n_workers,
                q_star=q_star, hypothesis=hypothesis,
                grain_latitude_mode=grain_latitude_mode,
                grain_strain_exponent=grain_strain_exponent,
            )
            scenarios[name] = self._mc_to_dict(mc)

        consistency_error = self._run_calibration_check()
        return compute_latitude_score(scenarios, consistency_error)
```

- [ ] **Step 3: Verify existing harness init still works**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -c "from autoresearch.harness import ExperimentHarness; print('OK')"` 
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/harness.py
git commit -m "feat: thread hypothesis through experiment harness"
```

---

## Task 12: Campaign Config and Runner Script

**Files:**
- Create: `autoresearch/experiments/hypothesis_config.json`
- Create: `autoresearch/experiments/run_hypothesis_campaign.py`

- [ ] **Step 1: Create experiments directory**

Run: `mkdir -p /c/Users/Joshu/.cursor/projects/EuropaConvection/autoresearch/experiments`

- [ ] **Step 2: Create hypothesis_config.json**

Create `autoresearch/experiments/hypothesis_config.json`:

```json
{
  "n_samples": 150,
  "n_workers": 7,
  "experiments": [
    {
      "name": "baseline_150",
      "hypothesis": null,
      "config_overrides": {}
    },
    {
      "name": "grain_alpha05",
      "hypothesis": null,
      "config_overrides": {"grain_latitude_mode": "strain", "grain_strain_exponent": 0.5}
    },
    {
      "name": "grain_alpha10",
      "hypothesis": null,
      "config_overrides": {"grain_latitude_mode": "strain", "grain_strain_exponent": 1.0}
    },
    {
      "name": "heatbal_ocean",
      "hypothesis": {"mechanism": "heat_balance", "params": {"include_tidal": false, "max_iterations": 5, "tolerance": 1e-4}},
      "config_overrides": {}
    },
    {
      "name": "heatbal_total",
      "hypothesis": {"mechanism": "heat_balance", "params": {"include_tidal": true, "max_iterations": 5, "tolerance": 1e-4}},
      "config_overrides": {}
    },
    {
      "name": "ra_crit_800",
      "hypothesis": {"mechanism": "ra_onset", "params": {"ra_crit_override": 800}},
      "config_overrides": {}
    },
    {
      "name": "ra_crit_1200",
      "hypothesis": {"mechanism": "ra_onset", "params": {"ra_crit_override": 1200}},
      "config_overrides": {}
    },
    {
      "name": "tidal_visc_n1",
      "hypothesis": {"mechanism": "tidal_viscosity", "params": {"epsilon_ref": 6e-6, "softening_exponent": 1.0}},
      "config_overrides": {}
    }
  ]
}
```

- [ ] **Step 3: Create run_hypothesis_campaign.py**

Create `autoresearch/experiments/run_hypothesis_campaign.py`:

```python
"""Run the 10-experiment hypothesis campaign and print ranked results."""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add paths
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / 'Europa2D' / 'src'))
sys.path.insert(0, str(_REPO / 'EuropaProjectDJ' / 'src'))
sys.path.insert(0, str(_REPO / 'autoresearch'))

import numpy as np
from convection_2d import ConvectionHypothesis
from harness import ExperimentHarness


def _load_config():
    config_path = Path(__file__).parent / 'hypothesis_config.json'
    with open(config_path) as f:
        return json.load(f)


def _load_log(log_path):
    """Load completed experiment names from JSONL log."""
    completed = set()
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line)
                completed.add(entry['experiment'])
    return completed


def _append_log(log_path, entry):
    """Append one experiment result to JSONL log."""
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(log_path, 'a') as f:
        f.write(json.dumps(entry, cls=_Enc) + '\n')


def run_campaign():
    config = _load_config()
    n_samples = config['n_samples']
    n_workers = config['n_workers']
    experiments = config['experiments']

    log_path = Path(__file__).parent / 'hypothesis_results.jsonl'
    completed = _load_log(log_path)

    harness = ExperimentHarness()

    results = []
    for exp in experiments:
        name = exp['name']
        if name in completed:
            print(f"[SKIP] {name} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'='*60}")

        # Build hypothesis
        hyp_def = exp.get('hypothesis')
        hypothesis = None
        if hyp_def is not None:
            hypothesis = ConvectionHypothesis(
                mechanism=hyp_def['mechanism'],
                params=hyp_def['params'],
            )

        # Config overrides for grain-strain experiments
        config_overrides = exp.get('config_overrides', {})
        grain_mode = config_overrides.get('grain_latitude_mode', 'global')
        grain_exp = config_overrides.get('grain_strain_exponent', 0.5)

        t0 = time.time()

        # Run the 3-scenario latitude experiment
        score, metrics = harness._run_latitude_experiment(
            n_samples=n_samples,
            n_workers=n_workers,
            hypothesis=hypothesis,
            grain_latitude_mode=grain_mode,
            grain_strain_exponent=grain_exp,
        )

        runtime = time.time() - t0

        entry = {
            'experiment': name,
            'hypothesis': exp.get('hypothesis'),
            'config_overrides': config_overrides,
            'latitude_score': float(score),
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in metrics.items()},
            'runtime_seconds': runtime,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        _append_log(log_path, entry)
        results.append(entry)

        print(f"  Score: {score:.2f}")
        print(f"  JS(D_cond): {metrics.get('JS_discriminability', 0):.4f}")
        print(f"  JS(D_conv): {metrics.get('JS_discriminability_Dconv', 0):.4f}")
        print(f"  D_conv contrast: {metrics.get('D_conv_contrast', 0):.2f} km")
        print(f"  Runtime: {runtime:.1f}s")

    # Print leaderboard
    if not results:
        print("\nNo new experiments run. Loading from log...")
        if log_path.exists():
            with open(log_path) as f:
                results = [json.loads(line) for line in f]

    if results:
        print(f"\n{'='*60}")
        print("  LEADERBOARD (lower score = better)")
        print(f"{'='*60}")
        ranked = sorted(results, key=lambda r: r['latitude_score'])
        for i, r in enumerate(ranked):
            m = r['metrics']
            print(f"  {i+1}. {r['experiment']:20s}  score={r['latitude_score']:+8.2f}"
                  f"  JS_Dcond={m.get('JS_discriminability', 0):.4f}"
                  f"  JS_Dconv={m.get('JS_discriminability_Dconv', 0):.4f}"
                  f"  D_conv_contrast={m.get('D_conv_contrast', 0):.1f}km")


if __name__ == '__main__':
    run_campaign()
```

- [ ] **Step 4: Verify campaign script imports work**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -c "from autoresearch.experiments.run_hypothesis_campaign import _load_config; print(_load_config()['n_samples'])"`
Expected: `150`

- [ ] **Step 5: Commit**

```bash
git add autoresearch/experiments/hypothesis_config.json autoresearch/experiments/run_hypothesis_campaign.py
git commit -m "feat: add hypothesis campaign config and runner script"
```

---

## Task 13: Run Baseline Experiment and Validate

**Files:** None (validation only)

- [ ] **Step 1: Run full test suite**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python -m pytest Europa2D/tests/ autoresearch/tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run baseline experiment (experiment 0)**

Run: `cd /c/Users/Joshu/.cursor/projects/EuropaConvection && python autoresearch/experiments/run_hypothesis_campaign.py`

This will run all experiments in sequence (starting from baseline_150). Monitor output for:
- baseline_150: score should be close to existing -5.20
- grain experiments: may or may not improve
- adjuster experiments: this is where discriminability should change

Expected runtime: ~30-45 minutes total.

- [ ] **Step 3: Analyze results**

After campaign completes, check `autoresearch/experiments/hypothesis_results.jsonl` for the leaderboard. Key questions:
- Did any mechanism improve JS_discriminability (D_cond) above 0.01?
- Did heat-balance improve JS_discriminability (D_conv) even if D_cond didn't move?
- Are Juno constraints still satisfied (D_cond @ 35 deg within 19-39 km)?

- [ ] **Step 4: Commit results**

```bash
git add autoresearch/experiments/hypothesis_results.jsonl
git commit -m "data: hypothesis campaign baseline + adjuster results"
```
