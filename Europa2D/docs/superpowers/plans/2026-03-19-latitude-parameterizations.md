# Improved Latitude Parameterizations — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade three latitude-dependent forcing parameterizations in `LatitudeProfile` to literature-grounded closed-form expressions, and propagate the new fields through the MC archive and thesis plotter.

**Architecture:** Additive fields on the existing frozen dataclass. Three formula changes (surface temperature energy balance floor, Beuthe 2013 tidal strain, Lemasquerier 2023 q_star), plus MC results and plotter carry the new metadata. No solver or EuropaProjectDJ changes.

**Tech Stack:** Python 3.10+, NumPy, SciPy (for integration tests), pytest.

**Spec:** `Europa2D/docs/superpowers/specs/2026-03-19-latitude-parameterizations-design.md`

**Literature references:**
- Ojakangas, G. W., and Stevenson, D. J. (1989), Thermal state of an ice shell on Europa, Icarus 81, 220-241.
- Ashkenazy, Y. (2019), The surface temperature of Europa, Planetary and Space Science 173, 20-30. DOI: 10.1016/j.pss.2019.06.002
- Beuthe, M. (2013), Spatial patterns of tidal heating, Icarus 223, 308-329. DOI: 10.1016/j.icarus.2012.11.020
- Tobie, G., Choblet, G., and Sotin, C. (2003), Tidally heated convection: Constraints on Europa's ice shell thickness, JGR 108, 5124. DOI: 10.1029/2003JE002099
- Lemasquerier, D. G., Bierson, C. J., and Soderlund, K. M. (2023), Europa's ocean translates interior tidal heating patterns to the ice-ocean boundary, AGU Advances. DOI: 10.1029/2023AV000994
- Soderlund, K. M., Schmidt, B. E., Wicht, J., and Blankenship, D. D. (2014), Ocean-driven heating of Europa's icy shell at low latitudes, Nature Geoscience. DOI: 10.1038/ngeo2021
- Ashkenazy, Y., and Tziperman, E. (2021), Dynamic Europa ocean shows transient Taylor columns and convection driven by ice melting and salinity, Nature Communications. DOI: 10.1038/s41467-021-26710-0

---

## File Structure

```
Europa2D/
├── src/
│   ├── latitude_profile.py     # MODIFY: new fields, updated formulas
│   ├── literature_scenarios.py  # MODIFY: scenarios use q_star
│   ├── latitude_sampler.py      # MODIFY: sample T_floor, mantle_tidal_fraction
│   ├── monte_carlo_2d.py        # MODIFY: results carry new metadata
│   └── profile_diagnostics.py   # MODIFY: diagnostics include q_star
├── scripts/
│   └── plot_thickness_profile.py # MODIFY: read new fields from NPZ
└── tests/
    ├── test_latitude_profile.py  # MODIFY: new tests for all three changes
    ├── test_literature_scenarios.py # MODIFY: updated expected ratios
    └── test_latitude_sampler.py  # MODIFY: test new sampled fields
```

---

## Task 1: Surface Temperature — Energy Balance Floor

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing tests for the energy balance floor**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestSurfaceTemperatureEnergyBalance:
    """Tests for T_s(phi) = ((T_eq^4 - T_floor^4)*cos(phi) + T_floor^4)^(1/4)."""

    def test_equator_returns_T_eq_exactly(self):
        """T_s(0) must equal T_eq — the reparameterized formula preserves this."""
        profile = LatitudeProfile(T_eq=110.0, T_floor=52.0)
        assert profile.surface_temperature(0.0) == pytest.approx(110.0, abs=1e-10)

    def test_pole_returns_T_floor(self):
        """At phi=pi/2, only T_floor^4 survives."""
        profile = LatitudeProfile(T_eq=110.0, T_floor=52.0)
        assert profile.surface_temperature(np.pi / 2) == pytest.approx(52.0, abs=1e-10)

    def test_monotonically_decreasing(self):
        profile = LatitudeProfile(T_eq=110.0, T_floor=52.0)
        lats = np.linspace(0, np.pi / 2, 50)
        temps = profile.surface_temperature(lats)
        assert np.all(np.diff(temps) <= 0)

    def test_T_floor_must_be_less_than_T_eq(self):
        with pytest.raises(ValueError, match="T_floor.*T_eq"):
            LatitudeProfile(T_eq=100.0, T_floor=100.0)
        with pytest.raises(ValueError, match="T_floor.*T_eq"):
            LatitudeProfile(T_eq=100.0, T_floor=110.0)

    def test_default_T_floor_is_52(self):
        profile = LatitudeProfile()
        assert profile.T_floor == 52.0

    def test_pole_is_warmer_than_old_clamp(self):
        """New floor (52 K) gives a warmer pole than old cos(89.5) clamp (~33.6 K)."""
        profile = LatitudeProfile(T_eq=110.0, T_floor=52.0)
        T_pole = profile.surface_temperature(np.pi / 2)
        assert T_pole > 40.0

    def test_array_input_with_floor(self):
        profile = LatitudeProfile(T_eq=110.0, T_floor=52.0)
        lats = np.array([0.0, np.radians(45), np.pi / 2])
        temps = profile.surface_temperature(lats)
        assert temps.shape == (3,)
        assert temps[0] == pytest.approx(110.0, abs=1e-10)
        assert temps[2] == pytest.approx(52.0, abs=1e-10)
        assert temps[0] > temps[1] > temps[2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestSurfaceTemperatureEnergyBalance -v`

Expected: FAIL (T_floor field does not exist, TypeError or AttributeError)

- [ ] **Step 3: Add T_floor field and __post_init__ validation to LatitudeProfile**

In `Europa2D/src/latitude_profile.py`, add the new field after `ocean_amplitude`:

```python
    T_floor: float = 52.0
```

Add a `__post_init__` method for validation (frozen dataclasses support this):

```python
    def __post_init__(self):
        if self.T_floor <= 0:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be positive."
            )
        if self.T_floor >= self.T_eq:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be less than T_eq ({self.T_eq} K). "
                "A polar floor >= equatorial temperature is non-physical for Europa."
            )
```

- [ ] **Step 4: Replace the surface_temperature formula**

Replace the body of `surface_temperature()` in `Europa2D/src/latitude_profile.py`:

```python
    def surface_temperature(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Surface temperature as a function of latitude.

        T_s(phi) = ((T_eq^4 - T_floor^4) * cos(phi) + T_floor^4)^(1/4)

        Reparameterized energy balance: T_s(0) = T_eq exactly, T_s(pi/2) = T_floor.
        The T_floor default (52 K) is from Ashkenazy (2019), absorbing obliquity,
        seasonal insolation, thermal inertia, and Jupiter longwave radiation.

        References:
            Ojakangas & Stevenson (1989): radiative equilibrium framework
            Ashkenazy (2019): full seasonal energy balance, T_pole = 51-52 K

        Args:
            phi: Geographic latitude in radians (0=equator, pi/2=pole)

        Returns:
            Surface temperature (K)
        """
        phi_arr = np.asarray(phi)
        T_eq4 = self.T_eq ** 4
        T_fl4 = self.T_floor ** 4
        result = ((T_eq4 - T_fl4) * np.cos(phi_arr) + T_fl4) ** 0.25
        return float(result) if np.ndim(phi) == 0 else result
```

Remove the module-level constants `_PHI_FLOOR` and `_COS_FLOOR` (lines 30-34 of current file).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All new tests PASS. Check that existing `TestSurfaceTemperature` tests also pass — some may need updating since the formula changed (e.g., `test_87_5_degrees_uses_true_cosine_profile` expects the old formula). Update any broken existing tests to match the new formula.

- [ ] **Step 6: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: replace surface temp clamp with energy balance floor (Ashkenazy 2019)

T_s(phi) = ((T_eq^4 - T_floor^4)*cos(phi) + T_floor^4)^(1/4)
T_floor default 52 K from Ashkenazy (2019) full seasonal energy balance.
T_s(0) = T_eq preserved exactly. Removes ad hoc cos(89.5 deg) clamp."
```

---

## Task 2: Tidal Strain — Beuthe (2013) Square-Root Form

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing tests for the sqrt strain form**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestTidalStrainBeuthe:
    """Tests for eps_0(phi) = eps_eq * sqrt(1 + c * sin^2(phi)), c = (eps_pole/eps_eq)^2 - 1."""

    def test_equator_returns_epsilon_eq(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(0.0) == pytest.approx(6e-6, rel=1e-10)

    def test_pole_returns_epsilon_pole(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        assert profile.tidal_strain(np.pi / 2) == pytest.approx(1.2e-5, rel=1e-10)

    def test_midlatitude_heating_ratio_matches_beuthe(self):
        """At 45 deg, eps^2/eps_eq^2 should be 1 + c*0.5, not 2.25 (old linear)."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        c = (1.2e-5 / 6e-6) ** 2 - 1  # = 3.0
        eps_45 = profile.tidal_strain(np.radians(45))
        ratio = (eps_45 / 6e-6) ** 2
        expected = 1.0 + c * 0.5  # = 2.5
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_default_c_equals_3_for_beuthe_pattern(self):
        """With defaults (6e-6, 1.2e-5), c = (2)^2 - 1 = 3, giving the exact
        Beuthe (2013) whole-shell eccentricity-tide pattern."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        c = (profile.epsilon_pole / profile.epsilon_eq) ** 2 - 1
        assert c == pytest.approx(3.0, rel=1e-10)

    def test_monotonically_increasing(self):
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        lats = np.linspace(0, np.pi / 2, 50)
        strains = profile.tidal_strain(lats)
        assert np.all(np.diff(strains) >= 0)

    def test_heating_ratio_pole_to_equator_is_4(self):
        """eps_pole^2 / eps_eq^2 = (1.2e-5)^2 / (6e-6)^2 = 4."""
        profile = LatitudeProfile(epsilon_eq=6e-6, epsilon_pole=1.2e-5)
        ratio = (profile.tidal_strain(np.pi / 2) / profile.tidal_strain(0.0)) ** 2
        assert ratio == pytest.approx(4.0, rel=1e-10)
```

- [ ] **Step 2: Run tests to verify the midlatitude test fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestTidalStrainBeuthe::test_midlatitude_heating_ratio_matches_beuthe -v`

Expected: FAIL (old linear formula gives ratio = 2.25, not 2.5)

- [ ] **Step 3: Replace the tidal_strain formula**

Replace the body of `tidal_strain()` in `Europa2D/src/latitude_profile.py`:

```python
    def tidal_strain(self, phi: FloatOrArray) -> FloatOrArray:
        """
        Tidal strain amplitude as a function of latitude.

        eps_0(phi) = eps_eq * sqrt(1 + c * sin^2(phi))
        where c = (eps_pole / eps_eq)^2 - 1

        This ensures eps_0^2(phi) = eps_eq^2 * (1 + c*sin^2(phi)), which
        reproduces the Beuthe (2013) zonally-averaged whole-shell eccentricity-tide
        dissipation pattern: q_tidal ~ 1 + 3*sin^2(phi) when c = 3.

        References:
            Beuthe (2013): spatial patterns of tidal heating, Icarus 223, 308-329
            Tobie et al. (2003): ~4:1 pole-to-equator dissipation ratio

        Args:
            phi: Geographic latitude in radians (0=equator, pi/2=pole)

        Returns:
            Tidal strain amplitude (dimensionless)
        """
        phi_arr = np.asarray(phi)
        c = (self.epsilon_pole / self.epsilon_eq) ** 2 - 1.0
        sin2 = np.sin(phi_arr) ** 2
        result = self.epsilon_eq * np.sqrt(1.0 + c * sin2)
        return float(result) if np.ndim(phi) == 0 else result
```

- [ ] **Step 4: Run all tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All PASS. The existing `TestTidalStrain` endpoint tests should still pass since endpoints are unchanged. The midlatitude test `test_midlatitude_is_between` should still pass since the value is still between eps_eq and eps_pole.

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: use Beuthe (2013) sqrt tidal strain for correct mid-latitude heating

eps_0(phi) = eps_eq * sqrt(1 + c * sin^2(phi)), c = (eps_pole/eps_eq)^2 - 1
Fixes ~10% mid-latitude underestimate. Endpoints unchanged."
```

---

## Task 3: Ocean Heat Flux — q_star, mantle_tidal_fraction, strict_q_star

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py`
- Modify: `Europa2D/src/latitude_profile.py`

- [ ] **Step 1: Write failing tests for q_star derivation and validation**

Append to `Europa2D/tests/test_latitude_profile.py`:

```python
class TestQStarDerivation:
    """Tests for q_star / mantle_tidal_fraction ocean heat flux parameterization."""

    def test_resolved_q_star_from_mantle_tidal_fraction(self):
        """When q_star and ocean_amplitude are both None, derive from mantle_tidal_fraction."""
        profile = LatitudeProfile(mantle_tidal_fraction=0.5, ocean_pattern="polar_enhanced")
        assert profile.resolved_q_star() == pytest.approx(0.91 * 0.5, rel=1e-10)

    def test_explicit_q_star_overrides_mantle_tidal_fraction(self):
        profile = LatitudeProfile(q_star=0.4, mantle_tidal_fraction=0.9, ocean_pattern="polar_enhanced")
        assert profile.resolved_q_star() == pytest.approx(0.4, rel=1e-10)

    def test_ocean_amplitude_overrides_q_star(self):
        """ocean_amplitude takes highest priority (backward compat)."""
        profile = LatitudeProfile(
            ocean_amplitude=1.0, q_star=0.4,
            ocean_pattern="polar_enhanced", q_ocean_mean=0.02,
        )
        assert profile.resolved_ocean_amplitude() == 1.0

    def test_q_star_to_amplitude_polar(self):
        """a = 3*q_star / (3 - q_star) for polar_enhanced."""
        profile = LatitudeProfile(q_star=0.455, ocean_pattern="polar_enhanced")
        a = profile.resolved_ocean_amplitude()
        expected = 3.0 * 0.455 / (3.0 - 0.455)  # = 0.536
        assert a == pytest.approx(expected, rel=1e-6)

    def test_q_star_to_amplitude_equator(self):
        """a = 3*q_star / (3 - 2*q_star) for equator_enhanced."""
        profile = LatitudeProfile(q_star=0.4, ocean_pattern="equator_enhanced")
        a = profile.resolved_ocean_amplitude()
        expected = 3.0 * 0.4 / (3.0 - 2.0 * 0.4)  # = 0.545
        assert a == pytest.approx(expected, rel=1e-6)

    def test_uniform_pattern_ignores_q_star(self):
        profile = LatitudeProfile(q_star=0.5, ocean_pattern="uniform")
        assert profile.resolved_ocean_amplitude() == 0.0

    def test_strict_q_star_rejects_above_091(self):
        with pytest.raises(ValueError, match="q_star.*0.91"):
            LatitudeProfile(q_star=0.95, ocean_pattern="polar_enhanced", strict_q_star=True)

    def test_relaxed_q_star_allows_above_091(self):
        """With strict_q_star=False, values up to the math singularity are OK."""
        profile = LatitudeProfile(q_star=1.5, ocean_pattern="polar_enhanced", strict_q_star=False)
        assert profile.resolved_q_star() == pytest.approx(1.5)

    def test_relaxed_q_star_rejects_at_singularity_polar(self):
        with pytest.raises(ValueError):
            LatitudeProfile(q_star=3.0, ocean_pattern="polar_enhanced", strict_q_star=False)

    def test_relaxed_q_star_rejects_at_singularity_equator(self):
        with pytest.raises(ValueError):
            LatitudeProfile(q_star=1.5, ocean_pattern="equator_enhanced", strict_q_star=False)

    def test_normalization_preserved_with_q_star(self):
        """Global mean must be preserved regardless of q_star derivation path."""
        from scipy.integrate import quad
        cases = [
            ("polar_enhanced", 0.2),
            ("polar_enhanced", 0.455),
            ("polar_enhanced", 0.8),
            ("equator_enhanced", 0.2),
            ("equator_enhanced", 0.4),
            ("equator_enhanced", 0.7),
        ]
        for pattern, q_star_val in cases:
            profile = LatitudeProfile(
                q_ocean_mean=0.02, ocean_pattern=pattern,
                q_star=q_star_val, strict_q_star=False,
            )
            numerator, _ = quad(
                lambda phi: profile.ocean_heat_flux(phi) * np.cos(phi),
                0, np.pi / 2
            )
            assert numerator == pytest.approx(0.02, rel=0.01), \
                f"Failed for {pattern} q_star={q_star_val}"

    def test_default_mantle_tidal_fraction_is_05(self):
        profile = LatitudeProfile()
        assert profile.mantle_tidal_fraction == 0.5

    def test_default_strict_q_star_is_true(self):
        profile = LatitudeProfile()
        assert profile.strict_q_star is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py::TestQStarDerivation -v`

Expected: FAIL (q_star, mantle_tidal_fraction, strict_q_star fields do not exist)

- [ ] **Step 3: Add new fields to LatitudeProfile**

In `Europa2D/src/latitude_profile.py`, add after the `T_floor` field:

```python
    q_star: Optional[float] = None
    mantle_tidal_fraction: float = 0.5
    strict_q_star: bool = True
```

- [ ] **Step 4: Add validation to __post_init__**

Extend `__post_init__` in `Europa2D/src/latitude_profile.py`:

```python
    def __post_init__(self):
        if self.T_floor <= 0:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be positive."
            )
        if self.T_floor >= self.T_eq:
            raise ValueError(
                f"T_floor ({self.T_floor} K) must be less than T_eq ({self.T_eq} K). "
                "A polar floor >= equatorial temperature is non-physical for Europa."
            )
        if self.q_star is not None:
            if self.strict_q_star and self.q_star > 0.91:
                raise ValueError(
                    f"q_star ({self.q_star}) exceeds Lemasquerier (2023) physical range "
                    f"(max ~0.91 for pure tidal mantle heating). "
                    f"Set strict_q_star=False for exploratory runs."
                )
            if self.ocean_pattern == "polar_enhanced" and self.q_star >= 3.0:
                raise ValueError(
                    f"q_star ({self.q_star}) >= 3.0 causes singularity in polar_enhanced "
                    f"amplitude inversion: a = 3*q_star/(3-q_star)."
                )
            if self.ocean_pattern == "equator_enhanced" and self.q_star >= 1.5:
                raise ValueError(
                    f"q_star ({self.q_star}) >= 1.5 causes singularity in equator_enhanced "
                    f"amplitude inversion: a = 3*q_star/(3-2*q_star)."
                )
```

- [ ] **Step 5: Add resolved_q_star() and update resolved_ocean_amplitude()**

In `Europa2D/src/latitude_profile.py`, add a new method and update the existing one:

```python
    def resolved_q_star(self) -> float:
        """
        Return the effective q_star (Lemasquerier 2023 contrast parameter).

        Resolution order:
        1. Explicit q_star field (if not None)
        2. Derived from mantle_tidal_fraction: q_star = 0.91 * mantle_tidal_fraction

        Returns 0.0 for uniform pattern.

        References:
            Lemasquerier et al. (2023): q* = 0.91 for pure tidal mantle heating
        """
        if self.ocean_pattern == "uniform":
            return 0.0
        if self.q_star is not None:
            return float(self.q_star)
        return 0.91 * self.mantle_tidal_fraction

    def _q_star_to_amplitude(self, q_star: float) -> float:
        """
        Convert q_star to shape function amplitude a.

        polar_enhanced:   a = 3*q_star / (3 - q_star)
        equator_enhanced: a = 3*q_star / (3 - 2*q_star)
        uniform:          a = 0
        """
        if self.ocean_pattern == "uniform":
            return 0.0
        if self.ocean_pattern == "polar_enhanced":
            return 3.0 * q_star / (3.0 - q_star)
        if self.ocean_pattern == "equator_enhanced":
            return 3.0 * q_star / (3.0 - 2.0 * q_star)
        raise ValueError(f"Unknown ocean pattern: {self.ocean_pattern}")

    def resolved_ocean_amplitude(self) -> float:
        """
        Return the contrast amplitude used by the ocean heat-flux pattern.

        Resolution order:
        1. Explicit ocean_amplitude (if not None) — backward compat
        2. Derived from q_star via pattern-specific inversion
        3. Derived from mantle_tidal_fraction -> q_star -> amplitude

        References:
            Lemasquerier et al. (2023): q* contrast parameter
            Soderlund et al. (2014): ~40% zonal-mean variation
        """
        if self.ocean_amplitude is not None:
            return float(self.ocean_amplitude)
        if self.ocean_pattern == "uniform":
            return 0.0
        q = self.resolved_q_star()
        return self._q_star_to_amplitude(q)
```

- [ ] **Step 6: Run all tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_profile.py -v`

Expected: All PASS. Existing `TestOceanHeatFlux` tests that rely on default amplitudes may need updating since the default derivation path now uses `mantle_tidal_fraction=0.5` -> `q_star=0.455` -> `a=0.536` instead of hardcoded defaults. Update these if they fail.

- [ ] **Step 7: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/tests/test_latitude_profile.py
git commit -m "feat: add q_star and mantle_tidal_fraction (Lemasquerier 2023)

Two-tiered validation: science bounds (<=0.91) by default,
math-safe bounds via strict_q_star=False.
Resolution: ocean_amplitude > q_star > mantle_tidal_fraction."
```

---

## Task 4: Update Literature Scenarios

**Files:**
- Modify: `Europa2D/src/literature_scenarios.py`
- Modify: `Europa2D/tests/test_literature_scenarios.py`

- [ ] **Step 1: Write updated tests**

Replace `Europa2D/tests/test_literature_scenarios.py`:

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import src

import pytest

from literature_scenarios import DEFAULT_SCENARIO, get_scenario, list_scenarios


def test_default_scenario_is_conservative_lemasquerier_case():
    assert DEFAULT_SCENARIO == "lemasquerier2023_polar"


def test_list_scenarios_contains_core_literature_cases():
    scenarios = set(list_scenarios())
    assert "uniform_transport" in scenarios
    assert "soderlund2014_equator" in scenarios
    assert "lemasquerier2023_polar" in scenarios
    assert "lemasquerier2023_polar_strong" in scenarios


def test_soderlund_preset_has_expected_endpoint_ratio():
    """q_star=0.4, equator_enhanced -> a=0.545 -> q_eq/q_pole = 1.55."""
    scenario = get_scenario("soderlund2014_equator")
    profile = scenario.build_profile(
        T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02,
    )
    q_eq, q_pole = profile.ocean_endpoint_fluxes()
    assert q_eq / q_pole == pytest.approx(1.545, rel=0.01)


def test_lemasquerier_conservative_endpoint_ratio():
    """q_star=0.455, polar_enhanced -> a=0.536 -> q_pole/q_eq = 1.54."""
    scenario = get_scenario("lemasquerier2023_polar")
    profile = scenario.build_profile(
        T_eq=110.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5, q_ocean_mean=0.02,
    )
    assert profile.ocean_endpoint_ratio() == pytest.approx(1.54, rel=0.01)


def test_strong_lemasquerier_stronger_than_conservative():
    conservative = get_scenario("lemasquerier2023_polar")
    strong = get_scenario("lemasquerier2023_polar_strong")
    c_prof = conservative.build_profile(110.0, 6e-6, 1.2e-5, 0.02)
    s_prof = strong.build_profile(110.0, 6e-6, 1.2e-5, 0.02)
    assert s_prof.ocean_endpoint_ratio() > c_prof.ocean_endpoint_ratio()


def test_scenarios_use_q_star_not_ocean_amplitude():
    """All scenarios should set q_star, not ocean_amplitude."""
    for name in list_scenarios():
        scenario = get_scenario(name)
        assert hasattr(scenario, 'q_star')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_literature_scenarios.py -v`

Expected: FAIL (scenarios still use ocean_amplitude, endpoint ratios differ)

- [ ] **Step 3: Update LiteratureScenario to use q_star**

Replace `Europa2D/src/literature_scenarios.py`:

```python
"""
Named literature-backed forcing presets for Europa2D.

These presets provide reproducible labels and default amplitudes for the
latitude-only ocean heat-flux proxies used by the 2D shell model.

References:
    Lemasquerier et al. (2023): q* contrast parameter, DOI: 10.1029/2023AV000994
    Soderlund et al. (2014): low-latitude ocean heating, DOI: 10.1038/ngeo2021
    Ashkenazy & Tziperman (2021): efficient meridional transport, DOI: 10.1038/s41467-021-26710-0
"""
from dataclasses import dataclass
from typing import Literal

from latitude_profile import LatitudeProfile, OceanPattern


ScenarioName = Literal[
    "uniform_transport",
    "soderlund2014_equator",
    "lemasquerier2023_polar",
    "lemasquerier2023_polar_strong",
]


@dataclass(frozen=True)
class LiteratureScenario:
    """Container for a named literature-backed forcing preset."""

    name: ScenarioName
    title: str
    citation: str
    reference_url: str
    description: str
    ocean_pattern: OceanPattern
    q_star: float

    def build_profile(
        self,
        T_eq: float,
        epsilon_eq: float,
        epsilon_pole: float,
        q_ocean_mean: float,
        T_floor: float = 52.0,
    ) -> LatitudeProfile:
        """Create a LatitudeProfile using the preset forcing family."""
        return LatitudeProfile(
            T_eq=T_eq,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
            q_star=self.q_star if self.q_star > 0 else None,
            T_floor=T_floor,
            strict_q_star=False,  # scenarios are pre-validated
        )


SCENARIOS: dict[ScenarioName, LiteratureScenario] = {
    "uniform_transport": LiteratureScenario(
        name="uniform_transport",
        title="Uniform transport proxy",
        citation="Ashkenazy & Tziperman (2021)",
        reference_url="https://www.nature.com/articles/s41467-021-26710-0",
        description="Efficient meridional transport benchmark with no imposed latitude contrast.",
        ocean_pattern="uniform",
        q_star=0.0,
    ),
    "soderlund2014_equator": LiteratureScenario(
        name="soderlund2014_equator",
        title="Equator-enhanced proxy",
        citation="Soderlund et al. (2014)",
        reference_url="https://doi.org/10.1038/ngeo2021",
        description="Low-latitude ocean heat-delivery benchmark with q* = 0.4.",
        ocean_pattern="equator_enhanced",
        q_star=0.4,
    ),
    "lemasquerier2023_polar": LiteratureScenario(
        name="lemasquerier2023_polar",
        title="Polar-enhanced proxy",
        citation="Lemasquerier et al. (2023)",
        reference_url="https://doi.org/10.1029/2023AV000994",
        description="Conservative polar-enhanced mantle-tidal benchmark, mantle_tidal_fraction = 0.5.",
        ocean_pattern="polar_enhanced",
        q_star=0.455,
    ),
    "lemasquerier2023_polar_strong": LiteratureScenario(
        name="lemasquerier2023_polar_strong",
        title="Strong polar-enhanced proxy",
        citation="Lemasquerier et al. (2023)",
        reference_url="https://doi.org/10.1029/2023AV000994",
        description="Upper-end polar-tidal sensitivity, mantle_tidal_fraction = 0.9.",
        ocean_pattern="polar_enhanced",
        q_star=0.819,
    ),
}

DEFAULT_SCENARIO: ScenarioName = "lemasquerier2023_polar"


def get_scenario(name: ScenarioName) -> LiteratureScenario:
    """Return a named literature-backed forcing preset."""
    return SCENARIOS[name]


def list_scenarios() -> tuple[ScenarioName, ...]:
    """Return the available preset names in stable order."""
    return tuple(SCENARIOS.keys())
```

- [ ] **Step 4: Run tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_literature_scenarios.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/literature_scenarios.py Europa2D/tests/test_literature_scenarios.py
git commit -m "feat: update literature scenarios to use q_star (Lemasquerier 2023)

Scenarios now specify q_star instead of ocean_amplitude.
Soderlund: q*=0.4, Lemasquerier conservative: q*=0.455, strong: q*=0.819."
```

---

## Task 5: Update Latitude Sampler

**Files:**
- Modify: `Europa2D/src/latitude_sampler.py`
- Modify: `Europa2D/tests/test_latitude_sampler.py`

- [ ] **Step 1: Write failing tests for new sampled fields**

Append to `Europa2D/tests/test_latitude_sampler.py`:

```python
class TestSamplerNewFields:

    def test_profile_has_T_floor(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'T_floor')
        assert 40 <= profile.T_floor <= 70

    def test_profile_has_mantle_tidal_fraction(self):
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        assert hasattr(profile, 'mantle_tidal_fraction')
        assert 0.0 < profile.mantle_tidal_fraction < 1.0

    def test_T_floor_independent_of_q_ocean(self):
        """T_floor should NOT be derived from q_ocean_mean."""
        sampler = LatitudeParameterSampler(seed=42)
        _, profile = sampler.sample()
        # T_floor should be sampled from Normal(52, 5), not 52 + 240*q
        assert profile.T_floor != pytest.approx(52.0 + 240.0 * profile.q_ocean_mean, rel=0.01)

    def test_T_floor_less_than_T_eq(self):
        """Guard: T_floor must always be < T_eq."""
        sampler = LatitudeParameterSampler(seed=0)
        for _ in range(200):
            _, profile = sampler.sample()
            assert profile.T_floor < profile.T_eq
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_sampler.py::TestSamplerNewFields -v`

Expected: FAIL

- [ ] **Step 3: Update LatitudeParameterSampler.sample()**

In `Europa2D/src/latitude_sampler.py`, add sampling of T_floor and mantle_tidal_fraction in the `sample()` method, after the T_eq sampling block:

```python
        # T_floor: independent of q_ocean_mean (avoids double-counting)
        # Ashkenazy (2019): T_pole = 51-52 K at zero internal heating
        T_floor = self.rng.normal(52.0, 5.0)
        T_floor = np.clip(T_floor, 40.0, 70.0)
        # Ensure T_floor < T_eq
        T_floor = min(T_floor, T_eq - 1.0)

        # mantle_tidal_fraction: radiogenic vs tidal partition
        # Lemasquerier (2023): q* = 0.91 * mantle_tidal_fraction
        mantle_tidal_fraction = self.rng.uniform(0.1, 0.9)

        # q_star sampling depends on ocean pattern (spec: Change 3, MC sampling)
        # - For polar_enhanced: derived from mantle_tidal_fraction (default path)
        # - For equator_enhanced: sampled directly, Normal(0.4, 0.1) clipped [0.1, 0.8]
        # - For uniform: not used (q_star = 0)
        q_star_explicit = None
        if self.ocean_pattern == "equator_enhanced":
            q_star_explicit = self.rng.normal(0.4, 0.1)
            q_star_explicit = float(np.clip(q_star_explicit, 0.1, 0.8))
```

Update the `LatitudeProfile` construction to include the new fields:

```python
        profile = LatitudeProfile(
            T_eq=T_eq,
            T_floor=T_floor,
            epsilon_eq=epsilon_eq,
            epsilon_pole=epsilon_pole,
            q_ocean_mean=q_ocean_mean,
            ocean_pattern=self.ocean_pattern,
            ocean_amplitude=self.ocean_amplitude,
            q_star=q_star_explicit,
            mantle_tidal_fraction=mantle_tidal_fraction,
        )
```

- [ ] **Step 4: Run all sampler tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_latitude_sampler.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/latitude_sampler.py Europa2D/tests/test_latitude_sampler.py
git commit -m "feat: sample T_floor and mantle_tidal_fraction in 2D MC sampler

T_floor ~ Normal(52, 5) clipped [40, 70], independent of q_ocean_mean.
mantle_tidal_fraction ~ Uniform(0.1, 0.9)."
```

---

## Task 6: MC Results Carry New Metadata

**Files:**
- Modify: `Europa2D/src/monte_carlo_2d.py`

- [ ] **Step 1: Add fields to MonteCarloResults2D**

In `Europa2D/src/monte_carlo_2d.py`, add to the `MonteCarloResults2D` dataclass after `ocean_amplitude`:

```python
    T_floor: float = 52.0
    q_star: float = 0.0
    mantle_tidal_fraction: float = 0.5
```

- [ ] **Step 2: Pass new fields through MonteCarloRunner2D**

Add constructor parameters to `MonteCarloRunner2D.__init__()`:

```python
        self.T_floor = T_floor  # (add T_floor: float = 52.0 to __init__ signature)
        self.mantle_tidal_fraction = mantle_tidal_fraction  # (add to __init__)
```

Update the `MonteCarloResults2D` construction in `run()` to include:

```python
            T_floor=self.T_floor,
            q_star=LatitudeProfile(
                ocean_pattern=self.ocean_pattern,
                ocean_amplitude=self.ocean_amplitude,
                mantle_tidal_fraction=self.mantle_tidal_fraction,
            ).resolved_q_star(),
            mantle_tidal_fraction=self.mantle_tidal_fraction,
```

- [ ] **Step 3: Update save_results_2d to write new fields**

In `save_results_2d()`, add to `save_dict`:

```python
        'T_floor': results.T_floor,
        'q_star': results.q_star,
        'mantle_tidal_fraction': results.mantle_tidal_fraction,
```

- [ ] **Step 4: Run existing MC tests to verify nothing broke**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_monte_carlo_2d.py -v --timeout=120`

Expected: All PASS (defaults preserve existing behavior)

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py
git commit -m "feat: MC results carry T_floor, q_star, mantle_tidal_fraction metadata"
```

---

## Task 7: Plotter Reads New Fields

**Files:**
- Modify: `Europa2D/scripts/plot_thickness_profile.py`

- [ ] **Step 1: Update _extract_pattern to read new fields**

Replace `_extract_pattern` in `Europa2D/scripts/plot_thickness_profile.py`:

```python
def _extract_pattern(data: np.lib.npyio.NpzFile) -> dict:
    """Read forcing metadata from the NPZ archive, with backward-compatible defaults."""
    pattern = str(np.asarray(data["ocean_pattern"]).item()) if "ocean_pattern" in data else "polar_enhanced"
    amplitude = float(data["ocean_amplitude"]) if "ocean_amplitude" in data else None
    T_floor = float(data["T_floor"]) if "T_floor" in data else 52.0
    q_star = float(data["q_star"]) if "q_star" in data else None
    mantle_tidal_fraction = float(data["mantle_tidal_fraction"]) if "mantle_tidal_fraction" in data else 0.5
    return {
        "pattern": pattern,
        "amplitude": amplitude,
        "T_floor": T_floor,
        "q_star": q_star,
        "mantle_tidal_fraction": mantle_tidal_fraction,
    }
```

- [ ] **Step 2: Update callers of _extract_pattern**

In `plot_thickness_profile()`, update the profile construction:

```python
    meta = _extract_pattern(data)
    profile = LatitudeProfile(
        q_ocean_mean=1.0,
        ocean_pattern=meta["pattern"],
        ocean_amplitude=meta["amplitude"],
        T_floor=meta["T_floor"],
        q_star=meta["q_star"],
        mantle_tidal_fraction=meta["mantle_tidal_fraction"],
        strict_q_star=False,
    )
```

- [ ] **Step 3: Add q_star and T_floor to summary annotation**

In the `summary_lines` list, add after the `q_pole/q_eq` line:

```python
        f"q* = {meta['q_star']:.3f}" if meta['q_star'] is not None else "q* = N/A (legacy)",
        f"T_floor = {meta['T_floor']:.1f} K",
```

- [ ] **Step 4: Verify plotter still works with old NPZ files (no new fields)**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -c "from scripts.plot_thickness_profile import _extract_pattern; import numpy as np; print('Import OK')"` (or test with an existing NPZ if available)

Expected: No crash. Backward-compatible defaults kick in.

- [ ] **Step 5: Commit**

```bash
git add Europa2D/scripts/plot_thickness_profile.py
git commit -m "feat: plotter reads and annotates T_floor, q_star from MC archives"
```

---

## Task 8: Update Profile Diagnostics

**Files:**
- Modify: `Europa2D/src/profile_diagnostics.py`

- [ ] **Step 1: Add q_star, mantle_tidal_fraction, and T_floor to ProfileDiagnostics**

Add to the `ProfileDiagnostics` frozen dataclass:

```python
    q_star: float
    mantle_tidal_fraction: float
    T_floor: float
```

- [ ] **Step 2: Update compute_profile_diagnostics**

In `compute_profile_diagnostics()`, add to the return:

```python
        q_star=profile.resolved_q_star(),
        mantle_tidal_fraction=profile.mantle_tidal_fraction,
        T_floor=profile.T_floor,
```

- [ ] **Step 3: Update format_diagnostic_lines**

Add lines:

```python
        f"q* = {diagnostics.q_star:.3f} (Lemasquerier 2023)",
        f"mantle tidal fraction = {diagnostics.mantle_tidal_fraction:.2f}",
        f"T_floor = {diagnostics.T_floor:.1f} K (Ashkenazy 2019)",
```

- [ ] **Step 4: Run tests**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_profile_diagnostics.py -v` (if exists, otherwise run full suite)

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/ -v --timeout=120`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/profile_diagnostics.py
git commit -m "feat: diagnostics include q_star and T_floor for thesis annotations"
```

---

## Task 9: Regression — Existing Tests and Validation

**Files:**
- Modify: `Europa2D/tests/test_latitude_profile.py` (cleanup old tests if needed)
- Run: `Europa2D/tests/test_validation.py`

- [ ] **Step 1: Run full test suite**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/ -v --timeout=120`

Expected: Some tests will fail due to changed defaults. Specifically:

**Tests that WILL break and need updating:**
- `test_latitude_profile.py::TestSurfaceTemperature::test_87_5_degrees_uses_true_cosine_profile` — expects old cos^(1/4) formula
- `test_latitude_profile.py::TestOceanHeatFlux::test_polar_enhanced_default_ratio_is_conservative_two_to_one` — expects 2:1 ratio, now ~1.54:1 with default `mantle_tidal_fraction=0.5`
- `test_latitude_profile.py::TestOceanHeatFlux::test_equator_enhanced_default_ratio_matches_soderlund_proxy` — expects 1.4:1, now different with default derivation chain
- `test_latitude_profile.py::TestOceanHeatFlux::test_explicit_ocean_amplitude_override_changes_ratio` — may need q_star/strict_q_star adjustment

Update each failing test to reflect the new physics or add explicit `ocean_amplitude` overrides where the test is specifically about the old shape function behavior.

- [ ] **Step 2: Run 2D single-column validation**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/test_validation.py -v`

Expected: PASS. The validation test uses explicit `T_eq=104.0` with `ocean_pattern="uniform"`. With the new formula, `T_s(0) = T_eq = 104.0` exactly (reparameterized). The result should match 1D within 1%.

- [ ] **Step 3: Commit any test fixups**

```bash
git add Europa2D/tests/
git commit -m "test: update test expectations for new latitude parameterizations"
```

---

## Task 10: Final Verification and Cleanup

- [ ] **Step 1: Run full test suite one final time**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\Europa2D && python -m pytest tests/ -v --timeout=120`

Expected: All PASS, zero failures

- [ ] **Step 2: Verify no unused imports or dead code**

Check that `_PHI_FLOOR`, `_COS_FLOOR`, and any references to them are removed from `latitude_profile.py`.

- [ ] **Step 3: Commit cleanup if needed**

```bash
git add -A Europa2D/
git commit -m "chore: remove dead code from latitude parameterization upgrade"
```
