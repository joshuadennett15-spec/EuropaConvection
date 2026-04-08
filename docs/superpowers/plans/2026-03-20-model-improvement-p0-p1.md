# Europa Model Improvement Plan (P0 + P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up internal consistency issues identified in the 2026-03-20 research note, add higher-value physics upgrades, and validate with benchmark experiments — all before any new large MC campaigns.

**Architecture:** Modifications to existing Europa2D modules (no new files except one test file). P0 fixes parameter defaults, reporting, and documentation. P1 generalizes tidal-heating and surface-boundary parameterizations. Validation tasks confirm single-column and multi-scenario benchmarks.

**Tech Stack:** Python 3.10+, NumPy, pytest, existing EuropaProjectDJ physics modules

**Research note:** `docs/research/2026-03-20-europa-model-improvement-report.md`

---

## File Structure

All changes modify existing files. No new source files are required.

| File | Role | Tasks |
|------|------|-------|
| `Europa2D/src/monte_carlo_2d.py` | MC results dataclass and runner | 1, 3, 4 |
| `Europa2D/src/literature_scenarios.py` | Scenario presets and DEFAULT_SCENARIO | 2 |
| `Europa2D/src/latitude_profile.py` | Surface temp, tidal strain, ocean flux | 5, 6, 7 |
| `Europa2D/src/latitude_sampler.py` | MC parameter sampling | 6, 7 |
| `Europa2D/src/profile_diagnostics.py` | Post-hoc band diagnostics | 4 |
| `Europa2D/tests/test_monte_carlo_2d.py` | MC runner tests | 1, 3, 4 |
| `Europa2D/tests/test_latitude_profile.py` | Profile physics tests | 5, 6, 7 |
| `Europa2D/tests/test_latitude_sampler.py` | Sampler tests | 6, 7 |
| `Europa2D/tests/test_literature_scenarios.py` | Scenario preset tests | 2 |
| `Europa2D/tests/test_validation.py` | Cross-solver validation | 9 |
| `Europa2D/scripts/run_2d_single.py` | Single-run diagnostic script | 10 |

---

## Phase A — Internal Consistency (P0)

These tasks must all complete before any new MC campaigns. Tasks 1-4 can be executed in parallel. Task 5 must run after Task 1 (they touch overlapping code in `monte_carlo_2d.py`).

---

### Task 1: Fix MonteCarloResults2D T_floor default

The `MonteCarloResults2D` dataclass has `T_floor: float = 52.0`, but the sampler and profile both use 46.0 K (Ashkenazy 2019). This creates a metadata inconsistency in saved NPZ archives.

**Files:**
- Modify: `Europa2D/src/monte_carlo_2d.py` (MonteCarloResults2D dataclass, ~line 50)
- Test: `Europa2D/tests/test_monte_carlo_2d.py`

- [ ] **Step 1: Write the failing test**

```python
def test_results_default_t_floor_matches_ashkenazy():
    """MonteCarloResults2D default T_floor must be 46.0 K (Ashkenazy 2019)."""
    import numpy as np
    from Europa2D.src.monte_carlo_2d import MonteCarloResults2D

    results = MonteCarloResults2D(
        H_profiles=np.zeros((1, 5)),
        latitudes_deg=np.linspace(0, 90, 5),
        n_iterations=1,
        n_valid=1,
        H_median=np.zeros(5),
        H_mean=np.zeros(5),
        H_sigma_low=np.zeros(5),
        H_sigma_high=np.zeros(5),
        runtime_seconds=0.0,
        ocean_pattern="uniform",
        ocean_amplitude=0.0,
    )
    assert results.T_floor == 46.0, (
        f"Default T_floor={results.T_floor}, expected 46.0 K (Ashkenazy 2019)"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_results_default_t_floor_matches_ashkenazy -v`
Expected: FAIL — `AssertionError: Default T_floor=52.0, expected 46.0`

- [ ] **Step 3: Fix the default**

In `Europa2D/src/monte_carlo_2d.py`, in the `MonteCarloResults2D` dataclass, change:

```python
# Before
T_floor: float = 52.0

# After
T_floor: float = 46.0  # Ashkenazy (2019) annual-mean polar floor at Q=0.05 W/m²
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_results_default_t_floor_matches_ashkenazy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py Europa2D/tests/test_monte_carlo_2d.py
git commit -m "fix: correct MonteCarloResults2D T_floor default from 52 to 46 K

Ashkenazy (2019) annual-mean polar floor is 46 K at Q=0.05 W/m².
The 52 K value was a stale reference from earlier design notes.
Closes P0.1 of the model improvement report."
```

---

### Task 2: Change DEFAULT_SCENARIO to uniform_transport

The current `DEFAULT_SCENARIO = "lemasquerier2023_polar"` in `literature_scenarios.py` makes polar-enhanced look like a neutral baseline. The global Howell (2021) parameters with uniform transport is the correct neutral baseline for the model hierarchy framing.

**Files:**
- Modify: `Europa2D/src/literature_scenarios.py` (~line where DEFAULT_SCENARIO is set)
- Test: `Europa2D/tests/test_literature_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
def test_default_scenario_is_uniform_transport():
    """Neutral baseline must be uniform_transport (global Howell 2021 params)."""
    from Europa2D.src.literature_scenarios import DEFAULT_SCENARIO
    assert DEFAULT_SCENARIO == "uniform_transport", (
        f"DEFAULT_SCENARIO={DEFAULT_SCENARIO!r}, expected 'uniform_transport'"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_literature_scenarios.py::test_default_scenario_is_uniform_transport -v`
Expected: FAIL — `DEFAULT_SCENARIO='lemasquerier2023_polar'`

- [ ] **Step 3: Change the default in ALL four locations**

This is a coordinated change across four files. All must be updated together.

**3a.** In `Europa2D/src/literature_scenarios.py`, change:

```python
# Before
DEFAULT_SCENARIO: ScenarioName = "lemasquerier2023_polar"

# After
DEFAULT_SCENARIO: ScenarioName = "uniform_transport"
```

**3b.** In `Europa2D/src/latitude_profile.py`, in the `LatitudeProfile` dataclass:

```python
# Before
ocean_pattern: str = "polar_enhanced"

# After
ocean_pattern: str = "uniform"
```

**3c.** In `Europa2D/src/latitude_sampler.py`, in `LatitudeParameterSampler.__init__`:

```python
# Before
ocean_pattern: OceanPattern = "polar_enhanced"

# After
ocean_pattern: OceanPattern = "uniform"
```

**3d.** In `Europa2D/src/monte_carlo_2d.py`, in `MonteCarloRunner2D.__init__`:

```python
# Before
ocean_pattern="polar_enhanced"

# After
ocean_pattern="uniform"
```

- [ ] **Step 4: Update tests that assumed the old default**

Known tests that will break:

1. `Europa2D/tests/test_monte_carlo_2d.py::test_runs_and_returns_results` — asserts `results.ocean_pattern == "polar_enhanced"`. Change to `"uniform"` and update the corresponding amplitude assertion.

2. `Europa2D/tests/test_literature_scenarios.py::test_default_scenario_is_conservative_lemasquerier_case` — remove this test (replaced by the new `test_default_scenario_is_uniform_transport`).

3. Any other test that constructs a `LatitudeProfile()` or `MonteCarloRunner2D()` without explicit `ocean_pattern` and then checks the pattern.

Search for all: `grep -rn "polar_enhanced\|lemasquerier2023_polar" Europa2D/tests/`

- [ ] **Step 5: Run full test suite to verify**

Run: `python -m pytest Europa2D/tests/ -v`
Expected: All pass after Step 4 updates.

- [ ] **Step 6: Commit**

```bash
git add Europa2D/src/literature_scenarios.py Europa2D/src/latitude_profile.py \
      Europa2D/src/latitude_sampler.py Europa2D/src/monte_carlo_2d.py Europa2D/tests/
git commit -m "fix: change default scenario to uniform_transport

The neutral baseline for the model hierarchy is global Howell (2021) params
with uniform transport. Updated defaults in literature_scenarios,
latitude_profile, latitude_sampler, and monte_carlo_2d. Updated tests
that assumed polar_enhanced default. Closes P0.2 of the model improvement
report."
```

---

### Task 3: Add D_cond, D_conv, and convective fraction aggregate statistics to MC results

Currently `D_cond_profiles`, `D_conv_profiles`, and `lid_fraction_profiles` are saved per-iteration but never aggregated into median/percentile bands like `H_total` is. The research note says "split all reporting of shell structure into: H_total, D_cond, D_conv, convective fraction." All four must be first-class reported quantities.

**Files:**
- Modify: `Europa2D/src/monte_carlo_2d.py` (MonteCarloResults2D dataclass + `run()` method + `save_results_2d()`)
- Test: `Europa2D/tests/test_monte_carlo_2d.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mc_results_have_d_cond_statistics():
    """MC results must include D_cond median and percentile bands."""
    from Europa2D.src.monte_carlo_2d import MonteCarloRunner2D

    runner = MonteCarloRunner2D(
        n_iterations=10,
        n_lat=5,
        nx=17,
        n_workers=1,
        seed=42,
        ocean_pattern="uniform",
    )
    results = runner.run()

    # D_cond aggregate statistics must exist and have correct shape
    assert results.D_cond_median is not None, "D_cond_median missing"
    assert results.D_cond_mean is not None, "D_cond_mean missing"
    assert results.D_cond_sigma_low is not None, "D_cond_sigma_low missing"
    assert results.D_cond_sigma_high is not None, "D_cond_sigma_high missing"
    assert results.D_cond_median.shape == (5,)
    # D_cond <= H_total at every latitude
    assert np.all(results.D_cond_median <= results.H_median + 0.01)

    # D_conv aggregate statistics
    assert results.D_conv_median is not None, "D_conv_median missing"
    assert results.D_conv_mean is not None, "D_conv_mean missing"
    assert results.D_conv_median.shape == (5,)
    # D_cond + D_conv ≈ H_total
    assert np.allclose(
        results.D_cond_median + results.D_conv_median,
        results.H_median, atol=1.0,
    )

    # Convective fraction aggregate statistics
    assert results.conv_fraction_median is not None, "conv_fraction_median missing"
    assert results.conv_fraction_mean is not None, "conv_fraction_mean missing"
    assert results.conv_fraction_median.shape == (5,)
    assert np.all(results.conv_fraction_median >= 0.0)
    assert np.all(results.conv_fraction_median <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_mc_results_have_d_cond_statistics -v`
Expected: FAIL — `AttributeError: 'MonteCarloResults2D' object has no attribute 'D_cond_median'`

- [ ] **Step 3: Add D_cond fields to MonteCarloResults2D**

In `Europa2D/src/monte_carlo_2d.py`, add to the `MonteCarloResults2D` dataclass:

```python
# D_cond aggregate statistics (n_lat,) — same shape as H_median
D_cond_median: Optional[np.ndarray] = None
D_cond_mean: Optional[np.ndarray] = None
D_cond_sigma_low: Optional[np.ndarray] = None   # 15.87th percentile
D_cond_sigma_high: Optional[np.ndarray] = None   # 84.13th percentile

# D_conv aggregate statistics (n_lat,)
D_conv_median: Optional[np.ndarray] = None
D_conv_mean: Optional[np.ndarray] = None
D_conv_sigma_low: Optional[np.ndarray] = None
D_conv_sigma_high: Optional[np.ndarray] = None

# Convective fraction aggregate statistics (n_lat,)
# Fraction of shell thickness occupied by the convective sublayer: D_conv / H_total
conv_fraction_median: Optional[np.ndarray] = None
conv_fraction_mean: Optional[np.ndarray] = None
conv_fraction_sigma_low: Optional[np.ndarray] = None
conv_fraction_sigma_high: Optional[np.ndarray] = None
```

- [ ] **Step 4: Compute D_cond statistics in the runner**

In the `run()` method, after stacking `D_cond_profiles`, add:

```python
if D_cond_stack.size > 0:
    D_cond_median = np.median(D_cond_stack, axis=0)
    D_cond_mean = np.mean(D_cond_stack, axis=0)
    D_cond_sigma_low = np.percentile(D_cond_stack, 15.87, axis=0)
    D_cond_sigma_high = np.percentile(D_cond_stack, 84.13, axis=0)

if D_conv_stack.size > 0:
    D_conv_median = np.median(D_conv_stack, axis=0)
    D_conv_mean = np.mean(D_conv_stack, axis=0)
    D_conv_sigma_low = np.percentile(D_conv_stack, 15.87, axis=0)
    D_conv_sigma_high = np.percentile(D_conv_stack, 84.13, axis=0)

# Convective fraction: D_conv / H_total per sample, then aggregate
conv_fraction_stack = np.where(
    H_stack > 0, D_conv_stack / H_stack, 0.0
)
conv_fraction_median = np.median(conv_fraction_stack, axis=0)
conv_fraction_mean = np.mean(conv_fraction_stack, axis=0)
conv_fraction_sigma_low = np.percentile(conv_fraction_stack, 15.87, axis=0)
conv_fraction_sigma_high = np.percentile(conv_fraction_stack, 84.13, axis=0)
```

Pass all twelve arrays to the `MonteCarloResults2D` constructor.

- [ ] **Step 5: Save D_cond statistics in save_results_2d()**

Add all twelve new aggregate arrays to the NPZ save dict (same pattern as H_median/H_mean): D_cond_{median,mean,sigma_low,sigma_high}, D_conv_{...}, conv_fraction_{...}.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_mc_results_have_d_cond_statistics -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py Europa2D/tests/test_monte_carlo_2d.py
git commit -m "feat: add D_cond aggregate statistics to MC results

D_cond, D_conv, and convective fraction now reported with median,
mean, and ±1σ percentile bands alongside H_total. All four shell-
structure quantities are first-class. Closes P0.3 of the model
improvement report."
```

---

### Task 4: Add latitude-band mean distributions to MC results

The research note says "report latitude-band means instead of the single 90° node." The `profile_diagnostics.py` module already has `band_mean_samples()` but it's not called from the MC runner.

**Files:**
- Modify: `Europa2D/src/monte_carlo_2d.py` (MonteCarloResults2D dataclass + `run()` method + `save_results_2d()`)
- Read: `Europa2D/src/profile_diagnostics.py` (existing `band_mean_samples()` function)
- Test: `Europa2D/tests/test_monte_carlo_2d.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mc_results_have_band_means():
    """MC results must include area-weighted band-mean distributions."""
    from Europa2D.src.monte_carlo_2d import MonteCarloRunner2D

    runner = MonteCarloRunner2D(
        n_iterations=10,
        n_lat=19,
        nx=17,
        n_workers=1,
        seed=42,
        ocean_pattern="uniform",
    )
    results = runner.run()

    # Band-mean distributions: one value per valid MC sample
    assert results.H_low_band is not None, "H_low_band missing"
    assert results.H_high_band is not None, "H_high_band missing"
    assert results.D_cond_low_band is not None, "D_cond_low_band missing"
    assert results.D_cond_high_band is not None, "D_cond_high_band missing"
    assert results.H_low_band.shape == (results.n_valid,)
    assert results.H_high_band.shape == (results.n_valid,)
    # Low-latitude band mean should be finite and positive
    assert np.all(np.isfinite(results.H_low_band))
    assert np.all(results.H_low_band > 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_mc_results_have_band_means -v`
Expected: FAIL — `AttributeError: ... has no attribute 'H_low_band'`

- [ ] **Step 3: Add band-mean fields to MonteCarloResults2D**

```python
# Latitude-band distributions: (n_valid,) — one value per MC sample
# Low band = area-weighted mean over 0-10° latitude
# High band = area-weighted mean over 80-90° latitude
H_low_band: Optional[np.ndarray] = None
H_high_band: Optional[np.ndarray] = None
D_cond_low_band: Optional[np.ndarray] = None
D_cond_high_band: Optional[np.ndarray] = None
```

- [ ] **Step 4: Compute band means in the runner**

In the `run()` method, after stacking profiles, import and call `band_mean_samples`:

```python
from Europa2D.src.profile_diagnostics import band_mean_samples, LOW_LAT_BAND, HIGH_LAT_BAND

H_low_band = band_mean_samples(latitudes_deg, H_stack, LOW_LAT_BAND)
H_high_band = band_mean_samples(latitudes_deg, H_stack, HIGH_LAT_BAND)
D_cond_low_band = band_mean_samples(latitudes_deg, D_cond_stack, LOW_LAT_BAND)
D_cond_high_band = band_mean_samples(latitudes_deg, D_cond_stack, HIGH_LAT_BAND)
```

Pass these to the `MonteCarloResults2D` constructor.

- [ ] **Step 5: Save band means in save_results_2d()**

Add `H_low_band`, `H_high_band`, `D_cond_low_band`, `D_cond_high_band` to the NPZ save dict.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest Europa2D/tests/test_monte_carlo_2d.py::test_mc_results_have_band_means -v`
Expected: PASS

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest Europa2D/tests/ -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add Europa2D/src/monte_carlo_2d.py Europa2D/tests/test_monte_carlo_2d.py
git commit -m "feat: add latitude-band mean distributions to MC results

Area-weighted band means over 0-10° and 80-90° for both H_total and
D_cond. These replace single-node pole readings as the primary
high-latitude diagnostic. Uses existing band_mean_samples() from
profile_diagnostics. Closes P0.4 of the model improvement report."
```

---

### Task 5: Reconcile stale doc/comment references

Several comments and design notes still reference `T_eq=110 K`, `T_floor=52 K`, or other outdated defaults. These must be reconciled before the thesis to prevent reviewer confusion.

**Files:**
- Modify: `Europa2D/src/latitude_profile.py` (verify comments match defaults)
- Modify: `Europa2D/src/latitude_sampler.py` (verify comments match defaults)
- Modify: `Europa2D/src/monte_carlo_2d.py` (verify after Task 1 fix)
- Search: Any `.md` design notes in `Europa2D/docs/` for stale references

- [ ] **Step 1: Search for stale references**

Run:
```bash
grep -rn "110\|52.*K\|T_floor.*52\|T_eq.*110" Europa2D/src/ Europa2D/docs/ --include="*.py" --include="*.md"
```

- [ ] **Step 2: For each stale reference, update comment or value**

For comments that say `52 K` when the code uses `46 K`: update the comment to `46 K` with `(Ashkenazy 2019)` citation.

For design notes that use `110/52`: add a note that these are legacy values superseded by `96/46 K`.

Do NOT change physics defaults — only comments and documentation.

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `python -m pytest Europa2D/tests/ -v`
Expected: All pass (comment-only changes)

- [ ] **Step 4: Commit**

```bash
git add Europa2D/
git commit -m "docs: reconcile stale T_eq/T_floor references to 96/46 K

All comments and design notes now consistently reference 96 K equatorial
and 46 K polar floor (Ashkenazy 2019). Legacy 110/52 K values are
explicitly marked as superseded. Closes P0.1 of the model improvement report."
```

---

## Phase B — Physics Upgrades (P1)

Phase B depends on Phase A being complete (clean defaults before adding new physics). Tasks 6 and 7 are independent of each other.

---

### Task 6: Add tidal-heating pattern families

Currently `tidal_strain(phi)` always uses a monotonic `sin²(phi)` law (polar-enhanced). Beuthe (2013) shows that tidal-heating patterns depend on dissipation geometry and can peak at the equator or mid-latitudes.

Add a `tidal_pattern` field to `LatitudeProfile` with three families:
- `mantle_core` (current default): `q_tidal ∝ 1 + c·sin²(φ)` — poles hotter
- `shell_dominated`: `q_tidal ∝ 1 + c·cos²(φ)` — equator hotter
- `non_monotonic`: base profile × `(1 + A·sin²(2φ))` — mid-latitude amplification over the mantle_core base

**Parameter semantics:** `epsilon_eq` and `epsilon_pole` always mean the strain at the equator and pole respectively, regardless of pattern. For `mantle_core`, set `epsilon_pole > epsilon_eq`. For `shell_dominated`, set `epsilon_eq > epsilon_pole`. For `non_monotonic`, the mid-latitude amplification is controlled by a new `mid_latitude_amplification` field (default 0.3, i.e. 30% boost at 45°).

**Files:**
- Modify: `Europa2D/src/latitude_profile.py` (LatitudeProfile dataclass + `tidal_strain()`)
- Modify: `Europa2D/src/latitude_sampler.py` (sample `tidal_pattern` for MC)
- Test: `Europa2D/tests/test_latitude_profile.py`
- Test: `Europa2D/tests/test_latitude_sampler.py`

- [ ] **Step 1: Write failing tests for endpoint values**

```python
def test_tidal_strain_mantle_core_unchanged():
    """mantle_core pattern reproduces current behavior: poles > equator."""
    profile = LatitudeProfile(
        epsilon_eq=6e-6, epsilon_pole=1.2e-5,
        tidal_pattern="mantle_core",
    )
    assert profile.tidal_strain(0.0) == pytest.approx(6e-6)
    assert profile.tidal_strain(np.pi / 2) == pytest.approx(1.2e-5)


def test_tidal_strain_shell_dominated_equator_peak():
    """shell_dominated pattern: equator > poles."""
    profile = LatitudeProfile(
        epsilon_eq=1.2e-5, epsilon_pole=6e-6,
        tidal_pattern="shell_dominated",
    )
    eq = profile.tidal_strain(0.0)
    pole = profile.tidal_strain(np.pi / 2)
    assert eq > pole, "Shell-dominated should peak at equator"
    assert eq == pytest.approx(1.2e-5)
    assert pole == pytest.approx(6e-6)


def test_tidal_strain_non_monotonic_mid_latitude_peak():
    """non_monotonic pattern: mid-latitude peak exceeds both endpoints."""
    profile = LatitudeProfile(
        epsilon_eq=6e-6, epsilon_pole=1.2e-5,
        tidal_pattern="non_monotonic",
        mid_latitude_amplification=0.5,
    )
    eq = profile.tidal_strain(0.0)
    mid = profile.tidal_strain(np.pi / 4)  # 45 degrees
    pole = profile.tidal_strain(np.pi / 2)
    # Endpoints match mantle_core profile
    assert eq == pytest.approx(6e-6)
    assert pole == pytest.approx(1.2e-5)
    # Mid-latitude is amplified above the mantle_core base at 45°
    base_at_45 = 6e-6 * np.sqrt(1 + 3.0 * 0.5)  # c=3 for 2:1 ratio
    assert mid == pytest.approx(base_at_45 * 1.5)  # 50% amplification
    assert mid > eq, "Mid-latitude should exceed equator"
    assert mid > pole, "Mid-latitude should exceed pole"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Europa2D/tests/test_latitude_profile.py -k "tidal_strain_mantle_core or tidal_strain_shell or tidal_strain_non_monotonic" -v`
Expected: FAIL — unknown field `tidal_pattern`

- [ ] **Step 3: Add tidal_pattern field to LatitudeProfile**

In `Europa2D/src/latitude_profile.py`, add to the dataclass:

```python
tidal_pattern: str = "mantle_core"  # "mantle_core" | "shell_dominated" | "non_monotonic"
mid_latitude_amplification: float = 0.3  # used only for non_monotonic pattern
```

Add validation in `__post_init__`:

```python
_TIDAL_PATTERNS = {"mantle_core", "shell_dominated", "non_monotonic"}
if self.tidal_pattern not in _TIDAL_PATTERNS:
    raise ValueError(
        f"Unknown tidal_pattern={self.tidal_pattern!r}, "
        f"must be one of {_TIDAL_PATTERNS}"
    )
```

- [ ] **Step 4: Generalize tidal_strain()**

Replace the body of `tidal_strain()` with pattern dispatch:

```python
def tidal_strain(self, phi: FloatOrArray) -> FloatOrArray:
    phi = np.asarray(phi, dtype=float)
    if self.tidal_pattern == "mantle_core":
        # Current: monotonic increase toward poles (Beuthe 2013 whole-shell)
        c = (self.epsilon_pole / self.epsilon_eq) ** 2 - 1
        return self.epsilon_eq * np.sqrt(1 + c * np.sin(phi) ** 2)
    elif self.tidal_pattern == "shell_dominated":
        # Monotonic increase toward equator (Beuthe 2013 thin-shell/membrane)
        c = (self.epsilon_eq / self.epsilon_pole) ** 2 - 1
        return self.epsilon_pole * np.sqrt(1 + c * np.cos(phi) ** 2)
    elif self.tidal_pattern == "non_monotonic":
        # Mantle_core base profile with degree-4 mid-latitude amplification
        # Base: same as mantle_core (monotonic eq→pole)
        c = (self.epsilon_pole / self.epsilon_eq) ** 2 - 1
        base = self.epsilon_eq * np.sqrt(1 + c * np.sin(phi) ** 2)
        # Amplify at mid-latitudes: sin²(2φ) peaks at 45°, zero at 0° and 90°
        return base * (1 + self.mid_latitude_amplification * np.sin(2 * phi) ** 2)
    raise ValueError(f"Unknown tidal_pattern: {self.tidal_pattern!r}")
```

**Parameter semantics:** `epsilon_eq` and `epsilon_pole` are always the literal strain values at 0° and 90° respectively. For `shell_dominated`, the formula uses `cos²(φ)` so that the equatorial value (`epsilon_eq`) is naturally the peak when `epsilon_eq > epsilon_pole`. For `non_monotonic`, the base profile matches `mantle_core` exactly, and `mid_latitude_amplification` adds a percentage boost at 45° (e.g. 0.3 = 30% boost).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest Europa2D/tests/test_latitude_profile.py -k "tidal_strain" -v`
Expected: All PASS (including existing tests — `mantle_core` is the default, backward-compatible)

- [ ] **Step 6: Add tidal_pattern to sampler**

In `Europa2D/src/latitude_sampler.py`, add `tidal_pattern` to `LATITUDE_STRUCTURE_KEYS`. In `sample()`, pass through the constructor parameter or default to `"mantle_core"`:

```python
# tidal_pattern is held fixed per MC campaign, not sampled
tidal_pattern = self._tidal_pattern  # set in constructor, default "mantle_core"
```

Add a `tidal_pattern` constructor parameter to `LatitudeParameterSampler`.

- [ ] **Step 7: Write and run sampler test**

```python
def test_sampler_passes_tidal_pattern():
    """Sampler propagates tidal_pattern to LatitudeProfile."""
    sampler = LatitudeParameterSampler(seed=42, tidal_pattern="shell_dominated")
    _, profile = sampler.sample()
    assert profile.tidal_pattern == "shell_dominated"
```

Run: `python -m pytest Europa2D/tests/test_latitude_sampler.py::test_sampler_passes_tidal_pattern -v`
Expected: PASS

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest Europa2D/tests/ -v`
Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add Europa2D/src/latitude_profile.py Europa2D/src/latitude_sampler.py Europa2D/tests/
git commit -m "feat: add tidal-heating pattern families (Beuthe 2013)

Three tidal dissipation patterns: mantle_core (poles, default),
shell_dominated (equator), non_monotonic (mid-latitude degree-4).
The existing behavior is preserved under the mantle_core default.
Closes P1.1 of the model improvement report."
```

---

### Task 7: Add surface-boundary temperature presets

The research note identifies code/doc drift around `T_eq` and `T_floor` values and recommends named presets for sensitivity analysis. Add presets corresponding to different internal-heating assumptions.

**Files:**
- Modify: `Europa2D/src/literature_scenarios.py` (add surface boundary presets)
- Test: `Europa2D/tests/test_literature_scenarios.py`

- [ ] **Step 1: Write failing tests for presets**

```python
def test_surface_presets_exist():
    """Named surface temperature presets for sensitivity analysis."""
    from Europa2D.src.literature_scenarios import SURFACE_PRESETS

    assert "ashkenazy_low_q" in SURFACE_PRESETS
    assert "ashkenazy_high_q" in SURFACE_PRESETS
    assert "legacy_110_52" in SURFACE_PRESETS

    low = SURFACE_PRESETS["ashkenazy_low_q"]
    assert low.T_eq == 96.0
    assert low.T_floor == 46.0

    high = SURFACE_PRESETS["ashkenazy_high_q"]
    assert high.T_eq == 96.0
    assert high.T_floor == 53.0

    legacy = SURFACE_PRESETS["legacy_110_52"]
    assert legacy.T_eq == 110.0
    assert legacy.T_floor == 52.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_literature_scenarios.py::test_surface_presets_exist -v`
Expected: FAIL — `ImportError` or `KeyError`

- [ ] **Step 3: Add SURFACE_PRESETS dict**

In `Europa2D/src/literature_scenarios.py`, add:

```python
@dataclass(frozen=True)
class SurfacePreset:
    """Named surface temperature boundary condition."""
    name: str
    T_eq: float
    T_floor: float
    citation: str
    description: str


SURFACE_PRESETS: Final[dict[str, SurfacePreset]] = {
    "ashkenazy_low_q": SurfacePreset(
        name="ashkenazy_low_q",
        T_eq=96.0,
        T_floor=46.0,
        citation="Ashkenazy (2019)",
        description="Annual-mean at Q=0.05 W/m². Default for MC runs.",
    ),
    "ashkenazy_high_q": SurfacePreset(
        name="ashkenazy_high_q",
        T_eq=96.0,
        T_floor=53.0,
        citation="Ashkenazy (2019)",
        description="Annual-mean at Q=0.2 W/m². Higher internal heating raises polar floor.",
    ),
    "legacy_110_52": SurfacePreset(
        name="legacy_110_52",
        T_eq=110.0,
        T_floor=52.0,
        citation="pre-Ashkenazy estimate",
        description="Legacy values from early design notes. Use only for sensitivity comparison.",
    ),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Europa2D/tests/test_literature_scenarios.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/literature_scenarios.py Europa2D/tests/test_literature_scenarios.py
git commit -m "feat: add surface temperature presets for sensitivity analysis

Three named presets (ashkenazy_low_q, ashkenazy_high_q, legacy_110_52)
with citations. The legacy preset is explicitly marked as superseded.
Closes P1.3 of the model improvement report."
```

---

### Task 8: Add mixed-heating diagnostic (stretch goal)

Deschamps & Vilella (2021) show that the ratio of internal (shell tidal) to basal (ocean) heating changes convection. Add a diagnostic that flags when a column is near the mixed-heating critical regime.

**Note:** This task is a stretch goal. If the thesis timeline is tight, defer it. The diagnostic is informational — it does not change any physics.

**Files:**
- Modify: `Europa2D/src/profile_diagnostics.py` (add `mixed_heating_ratio()`)
- Test: `Europa2D/tests/test_profile_diagnostics.py`

- [ ] **Step 1: Write failing test**

```python
def test_mixed_heating_ratio():
    """Compute internal-to-total heating fraction per column."""
    from Europa2D.src.profile_diagnostics import mixed_heating_ratio

    # Pure basal heating: ratio = 0
    assert mixed_heating_ratio(q_tidal_internal=0.0, q_ocean=0.02) == pytest.approx(0.0)
    # Equal heating: ratio = 0.5
    assert mixed_heating_ratio(q_tidal_internal=0.02, q_ocean=0.02) == pytest.approx(0.5)
    # Pure internal: ratio = 1
    assert mixed_heating_ratio(q_tidal_internal=0.02, q_ocean=0.0) == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Europa2D/tests/test_profile_diagnostics.py::test_mixed_heating_ratio -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement mixed_heating_ratio**

In `Europa2D/src/profile_diagnostics.py`, add:

```python
def mixed_heating_ratio(q_tidal_internal: float, q_ocean: float) -> float:
    """Fraction of total column heating from internal (shell tidal) sources.

    Deschamps & Vilella (2021): when this ratio exceeds ~0.5-0.7, the
    convective regime changes qualitatively and bottom heat flux can
    reverse sign.

    Parameters
    ----------
    q_tidal_internal : Tidal heat flux generated within the convective sublayer (W/m²)
    q_ocean : Ocean heat flux at the shell base (W/m²)

    Returns
    -------
    Fraction in [0, 1]. Returns 0 if both inputs are zero.
    """
    total = q_tidal_internal + q_ocean
    if total <= 0:
        return 0.0
    return q_tidal_internal / total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest Europa2D/tests/test_profile_diagnostics.py::test_mixed_heating_ratio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Europa2D/src/profile_diagnostics.py Europa2D/tests/test_profile_diagnostics.py
git commit -m "feat: add mixed-heating ratio diagnostic (Deschamps & Vilella 2021)

Computes internal-to-total heating fraction per column. Flags proximity
to the critical regime where bottom heat flux can reverse sign.
Informational diagnostic only — does not change physics.
Closes P1.2 of the model improvement report."
```

---

## Phase C — Validation Experiments

Phase C depends on Phase A being complete. These are run-and-verify tasks, not code changes.

---

### Task 9: Single-column 2D validation against 1D solver

The research note calls for a strict apples-to-apples check: run the 2D solver with `n_lat=1`, `uniform` forcing, and confirm it matches the 1D solver output.

**Files:**
- Modify: `Europa2D/tests/test_validation.py` (add explicit single-column regression test)

- [ ] **Step 1: Write the validation test**

```python
def test_single_column_2d_matches_1d():
    """2D solver with n_lat=1, uniform forcing must match 1D solver."""
    from Europa2D.src.axial_solver import AxialSolver2D
    from Europa2D.src.latitude_profile import LatitudeProfile

    profile = LatitudeProfile(
        T_eq=96.0, T_floor=46.0,
        epsilon_eq=6e-6, epsilon_pole=6e-6,
        q_ocean_mean=0.02,
        ocean_pattern="uniform",
    )
    solver = AxialSolver2D(
        n_lat=1,
        nx=31,
        dt=5e12,
        latitude_profile=profile,
        use_convection=True,
    )
    result = solver.run_to_equilibrium(threshold=1e-12, max_steps=500)
    H_2d = result["H_profile_km"][0]

    # Compare against 1D solver with same parameters
    # (existing test_validation.py should already have this reference)
    # Tolerance: 1% relative or 0.5 km absolute
    assert H_2d > 5.0, "Unphysical: shell too thin"
    assert H_2d < 80.0, "Unphysical: shell too thick"
    # The exact match value depends on the 1D solver output.
    # Store as a regression anchor once confirmed.
```

- [ ] **Step 2: Add the warm-variant test**

The research note asks for a second variant with the warmer legacy `T_eq/T_floor` pair:

```python
def test_single_column_2d_warm_variant():
    """Repeat with legacy 110/52 K surface BCs for sensitivity comparison."""
    profile = LatitudeProfile(
        T_eq=110.0, T_floor=52.0,
        epsilon_eq=6e-6, epsilon_pole=6e-6,
        q_ocean_mean=0.02,
        ocean_pattern="uniform",
    )
    solver = AxialSolver2D(
        n_lat=1, nx=31, dt=5e12,
        latitude_profile=profile,
        use_convection=True,
    )
    result = solver.run_to_equilibrium(threshold=1e-12, max_steps=500)
    H_warm = result["H_profile_km"][0]
    assert H_warm > 5.0
    assert H_warm < 80.0
    # Warmer surface → thinner conductive lid (lower T gradient)
    # Record value for comparison against cold variant
```

- [ ] **Step 3: Run both tests**

Run: `python -m pytest Europa2D/tests/test_validation.py -k "single_column" -v`
Expected: Both PASS — record exact H values for thesis Table.

- [ ] **Step 4: Commit**

```bash
git add Europa2D/tests/test_validation.py
git commit -m "test: add single-column 2D vs 1D validation check

Verifies that AxialSolver2D with n_lat=1 and uniform forcing
reproduces the 1D solver output. Closes validation experiment 1
from the model improvement report."
```

---

### Task 10: Three-scenario benchmark suite

Run the three core literature scenarios and record the band diagnostics specified in the research note. This is a diagnostic run, not a large MC campaign.

**Files:**
- Modify: `Europa2D/scripts/run_2d_single.py` (add benchmark mode)

- [ ] **Step 1: Add benchmark function to run_2d_single.py**

```python
def run_benchmark_suite():
    """Run three-scenario benchmark and print band diagnostics."""
    from Europa2D.src.literature_scenarios import SCENARIOS
    from Europa2D.src.profile_diagnostics import compute_profile_diagnostics

    scenarios = ["uniform_transport", "soderlund2014_equator", "lemasquerier2023_polar"]
    for name in scenarios:
        scenario = SCENARIOS[name]
        # Build profile with fixed reference params
        profile = scenario.build_profile(
            T_eq=96.0, epsilon_eq=6e-6, epsilon_pole=1.2e-5,
            q_ocean_mean=0.02, T_floor=46.0,
        )
        solver = AxialSolver2D(n_lat=37, nx=31, dt=5e12, latitude_profile=profile)
        result = solver.run_to_equilibrium()
        diag = compute_profile_diagnostics(
            result["latitudes_deg"], result["H_profile_km"], profile,
        )
        print(f"\n=== {name} ===")
        print(f"  H_low (0-10°):   {diag.low_band_mean_km:.2f} km")
        print(f"  H_high (80-90°): {diag.high_band_mean_km:.2f} km")
        print(f"  ΔH:              {diag.high_minus_low_km:.2f} km")
        print(f"  Min H:           {diag.min_thickness_km:.2f} km at {diag.min_latitude_deg:.1f}°")
```

- [ ] **Step 2: Run the benchmark**

Run: `python -c "from Europa2D.scripts.run_2d_single import run_benchmark_suite; run_benchmark_suite()"`

Record the output for the thesis and verify that:
- `uniform_transport` shows near-zero ΔH
- `soderlund2014_equator` shows equatorial thinning
- `lemasquerier2023_polar` shows polar thinning

- [ ] **Step 3: Commit**

```bash
git add Europa2D/scripts/run_2d_single.py
git commit -m "feat: add three-scenario benchmark suite with band diagnostics

Runs uniform_transport, soderlund2014_equator, and lemasquerier2023_polar
with fixed reference parameters and reports H_low, H_high, ΔH, and
min-thickness latitude. Closes validation experiment 2 from the model
improvement report."
```

---

## Dependency Graph

```
Phase A (P0):
  Task 1: Fix T_floor default          ──┐
  Task 2: Change DEFAULT_SCENARIO        │── can run in parallel
  Task 3: D_cond/D_conv/conv_frac stats  │
  Task 4: Band-mean distributions       ──┘
  Task 5: Reconcile stale references    ── must run AFTER Task 1

Phase B (P1) — depends on Phase A complete:
  Task 6: Tidal-heating patterns ──┐
  Task 7: Surface-boundary presets ├── independent of each other
  Task 8: Mixed-heating diagnostic ┘   (stretch goal)

Phase C (Validation) — depends on Phase A complete:
  Task 9:  Single-column validation (cold + warm variants)
  Task 10: Three-scenario benchmark ── depends on Task 9 passing
```

## Out of Scope

These items from the research note are explicitly deferred:

- **P1.4:** Improved 1D equatorial proxy transport diagnostic (different subsystem, covered by equatorial-proxy-suite plan)
- **P1.5:** Regime-split D_cond plots (plotting task, not model improvement)
- **P2:** Reduced-order ocean closure, geometry-sensitive forcing, transient/orbital branch, topography evolution (next-generation work, post-thesis)
- **Sobol sensitivity analysis** (separate plan already exists)
