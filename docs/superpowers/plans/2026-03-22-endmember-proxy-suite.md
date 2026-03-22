# 1D Equator/Pole Endmember Proxy Suite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone 1D MC suite that runs equatorial and polar endmember proxies under three ocean transport scenarios (uniform, Soderlund, Lemasquerier), producing 6 result files for publication analysis.

**Architecture:** A single `AuditedEndmemberSampler` class inherits from `AuditedShellSampler` and overrides T_surf, epsilon_0, and P_tidal via a `q_tidal_multiplier`. Six named subclasses (one per endmember × scenario) provide Windows-picklable samplers. A driver script runs all 6 ensembles sequentially.

**Tech Stack:** Python, NumPy, pytest, existing `MonteCarloRunner` / `AuditedShellSampler` infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-22-paired-1d-band-suite-design.md`

---

### Task 1: AuditedEndmemberSampler — tests

**Files:**
- Create: `EuropaProjectDJ/tests/test_endmember_sampler.py`

- [ ] **Step 1: Write tests for the endmember sampler**

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from audited_endmember_sampler import AuditedEndmemberSampler
from constants import Planetary


# Endmember presets (from spec)
EQ_PRESET = dict(
    T_surf_mean=110.0, T_surf_std=5.0, T_surf_clip=(95.0, 120.0),
    epsilon_0_log_center=np.log10(6e-6), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 2e-5),
)
POLE_PRESET = dict(
    T_surf_mean=50.0, T_surf_std=5.0, T_surf_clip=(45.0, 80.0),
    epsilon_0_log_center=np.log10(1.2e-5), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 3.4e-5),
)


def _reconstruct_q_tidal(params):
    """Recover q_tidal from P_tidal for test assertions."""
    return params['P_tidal'] / Planetary.AREA


class TestEndmemberOverrides:
    def test_eq_T_surf_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(95.0 <= t <= 120.0 for t in t_surfs)
        assert 105.0 < np.mean(t_surfs) < 115.0

    def test_pole_T_surf_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(45.0 <= t <= 80.0 for t in t_surfs)
        assert 45.0 < np.mean(t_surfs) < 58.0

    def test_eq_epsilon_0_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(2e-6 <= e <= 2e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(4e-6) < log_mean < np.log10(1e-5)

    def test_pole_epsilon_0_in_range(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(2e-6 <= e <= 3.4e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(8e-6) < log_mean < np.log10(2e-5)

    def test_d_grain_not_overridden(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        grains = [sampler.sample()['d_grain'] for _ in range(100)]
        assert all(5e-5 <= d <= 3e-3 for d in grains)


class TestQtidalMultiplier:
    def test_multiplier_1_no_change(self):
        from audited_sampler import AuditedShellSampler
        base = AuditedShellSampler(seed=42)
        endm = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        p_base = base.sample()
        p_endm = endm.sample()
        # P_tidal should be identical (multiplier=1.0 doesn't change it)
        np.testing.assert_allclose(p_endm['P_tidal'], p_base['P_tidal'], rtol=1e-10)

    def test_multiplier_scales_P_tidal(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.15, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        np.testing.assert_allclose(p2['P_tidal'], 1.15 * p1['P_tidal'], rtol=1e-10)

    def test_multiplier_less_than_one(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **POLE_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=0.70, **POLE_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        np.testing.assert_allclose(p2['P_tidal'], 0.70 * p1['P_tidal'], rtol=1e-10)

    def test_radiogenic_unchanged_by_multiplier(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.30, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        # H_rad and D_H2O are drawn before overrides, should be identical
        assert p1['H_rad'] == p2['H_rad']
        assert p1['D_H2O'] == p2['D_H2O']

    def test_multiplier_stored_in_params(self):
        sampler = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=0.85, **EQ_PRESET)
        params = sampler.sample()
        assert params.get('q_tidal_multiplier') == pytest.approx(0.85)


class TestReproducibility:
    def test_same_seed_same_output(self):
        s1 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        s2 = AuditedEndmemberSampler(seed=42, q_tidal_multiplier=1.0, **EQ_PRESET)
        p1 = s1.sample()
        p2 = s2.sample()
        assert p1['T_surf'] == p2['T_surf']
        assert p1['epsilon_0'] == p2['epsilon_0']
        assert p1['P_tidal'] == p2['P_tidal']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd EuropaProjectDJ && python -m pytest tests/test_endmember_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audited_endmember_sampler'`

- [ ] **Step 3: Commit failing tests**

```bash
git add EuropaProjectDJ/tests/test_endmember_sampler.py
git commit -m "test: add failing tests for AuditedEndmemberSampler"
```

---

### Task 2: AuditedEndmemberSampler — implementation

**Files:**
- Create: `EuropaProjectDJ/src/audited_endmember_sampler.py`

- [ ] **Step 1: Write the sampler**

```python
"""
Endmember proxy sampler for equator/pole 1D MC runs.

Inherits all audited global priors from AuditedShellSampler.
Overrides only endmember-specific terms:
  T_surf:    configurable normal distribution per endmember
  epsilon_0: configurable lognormal distribution per endmember
  P_tidal:   scaled by q_tidal_multiplier (local flux adjustment)

Does NOT override d_grain — grain size is sampled once per realization
from the audited global prior (PARAMETER_PRIOR_AUDIT_2026.md).

Multiplier values (derived from 2D ocean_heat_flux shape functions at
q_star=0.45):
  Uniform:            eq=1.00, pole=1.00
  Soderlund 2014:     eq=1.15, pole=0.70
  Lemasquerier 2023:  eq=0.85, pole=1.30
"""
import numpy as np
from typing import Dict, Tuple

from audited_sampler import AuditedShellSampler


class AuditedEndmemberSampler(AuditedShellSampler):
    """
    Endmember proxy sampler with configurable T_surf, epsilon_0, and
    local tidal flux multiplier.

    Scales only the tidal component of q_basal (P_tidal), not the
    radiogenic component, to avoid implicitly enhancing spatially
    uniform radiogenic heating.
    """

    def __init__(
        self,
        T_surf_mean: float,
        T_surf_std: float,
        T_surf_clip: Tuple[float, float],
        epsilon_0_log_center: float,
        epsilon_0_log_sigma: float,
        epsilon_0_clip: Tuple[float, float],
        q_tidal_multiplier: float = 1.0,
        seed=None,
    ):
        super().__init__(seed=seed)
        self._T_surf_mean = T_surf_mean
        self._T_surf_std = T_surf_std
        self._T_surf_clip = T_surf_clip
        self._eps_log_center = epsilon_0_log_center
        self._eps_log_sigma = epsilon_0_log_sigma
        self._eps_clip = epsilon_0_clip
        self._q_tidal_multiplier = q_tidal_multiplier

    def sample(self) -> Dict[str, float]:
        params = super().sample()

        # Override T_surf with endmember distribution
        while True:
            t = self.rng.normal(self._T_surf_mean, self._T_surf_std)
            if self._T_surf_clip[0] <= t <= self._T_surf_clip[1]:
                break
        params['T_surf'] = t

        # Override epsilon_0 with endmember distribution
        while True:
            eps = 10 ** self.rng.normal(self._eps_log_center, self._eps_log_sigma)
            if self._eps_clip[0] <= eps <= self._eps_clip[1]:
                break
        params['epsilon_0'] = eps

        # Scale tidal component of q_basal
        params['P_tidal'] = self._q_tidal_multiplier * params['P_tidal']

        # Store diagnostic
        params['q_tidal_multiplier'] = self._q_tidal_multiplier

        return params
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd EuropaProjectDJ && python -m pytest tests/test_endmember_sampler.py -v`
Expected: All 12 tests PASS

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/src/audited_endmember_sampler.py
git commit -m "feat: add AuditedEndmemberSampler with configurable T_surf, epsilon_0, q_tidal_multiplier"
```

---

### Task 3: Named subclasses — tests

**Files:**
- Modify: `EuropaProjectDJ/tests/test_endmember_sampler.py`

- [ ] **Step 1: Add pickling and subclass tests**

Append to `test_endmember_sampler.py`:

```python
import pickle


class TestNamedSubclasses:
    """Verify each named subclass is picklable and produces correct overrides."""

    SUBCLASS_CONFIGS = [
        ("UniformEqSampler",        110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   1.00),
        ("UniformPoleSampler",       50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 1.00),
        ("SoderlundEqSampler",      110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   1.15),
        ("SoderlundPoleSampler",     50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 0.70),
        ("LemasquerierEqSampler",   110.0, (95.0, 120.0), 6e-6,  (2e-6, 2e-5),   0.85),
        ("LemasquerierPoleSampler",  50.0, (45.0, 80.0),  1.2e-5,(2e-6, 3.4e-5), 1.30),
    ]

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_is_picklable(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        roundtripped = pickle.loads(pickle.dumps(sampler))
        p1 = sampler.sample()
        # Re-create (pickle roundtrip resets RNG state, so just verify it works)
        p2 = cls(seed=42).sample()
        assert p1['T_surf'] == p2['T_surf']

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_T_surf_in_range(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(200)]
        assert all(t_clip[0] <= t <= t_clip[1] for t in t_surfs)

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_multiplier_correct(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=42)
        params = sampler.sample()
        assert params['q_tidal_multiplier'] == pytest.approx(mult)

    @pytest.mark.parametrize("cls_name,t_mean,t_clip,eps_center,eps_clip,mult",
                             SUBCLASS_CONFIGS)
    def test_subclass_accepts_only_seed(self, cls_name, t_mean, t_clip, eps_center, eps_clip, mult):
        """MonteCarloRunner calls sampler_class(seed=N). Verify this works."""
        import audited_endmember_sampler as mod
        cls = getattr(mod, cls_name)
        sampler = cls(seed=99)
        params = sampler.sample()
        assert 'T_surf' in params
        assert 'P_tidal' in params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd EuropaProjectDJ && python -m pytest tests/test_endmember_sampler.py::TestNamedSubclasses -v`
Expected: FAIL — `AttributeError: module 'audited_endmember_sampler' has no attribute 'UniformEqSampler'`

- [ ] **Step 3: Commit failing tests**

```bash
git add EuropaProjectDJ/tests/test_endmember_sampler.py
git commit -m "test: add named subclass pickling and parameter tests"
```

---

### Task 4: Named subclasses — implementation

**Files:**
- Modify: `EuropaProjectDJ/src/audited_endmember_sampler.py`

- [ ] **Step 1: Add named subclasses**

Append to `audited_endmember_sampler.py`:

```python
# ── Endmember presets ────────────────────────────────────────────────────────

_EQ_PRESET = dict(
    T_surf_mean=110.0, T_surf_std=5.0, T_surf_clip=(95.0, 120.0),
    epsilon_0_log_center=np.log10(6e-6), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 2e-5),
)

_POLE_PRESET = dict(
    T_surf_mean=50.0, T_surf_std=5.0, T_surf_clip=(45.0, 80.0),
    epsilon_0_log_center=np.log10(1.2e-5), epsilon_0_log_sigma=0.2,
    epsilon_0_clip=(2e-6, 3.4e-5),
)

# q_tidal_multiplier values from 2D ocean_heat_flux() shape functions
# at q_star=0.45 (mantle_tidal_fraction=0.5):
#   Uniform:            eq=1.00, pole=1.00
#   Soderlund 2014:     eq=1.15, pole=0.70  (equator-enhanced)
#   Lemasquerier 2023:  eq=0.85, pole=1.30  (polar-enhanced)


# ── Named subclasses (picklable for Windows multiprocessing) ─────────────────

class UniformEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=1.00, seed=seed)


class UniformPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=1.00, seed=seed)


class SoderlundEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=1.15, seed=seed)


class SoderlundPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=0.70, seed=seed)


class LemasquerierEqSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_EQ_PRESET, q_tidal_multiplier=0.85, seed=seed)


class LemasquerierPoleSampler(AuditedEndmemberSampler):
    def __init__(self, seed=None):
        super().__init__(**_POLE_PRESET, q_tidal_multiplier=1.30, seed=seed)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd EuropaProjectDJ && python -m pytest tests/test_endmember_sampler.py -v`
Expected: All tests PASS (12 original + 24 subclass tests)

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/src/audited_endmember_sampler.py
git commit -m "feat: add 6 named endmember subclasses for Windows pickling"
```

---

### Task 5: Driver script

**Files:**
- Create: `EuropaProjectDJ/scripts/run_endmember_suite.py`

- [ ] **Step 1: Write the driver**

```python
"""
1D endmember proxy Monte Carlo suite.

Runs equatorial and polar endmember proxies under three ocean transport
scenarios using audited 2026 priors with Andrade rheology.

Produces 6 NPZ result files (3 scenarios x 2 endmembers).

q_tidal_multiplier values derived from 2D ocean_heat_flux() shape
functions at q_star=0.45:
  Uniform:            eq=1.00, pole=1.00
  Soderlund 2014:     eq=1.15, pole=0.70
  Lemasquerier 2023:  eq=0.85, pole=1.30

reject_subcritical=False for all runs (no asymmetric branch handling).

Config must be set to Andrade in src/config.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import multiprocessing as mp
from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_endmember_sampler import (
    UniformEqSampler, UniformPoleSampler,
    SoderlundEqSampler, SoderlundPoleSampler,
    LemasquerierEqSampler, LemasquerierPoleSampler,
)
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

ENSEMBLES = [
    ("endmember_uniform_eq",          UniformEqSampler),
    ("endmember_uniform_pole",        UniformPoleSampler),
    ("endmember_soderlund_eq",        SoderlundEqSampler),
    ("endmember_soderlund_pole",      SoderlundPoleSampler),
    ("endmember_lemasquerier_eq",     LemasquerierEqSampler),
    ("endmember_lemasquerier_pole",   LemasquerierPoleSampler),
]


def run_ensemble(name, sampler_cls, n_iter, seed, n_workers, skip_existing):
    outpath = os.path.join(RESULTS_DIR, f"{name}_andrade.npz")
    if skip_existing and os.path.exists(outpath):
        print(f"  [{name}] exists, skipping")
        return

    config = SolverConfig(reject_subcritical=False)
    runner = MonteCarloRunner(
        n_iterations=n_iter,
        seed=seed,
        verbose=True,
        n_workers=n_workers,
        config=config,
        sampler_class=sampler_cls,
    )

    print(f"\n{'='*60}")
    print(f"  {name}  (N={n_iter}, sampler={sampler_cls.__name__})")
    print(f"{'='*60}")

    results = runner.run()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, outpath)

    print(f"  CBE = {results.cbe_km:.1f} km, "
          f"median = {results.median_km:.1f} km, "
          f"valid = {results.n_valid}/{results.n_iterations}")


def main():
    parser = argparse.ArgumentParser(
        description="Run 1D endmember proxy MC suite")
    parser.add_argument("-n", type=int, default=5000,
                        help="Iterations per ensemble (default 5000)")
    parser.add_argument("--seed", type=int, default=10042)
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count - 1)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip ensembles whose .npz already exists")
    args = parser.parse_args()

    n_workers = args.workers or max(1, mp.cpu_count() - 1)

    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    print(f"1D Endmember Proxy Suite")
    print(f"  Iterations per ensemble: {args.n}")
    print(f"  Seed: {args.seed}")
    print(f"  Workers: {n_workers}")
    print(f"  Output: {RESULTS_DIR}/endmember_*.npz")

    for name, sampler_cls in ENSEMBLES:
        run_ensemble(name, sampler_cls, args.n, args.seed, n_workers,
                     args.skip_existing)

    print(f"\n{'='*60}")
    print("ENDMEMBER SUITE COMPLETE")
    for name, _ in ENSEMBLES:
        print(f"  {os.path.join(RESULTS_DIR, f'{name}_andrade.npz')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
```

- [ ] **Step 2: Verify the script imports cleanly**

Run: `cd EuropaProjectDJ && python -c "import sys; sys.path.insert(0,'src'); from audited_endmember_sampler import UniformEqSampler; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/scripts/run_endmember_suite.py
git commit -m "feat: add run_endmember_suite.py driver for 6-ensemble MC suite"
```

---

### Task 6: Smoke test with small N

**Files:** None (runtime verification only)

- [ ] **Step 1: Run the suite with N=10 to verify end-to-end**

Run: `cd EuropaProjectDJ && python scripts/run_endmember_suite.py -n 10 --seed 42`
Expected: 6 `.npz` files created in `results/`, no crashes. Output shows CBE/median for each ensemble.

- [ ] **Step 2: Verify output files exist**

Run: `ls EuropaProjectDJ/results/endmember_*.npz`
Expected: 6 files:
- `endmember_uniform_eq_andrade.npz`
- `endmember_uniform_pole_andrade.npz`
- `endmember_soderlund_eq_andrade.npz`
- `endmember_soderlund_pole_andrade.npz`
- `endmember_lemasquerier_eq_andrade.npz`
- `endmember_lemasquerier_pole_andrade.npz`

- [ ] **Step 3: Clean up smoke test results**

Run: `rm EuropaProjectDJ/results/endmember_*_andrade.npz`

---

### Task 7: Run full production suite

**Files:** None (production run)

- [ ] **Step 1: Run the full suite at N=5000**

Run: `cd EuropaProjectDJ && python scripts/run_endmember_suite.py -n 5000 --seed 10042`
Expected: 6 `.npz` files, each with ~5000 valid samples. This will take a while.

- [ ] **Step 2: Verify results are reasonable**

Quick sanity check — equatorial uniform should give thinner shells than polar uniform:

```bash
cd EuropaProjectDJ && python -c "
import numpy as np
eq = np.load('results/endmember_uniform_eq_andrade.npz')
pole = np.load('results/endmember_uniform_pole_andrade.npz')
print(f'Eq uniform:   median={np.median(eq[\"thicknesses_km\"]):.1f} km')
print(f'Pole uniform: median={np.median(pole[\"thicknesses_km\"]):.1f} km')
"
```

Expected: Equatorial median < Polar median (warmer surface → thinner shell).
