# Equatorial-Proxy Juno Comparison Suite — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 1D equatorial-proxy Monte Carlo suite in `EuropaProjectDJ` that produces three ocean-heat-transport scenarios, compares each to the Juno MWR constraint via D_cond importance reweighting, and reports per-mode Bayes factors.

**Architecture:** A single `AuditedEquatorialSampler` inherits from `AuditedShellSampler`, overriding only equatorial-specific terms (T_surf, epsilon_0, q_tidal scaling). A runner script loops over three enhancement modes (1.0x, 1.2x, 1.5x), producing one NPZ per mode. A plotter generates per-mode figures. The existing `bayesian_inversion_juno.py` machinery is reused via a thin wrapper that collects cross-mode evidence for a Bayes factor table.

**Tech Stack:** Python 3.10+, NumPy, SciPy, matplotlib. Inherits from `EuropaProjectDJ/src/`.

**Spec:** Agreed in conversation 2026-03-19. Key constraints: scale only the tidal component of q_basal (not radiogenic); do not reintroduce equator-specific d_grain; cap T_surf at 120 K; compare Juno to D_cond, not H_total.

---

## File Structure

```
EuropaProjectDJ/
├── src/
│   ├── audited_sampler.py                    # Existing (no changes)
│   └── audited_equatorial_sampler.py         # NEW: equatorial overrides
├── scripts/
│   ├── run_equatorial_suite.py               # NEW: runs 3 modes, saves NPZs
│   ├── plot_equatorial_results.py            # NEW: per-mode MC figures
│   └── bayesian_equatorial_juno.py           # NEW: Juno comparison + BF table
├── tests/
│   └── test_equatorial_sampler.py            # NEW: sampler unit tests
└── results/
    ├── eq_baseline_andrade.npz               # Output: mode 1.0x
    ├── eq_moderate_andrade.npz               # Output: mode 1.2x
    └── eq_strong_andrade.npz                 # Output: mode 1.5x
```

---

## Chunk 1: Equatorial Sampler

### Task 1: Write failing tests for AuditedEquatorialSampler

**Files:**
- Create: `EuropaProjectDJ/tests/test_equatorial_sampler.py`

- [ ] **Step 1: Write the failing test file**

```python
# EuropaProjectDJ/tests/test_equatorial_sampler.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from audited_equatorial_sampler import AuditedEquatorialSampler
from constants import Planetary


class TestEquatorialOverrides:
    """Equatorial sampler overrides T_surf, epsilon_0, and q_tidal scaling."""

    def test_T_surf_equatorial_range(self):
        """T_surf should be centered near 110 K, clipped to [95, 120] K."""
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        t_surfs = [sampler.sample()['T_surf'] for _ in range(500)]
        assert all(95.0 <= t <= 120.0 for t in t_surfs)
        mean_t = np.mean(t_surfs)
        assert 105.0 < mean_t < 115.0

    def test_epsilon_0_equatorial_range(self):
        """epsilon_0 should be centered near 6e-6, clipped to [2e-6, 2e-5]."""
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        epsilons = [sampler.sample()['epsilon_0'] for _ in range(500)]
        assert all(2e-6 <= e <= 2e-5 for e in epsilons)
        log_mean = np.mean(np.log10(epsilons))
        assert np.log10(4e-6) < log_mean < np.log10(1e-5)

    def test_d_grain_in_audited_range(self):
        """d_grain should remain in audited global range (not overridden)."""
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        grains = [sampler.sample()['d_grain'] for _ in range(100)]
        assert all(5e-5 <= d <= 3e-3 for d in grains)


class TestTidalEnhancement:
    """q_tidal component is scaled, not full q_basal."""

    def _reconstruct_q_basal(self, params):
        """Reconstruct q_basal from P_tidal and H_rad (same as solver)."""
        H_rad = params['H_rad']
        D_H2O = params['D_H2O']
        R_rock = Planetary.RADIUS - D_H2O
        M_rock = (4.0 / 3.0) * np.pi * (R_rock ** 3) * 3500.0
        q_rad = (H_rad * M_rock) / Planetary.AREA
        q_tidal = params['P_tidal'] / Planetary.AREA
        return q_rad + q_tidal, q_rad, q_tidal

    def test_baseline_no_scaling(self):
        """enhancement_factor=1.0 should match audited global q_basal range."""
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        params = sampler.sample()
        q_basal, _, _ = self._reconstruct_q_basal(params)
        # q_basal should be in [5, 25] mW/m^2 (audited range)
        assert 0.005 <= q_basal <= 0.025

    def test_moderate_increases_or_matches_q_basal(self):
        """enhancement_factor=1.2 should increase q_basal (or match if q_tidal=0)."""
        rng_seed = 99
        s1 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.0)
        p1 = s1.sample()
        s2 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.2)
        p2 = s2.sample()
        q1, _, q_tid1 = self._reconstruct_q_basal(p1)
        q2, _, q_tid2 = self._reconstruct_q_basal(p2)
        # q_tidal can be 0 if radiogenic exceeds the sampled q_basal target,
        # in which case enhancement has no effect. Otherwise q2 > q1.
        if q_tid1 > 0:
            assert q2 > q1
        else:
            np.testing.assert_allclose(q1, q2, rtol=1e-10)

    def test_only_tidal_component_scales(self):
        """Radiogenic is unchanged; difference equals 0.5 * q_tidal_global."""
        rng_seed = 77
        s1 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.0)
        p1 = s1.sample()
        s2 = AuditedEquatorialSampler(seed=rng_seed, enhancement_factor=1.5)
        p2 = s2.sample()
        q1, q_rad1, q_tid1 = self._reconstruct_q_basal(p1)
        q2, q_rad2, q_tid2 = self._reconstruct_q_basal(p2)
        # Radiogenic unchanged (same seed -> same H_rad, D_H2O)
        np.testing.assert_allclose(q_rad1, q_rad2, rtol=1e-10)
        # Tidal scaled by exactly 1.5x
        np.testing.assert_allclose(q_tid2, 1.5 * q_tid1, rtol=1e-10)

    def test_enhancement_factor_stored_in_params(self):
        """The enhancement factor should be stored for diagnostics."""
        sampler = AuditedEquatorialSampler(seed=42, enhancement_factor=1.2)
        params = sampler.sample()
        assert params.get('eq_enhancement') == pytest.approx(1.2)


class TestReproducibility:
    """Same seed produces identical draws."""

    def test_same_seed_same_output(self):
        s1 = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        s2 = AuditedEquatorialSampler(seed=42, enhancement_factor=1.0)
        p1 = s1.sample()
        p2 = s2.sample()
        assert p1['T_surf'] == p2['T_surf']
        assert p1['epsilon_0'] == p2['epsilon_0']
        assert p1['P_tidal'] == p2['P_tidal']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python -m pytest tests/test_equatorial_sampler.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'audited_equatorial_sampler'`

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/tests/test_equatorial_sampler.py
git commit -m "test: add failing tests for AuditedEquatorialSampler"
```

---

### Task 2: Implement AuditedEquatorialSampler

**Files:**
- Create: `EuropaProjectDJ/src/audited_equatorial_sampler.py`

- [ ] **Step 1: Implement the sampler**

```python
# EuropaProjectDJ/src/audited_equatorial_sampler.py
"""
Equatorial-proxy sampler for Juno MWR comparison.

Inherits all audited global priors from AuditedShellSampler.
Overrides only equatorial-specific terms:
  T_surf:    N(110, 5) clipped [95, 120] K (Ojakangas & Stevenson 1989)
  epsilon_0: lognormal(6e-6, 0.2 dex) clipped [2e-6, 2e-5] (Tobie+ 2003)
  q_tidal:   scaled by enhancement_factor (ocean heat redistribution)

Does NOT override d_grain — grain size is sampled once per realization
from the audited global prior (PARAMETER_PRIOR_AUDIT_2026.md).

Enhancement modes:
  1.0x — uniform ocean transport (Ashkenazy+ 2021)
  1.2x — moderate equatorial enhancement (Soderlund+ 2014 proxy)
  1.5x — strong equatorial enhancement (upper-bound sensitivity)
"""
import numpy as np

from audited_sampler import AuditedShellSampler
from constants import Planetary


class AuditedEquatorialSampler(AuditedShellSampler):
    """
    Equatorial-proxy sampler with ocean heat enhancement modes.

    Scales only the tidal/ocean component of q_basal, not the
    radiogenic component, to avoid implicitly enhancing spatially
    uniform radiogenic heating.
    """

    def __init__(self, seed=None, enhancement_factor=1.0):
        super().__init__(seed=seed)
        self.enhancement_factor = enhancement_factor

    def sample(self):
        params = super().sample()

        # 1. Override T_surf: equatorial radiative equilibrium
        #    N(110, 5) clipped [95, 120] K
        while True:
            t = self.rng.normal(110.0, 5.0)
            if 95.0 <= t <= 120.0:
                break
        params['T_surf'] = t

        # 2. Override epsilon_0: equatorial tidal strain is lower
        #    lognormal(6e-6, 0.2 dex) clipped [2e-6, 2e-5]
        while True:
            eps = 10 ** self.rng.normal(np.log10(6e-6), 0.2)
            if 2e-6 <= eps <= 2e-5:
                break
        params['epsilon_0'] = eps

        # 3. Scale tidal component of q_basal (not radiogenic)
        #    Parent already set P_tidal = q_silicate_tidal * AREA.
        #    We read it back, scale, and rewrite. The solver reads P_tidal
        #    to compute q_basal = q_radiogenic + P_tidal / AREA.
        if self.enhancement_factor != 1.0:
            q_tidal_flux = params['P_tidal'] / Planetary.AREA
            params['P_tidal'] = self.enhancement_factor * q_tidal_flux * Planetary.AREA

        # 4. Store diagnostics
        params['eq_enhancement'] = self.enhancement_factor

        return params
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python -m pytest tests/test_equatorial_sampler.py -v`

Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/src/audited_equatorial_sampler.py
git commit -m "feat: add AuditedEquatorialSampler with tidal enhancement modes"
```

---

## Chunk 2: Runner Script

### Task 3: Implement run_equatorial_suite.py

**Files:**
- Create: `EuropaProjectDJ/scripts/run_equatorial_suite.py`

- [ ] **Step 1: Write the runner**

```python
# EuropaProjectDJ/scripts/run_equatorial_suite.py
"""
Equatorial-proxy Monte Carlo suite with three ocean heat transport modes.

Runs Andrade rheology with audited priors + equatorial overrides.
Produces one NPZ per mode for downstream Juno comparison.

Shared seed: all three modes use the same base seed so that the
sampler-side parameter draws (d_grain, Q_v, Q_b, etc.) are drawn
from the same RNG sequence. This reduces sampler-side noise in
mode-to-mode comparisons. Note: the saved NPZ arrays are NOT
index-aligned across modes because MonteCarloRunner uses
imap_unordered and filters invalid draws. For true paired
analysis, match draws by sample ID; for aggregate statistics
(CDFs, Bayes factors), the shared seed is sufficient.

reject_subcritical=False matches the audited global Andrade baseline
(run_andrade_15k.py) so that the comparison isolates equatorial
forcing, not branch-handling differences.

Modes:
  baseline (1.0x) — uniform ocean heat transport
  moderate (1.2x) — Soderlund (2014) equatorial enhancement proxy
  strong   (1.5x) — upper-bound equatorial enhancement

Config must be set to Andrade in src/config.json.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import multiprocessing as mp
from Monte_Carlo import MonteCarloRunner, SolverConfig, save_results
from audited_equatorial_sampler import AuditedEquatorialSampler
from constants import Rheology

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Paired seed: same base draws for all modes so only tidal scaling differs
SEED = 10042

N_ITERATIONS = 15000

MODES = [
    ("baseline", 1.0),
    ("moderate", 1.2),
    ("strong",   1.5),
]


# Module-level sampler classes (picklable on Windows spawn-based mp)
class BaselineSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.0, **kwargs)


class ModerateSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.2, **kwargs)


class StrongSampler(AuditedEquatorialSampler):
    def __init__(self, **kwargs):
        super().__init__(enhancement_factor=1.5, **kwargs)


_SAMPLER_MAP = {
    "baseline": BaselineSampler,
    "moderate": ModerateSampler,
    "strong":   StrongSampler,
}


def run_mode(label, enhancement_factor):
    """Run one equatorial MC mode and save results."""
    print(f"\n{'=' * 60}")
    print(f"EQUATORIAL MODE: {label} ({enhancement_factor:.1f}x tidal)")
    print(f"  N = {N_ITERATIONS:,}, seed = {SEED} (paired)")
    print(f"{'=' * 60}")

    # reject_subcritical=False: matches audited global baseline
    config = SolverConfig(reject_subcritical=False)

    runner = MonteCarloRunner(
        n_iterations=N_ITERATIONS,
        seed=SEED,
        verbose=True,
        config=config,
        sampler_class=_SAMPLER_MAP[label],
    )
    results = runner.run()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"eq_{label}_andrade.npz")
    save_results(results, output_path)

    print(f"\n--- {label} ({enhancement_factor:.1f}x) RESULTS ---")
    print(f"  CBE:     {results.cbe_km:.1f} km")
    print(f"  Median:  {results.median_km:.1f} km")
    print(f"  1-sigma: [{results.sigma_1_low_km:.1f}, {results.sigma_1_high_km:.1f}] km")
    print(f"  Valid:   {results.n_valid}/{results.n_iterations}")

    return output_path


def main():
    print(f"Rheology model: {Rheology.MODEL}")
    assert Rheology.MODEL == "Andrade", f"Expected Andrade, got {Rheology.MODEL}"

    paths = {}
    for label, factor in MODES:
        paths[label] = run_mode(label, factor)

    print(f"\n{'=' * 60}")
    print("EQUATORIAL SUITE COMPLETE")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
```

- [ ] **Step 2: Verify it runs (quick smoke test with 50 iterations)**

Run: Edit `N_ITERATIONS = 50` temporarily, then:
`cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python scripts/run_equatorial_suite.py`

Expected: Three NPZ files created in `results/`. Revert to `N_ITERATIONS = 15000`.

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/scripts/run_equatorial_suite.py
git commit -m "feat: add equatorial-proxy runner with 3 ocean heat modes"
```

---

## Chunk 3: Equatorial Plotting

### Task 4: Implement plot_equatorial_results.py

**Files:**
- Create: `EuropaProjectDJ/scripts/plot_equatorial_results.py`

- [ ] **Step 1: Write the plotter**

```python
# EuropaProjectDJ/scripts/plot_equatorial_results.py
"""
Per-mode equatorial MC figures:
  (a) Total thickness PDF with regime split
  (b) D_cond PDF with Juno bands overlaid
  (c) Shell structure (D_cond/D_conv stacked bar)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pub_style import (
    apply_style, PAL, figsize_double_tall,
    label_panel, save_fig, add_minor_gridlines,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()

RA_CRIT = 1000.0

JUNO_MODELS = [
    (29.0, 10.0, 3.0, "Pure water"),
    (24.0, 10.0, 3.0, "Low salinity"),
]

MODES = [
    ("eq_baseline_andrade.npz", "Baseline (1.0x)", "baseline"),
    ("eq_moderate_andrade.npz", "Moderate (1.2x)", "moderate"),
    ("eq_strong_andrade.npz",   "Strong (1.5x)",   "strong"),
]


def _kde(values, n_pts=300):
    """Gaussian KDE with guard against small or degenerate samples."""
    if len(values) < 10:
        return None, None
    if np.std(values) < 1e-10:
        return None, None
    kde = gaussian_kde(values)
    lo = max(0, np.percentile(values, 0.5) - 2)
    hi = np.percentile(values, 99.5) + 2
    x = np.linspace(lo, hi, n_pts)
    return x, kde(x)


def plot_mode(filepath, title, tag):
    """Three-panel figure for one equatorial mode."""
    data = np.load(filepath)
    H = data['thicknesses_km']
    D_cond = data['D_cond_km']
    D_conv = data['D_conv_km']
    Ra = data['Ra_values'] if 'Ra_values' in data else np.zeros(len(H))
    n = len(H)

    conv_mask = Ra >= RA_CRIT
    frac_conv = conv_mask.mean()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.0, 3.2))

    # (a) Total thickness with regime split
    x_grid = np.linspace(0, np.percentile(H, 99.5) + 5, 400)

    H_cond = H[~conv_mask]
    H_conv = H[conv_mask]
    frac_cond = 1.0 - frac_conv

    x_c, pdf_c = _kde(H_cond)
    if pdf_c is not None:
        # Interpolate onto shared x_grid and scale by fraction
        pdf_c_grid = np.interp(x_grid, x_c, pdf_c, left=0, right=0) * frac_cond
        ax1.fill_between(x_grid, 0, pdf_c_grid, color=PAL.COND, alpha=0.25)
        ax1.plot(x_grid, pdf_c_grid, color=PAL.COND, lw=1.2,
                 label=f"Cond. ({frac_cond:.0%})")

    x_v, pdf_v = _kde(H_conv)
    if pdf_v is not None:
        pdf_v_grid = np.interp(x_grid, x_v, pdf_v, left=0, right=0) * frac_conv
        ax1.fill_between(x_grid, 0, pdf_v_grid, color=PAL.CONV, alpha=0.20)
        ax1.plot(x_grid, pdf_v_grid, color=PAL.CONV, lw=1.2,
                 label=f"Conv. ({frac_conv:.0%})")

    ax1.set_xlabel("Ice shell thickness (km)")
    ax1.set_ylabel("Probability density")
    ax1.set_xlim(0, np.percentile(H, 99.5) + 5)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=6)
    label_panel(ax1, "a")

    # (b) D_cond with Juno bands
    x_d, pdf_d = _kde(D_cond)
    if pdf_d is not None:
        ax2.fill_between(x_d, 0, pdf_d, color=PAL.COND, alpha=0.25)
        ax2.plot(x_d, pdf_d, color=PAL.COND, lw=1.5, label=r"$D_{\rm cond}$")

    x_juno = np.linspace(0, 70, 300)
    for D_obs, sigma_obs, sigma_model, jlabel in JUNO_MODELS:
        sigma_tot = np.sqrt(sigma_obs**2 + sigma_model**2)
        juno_pdf = np.exp(-0.5 * ((x_juno - D_obs) / sigma_tot)**2)
        juno_pdf /= (sigma_tot * np.sqrt(2 * np.pi))
        ls = "--" if "Pure" in jlabel else ":"
        ax2.plot(x_juno, juno_pdf, color=PAL.BLACK, lw=1.0, ls=ls,
                 alpha=0.7, label=f"Juno {jlabel}")

    ax2.set_xlabel(r"$D_{\rm cond}$ (km)")
    ax2.set_ylabel("Probability density")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=5.5, loc="upper right")
    label_panel(ax2, "b")

    # (c) Shell structure stacked bar
    h_max = np.percentile(H, 98)
    bin_edges = np.linspace(max(H.min(), 0), h_max, 25)
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    dig = np.digitize(H, bin_edges)
    w = bin_edges[1] - bin_edges[0]

    mc = np.array([D_cond[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mv = np.array([D_conv[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])
    mH = np.array([H[dig == i].mean() if np.sum(dig == i) > 5
                   else np.nan for i in range(1, len(bin_edges))])

    ok = ~np.isnan(mc) & ~np.isnan(mv)
    ax3.bar(bc[ok], mc[ok], width=w * 0.9, color=PAL.COND, alpha=0.75,
            label=r"$D_{\rm cond}$")
    ax3.bar(bc[ok], mv[ok], width=w * 0.9, bottom=mc[ok],
            color=PAL.CONV, alpha=0.75, label=r"$D_{\rm conv}$")
    ax3.plot(bc[ok], mH[ok], "k-", lw=1.0, label=r"$H_{\rm total}$")

    ax3.set_xlabel("Total thickness bin (km)")
    ax3.set_ylabel("Mean layer thickness (km)")
    ax3.legend(fontsize=6, loc="upper left")
    label_panel(ax3, "c")

    fig.suptitle(f"Equatorial proxy: {title} (N = {n:,})", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=1.5)
    save_fig(fig, f"fig_eq_{tag}", FIGURES_DIR)


def main():
    for filename, title, tag in MODES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            print(f"\nPlotting: {title}")
            plot_mode(filepath, title, tag)
        else:
            print(f"Skipping {title}: {filepath} not found")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add EuropaProjectDJ/scripts/plot_equatorial_results.py
git commit -m "feat: add per-mode equatorial MC plotter with Juno overlay"
```

---

## Chunk 4: Bayesian Juno Comparison + Bayes Factors

### Task 5: Implement bayesian_equatorial_juno.py

**Files:**
- Create: `EuropaProjectDJ/scripts/bayesian_equatorial_juno.py`

- [ ] **Step 1: Write the Bayesian comparison script**

```python
# EuropaProjectDJ/scripts/bayesian_equatorial_juno.py
"""
Bayesian Juno comparison across equatorial-proxy modes.

For each mode, runs importance reweighting against D_cond for both
Juno observation models (pure-water 29+/-10, low-salinity 24+/-10).
Computes marginal likelihoods and reports Bayes factors between modes.

Reuses core machinery from bayesian_inversion_juno.py.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from bayesian_inversion_juno import (
    compute_log_weights, normalize_weights, effective_sample_size,
    weighted_percentile, posterior_summary, run_model,
)
from pub_style import (
    apply_style, PAL, figsize_double, save_fig, add_minor_gridlines,
    label_panel,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures', 'pub')

apply_style()

MODES = [
    ("eq_baseline_andrade.npz", "Baseline (1.0x)", "baseline"),
    ("eq_moderate_andrade.npz", "Moderate (1.2x)", "moderate"),
    ("eq_strong_andrade.npz",   "Strong (1.5x)",   "strong"),
]

JUNO_OBS = [
    (29.0, 10.0, "Model A: 29\u00b110 km"),
    (24.0, 10.0, "Model B: 24\u00b110 km"),
]

SIGMA_MODEL = 3.0


def log_marginal_likelihood(D_cond, D_obs, sigma_obs, sigma_model):
    """
    Log marginal likelihood: log p(D_obs | mode) = log (1/N) sum_i p(D_obs | D_cond_i).

    Uses logsumexp for numerical stability.
    """
    log_w = compute_log_weights(D_cond, D_obs, sigma_obs, sigma_model)
    return logsumexp(log_w) - np.log(len(log_w))


def main():
    np.random.seed(42)

    # Collect results per mode
    results_table = []

    for filename, mode_label, tag in MODES:
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {mode_label}: {filepath} not found")
            continue

        print(f"\n{'#' * 60}")
        print(f"# MODE: {mode_label}")
        print(f"{'#' * 60}")

        data = np.load(filepath)
        D_cond = data['D_cond_km']
        n_valid = len(D_cond)
        n_total = int(data['n_iterations']) if 'n_iterations' in data else n_valid
        print(f"  N = {n_valid:,} valid / {n_total:,} total ({100*n_valid/n_total:.0f}%)")

        row = {'mode': mode_label, 'tag': tag, 'n': n_valid, 'n_total': n_total}

        for D_obs, sigma_obs, obs_label in JUNO_OBS:
            obs_tag = "A" if D_obs == 29.0 else "B"

            # Marginal likelihood
            log_ml = log_marginal_likelihood(D_cond, D_obs, sigma_obs, SIGMA_MODEL)
            row[f'log_ml_{obs_tag}'] = log_ml

            # Posterior summaries
            log_w = compute_log_weights(D_cond, D_obs, sigma_obs, SIGMA_MODEL)
            w = normalize_weights(log_w)
            ess = effective_sample_size(w)
            row[f'ess_{obs_tag}'] = ess

            print(f"\n  {obs_label}:")
            print(f"    Log marginal likelihood: {log_ml:.3f}")
            print(f"    ESS: {ess:.0f} / {n_valid} ({100 * ess / n_valid:.1f}%)")

            med_dc = weighted_percentile(D_cond, w, 50)
            row[f'D_cond_median_{obs_tag}'] = med_dc
            print(f"    Posterior D_cond median: {med_dc:.1f} km")

            # Posterior convective fraction
            if 'Ra_values' in data:
                conv_frac = np.sum(w[data['Ra_values'] >= 1000])
                row[f'conv_frac_{obs_tag}'] = conv_frac
                print(f"    Posterior convective fraction: {conv_frac:.1%}")

            # Per-mode Bayesian figures (prefixed to avoid collision
            # with global Bayesian figures in the same directory)
            full_label = f"eq_{tag}_{obs_tag}"
            run_model(data, D_obs, sigma_obs, SIGMA_MODEL, full_label)

        results_table.append(row)

    # ── Bayes factor table ────────────────────────────────────────────
    # NOTE: These Bayes factors are conditional on solver-valid draws.
    # If valid yield differs across modes, the comparison reflects
    # p(Juno | mode, valid) not full p(Juno | mode). The yield
    # difference is reported separately as additional context.
    if len(results_table) < 2:
        print("\nNot enough modes for Bayes factor comparison.")
        return

    print(f"\n{'=' * 60}")
    print("BAYES FACTOR TABLE (relative to baseline, conditional on valid draws)")
    print(f"{'=' * 60}")

    baseline = results_table[0]
    header = (f"{'Mode':<22s} | {'N_valid':>7s} | {'Yield':>6s} | "
              f"{'log BF_A':>10s} | {'BF_A':>8s} | {'log BF_B':>10s} | {'BF_B':>8s}")
    print(header)
    print("-" * len(header))

    for row in results_table:
        for obs_tag in ["A", "B"]:
            key = f'log_ml_{obs_tag}'
            row[f'log_bf_{obs_tag}'] = row[key] - baseline[key]

        log_bf_a = row['log_bf_A']
        log_bf_b = row['log_bf_B']
        bf_a = np.exp(log_bf_a)
        bf_b = np.exp(log_bf_b)
        n_valid = row['n']
        n_total = row.get('n_total', n_valid)
        yield_pct = f"{100 * n_valid / n_total:.0f}%" if n_total > 0 else "?"
        print(f"{row['mode']:<22s} | {n_valid:>7d} | {yield_pct:>6s} | "
              f"{log_bf_a:>10.3f} | {bf_a:>8.3f} | {log_bf_b:>10.3f} | {bf_b:>8.3f}")

    # ── Summary figure: BF bar chart ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double(0.40))

    labels = [r['mode'] for r in results_table]
    x = np.arange(len(labels))
    w = 0.35

    for ax, obs_tag, obs_label in [(ax1, "A", "Pure water (29 km)"),
                                    (ax2, "B", "Low salinity (24 km)")]:
        log_bfs = [r[f'log_bf_{obs_tag}'] for r in results_table]
        colors = [PAL.GREEN if bf > 0 else PAL.RED for bf in log_bfs]
        ax.bar(x, log_bfs, color=colors, alpha=0.7, edgecolor="0.3", lw=0.5)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=15, ha="right")
        ax.set_ylabel("log Bayes factor vs baseline")
        ax.set_title(obs_label, fontsize=8)
        add_minor_gridlines(ax, axis="y")

    label_panel(ax1, "a")
    label_panel(ax2, "b")

    fig.suptitle("Equatorial mode evidence comparison", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "fig_eq_bayes_factors", FIGURES_DIR)

    print(f"\nAll equatorial Juno figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add EuropaProjectDJ/scripts/bayesian_equatorial_juno.py
git commit -m "feat: add equatorial Juno Bayesian comparison with Bayes factors"
```

---

## Chunk 5: Production Run + Verification

### Task 6: Run the full equatorial suite (15k per mode)

- [ ] **Step 1: Verify config is Andrade**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python -c "import sys; sys.path.insert(0,'src'); from constants import Rheology; print(Rheology.MODEL)"`

Expected: `Andrade`

- [ ] **Step 2: Run all three modes**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python scripts/run_equatorial_suite.py`

Expected: Three NPZ files in `results/`: `eq_baseline_andrade.npz`, `eq_moderate_andrade.npz`, `eq_strong_andrade.npz`. Runtime ~60-70 minutes total.

- [ ] **Step 3: Generate per-mode figures**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python scripts/plot_equatorial_results.py`

Expected: Three figure sets in `figures/pub/`: `fig_eq_baseline`, `fig_eq_moderate`, `fig_eq_strong`.

- [ ] **Step 4: Run Bayesian Juno comparison**

Run: `cd C:\Users\Joshu\.cursor\projects\EuropaConvection\EuropaProjectDJ && python scripts/bayesian_equatorial_juno.py`

Expected: Per-mode Bayesian figures + Bayes factor table printed to console + `fig_eq_bayes_factors` saved.

- [ ] **Step 5: Verify Bayes factors are sensible**

Check:
- BF values should be O(1) (weak-to-moderate) given the broad Juno constraint
- ESS should be > 5% for all modes (otherwise the reweighting is collapsing)
- D_cond posterior median should shift toward the Juno observation

- [ ] **Step 6: Commit results**

```bash
git add EuropaProjectDJ/scripts/ EuropaProjectDJ/src/audited_equatorial_sampler.py EuropaProjectDJ/tests/test_equatorial_sampler.py
git commit -m "feat: complete equatorial-proxy Juno comparison suite"
```
