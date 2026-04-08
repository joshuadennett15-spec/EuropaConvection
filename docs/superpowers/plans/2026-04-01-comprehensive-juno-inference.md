# Comprehensive Juno Inference Study — Research & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rigorously determine what Juno's D_cond = 29 +/- 10 km at 35 deg tells us about Europa's ice shell — combining systematic hypothesis testing, Bayesian inference across all ocean transport scenarios, and critical evaluation of methodology and biases.

**Architecture:** Three phases — (1) hypothesis generation and critical evaluation of our approach, (2) data generation (0.6mm MC runs for all scenarios + Bayesian reweighting), (3) synthesis with three-way comparison figures and a structured findings report.

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib. Existing Europa2D MC runner, pub_style, autoresearch objectives.

---

## Phase 1: Scientific Framework

### Task 1: Generate and evaluate competing hypotheses

Before running experiments, formalize the scientific question and competing explanations. This task is **research only** — no code, just analysis written to a markdown file.

**Files:**
- Create: `Europa2D/docs/2026-04-01-juno-inference-hypotheses.md`

- [ ] **Step 1: Formulate the core scientific question**

Write the following to `Europa2D/docs/2026-04-01-juno-inference-hypotheses.md`:

```markdown
# Competing Hypotheses: What Controls Europa's Conductive Lid Thickness?

## Core Question

Juno MWR measured D_cond = 29 +/- 10 km at ~35 deg latitude (Levin et al. 2025).
Our 2D model with broad priors (0.6mm grain center) produces D_cond median ~19 km
at 35 deg — a ~10 km underprediction. What physical mechanism is responsible for
the discrepancy, and what does the Juno constraint actually tell us about Europa's
ice shell?

## Observation

The model prior underpredicts D_cond by ~10 km relative to Juno. Bayesian
reweighting shifts the posterior to ~24.5 km (still 4.5 km below Juno center).
Manually increasing grain size from 0.6mm to 1.5mm shifts D_cond to ~28 km but
suppresses convective vigor (Nu drops from 6-7 to 4-5 when convection occurs).

## Competing Hypotheses

### H1: Grain Size Dominance
The 0.6mm grain prior is too small. Europa's equilibrium grain size is closer to
1-2 mm (Barr & McKinnon 2007 support up to 30-80 mm), producing a stiffer ice
shell with thicker conductive lids.

- **Prediction:** Shifting grain center to 1.5mm matches Juno D_cond.
- **Testable:** Compare D_cond at 35 deg across grain sizes 0.6-2.0 mm.
- **Status:** CONFIRMED by autoresearch experiments (score 18.1 -> 5.0).
- **Concern:** Suppresses convective vigor globally. Is this physical?

### H2: Bayesian Selection (Data-Driven)
The broad 0.6mm prior is acceptable. Juno simply selects the subset of parameter
space with thicker lids. These samples have specific combinations of grain size,
viscosity, and tidal heating that produce thick D_cond while still allowing
vigorous convection underneath.

- **Prediction:** Juno-reweighted posterior has higher Nu|convecting than 1.5mm prior.
- **Testable:** Compare conditional Nu between reweighted and grain-shifted.
- **Status:** CONFIRMED for uniform scenario (Nu 7.6 vs 4.8).
- **Concern:** Reweighting only reaches D_cond ~24.5 km, not 29 km. N_eff = 167.

### H3: Combined Effect
Neither grain size alone nor Bayesian selection alone is sufficient. The truth
requires BOTH a modest grain size increase (e.g., 0.8-1.0 mm, physically motivated)
AND Bayesian reweighting to reach Juno's 29 km while preserving realistic convection.

- **Prediction:** A "compromise" grain prior (0.8-1.0mm) + Juno reweighting produces
  D_cond ~28-30 km with Nu|convecting ~6-7.
- **Testable:** Run intermediate grain priors and apply Juno reweighting to each.
- **Status:** UNTESTED.

### H4: Missing Physics
The model itself lacks a mechanism that thickens D_cond. Candidates:
- Tidal dissipation is overestimated (q_tidal_scale too high)
- Convection closure is too efficient (Nu too high at marginal Ra)
- Surface temperature profile needs refinement
- Lateral ice flow (not modeled) redistributes thickness

- **Prediction:** No parameter adjustment within current model fully resolves
  the tension without unrealistic side effects.
- **Testable:** If all grain/tidal/q combinations fail to match Juno while
  preserving convection, missing physics is implicated.
- **Status:** PARTIALLY TESTED (tidal scale reduction made things worse).

### H5: Ocean Transport Sensitivity
Different ocean transport regimes produce different D_cond at 35 deg, and the
correct regime naturally produces thicker lids at mid-latitudes without needing
grain size changes.

- **Prediction:** At least one ocean scenario produces D_cond closer to 29 km
  than others with the original 0.6mm grain prior.
- **Testable:** Compare D_cond at 35 deg across all 4 scenarios with 0.6mm grains.
- **Status:** TO BE TESTED in this study.

## Critical Evaluation of Methodology

### Potential Biases
1. **Confirmation bias:** We found grain size works and stopped looking for
   other mechanisms.
2. **Single-observable bias:** Optimizing only D_cond at 35 deg ignores
   constraints from convection, heat flux, and latitude structure.
3. **Prior sensitivity:** Results depend heavily on the chosen grain size prior,
   which is poorly constrained (0.05-80 mm in literature).
4. **Numerical bias:** The dt fix changed equilibrium states; old intuitions
   about parameter sensitivity may not hold.

### What Would Falsify Each Hypothesis?
- H1 falsified if: 1.5mm grains produce unrealistic convection patterns
  (e.g., no convection anywhere, or convection only at poles)
- H2 falsified if: Juno-reweighted posterior has LOWER Nu than 1.5mm prior
- H3 falsified if: no intermediate grain size balances D_cond and convection
- H4 falsified if: parameter adjustment within current model resolves tension
- H5 falsified if: all scenarios produce identical D_cond at 35 deg (ALREADY
  PARTIALLY OBSERVED — they're very similar at 35 deg)

## Multi-Constraint Scoring

Instead of optimizing D_cond alone, define a composite score:
1. D_cond at 35 deg vs Juno (primary)
2. Convecting fraction at equator (should be 30-60%, Green et al. 2021)
3. Conditional Nu when convecting (should be 3-10, stagnant-lid regime)
4. Shell thickness 20-40 km (Wakita et al. 2024, Howell 2021)
5. Scenario discriminability (latitude structure should vary with ocean regime)
```

- [ ] **Step 2: Review the hypotheses document for scientific rigor**

Read the document and verify:
- Each hypothesis is testable and falsifiable
- Predictions are specific enough to discriminate between hypotheses
- Biases are honestly acknowledged
- No logical fallacies (especially single-cause fallacy, confirmation bias)

---

## Phase 2: Data Generation

### Task 2: Run 0.6mm MC ensembles for all 4 scenarios

The uniform scenario already exists. Run the remaining 3.

**Files:**
- Verify: `Europa2D/src/latitude_sampler.py` has NO grain override (uses audited 0.6mm default)
- Output: `Europa2D/results/mc_2d_{scenario}_250_grain06mm.npz` (3 new files)

- [ ] **Step 1: Verify grain override is removed**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
grep -n "d_grain_2d" Europa2D/src/latitude_sampler.py
```

Expected: no matches. If the 1.5mm override is present, remove it temporarily (it will be restored in Task 5).

- [ ] **Step 2: Run Soderlund equator-enhanced (0.6mm grain)**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -c "
import sys, os
sys.path.insert(0, 'Europa2D/src')
sys.path.insert(0, 'EuropaProjectDJ/src')
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
mp.freeze_support()
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

runner = MonteCarloRunner2D(
    n_iterations=250, seed=10042, n_workers=15,
    n_lat=37, nx=31, dt=1e12, max_steps=1500,
    ocean_pattern='equator_enhanced', q_star=0.4, verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_soderlund2014_equator_250_grain06mm.npz')
print('Saved Soderlund 0.6mm')
"
```

Expected: ~250/250 valid, runtime ~10-13 min.

- [ ] **Step 3: Run Lemasquerier polar (0.6mm grain)**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -c "
import sys, os
sys.path.insert(0, 'Europa2D/src')
sys.path.insert(0, 'EuropaProjectDJ/src')
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
mp.freeze_support()
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

runner = MonteCarloRunner2D(
    n_iterations=250, seed=20042, n_workers=15,
    n_lat=37, nx=31, dt=1e12, max_steps=1500,
    ocean_pattern='polar_enhanced', q_star=0.455, verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_lemasquerier2023_polar_250_grain06mm.npz')
print('Saved Lemasquerier polar 0.6mm')
"
```

- [ ] **Step 4: Run Lemasquerier polar strong (0.6mm grain)**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -c "
import sys, os
sys.path.insert(0, 'Europa2D/src')
sys.path.insert(0, 'EuropaProjectDJ/src')
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
mp.freeze_support()
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

runner = MonteCarloRunner2D(
    n_iterations=250, seed=30042, n_workers=15,
    n_lat=37, nx=31, dt=1e12, max_steps=1500,
    ocean_pattern='polar_enhanced', q_star=0.819, verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_lemasquerier2023_polar_strong_250_grain06mm.npz')
print('Saved Lemasquerier polar strong 0.6mm')
"
```

- [ ] **Step 5: Verify all 4 grain06mm files exist**

```bash
ls -la Europa2D/results/mc_2d_*_grain06mm.npz
```

Expected: 4 files.

### Task 3: Run intermediate grain prior (H3 test — 1.0mm)

Test the "compromise" hypothesis: 1.0mm grain + Juno reweighting.

**Files:**
- Modify: `Europa2D/src/latitude_sampler.py` (add temporary 1.0mm override)
- Output: `Europa2D/results/mc_2d_uniform_transport_250_grain10mm.npz`

- [ ] **Step 1: Add 1.0mm grain override**

In `Europa2D/src/latitude_sampler.py`, after `H_rad = audited_params['H_rad']` (line ~101), add:

```python
        # Temporary 1.0mm grain override for H3 hypothesis test
        d_grain_2d = 10 ** self.rng.normal(np.log10(1.0e-3), 0.35)
        d_grain_2d = float(np.clip(d_grain_2d, 5e-5, 4e-3))
        audited_params['d_grain'] = d_grain_2d
```

- [ ] **Step 2: Run uniform scenario with 1.0mm grain**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -c "
import sys, os
sys.path.insert(0, 'Europa2D/src')
sys.path.insert(0, 'EuropaProjectDJ/src')
os.environ['OMP_NUM_THREADS'] = '1'
import multiprocessing as mp
mp.freeze_support()
from monte_carlo_2d import MonteCarloRunner2D, save_results_2d

runner = MonteCarloRunner2D(
    n_iterations=250, seed=42, n_workers=15,
    n_lat=37, nx=31, dt=1e12, max_steps=1500,
    ocean_pattern='uniform', verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_uniform_transport_250_grain10mm.npz')
print('Saved 1.0mm grain results')
"
```

- [ ] **Step 3: Remove the 1.0mm grain override from latitude_sampler.py**

Delete the 3-line block added in Step 1.

### Task 4: Compute Bayesian reweighting for all scenario x grain combinations

**Files:**
- Create: `Europa2D/scripts/juno_comprehensive_comparison.py`

- [ ] **Step 1: Create the comprehensive comparison script**

```python
# Europa2D/scripts/juno_comprehensive_comparison.py
"""
Comprehensive Juno inference: compare prior, Juno-reweighted, and grain-shifted
results across all 4 ocean scenarios and 3 grain priors (0.6, 1.0, 1.5 mm).
"""
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")

JUNO = 29.0
SIGMA_EFF = np.sqrt(10.0**2 + 3.0**2)

SCENARIOS = [
    ("uniform_transport",             "Uniform"),
    ("soderlund2014_equator",         "Eq-enhanced"),
    ("lemasquerier2023_polar",        "Polar-enh."),
    ("lemasquerier2023_polar_strong", "Strong polar"),
]


def interp_at(lat, profiles, target):
    return np.array([np.interp(target, lat, profiles[i])
                     for i in range(profiles.shape[0])])


def gaussian_lk(dc):
    return np.exp(-0.5 * ((dc - JUNO) / SIGMA_EFF)**2)


def analyze(data, weights=None):
    """Compute summary stats, optionally importance-weighted."""
    lat = data["latitudes_deg"]
    H = data["H_profiles"]
    Nu = data["Nu_profiles"]
    Ra = data["Ra_profiles"]
    Dc = data["D_cond_profiles"]

    dc35 = interp_at(lat, Dc, 35.0)

    if weights is not None:
        w = weights
        dc_val = float(np.sum(w * dc35))
        h_eq = float(np.sum(w * H[:, 0]))
        h_po = float(np.sum(w * H[:, -1]))
        conv_eq = float(np.sum(w * (Nu[:, 0] > 1.1)))
        conv_po = float(np.sum(w * (Nu[:, -1] > 1.1)))
        mask_eq = Nu[:, 0] > 1.1
        nu_eq = float(np.sum(w[mask_eq] * Nu[mask_eq, 0]) / max(np.sum(w[mask_eq]), 1e-12)) if mask_eq.sum() > 5 else 0
        mask_po = Nu[:, -1] > 1.1
        nu_po = float(np.sum(w[mask_po] * Nu[mask_po, -1]) / max(np.sum(w[mask_po]), 1e-12)) if mask_po.sum() > 5 else 0
        n_eff = 1.0 / np.sum(w**2)
    else:
        dc_val = float(np.median(dc35))
        h_eq = float(np.median(H[:, 0]))
        h_po = float(np.median(H[:, -1]))
        conv_eq = float(np.mean(Nu[:, 0] > 1.1))
        conv_po = float(np.mean(Nu[:, -1] > 1.1))
        mask_eq = Nu[:, 0] > 1.1
        nu_eq = float(np.median(Nu[mask_eq, 0])) if mask_eq.sum() > 5 else 0
        mask_po = Nu[:, -1] > 1.1
        nu_po = float(np.median(Nu[mask_po, -1])) if mask_po.sum() > 5 else 0
        n_eff = float(len(dc35))

    return {
        "dc35": dc_val, "h_eq": h_eq, "h_po": h_po,
        "conv_eq": conv_eq, "conv_po": conv_po,
        "nu_eq": nu_eq, "nu_po": nu_po, "n_eff": n_eff,
    }


def main():
    print("=" * 100)
    print("COMPREHENSIVE JUNO INFERENCE: 4 SCENARIOS x 4 VIEWS")
    print(f"Juno: D_cond = {JUNO} +/- {SIGMA_EFF:.1f} km (eff) at 35 deg")
    print("=" * 100)

    header = (f"{'Scenario':14s} {'View':22s} | {'Dc(35)':>7s} {'H_eq':>6s} {'H_po':>6s} "
              f"{'C%eq':>5s} {'C%po':>5s} {'Nu|eq':>6s} {'Nu|po':>6s} {'N_eff':>6s}")
    print(header)
    print("-" * len(header))

    for key, title in SCENARIOS:
        # Load all available grain results
        grain_files = [
            ("0.6mm prior", f"mc_2d_{key}_250_grain06mm.npz"),
            ("1.0mm prior", f"mc_2d_{key}_250_grain10mm.npz"),
            ("1.5mm prior", f"mc_2d_{key}_250.npz"),
        ]

        for grain_label, fname in grain_files:
            fpath = os.path.join(RESULTS_DIR, fname)
            if not os.path.exists(fpath):
                continue
            data = dict(np.load(fpath, allow_pickle=True))

            # Prior stats
            s = analyze(data)
            print(f"{title:14s} {grain_label:22s} | {s['dc35']:6.1f}  {s['h_eq']:5.1f}  {s['h_po']:5.1f}  "
                  f"{s['conv_eq']:4.0%}  {s['conv_po']:4.0%}  {s['nu_eq']:5.1f}  {s['nu_po']:5.1f}  {s['n_eff']:5.0f}")

            # Juno-reweighted stats
            dc35 = interp_at(data["latitudes_deg"], data["D_cond_profiles"], 35.0)
            lk = gaussian_lk(dc35)
            w = lk / lk.sum()
            s_j = analyze(data, w)
            print(f"{'':14s} {grain_label + '+Juno':22s} | {s_j['dc35']:6.1f}  {s_j['h_eq']:5.1f}  {s_j['h_po']:5.1f}  "
                  f"{s_j['conv_eq']:4.0%}  {s_j['conv_po']:4.0%}  {s_j['nu_eq']:5.1f}  {s_j['nu_po']:5.1f}  {s_j['n_eff']:5.0f}")

        print("-" * len(header))

    # Hypothesis evaluation
    print("\n" + "=" * 100)
    print("HYPOTHESIS EVALUATION")
    print("=" * 100)
    print("""
H1 (Grain dominance):  1.5mm grain matches Juno Dc but suppresses Nu to 4-5.
                       SUPPORTED for Dc match, CONCERN about convection realism.

H2 (Bayesian select):  0.6mm+Juno reaches Dc ~24.5 km with Nu ~7-8.
                       SUPPORTED for preserving convection, FALLS SHORT on Dc.

H3 (Combined):         1.0mm+Juno should reach Dc ~27-28 km with Nu ~6-7.
                       CHECK the 1.0mm+Juno row in the table above.

H5 (Ocean sensitivity): Compare Dc(35) across scenarios within each grain prior.
                        If they differ by >2 km, ocean regime matters at 35 deg.
""")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comprehensive comparison**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python Europa2D/scripts/juno_comprehensive_comparison.py
```

Expected: Table with all scenarios x views, plus hypothesis evaluation summary.

### Task 5: Restore 1.5mm grain override and commit

**Files:**
- Modify: `Europa2D/src/latitude_sampler.py`

- [ ] **Step 1: Add the 1.5mm grain override back**

In `Europa2D/src/latitude_sampler.py`, after `H_rad = audited_params['H_rad']` (line ~101), add:

```python
        # 2D grain-size override: shift prior center from 0.6 mm to 1.5 mm
        # to favour thicker conductive lids (Barr & McKinnon 2007 supports
        # equilibrium grains up to 30-80 mm; 1.5 mm is still conservative).
        d_grain_2d = 10 ** self.rng.normal(np.log10(1.5e-3), 0.35)
        d_grain_2d = float(np.clip(d_grain_2d, 5e-5, 5e-3))
        audited_params['d_grain'] = d_grain_2d
```

- [ ] **Step 2: Smoke test**

```bash
python -m pytest Europa2D/tests/test_validation.py -x -q
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add Europa2D/src/latitude_sampler.py
git commit -m "feat: restore 1.5mm grain prior as default for 2D runs"
```

---

## Phase 3: Synthesis

### Task 6: Generate 4-scenario three-way comparison figure

**Files:**
- Modify: `Europa2D/scripts/plot_prior_vs_posterior_convection.py`

- [ ] **Step 1: Update the plotting script for 4x4 panel layout**

Replace `Europa2D/scripts/plot_prior_vs_posterior_convection.py` with a 4-row (scenarios) x 4-column (H_total, D_cond, Conv%, Nu|conv) figure showing all three views per scenario. Use `C_PRIOR="0.55"` for 0.6mm, `C_JUNO=PAL.BLUE` for Juno-reweighted, `C_GRAIN=PAL.RED` for 1.5mm.

The script structure should match the existing `plot_prior_vs_posterior_convection.py` but loop over all 4 scenarios instead of just uniform. Load 0.6mm data from `*_grain06mm.npz` and 1.5mm data from the standard `*_250.npz` files.

See the full script in the earlier plan at `docs/superpowers/plans/2026-04-01-juno-reweighting-all-scenarios.md`, Task 3, Step 1 — use that code.

- [ ] **Step 2: Run the plotting script**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python Europa2D/scripts/plot_prior_vs_posterior_convection.py
```

Expected: `Saved: prior_vs_juno_vs_grain_4scenario.{png, pdf}`

- [ ] **Step 3: Commit**

```bash
git add Europa2D/scripts/plot_prior_vs_posterior_convection.py Europa2D/scripts/juno_comprehensive_comparison.py
git commit -m "feat: comprehensive Juno inference with 4-scenario comparison"
```

### Task 7: Write findings summary

**Files:**
- Create: `Europa2D/docs/2026-04-01-juno-inference-findings.md`

- [ ] **Step 1: Write findings based on the comprehensive comparison output**

After running the comparison script and viewing the figure, write a structured findings document addressing:

1. **Which hypothesis is best supported?** — Evaluate H1-H5 against the data
2. **What does Juno actually constrain?** — Grain size? Ocean regime? Both? Neither?
3. **Recommended reporting strategy** — What should go in the thesis/paper?
4. **Limitations and caveats** — What biases remain? What can't we resolve?
5. **Future work** — What additional observations would discriminate further?

The findings should include specific numbers from the comparison table, reference the figure panels, and use hedged language appropriate for a thesis.

- [ ] **Step 2: Commit**

```bash
git add Europa2D/docs/2026-04-01-juno-inference-hypotheses.md Europa2D/docs/2026-04-01-juno-inference-findings.md
git commit -m "docs: Juno inference hypothesis framework and findings"
```
