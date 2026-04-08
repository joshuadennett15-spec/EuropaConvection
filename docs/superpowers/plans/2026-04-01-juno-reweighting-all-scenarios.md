# Juno Reweighting All Scenarios — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the 0.6mm grain prior MC for all 4 ocean transport scenarios, then generate a 4-scenario three-way comparison figure (prior vs Juno-reweighted vs 1.5mm grain).

**Architecture:** Three steps — revert grain override, run 3 remaining MC ensembles (Soderlund, Lemasquerier polar, Lemasquerier polar strong) saving to `_grain06mm.npz` filenames, restore grain override, then update the plotting script to produce a 4×4 panel figure.

**Tech Stack:** Python 3.10+, numpy, matplotlib, existing Europa2D MC runner and pub_style.

---

### Task 1: Run 0.6mm MC ensembles for the 3 remaining scenarios

The uniform scenario already exists at `Europa2D/results/mc_2d_uniform_transport_250_grain06mm.npz`. We need the other three.

**Files:**
- Verify: `Europa2D/src/latitude_sampler.py` — confirm the 2D grain override block is NOT present (reverted to 0.6mm audited default)

**Prerequisite check:** The grain override was already reverted earlier in this session. Verify before running.

- [ ] **Step 1: Verify grain override is not present**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
grep -n "d_grain_2d" Europa2D/src/latitude_sampler.py
```

Expected: no matches (override was reverted). If matches found, remove the 3-line block that sets `audited_params['d_grain']`.

- [ ] **Step 2: Run Soderlund equator-enhanced MC (250 samples, 15 workers)**

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
    ocean_pattern='equator_enhanced', q_star=0.4,
    verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_soderlund2014_equator_250_grain06mm.npz')
print('Saved Soderlund 0.6mm results')
"
```

Expected: `Valid: ~245-250/250`, runtime ~10-13 min. File created at path above.

- [ ] **Step 3: Run Lemasquerier polar MC**

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
    ocean_pattern='polar_enhanced', q_star=0.455,
    verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_lemasquerier2023_polar_250_grain06mm.npz')
print('Saved Lemasquerier polar 0.6mm results')
"
```

Expected: `Valid: ~250/250`, runtime ~10-13 min.

- [ ] **Step 4: Run Lemasquerier polar strong MC**

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
    ocean_pattern='polar_enhanced', q_star=0.819,
    verbose=True,
)
results = runner.run()
save_results_2d(results, 'Europa2D/results/mc_2d_lemasquerier2023_polar_strong_250_grain06mm.npz')
print('Saved Lemasquerier polar strong 0.6mm results')
"
```

Expected: `Valid: ~250/250`, runtime ~10-13 min.

- [ ] **Step 5: Verify all 4 files exist**

```bash
ls -la Europa2D/results/mc_2d_*_grain06mm.npz
```

Expected: 4 files, all ~250 valid samples:
- `mc_2d_uniform_transport_250_grain06mm.npz`
- `mc_2d_soderlund2014_equator_250_grain06mm.npz`
- `mc_2d_lemasquerier2023_polar_250_grain06mm.npz`
- `mc_2d_lemasquerier2023_polar_strong_250_grain06mm.npz`

---

### Task 2: Restore 1.5mm grain override in latitude_sampler.py

After the 0.6mm runs complete, restore the grain override so the 1.5mm prior is the active default for future runs.

**Files:**
- Modify: `Europa2D/src/latitude_sampler.py:99-101`

- [ ] **Step 1: Add the grain override back**

In `Europa2D/src/latitude_sampler.py`, after the line `H_rad = audited_params['H_rad']` (around line 101), add:

```python
        # 2D grain-size override: shift prior center from 0.6 mm to 1.5 mm
        # to favour thicker conductive lids (Barr & McKinnon 2007 supports
        # equilibrium grains up to 30-80 mm; 1.5 mm is still conservative).
        d_grain_2d = 10 ** self.rng.normal(np.log10(1.5e-3), 0.35)
        d_grain_2d = float(np.clip(d_grain_2d, 5e-5, 5e-3))
        audited_params['d_grain'] = d_grain_2d
```

- [ ] **Step 2: Verify smoke test passes**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -m pytest Europa2D/tests/test_validation.py -x -q
```

Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add Europa2D/src/latitude_sampler.py
git commit -m "feat: restore 1.5mm grain prior as default for 2D runs"
```

---

### Task 3: Update plotting script for 4-scenario three-way comparison

**Files:**
- Modify: `Europa2D/scripts/plot_prior_vs_posterior_convection.py`

- [ ] **Step 1: Rewrite the plotting script for 4×4 panel layout**

Replace the contents of `Europa2D/scripts/plot_prior_vs_posterior_convection.py` with:

```python
"""
4-scenario three-way comparison: 0.6mm prior vs Juno-reweighted vs 1.5mm prior.

Layout: 4 rows (one per scenario) x 4 columns (H_total, D_cond, Conv%, Nu|conv).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
sys.path.insert(0, _SCRIPT_DIR)

from pub_style import apply_style, PAL, label_panel, save_fig, add_minor_gridlines, DOUBLE_COL

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")

JUNO = 29.0
SIGMA_EFF = np.sqrt(10.0**2 + 3.0**2)

C_PRIOR = "0.55"
C_JUNO = PAL.BLUE
C_GRAIN = PAL.RED

SCENARIOS = [
    ("uniform_transport",             "Uniform transport"),
    ("soderlund2014_equator",         "Equator-enhanced"),
    ("lemasquerier2023_polar",        "Polar-enhanced"),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced"),
]


def _interp_at(lat, profiles, target):
    return np.array([np.interp(target, lat, profiles[i])
                     for i in range(profiles.shape[0])])


def _gaussian_lk(dc):
    return np.exp(-0.5 * ((dc - JUNO) / SIGMA_EFF)**2)


def _weighted_profile(arr, weights):
    n_lat = arr.shape[1]
    med = np.zeros(n_lat)
    lo = np.zeros(n_lat)
    hi = np.zeros(n_lat)
    for j in range(n_lat):
        col = arr[:, j]
        idx = np.argsort(col)
        cw = np.cumsum(weights[idx])
        cw /= cw[-1]
        med[j] = col[idx[np.searchsorted(cw, 0.50)]]
        lo[j] = col[idx[np.searchsorted(cw, 0.16)]]
        hi[j] = col[idx[np.searchsorted(cw, 0.84)]]
    return med, lo, hi


def _weighted_conv_frac(Nu, weights):
    conv = (Nu > 1.1).astype(float)
    return np.sum(weights[:, None] * conv, axis=0)


def _conditional_median(arr, Nu):
    n_lat = arr.shape[1]
    med = np.full(n_lat, np.nan)
    for j in range(n_lat):
        mask = Nu[:, j] > 1.1
        if mask.sum() > 5:
            med[j] = np.median(arr[mask, j])
    return med


def _conditional_weighted_median(arr, Nu, weights):
    n_lat = arr.shape[1]
    med = np.full(n_lat, np.nan)
    for j in range(n_lat):
        mask = Nu[:, j] > 1.1
        if mask.sum() < 5:
            continue
        vals = arr[mask, j]
        w = weights[mask]
        w = w / w.sum()
        idx = np.argsort(vals)
        cw = np.cumsum(w[idx])
        med[j] = vals[idx[np.searchsorted(cw, 0.50)]]
    return med


def main():
    apply_style()

    fig, axes = plt.subplots(
        4, 4,
        figsize=(DOUBLE_COL, DOUBLE_COL * 1.10),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.38)

    col_titles = ["Shell thickness", "Conductive lid", "Convecting fraction", "Convective vigor"]

    for row, (key, title) in enumerate(SCENARIOS):
        # Load 0.6mm and 1.5mm results
        old = dict(np.load(os.path.join(RESULTS_DIR, f"mc_2d_{key}_250_grain06mm.npz"),
                           allow_pickle=True))
        new = dict(np.load(os.path.join(RESULTS_DIR, f"mc_2d_{key}_250.npz"),
                           allow_pickle=True))

        lat = old["latitudes_deg"]
        lat_n = new["latitudes_deg"]

        # Juno importance weights
        dc35 = _interp_at(lat, old["D_cond_profiles"], 35.0)
        lk = _gaussian_lk(dc35)
        w = lk / lk.sum()

        # --- Column 0: H_total ---
        ax = axes[row, 0]
        ax.plot(lat, np.median(old["H_profiles"], axis=0),
                color=C_PRIOR, lw=1.0, ls="--")
        H_w, _, _ = _weighted_profile(old["H_profiles"], w)
        ax.plot(lat, H_w, color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.median(new["H_profiles"], axis=0),
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(15, 65)
        if row == 0:
            ax.set_title(col_titles[0], fontsize=7, fontweight="bold")
        ax.set_ylabel(title, fontsize=6.5, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + row * 4))

        # --- Column 1: D_cond ---
        ax = axes[row, 1]
        ax.plot(lat, np.median(old["D_cond_profiles"], axis=0),
                color=C_PRIOR, lw=1.0, ls="--")
        Dc_w, _, _ = _weighted_profile(old["D_cond_profiles"], w)
        ax.plot(lat, Dc_w, color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.median(new["D_cond_profiles"], axis=0),
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.errorbar(35.0, JUNO, yerr=10.0, fmt="D", ms=3, color=PAL.ORANGE,
                    ecolor=PAL.ORANGE, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=5)
        ax.set_ylim(10, 55)
        if row == 0:
            ax.set_title(col_titles[1], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + row * 4 + 1))

        # --- Column 2: Convecting fraction ---
        ax = axes[row, 2]
        ax.plot(lat, np.mean(old["Nu_profiles"] > 1.1, axis=0) * 100,
                color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _weighted_conv_frac(old["Nu_profiles"], w) * 100,
                color=C_JUNO, lw=1.5)
        ax.plot(lat_n, np.mean(new["Nu_profiles"] > 1.1, axis=0) * 100,
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(0, 70)
        if row == 0:
            ax.set_title(col_titles[2], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + row * 4 + 2))

        # --- Column 3: Conditional Nu ---
        ax = axes[row, 3]
        ax.plot(lat, _conditional_median(old["Nu_profiles"], old["Nu_profiles"]),
                color=C_PRIOR, lw=1.0, ls="--")
        ax.plot(lat, _conditional_weighted_median(old["Nu_profiles"], old["Nu_profiles"], w),
                color=C_JUNO, lw=1.5)
        ax.plot(lat_n, _conditional_median(new["Nu_profiles"], new["Nu_profiles"]),
                color=C_GRAIN, lw=1.0, ls="-.")
        ax.set_ylim(1, 15)
        if row == 0:
            ax.set_title(col_titles[3], fontsize=7, fontweight="bold")
        add_minor_gridlines(ax)
        label_panel(ax, chr(97 + row * 4 + 3))

    # X-axis labels on bottom row only
    for ax in axes[3, :]:
        ax.set_xlabel(r"Latitude ($\degree$)", fontsize=7)
        ax.set_xlim(0, 90)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(30))

    # Shared legend at bottom
    legend_elements = [
        Line2D([0], [0], color=C_PRIOR, lw=1.0, ls="--", label="0.6 mm prior"),
        Line2D([0], [0], color=C_JUNO, lw=1.5, label="0.6 mm + Juno reweighted"),
        Line2D([0], [0], color=C_GRAIN, lw=1.0, ls="-.", label="1.5 mm prior"),
    ]
    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.01),
               columnspacing=2.0, handletextpad=0.5)

    save_fig(fig, "prior_vs_juno_vs_grain_4scenario", FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the plotting script**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python Europa2D/scripts/plot_prior_vs_posterior_convection.py
```

Expected: `Saved: prior_vs_juno_vs_grain_4scenario.{png, pdf}` in `Europa2D/figures/`.

- [ ] **Step 3: Commit**

```bash
git add Europa2D/scripts/plot_prior_vs_posterior_convection.py
git commit -m "feat: 4-scenario Juno reweighting comparison figure"
```

---

### Task 4: Print summary statistics table

- [ ] **Step 1: Run summary comparison across all 4 scenarios**

```bash
cd C:/Users/Joshu/.cursor/projects/EuropaConvection
python -c "
import numpy as np

JUNO = 29.0
SIGMA_EFF = np.sqrt(10.0**2 + 3.0**2)

def interp_at(lat, profiles, target):
    return np.array([np.interp(target, lat, profiles[i]) for i in range(profiles.shape[0])])

def gaussian_lk(dc):
    return np.exp(-0.5 * ((dc - JUNO) / SIGMA_EFF)**2)

SCENARIOS = [
    ('uniform_transport',             'Uniform'),
    ('soderlund2014_equator',         'Equator-enh.'),
    ('lemasquerier2023_polar',        'Polar-enh.'),
    ('lemasquerier2023_polar_strong', 'Strong polar'),
]

header = f\"\"\"{'Scenario':16s} | {'View':20s} | {'Dc(35)':>7s} | {'Conv%eq':>7s} | {'Conv%po':>7s} | {'Nu|eq':>6s} | {'Nu|po':>6s} | {'H_eq':>6s} | {'H_po':>6s}\"\"\"
print('=' * len(header))
print('JUNO REWEIGHTING: ALL SCENARIOS x 3 VIEWS')
print('=' * len(header))
print(header)
print('-' * len(header))

for key, title in SCENARIOS:
    old = dict(np.load(f'Europa2D/results/mc_2d_{key}_250_grain06mm.npz', allow_pickle=True))
    new = dict(np.load(f'Europa2D/results/mc_2d_{key}_250.npz', allow_pickle=True))
    lat = old['latitudes_deg']

    dc35 = interp_at(lat, old['D_cond_profiles'], 35.0)
    lk = gaussian_lk(dc35)
    w = lk / lk.sum()

    for label, data, weights in [
        ('0.6mm prior',   old, None),
        ('0.6mm+Juno',    old, w),
        ('1.5mm prior',   new, None),
    ]:
        H = data['H_profiles']
        Nu = data['Nu_profiles']
        Dc = data['D_cond_profiles']
        lt = data['latitudes_deg']
        dc = interp_at(lt, Dc, 35.0)

        if weights is not None:
            dc_v = float(np.sum(weights * dc))
            conv_eq = float(np.sum(weights * (Nu[:,0] > 1.1)))
            conv_po = float(np.sum(weights * (Nu[:,-1] > 1.1)))
            mask_eq = Nu[:,0] > 1.1
            nu_eq = float(np.sum(weights[mask_eq]*Nu[mask_eq,0])/max(np.sum(weights[mask_eq]),1e-12)) if mask_eq.sum()>5 else 0
            mask_po = Nu[:,-1] > 1.1
            nu_po = float(np.sum(weights[mask_po]*Nu[mask_po,-1])/max(np.sum(weights[mask_po]),1e-12)) if mask_po.sum()>5 else 0
            h_eq = float(np.sum(weights * H[:,0]))
            h_po = float(np.sum(weights * H[:,-1]))
        else:
            dc_v = float(np.median(dc))
            conv_eq = float(np.mean(Nu[:,0] > 1.1))
            conv_po = float(np.mean(Nu[:,-1] > 1.1))
            mask_eq = Nu[:,0] > 1.1
            nu_eq = float(np.median(Nu[mask_eq,0])) if mask_eq.sum()>5 else 0
            mask_po = Nu[:,-1] > 1.1
            nu_po = float(np.median(Nu[mask_po,-1])) if mask_po.sum()>5 else 0
            h_eq = float(np.median(H[:,0]))
            h_po = float(np.median(H[:,-1]))

        print(f'{title:16s} | {label:20s} | {dc_v:6.1f}  | {conv_eq:6.0%}  | {conv_po:6.0%}  | {nu_eq:5.1f}  | {nu_po:5.1f}  | {h_eq:5.1f}  | {h_po:5.1f}')
    print('-' * len(header))
"
```

Expected: A table showing all 4 scenarios × 3 views with consistent formatting.

- [ ] **Step 2: Verify results are physically consistent**

Check that for each scenario:
- 0.6mm+Juno convecting fraction is between 0.6mm prior and 1.5mm prior
- 0.6mm+Juno Nu|convecting is higher than or equal to 1.5mm Nu|convecting
- D_cond at 35 deg follows: 0.6mm < 0.6mm+Juno < 1.5mm
