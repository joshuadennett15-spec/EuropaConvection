# MC Statistical Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-module statistical analysis framework that compares four production MC ensembles and produces thesis-ready tables and figures.

**Architecture:** `thesis_stats.py` computes all statistics and saves structured JSON/CSV; `thesis_figures.py` reads those results and produces publication-quality plots. Both live in `EuropaProjectDJ/scripts/`. Custom implementations of Jonckheere-Terpstra and Kendall's W are inlined (~45 lines total). All other stats use scipy/statsmodels.

**Tech Stack:** Python 3, numpy, scipy, matplotlib, statsmodels (new dep — `QuantReg`, `multipletests`)

**Spec:** `docs/superpowers/specs/2026-03-19-mc-statistical-comparison-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `EuropaProjectDJ/scripts/thesis_stats.py` | All statistical computation (Blocks 1-6), JSON/CSV export |
| Create | `EuropaProjectDJ/scripts/thesis_figures.py` | All 7 publication figures from saved results |
| Create | `EuropaProjectDJ/tests/test_thesis_stats.py` | Unit tests for custom stats + smoke test |
| Reuse | `EuropaProjectDJ/scripts/pub_style.py` | Existing Wong 2011 palette + AGU column widths |
| Reuse | `EuropaProjectDJ/results/mc_15000_optionA_v2_andrade.npz` | Global Audited archive |
| Reuse | `EuropaProjectDJ/results/eq_baseline_andrade.npz` | Equatorial 1.0x archive |
| Reuse | `EuropaProjectDJ/results/eq_moderate_andrade.npz` | Equatorial 1.2x archive |
| Reuse | `EuropaProjectDJ/results/eq_strong_andrade.npz` | Equatorial 1.5x archive |
| Output | `EuropaProjectDJ/results/thesis_stats/` | JSON, CSV results |
| Output | `EuropaProjectDJ/figures/thesis_stats/` | PNG + PDF figures |

---

## Task 0: Install statsmodels

**Files:**
- None (environment setup)

- [ ] **Step 1: Install statsmodels**

Run: `pip install statsmodels`

- [ ] **Step 2: Verify import**

Run: `python -c "from statsmodels.regression.quantile_regression import QuantReg; from statsmodels.stats.multitest import multipletests; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit** — skip, no code change

---

## Task 1: load_scenario + zero-variance filter

**Files:**
- Create: `EuropaProjectDJ/scripts/thesis_stats.py`
- Create: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write the failing test for zero-variance filter**

In `EuropaProjectDJ/tests/test_thesis_stats.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest
from thesis_stats import load_scenario


def test_load_scenario_filters_zero_variance_params(tmp_path):
    """Zero-variance columns are excluded; varying columns are kept."""
    npz_path = tmp_path / "fake.npz"
    np.savez(
        npz_path,
        thicknesses_km=np.random.default_rng(0).normal(30, 5, 100),
        D_cond_km=np.random.default_rng(0).normal(15, 3, 100),
        D_conv_km=np.random.default_rng(0).normal(15, 3, 100),
        lid_fractions=np.random.default_rng(0).uniform(0.3, 1.0, 100),
        Ra_values=np.random.default_rng(0).lognormal(8, 2, 100),
        Nu_values=np.random.default_rng(0).lognormal(1, 0.5, 100),
        param_good=np.random.default_rng(0).normal(10, 1, 100),
        param_constant=np.full(100, 42.0),
        param_tiny_constant=np.full(100, 1e-12),
    )
    data = load_scenario(str(npz_path))
    assert "good" in data["params"]
    assert "constant" not in data["params"]
    assert "tiny_constant" not in data["params"]
    assert "thickness_km" in data["qois"]
    assert "lid_fraction" in data["qois"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py::test_load_scenario_filters_zero_variance_params -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'thesis_stats'`

- [ ] **Step 3: Write load_scenario**

In `EuropaProjectDJ/scripts/thesis_stats.py`:

```python
"""
Statistical comparison of Europa MC ensembles for thesis chapter.

Spec: docs/superpowers/specs/2026-03-19-mc-statistical-comparison-design.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# NPZ key -> internal name
_QOI_KEY_MAP = {
    "thicknesses_km": "thickness_km",
    "D_cond_km": "D_cond_km",
    "D_conv_km": "D_conv_km",
    "lid_fractions": "lid_fraction",
    "Ra_values": "Ra",
    "Nu_values": "Nu",
}

# Whole-population QoIs for Block 2 pairwise tests
WHOLE_POP_QOIS = ("thickness_km", "D_cond_km", "lid_fraction")

# QoIs only meaningful for convective subpopulation
CONV_ONLY_QOIS = ("D_conv_km", "Ra", "Nu")

# All QoIs
ALL_QOIS = WHOLE_POP_QOIS + CONV_ONLY_QOIS

# JT-eligible QoIs (continuous, no heavy boundary mass)
JT_QOIS = ("thickness_km", "D_cond_km", "lid_fraction")

# Conductive threshold (see spec for rationale)
LID_FRACTION_COND_THRESHOLD = 0.999

# Default scenario paths relative to results/
DEFAULT_SCENARIOS = {
    "Global Audited": "mc_15000_optionA_v2_andrade.npz",
    "Eq Baseline": "eq_baseline_andrade.npz",
    "Eq Moderate": "eq_moderate_andrade.npz",
    "Eq Strong": "eq_strong_andrade.npz",
}


def load_scenario(path: str) -> Dict[str, Any]:
    """Load an NPZ archive and return standardized qois, params, metadata."""
    d = np.load(path, allow_pickle=True)

    qois = {}
    for npz_key, internal_name in _QOI_KEY_MAP.items():
        if npz_key in d:
            qois[internal_name] = np.asarray(d[npz_key], dtype=float)

    # Auto-detect and filter zero-variance parameters
    params = {}
    excluded = []
    for key in sorted(d.keys()):
        if not key.startswith("param_"):
            continue
        arr = np.asarray(d[key], dtype=float)
        name = key[len("param_"):]
        mean_abs = max(abs(np.mean(arr)), 1e-30)
        rel_std = np.std(arr) / mean_abs
        if rel_std < 1e-6:
            excluded.append(name)
            continue
        params[name] = arr

    if excluded:
        logger.info("Excluded zero-variance params from %s: %s", path, excluded)

    metadata = {
        "path": str(path),
        "n_valid": len(next(iter(qois.values()))),
        "excluded_params": excluded,
    }
    # Route eq_enhancement to metadata if present
    if "param_eq_enhancement" in d:
        metadata["eq_enhancement"] = float(d["param_eq_enhancement"][0])

    return {"qois": qois, "params": params, "metadata": metadata}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py::test_load_scenario_filters_zero_variance_params -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add load_scenario with zero-variance filter"
```

---

## Task 2: Custom statistics — Cliff's delta, Jonckheere-Terpstra, Kendall's W

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing tests for all three custom stats**

Append to `test_thesis_stats.py`:

```python
from thesis_stats import _cliffs_delta_from_u, _jonckheere_terpstra, _kendall_w


def test_cliffs_delta_identical_distributions():
    a = np.arange(100, dtype=float)
    u_stat, _ = stats.mannwhitneyu(a, a, alternative="two-sided")
    d = _cliffs_delta_from_u(u_stat, len(a), len(a))
    assert abs(d) < 0.1


def test_cliffs_delta_fully_separated():
    a = np.arange(100, dtype=float)
    b = np.arange(200, 300, dtype=float)
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = _cliffs_delta_from_u(u_stat, len(a), len(b))
    assert d < -0.9  # a is smaller than b


def test_jt_monotonic_shift():
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 500), rng.normal(1, 1, 500), rng.normal(2, 1, 500)]
    j_stat, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value < 0.001


def test_jt_identical_groups():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 500)
    groups = [base.copy(), base.copy(), base.copy()]
    _, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value > 0.05


def test_jt_reversed_order_one_sided():
    rng = np.random.default_rng(42)
    groups = [rng.normal(2, 1, 500), rng.normal(1, 1, 500), rng.normal(0, 1, 500)]
    _, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value > 0.5


def test_jt_tie_correction_differs():
    """Heavy ties should produce different variance than no-tie formula."""
    rng = np.random.default_rng(42)
    # Groups with many tied values (discretized)
    groups = [
        np.round(rng.normal(0, 1, 300), 0),
        np.round(rng.normal(1, 1, 300), 0),
        np.round(rng.normal(2, 1, 300), 0),
    ]
    j_stat, p_value = _jonckheere_terpstra(groups, alternative="increasing")
    assert p_value < 0.05  # still detects the trend despite ties


def test_kendall_w_identical_rankings():
    rankings = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    w, _, _ = _kendall_w(rankings)
    assert abs(w - 1.0) < 1e-10


def test_kendall_w_random_rankings():
    rng = np.random.default_rng(42)
    rankings = np.array([rng.permutation(10) + 1 for _ in range(4)])
    w, _, _ = _kendall_w(rankings)
    assert 0.0 <= w <= 1.0
    assert w < 0.5  # random rankings should have low concordance
```

Add `from scipy import stats` to the test file imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "cliffs or jt_ or kendall" -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the three custom stats**

Append to `thesis_stats.py`:

```python
def _cliffs_delta_from_u(U: float, n1: int, n2: int) -> float:
    """Cliff's delta derived from Mann-Whitney U. Range [-1, 1]."""
    return (2.0 * U) / (n1 * n2) - 1.0


def _jonckheere_terpstra(
    groups: Sequence[np.ndarray], alternative: str = "two-sided"
) -> tuple:
    """
    Jonckheere-Terpstra test for ordered alternatives.

    Tie-corrected asymptotic normal approximation (Hollander & Wolfe 1999).
    Returns (J_statistic, p_value).
    """
    k = len(groups)
    ns = [len(g) for g in groups]
    N = sum(ns)

    # J = sum of U_{ij} for i < j
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            u_stat, _ = stats.mannwhitneyu(
                groups[i], groups[j], alternative="two-sided"
            )
            J += u_stat

    # Expected value under H0
    N_sq_sum = sum(n * n for n in ns)
    E_J = (N * N - N_sq_sum) / 4.0

    # Tie-corrected variance (Hollander & Wolfe 1999, eq. 6.18)
    all_values = np.concatenate(groups)
    _, tie_counts = np.unique(all_values, return_counts=True)

    # Terms for variance
    A = (N * (N - 1) * (2 * N + 5)
         - sum(n * (n - 1) * (2 * n + 5) for n in ns)
         - sum(int(t) * (int(t) - 1) * (2 * int(t) + 5) for t in tie_counts))

    B = (sum(n * (n - 1) * (n - 2) for n in ns)
         * sum(int(t) * (int(t) - 1) * (int(t) - 2) for t in tie_counts))

    C = (sum(n * (n - 1) for n in ns)
         * sum(int(t) * (int(t) - 1) for t in tie_counts))

    Var_J = A / 72.0 + B / (36.0 * N * (N - 1) * (N - 2)) + C / (8.0 * N * (N - 1))
    Var_J = max(Var_J, 1e-30)

    z = (J - E_J) / np.sqrt(Var_J)

    if alternative == "increasing":
        p_value = 1.0 - stats.norm.cdf(z)
    elif alternative == "decreasing":
        p_value = stats.norm.cdf(z)
    else:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return float(J), float(p_value)


def _kendall_w(rankings: np.ndarray) -> tuple:
    """
    Kendall's W (coefficient of concordance).

    rankings: (k_raters, n_items) array of ranks.
    Returns (W, chi2, p_value).
    """
    k, n = rankings.shape
    rank_sums = rankings.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    SS = float(np.sum((rank_sums - mean_rank_sum) ** 2))
    W = (12.0 * SS) / (k * k * (n ** 3 - n))
    chi2 = k * (n - 1) * W
    p_value = 1.0 - stats.chi2.cdf(chi2, df=n - 1)
    return float(W), float(chi2), float(p_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "cliffs or jt_ or kendall" -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Cliff's delta, Jonckheere-Terpstra, Kendall's W"
```

---

## Task 3: Block 1 — descriptive_summary

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing test**

```python
from thesis_stats import descriptive_summary, LID_FRACTION_COND_THRESHOLD


def test_descriptive_summary_known_array():
    data = {
        "qois": {"thickness_km": np.arange(1.0, 101.0)},
        "params": {},
        "metadata": {"n_valid": 100},
    }
    result = descriptive_summary(data, n_boot=500)
    t = result["thickness_km"]
    assert abs(t["median"] - 50.5) < 0.01
    assert abs(t["P25"] - 25.75) < 0.01
    assert abs(t["P75"] - 75.25) < 0.01
    assert "ci_median_low" in t
    assert "ci_median_high" in t
    assert t["ci_median_low"] < t["median"] < t["ci_median_high"]


def test_descriptive_summary_conductive_fraction():
    lid = np.concatenate([np.full(40, 1.0), np.linspace(0.3, 0.95, 60)])
    data = {
        "qois": {"lid_fraction": lid},
        "params": {},
        "metadata": {"n_valid": 100},
    }
    result = descriptive_summary(data, n_boot=500)
    assert abs(result["conductive_fraction"] - 0.40) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "descriptive" -v`
Expected: FAIL

- [ ] **Step 3: Implement descriptive_summary**

Add to `thesis_stats.py`:

```python
from scipy.signal import savgol_filter


def _bca_ci(data: np.ndarray, stat_func, n_boot: int = 10000,
            alpha: float = 0.05, rng=None) -> tuple:
    """BCa bootstrap confidence interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(data)
    observed = stat_func(data)

    boot_stats = np.array([
        stat_func(data[rng.integers(0, n, n)]) for _ in range(n_boot)
    ])

    # Bias correction (clip to avoid -inf/+inf at edges)
    prop = np.clip(np.mean(boot_stats < observed), 0.5 / n_boot, 1.0 - 0.5 / n_boot)
    z0 = stats.norm.ppf(prop)

    # Acceleration (jackknife, subsampled for large n to avoid O(n^2))
    jack_n = min(n, 2000)
    jack_idx = rng.choice(n, jack_n, replace=False) if jack_n < n else np.arange(n)
    jackknife = np.array([stat_func(np.delete(data, i)) for i in jack_idx])
    jack_mean = jackknife.mean()
    num = np.sum((jack_mean - jackknife) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2.0)
    z_1alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
    p_high = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1.0 - a * (z0 + z_1alpha)))

    p_low = np.clip(p_low, 0.5 / n_boot, 1.0 - 0.5 / n_boot)
    p_high = np.clip(p_high, 0.5 / n_boot, 1.0 - 0.5 / n_boot)

    ci_low = float(np.percentile(boot_stats, 100 * p_low))
    ci_high = float(np.percentile(boot_stats, 100 * p_high))
    return ci_low, ci_high


def _cbe_savgol(data: np.ndarray) -> float:
    """Current Best Estimate = mode of Savitzky-Golay smoothed PDF."""
    if len(data) < 50:
        return float(np.median(data))
    n_bins = min(80, max(20, len(data) // 200))
    counts, edges = np.histogram(data, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    window = min(len(counts) - 1, 7)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return float(centers[np.argmax(counts)])
    smoothed = savgol_filter(counts, window, 2)
    return float(centers[np.argmax(smoothed)])


def descriptive_summary(data: Mapping[str, Any],
                        n_boot: int = 10000) -> Dict[str, Any]:
    """Block 1: descriptive statistics for all QoIs in data."""
    rng = np.random.default_rng(0)
    result: Dict[str, Any] = {}

    for name, arr in data["qois"].items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        ci_median = _bca_ci(finite, np.median, n_boot, rng=rng)
        ci_mean = _bca_ci(finite, np.mean, n_boot, rng=rng)
        entry = {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "cbe": _cbe_savgol(finite),
            "std": float(np.std(finite, ddof=1)),
            "P5": float(np.percentile(finite, 5)),
            "P16": float(np.percentile(finite, 16)),
            "P25": float(np.percentile(finite, 25)),
            "P50": float(np.percentile(finite, 50)),
            "P75": float(np.percentile(finite, 75)),
            "P84": float(np.percentile(finite, 84)),
            "P95": float(np.percentile(finite, 95)),
            "IQR": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
            "skewness": float(stats.skew(finite)),
            "kurtosis": float(stats.kurtosis(finite)),
            "ci_median_low": ci_median[0],
            "ci_median_high": ci_median[1],
            "ci_mean_low": ci_mean[0],
            "ci_mean_high": ci_mean[1],
            "n": len(finite),
        }
        # log10 stats for Ra and Nu
        if name in ("Ra", "Nu"):
            log_arr = np.log10(finite[finite > 0])
            if len(log_arr) > 0:
                entry["log10_mean"] = float(np.mean(log_arr))
                entry["log10_median"] = float(np.median(log_arr))
                entry["log10_std"] = float(np.std(log_arr, ddof=1))
        result[name] = entry

    # Conductive fraction
    if "lid_fraction" in data["qois"]:
        lf = data["qois"]["lid_fraction"]
        cond_frac = float(np.mean(lf >= LID_FRACTION_COND_THRESHOLD))
        ci = _bca_ci(
            (lf >= LID_FRACTION_COND_THRESHOLD).astype(float),
            np.mean, n_boot, rng=rng
        )
        result["conductive_fraction"] = cond_frac
        result["conductive_fraction_ci"] = {"low": ci[0], "high": ci[1]}

    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "descriptive" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 1 descriptive_summary with BCa bootstrap"
```

---

## Task 4: Block 2 — pairwise_comparison + fdr_correct

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing tests**

```python
from thesis_stats import pairwise_comparison, fdr_correct, WHOLE_POP_QOIS


def test_pairwise_identical_distributions():
    rng = np.random.default_rng(42)
    arr = rng.normal(30, 5, 1000)
    data_a = {"qois": {"thickness_km": arr.copy()}}
    data_b = {"qois": {"thickness_km": arr.copy()}}
    result = pairwise_comparison(data_a, data_b, ["thickness_km"])
    t = result["thickness_km"]
    assert t["ks_D"] < 0.05
    assert t["ks_p"] > 0.1
    assert abs(t["cliff_d"]) < 0.1
    assert abs(t["cohen_d"]) < 0.1


def test_pairwise_separated_distributions():
    rng = np.random.default_rng(42)
    data_a = {"qois": {"thickness_km": rng.normal(20, 2, 1000)}}
    data_b = {"qois": {"thickness_km": rng.normal(40, 2, 1000)}}
    result = pairwise_comparison(data_a, data_b, ["thickness_km"])
    t = result["thickness_km"]
    assert t["ks_D"] > 0.8
    assert t["ks_p"] < 1e-10
    assert abs(t["cliff_d"]) > 0.9
    assert abs(t["cohen_d"]) > 5.0


def test_fdr_correct_known_pvalues():
    from statsmodels.stats.multitest import multipletests
    raw_p = [0.001, 0.01, 0.03, 0.04, 0.8, 0.9]
    results = {
        f"pair_{i}": {"thickness_km": {"ks_p": p}}
        for i, p in enumerate(raw_p)
    }
    corrected = fdr_correct(results, qoi="thickness_km", p_key="ks_p")
    _, expected, _, _ = multipletests(raw_p, method="fdr_bh")
    for i, pair_key in enumerate(sorted(corrected)):
        np.testing.assert_allclose(
            corrected[pair_key]["thickness_km"]["ks_p_fdr"],
            expected[i], rtol=1e-10,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "pairwise or fdr" -v`
Expected: FAIL

- [ ] **Step 3: Implement pairwise_comparison and fdr_correct**

Add to `thesis_stats.py`:

```python
from statsmodels.stats.multitest import multipletests


def pairwise_comparison(
    data_a: Mapping[str, Any],
    data_b: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 2: KS, Mann-Whitney, Cliff's delta, Cohen's d for each QoI."""
    result = {}
    for qoi in qois:
        a = data_a["qois"][qoi]
        b = data_b["qois"][qoi]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        ks_D, ks_p = stats.ks_2samp(a, b)
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        # Note: r_rb = 1 - 2U/(n1*n2) and cliff_d = 2U/(n1*n2) - 1,
        # so r_rb = -cliff_d by definition. Both are reported; cliff_d > 0
        # means sample A tends to exceed sample B.
        r_rb = 1.0 - (2.0 * u_stat) / (n1 * n2)
        cliff_d = _cliffs_delta_from_u(u_stat, n1, n2)
        pooled_std = np.sqrt(
            (np.var(a, ddof=1) * (n1 - 1) + np.var(b, ddof=1) * (n2 - 1))
            / (n1 + n2 - 2)
        )
        cohen_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

        result[qoi] = {
            "ks_D": float(ks_D),
            "ks_p": float(ks_p),
            "mw_U": float(u_stat),
            "mw_p": float(u_p),
            "r_rb": float(r_rb),
            "cliff_d": float(cliff_d),
            "cohen_d": float(cohen_d),
            "n_a": n1,
            "n_b": n2,
        }
    return result


def fdr_correct(
    pairwise_results: Mapping[str, Any],
    qoi: str,
    p_key: str = "ks_p",
) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction within a QoI."""
    pair_keys = sorted(pairwise_results.keys())
    raw_p = [pairwise_results[pk][qoi][p_key] for pk in pair_keys]
    _, corrected_p, _, _ = multipletests(raw_p, method="fdr_bh")

    out = {}
    for i, pk in enumerate(pair_keys):
        entry = dict(pairwise_results[pk][qoi])
        entry[f"{p_key}_fdr"] = float(corrected_p[i])
        out[pk] = {qoi: entry}
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "pairwise or fdr" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 2 pairwise comparison with FDR correction"
```

---

## Task 5: Block 3 — enhancement_trend

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing test**

```python
from thesis_stats import enhancement_trend


def test_enhancement_trend_detects_monotonic_decrease():
    rng = np.random.default_rng(42)
    scenarios = [
        {"qois": {"thickness_km": rng.normal(40, 5, 1000)}, "enhancement": 1.0},
        {"qois": {"thickness_km": rng.normal(30, 5, 1000)}, "enhancement": 1.2},
        {"qois": {"thickness_km": rng.normal(20, 5, 1000)}, "enhancement": 1.5},
    ]
    result = enhancement_trend(scenarios, ["thickness_km"])
    t = result["thickness_km"]
    assert t["kw_p"] < 0.001
    assert t["jt_p"] < 0.001
    assert t["qr_slope_P50"] < 0  # thickness decreases with enhancement
    assert t["qr_slope_P50_ci_low"] < t["qr_slope_P50"] < t["qr_slope_P50_ci_high"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "enhancement" -v`
Expected: FAIL

- [ ] **Step 3: Implement enhancement_trend**

Add to `thesis_stats.py`:

```python
from statsmodels.regression.quantile_regression import QuantReg


def enhancement_trend(
    eq_scenarios: Sequence[Mapping[str, Any]],
    qois: Sequence[str],
    n_boot_qr: int = 5000,
) -> Dict[str, Any]:
    """Block 3: Kruskal-Wallis, JT, and quantile regression for enhancement sweep."""
    rng = np.random.default_rng(0)
    result = {}

    enhancements = [s["enhancement"] for s in eq_scenarios]
    groups_by_qoi = {
        qoi: [s["qois"][qoi][np.isfinite(s["qois"][qoi])] for s in eq_scenarios]
        for qoi in qois
    }

    for qoi in qois:
        groups = groups_by_qoi[qoi]
        entry: Dict[str, Any] = {}

        # Kruskal-Wallis
        kw_H, kw_p = stats.kruskal(*groups)
        entry["kw_H"] = float(kw_H)
        entry["kw_p"] = float(kw_p)

        # Jonckheere-Terpstra (only for JT-eligible QoIs)
        if qoi in JT_QOIS:
            if qoi == "lid_fraction":
                alt = "increasing"
            elif qoi == "D_cond_km":
                alt = "two-sided"
            else:
                alt = "decreasing"
            jt_J, jt_p = _jonckheere_terpstra(groups, alternative=alt)
            entry["jt_J"] = jt_J
            entry["jt_p"] = jt_p
            entry["jt_alternative"] = alt

        # Quantile regression — descriptive slopes with bootstrap CIs
        y_all = np.concatenate(groups)
        x_all = np.concatenate([
            np.full(len(g), enh) for g, enh in zip(groups, enhancements)
        ])
        X = np.column_stack([np.ones_like(x_all), x_all])

        for tau_label, tau in [("P5", 0.05), ("P50", 0.50), ("P95", 0.95)]:
            try:
                model = QuantReg(y_all, X)
                fit = model.fit(q=tau, max_iter=1000)
                slope = float(fit.params[1])
            except Exception:
                slope = np.nan

            # Bootstrap CI on slope
            boot_slopes = []
            for _ in range(n_boot_qr):
                idx = [rng.choice(len(g), len(g), replace=True) for g in groups]
                y_b = np.concatenate([g[i] for g, i in zip(groups, idx)])
                x_b = np.concatenate([
                    np.full(len(i), enh) for i, enh in zip(idx, enhancements)
                ])
                X_b = np.column_stack([np.ones_like(x_b), x_b])
                try:
                    fit_b = QuantReg(y_b, X_b).fit(q=tau, max_iter=500)
                    boot_slopes.append(float(fit_b.params[1]))
                except Exception:
                    pass

            if boot_slopes:
                ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
            else:
                ci_low, ci_high = np.nan, np.nan

            entry[f"qr_slope_{tau_label}"] = slope
            entry[f"qr_slope_{tau_label}_ci_low"] = float(ci_low)
            entry[f"qr_slope_{tau_label}_ci_high"] = float(ci_high)

        result[qoi] = entry
    return result
```

- [ ] **Step 4: Run test**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "enhancement" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 3 enhancement trend (KW, JT, quantile regression)"
```

---

## Task 6: Block 4 — parameter_ranking + ranking_concordance

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing test**

```python
from thesis_stats import parameter_ranking, ranking_concordance


def test_parameter_ranking_detects_dominant_control():
    rng = np.random.default_rng(42)
    n = 1000
    x_strong = rng.uniform(0, 1, n)
    x_weak = rng.uniform(0, 1, n)
    y = 10 * x_strong + 0.1 * x_weak + rng.normal(0, 0.5, n)
    data = {
        "qois": {"thickness_km": y},
        "params": {"strong": x_strong, "weak": x_weak},
        "metadata": {"n_valid": n},
    }
    result = parameter_ranking(data, ["thickness_km"])
    ranks = result["thickness_km"]
    # "strong" should have higher |rho| than "weak"
    assert abs(ranks["strong"]["rho"]) > abs(ranks["weak"]["rho"])
    assert ranks["strong"]["significant"]


def test_ranking_concordance_perfect_agreement():
    rankings = {
        "scenario_a": {"thickness_km": {"p1": {"rank": 1}, "p2": {"rank": 2}}},
        "scenario_b": {"thickness_km": {"p1": {"rank": 1}, "p2": {"rank": 2}}},
    }
    result = ranking_concordance(rankings, ["thickness_km"])
    assert result["thickness_km"]["W"] > 0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "ranking" -v`
Expected: FAIL

- [ ] **Step 3: Implement parameter_ranking and ranking_concordance**

Add to `thesis_stats.py`:

```python
def parameter_ranking(
    data: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 4: Spearman rank correlations for each param -> QoI."""
    n_params = len(data["params"])

    result = {}
    for qoi in qois:
        y = data["qois"][qoi]
        qoi_result = {}
        for param_name, x in data["params"].items():
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 10:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            qoi_result[param_name] = {
                "rho": float(rho),
                "p": float(p),
                "significant": p < (0.001 / n_params),
            }
        # Assign ranks by |rho|
        sorted_params = sorted(
            qoi_result.keys(), key=lambda k: abs(qoi_result[k]["rho"]), reverse=True
        )
        for rank, param_name in enumerate(sorted_params, start=1):
            qoi_result[param_name]["rank"] = rank
        result[qoi] = qoi_result
    return result


def ranking_concordance(
    rankings_by_scenario: Mapping[str, Any],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """Block 4: Kendall's W across scenarios for each QoI."""
    scenario_names = list(rankings_by_scenario.keys())
    result = {}
    for qoi in qois:
        # Collect parameter names present in all scenarios
        all_param_sets = [
            set(rankings_by_scenario[s][qoi].keys()) for s in scenario_names
        ]
        common_params = sorted(set.intersection(*all_param_sets))
        if len(common_params) < 2:
            continue
        # Build rankings matrix: (k_raters, n_items)
        matrix = np.array([
            [rankings_by_scenario[s][qoi][p]["rank"] for p in common_params]
            for s in scenario_names
        ], dtype=float)
        W, chi2, p = _kendall_w(matrix)
        result[qoi] = {
            "W": W,
            "chi2": chi2,
            "p": p,
            "n_params": len(common_params),
            "n_scenarios": len(scenario_names),
        }
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "ranking" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 4 parameter ranking with Kendall's W concordance"
```

---

## Task 7: Block 5 — bootstrap_convergence

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing test**

```python
from thesis_stats import bootstrap_convergence


def test_bootstrap_convergence_ci_narrows():
    rng = np.random.default_rng(42)
    data = {
        "qois": {"thickness_km": rng.normal(30, 5, 15000)},
        "params": {},
        "metadata": {"n_valid": 15000},
    }
    result = bootstrap_convergence(data, ["thickness_km"], [500, 5000, 14997])
    t = result["thickness_km"]
    # CI should narrow with more samples
    ci_width_500 = t[500]["ci_median_high"] - t[500]["ci_median_low"]
    ci_width_14997 = t[14997]["ci_median_high"] - t[14997]["ci_median_low"]
    assert ci_width_500 > ci_width_14997
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "convergence" -v`
Expected: FAIL

- [ ] **Step 3: Implement bootstrap_convergence**

Add to `thesis_stats.py`:

```python
# Cap for consistent cross-scenario comparison
CONVERGENCE_SCHEDULE = [500, 1000, 2000, 5000, 10000, 14997]


def bootstrap_convergence(
    data: Mapping[str, Any],
    qois: Sequence[str],
    subsample_sizes: Optional[Sequence[int]] = None,
    n_boot: int = 5000,
) -> Dict[str, Any]:
    """Block 5: bootstrap CI width vs sample size."""
    if subsample_sizes is None:
        subsample_sizes = CONVERGENCE_SCHEDULE
    rng = np.random.default_rng(0)
    result: Dict[str, Any] = {}

    for qoi in qois:
        arr = data["qois"][qoi]
        arr = arr[np.isfinite(arr)]
        qoi_result = {}
        for n_sub in subsample_sizes:
            if n_sub > len(arr):
                continue
            sub = rng.choice(arr, n_sub, replace=False)
            ci_median = _bca_ci(sub, np.median, n_boot, rng=rng)
            ci_p5 = _bca_ci(sub, lambda x: np.percentile(x, 5), n_boot, rng=rng)
            ci_p95 = _bca_ci(sub, lambda x: np.percentile(x, 95), n_boot, rng=rng)
            qoi_result[n_sub] = {
                "median": float(np.median(sub)),
                "ci_median_low": ci_median[0],
                "ci_median_high": ci_median[1],
                "ci_median_width": ci_median[1] - ci_median[0],
                "P5": float(np.percentile(sub, 5)),
                "ci_P5_low": ci_p5[0],
                "ci_P5_high": ci_p5[1],
                "P95": float(np.percentile(sub, 95)),
                "ci_P95_low": ci_p95[0],
                "ci_P95_high": ci_p95[1],
            }
        # Determine sample size where CI width < 1 km
        threshold_n = None
        for n_sub in sorted(qoi_result.keys()):
            if qoi_result[n_sub]["ci_median_width"] < 1.0:
                threshold_n = n_sub
                break
        qoi_result["convergence_threshold_n"] = threshold_n
        result[qoi] = qoi_result

    # Conductive fraction convergence
    if "lid_fraction" in data["qois"]:
        lf = data["qois"]["lid_fraction"]
        cond = (lf >= LID_FRACTION_COND_THRESHOLD).astype(float)
        cf_result = {}
        for n_sub in subsample_sizes:
            if n_sub > len(cond):
                continue
            sub = rng.choice(cond, n_sub, replace=False)
            ci = _bca_ci(sub, np.mean, n_boot, rng=rng)
            cf_result[n_sub] = {
                "fraction": float(np.mean(sub)),
                "ci_low": ci[0],
                "ci_high": ci[1],
                "ci_width": ci[1] - ci[0],
            }
        result["conductive_fraction"] = cf_result

    return result
```

- [ ] **Step 4: Run test**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "convergence" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 5 bootstrap convergence"
```

---

## Task 8: Block 6 — shell_structure

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write failing test**

```python
from thesis_stats import shell_structure


def test_shell_structure_splits_subpopulations():
    rng = np.random.default_rng(42)
    n = 2000
    lid = np.concatenate([np.full(800, 1.0), rng.uniform(0.3, 0.95, 1200)])
    data_by_scenario = {
        "A": {
            "qois": {
                "thickness_km": rng.normal(30, 5, n),
                "D_cond_km": rng.normal(15, 3, n),
                "D_conv_km": np.where(lid < 0.999, rng.uniform(1, 20, n), 0.0),
                "lid_fraction": lid,
                "Ra": np.where(lid < 0.999, rng.lognormal(8, 2, n), 10.0),
                "Nu": np.where(lid < 0.999, rng.lognormal(1, 0.5, n), 1.0),
            },
            "params": {},
            "metadata": {"n_valid": n},
        },
        "B": {
            "qois": {
                "thickness_km": rng.normal(25, 5, n),
                "D_cond_km": rng.normal(12, 3, n),
                "D_conv_km": np.where(lid < 0.999, rng.uniform(1, 15, n), 0.0),
                "lid_fraction": lid,
                "Ra": np.where(lid < 0.999, rng.lognormal(7, 2, n), 10.0),
                "Nu": np.where(lid < 0.999, rng.lognormal(0.8, 0.5, n), 1.0),
            },
            "params": {},
            "metadata": {"n_valid": n},
        },
    }
    result = shell_structure(data_by_scenario)
    assert "whole_pop" in result
    assert "convective_subpop" in result
    assert "conductive_fractions" in result
    # Check convective subpop pairwise exists
    assert "A_vs_B" in result["convective_subpop"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "shell_structure" -v`
Expected: FAIL

- [ ] **Step 3: Implement shell_structure**

Add to `thesis_stats.py`:

```python
def shell_structure(
    data_by_scenario: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Block 6: shell structure partitioning with subpopulation analysis."""
    scenario_names = list(data_by_scenario.keys())
    result: Dict[str, Any] = {
        "whole_pop": {},
        "convective_subpop": {},
        "conductive_fractions": {},
        "dcond_dconv_correlation": {},
    }

    # Conductive fractions with bootstrap CIs
    rng = np.random.default_rng(0)
    for name, data in data_by_scenario.items():
        lf = data["qois"]["lid_fraction"]
        frac = float(np.mean(lf >= LID_FRACTION_COND_THRESHOLD))
        ci = _bca_ci(
            (lf >= LID_FRACTION_COND_THRESHOLD).astype(float),
            np.mean, 5000, rng=rng,
        )
        result["conductive_fractions"][name] = {
            "fraction": frac, "ci_low": ci[0], "ci_high": ci[1],
        }

    # Whole-population pairwise (lid_fraction only — Block 2 handles thickness/D_cond)
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            result["whole_pop"][pair_key] = pairwise_comparison(
                data_by_scenario[name_a],
                data_by_scenario[name_b],
                ["lid_fraction"],
            )

    # Convective subpopulation
    conv_data = {}
    for name, data in data_by_scenario.items():
        mask = data["qois"]["lid_fraction"] < LID_FRACTION_COND_THRESHOLD
        conv_data[name] = {
            "qois": {
                qoi: data["qois"][qoi][mask]
                for qoi in ALL_QOIS
                if qoi in data["qois"]
            },
        }

    conv_qois = ["thickness_km", "D_cond_km", "D_conv_km", "Ra", "Nu"]
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            result["convective_subpop"][pair_key] = pairwise_comparison(
                conv_data[name_a], conv_data[name_b], conv_qois,
            )

    # FDR correction on convective subpop pairwise tests (same scope as Block 2)
    subpop_pairs = {k: v for k, v in result["convective_subpop"].items() if "_vs_" in k}
    for qoi in conv_qois:
        for p_key in ("ks_p", "mw_p"):
            corrected = fdr_correct(subpop_pairs, qoi, p_key)
            for pk, entry in corrected.items():
                result["convective_subpop"][pk][qoi].update(entry[qoi])

    # Ra/Nu conditional stats per scenario
    for name, cd in conv_data.items():
        for qoi in ("Ra", "Nu"):
            arr = cd["qois"].get(qoi, np.array([]))
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                continue
            log_arr = np.log10(finite[finite > 0])
            result["convective_subpop"].setdefault(f"{name}_stats", {})[qoi] = {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "IQR": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
                "log10_mean": float(np.mean(log_arr)) if len(log_arr) > 0 else None,
                "log10_median": float(np.median(log_arr)) if len(log_arr) > 0 else None,
                "log10_IQR": float(np.percentile(log_arr, 75) - np.percentile(log_arr, 25)) if len(log_arr) > 0 else None,
            }

    # D_cond vs D_conv Pearson r per scenario
    for name, cd in conv_data.items():
        dc = cd["qois"].get("D_cond_km", np.array([]))
        dv = cd["qois"].get("D_conv_km", np.array([]))
        mask = np.isfinite(dc) & np.isfinite(dv)
        if np.sum(mask) > 10:
            r, p = stats.pearsonr(dc[mask], dv[mask])
            result["dcond_dconv_correlation"][name] = {
                "pearson_r": float(r), "p": float(p),
            }

    return result
```

- [ ] **Step 4: Run test**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -k "shell_structure" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "feat(thesis-stats): add Block 6 shell structure with convective subpop analysis"
```

---

## Task 9: run_all + save_results orchestrator

**Files:**
- Modify: `EuropaProjectDJ/scripts/thesis_stats.py`

- [ ] **Step 1: Implement run_all and save_results**

Add to `thesis_stats.py`:

```python
import csv


def run_all(
    scenario_paths: Mapping[str, str],
    output_dir: str = "results/thesis_stats",
) -> Dict[str, Any]:
    """Orchestrate all 6 analysis blocks and return combined results."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load all scenarios
    scenarios = {name: load_scenario(path) for name, path in scenario_paths.items()}
    scenario_names = list(scenarios.keys())

    results: Dict[str, Any] = {"scenarios": {}}

    # Block 1: Descriptive summary
    for name, data in scenarios.items():
        results["scenarios"][name] = {
            "descriptive": descriptive_summary(data),
            "metadata": data["metadata"],
        }

    # Block 2: Pairwise comparison (whole-pop QoIs only)
    results["pairwise"] = {}
    for i, name_a in enumerate(scenario_names):
        for name_b in scenario_names[i + 1:]:
            pair_key = f"{name_a}_vs_{name_b}"
            pw = pairwise_comparison(
                scenarios[name_a], scenarios[name_b], list(WHOLE_POP_QOIS),
            )
            results["pairwise"][pair_key] = pw

    # FDR correction within each QoI
    results["pairwise_fdr"] = {}
    for qoi in WHOLE_POP_QOIS:
        for p_key in ("ks_p", "mw_p"):
            corrected = fdr_correct(results["pairwise"], qoi, p_key)
            for pk, entry in corrected.items():
                results["pairwise"].setdefault(pk, {}).setdefault(qoi, {}).update(
                    entry[qoi]
                )

    # Block 3: Enhancement trend (equatorial only)
    eq_names = [n for n in scenario_names if "Eq" in n]
    eq_enhancements = {
        "Eq Baseline": 1.0, "Eq Moderate": 1.2, "Eq Strong": 1.5,
    }
    eq_data = [
        {**scenarios[n], "enhancement": eq_enhancements[n]}
        for n in eq_names if n in eq_enhancements
    ]
    if len(eq_data) >= 2:
        results["enhancement_trend"] = enhancement_trend(eq_data, list(JT_QOIS))

    # Block 4: Parameter ranking
    results["parameter_ranking"] = {}
    for name, data in scenarios.items():
        results["parameter_ranking"][name] = parameter_ranking(data, list(WHOLE_POP_QOIS))
    results["ranking_concordance"] = ranking_concordance(
        results["parameter_ranking"], list(WHOLE_POP_QOIS),
    )

    # Block 5: Bootstrap convergence
    results["bootstrap_convergence"] = {}
    for name, data in scenarios.items():
        results["bootstrap_convergence"][name] = bootstrap_convergence(
            data, ["thickness_km"],
        )

    # Block 6: Shell structure
    results["shell_structure"] = shell_structure(scenarios)

    return results


def _to_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(i) for i in obj]
    return obj


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Write JSON + summary CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # JSON
    with (out_path / "comparison_results.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(results), f, indent=2)

    # Summary tables CSV (long format)
    rows = []
    for scenario_name, scenario_data in results.get("scenarios", {}).items():
        desc = scenario_data.get("descriptive", {})
        for qoi, stats_dict in desc.items():
            if not isinstance(stats_dict, dict):
                # scalar like conductive_fraction
                rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "statistic": "value", "value": stats_dict,
                    "ci_low": "", "ci_high": "",
                })
                continue
            for stat_name, value in stats_dict.items():
                if stat_name.startswith("ci_") or stat_name == "n":
                    continue
                ci_low = stats_dict.get(f"ci_{stat_name}_low", "")
                ci_high = stats_dict.get(f"ci_{stat_name}_high", "")
                rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "statistic": stat_name, "value": value,
                    "ci_low": ci_low, "ci_high": ci_high,
                })

    if rows:
        fieldnames = ["scenario", "qoi", "statistic", "value", "ci_low", "ci_high"]
        with (out_path / "summary_tables.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Parameter rankings CSV
    rank_rows = []
    for scenario_name, qoi_ranks in results.get("parameter_ranking", {}).items():
        for qoi, param_dict in qoi_ranks.items():
            for param_name, param_stats in param_dict.items():
                rank_rows.append({
                    "scenario": scenario_name, "qoi": qoi,
                    "parameter": param_name, "rho": param_stats["rho"],
                    "rank": param_stats["rank"],
                    "significant": param_stats["significant"],
                })
    if rank_rows:
        fieldnames = ["scenario", "qoi", "parameter", "rho", "rank", "significant"]
        with (out_path / "parameter_rankings.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rank_rows)
```

- [ ] **Step 2: Add `if __name__ == "__main__"` block**

Append to `thesis_stats.py`:

```python
if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    paths = {
        label: os.path.join(results_dir, filename)
        for label, filename in DEFAULT_SCENARIOS.items()
    }
    results = run_all(paths)
    save_results(results, os.path.join(results_dir, "thesis_stats"))
    print("Done. Results saved to results/thesis_stats/")
```

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_stats.py
git commit -m "feat(thesis-stats): add run_all orchestrator and save_results"
```

---

## Task 10: Smoke test on real data

**Files:**
- Modify: `EuropaProjectDJ/tests/test_thesis_stats.py`

- [ ] **Step 1: Write smoke test**

```python
def test_quantile_regression_recovers_known_slope():
    """Direct QR test: y = 2x + noise, slope CI should contain 2.0."""
    from statsmodels.regression.quantile_regression import QuantReg as QR
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 500)
    y = 2.0 * x + rng.normal(0, 1, 500)
    X = np.column_stack([np.ones_like(x), x])
    fit = QR(y, X).fit(q=0.5)
    slope = fit.params[1]
    # 95% CI from bootstrap
    boot_slopes = []
    for _ in range(1000):
        idx = rng.choice(len(x), len(x), replace=True)
        fit_b = QR(y[idx], X[idx]).fit(q=0.5)
        boot_slopes.append(fit_b.params[1])
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])
    assert ci_low < 2.0 < ci_high


def test_save_results_produces_valid_outputs(tmp_path):
    """save_results writes parseable JSON and CSV with expected columns."""
    from thesis_stats import save_results
    fake_results = {
        "scenarios": {
            "A": {
                "descriptive": {
                    "thickness_km": {"mean": 30.0, "median": 28.0},
                    "conductive_fraction": 0.35,
                },
                "metadata": {"n_valid": 100},
            }
        },
        "parameter_ranking": {
            "A": {
                "thickness_km": {
                    "P_tidal": {"rho": -0.8, "rank": 1, "significant": True},
                }
            }
        },
    }
    save_results(fake_results, str(tmp_path))
    import json, csv
    with open(tmp_path / "comparison_results.json") as f:
        loaded = json.load(f)
    assert "scenarios" in loaded

    with open(tmp_path / "summary_tables.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "scenario" in rows[0]
    assert "qoi" in rows[0]

    with open(tmp_path / "parameter_rankings.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert rows[0]["parameter"] == "P_tidal"


def test_smoke_run_all_on_real_archives():
    """Smoke test: run_all on all four production archives."""
    import os
    from thesis_stats import run_all, save_results, DEFAULT_SCENARIOS

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    paths = {
        label: os.path.join(results_dir, filename)
        for label, filename in DEFAULT_SCENARIOS.items()
    }
    # Skip if archives not present
    for p in paths.values():
        if not os.path.exists(p):
            pytest.skip(f"Archive not found: {p}")

    results = run_all(paths)

    assert "scenarios" in results
    assert len(results["scenarios"]) == 4
    assert "pairwise" in results
    assert "enhancement_trend" in results
    assert "parameter_ranking" in results
    assert "bootstrap_convergence" in results
    assert "shell_structure" in results

    # Check descriptive summary has expected keys
    for scenario_data in results["scenarios"].values():
        desc = scenario_data["descriptive"]
        assert "thickness_km" in desc
        assert "median" in desc["thickness_km"]
        assert "conductive_fraction" in desc
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -v`
Expected: all PASS (smoke test may take ~60s due to bootstrap)

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/tests/test_thesis_stats.py
git commit -m "test(thesis-stats): add smoke test on real MC archives"
```

---

## Task 11: thesis_figures.py — all 7 figures

**Files:**
- Create: `EuropaProjectDJ/scripts/thesis_figures.py`

- [ ] **Step 1: Create thesis_figures.py**

```python
"""
Publication figures for Europa MC statistical comparison thesis chapter.

Reads comparison_results.json and original NPZ archives.
Uses pub_style.py for Wong 2011 palette and AGU column widths.

Usage:
    python thesis_figures.py [--results-dir results/thesis_stats]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import gaussian_kde

from pub_style import apply_style, PAL, figsize_single, figsize_double, figsize_double_tall
from thesis_stats import (
    DEFAULT_SCENARIOS, LID_FRACTION_COND_THRESHOLD, load_scenario, CONVERGENCE_SCHEDULE,
)

apply_style()

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures" / "thesis_stats"

SCENARIO_COLORS = {
    "Global Audited": PAL.BLUE,
    "Eq Baseline": PAL.ORANGE,
    "Eq Moderate": PAL.GREEN,
    "Eq Strong": PAL.RED,
}

SCENARIO_LABELS = {
    "Global Audited": "Global",
    "Eq Baseline": r"Eq 1.0$\times$",
    "Eq Moderate": r"Eq 1.2$\times$",
    "Eq Strong": r"Eq 1.5$\times$",
}


def _load_results_and_data(results_dir):
    with open(os.path.join(results_dir, "comparison_results.json")) as f:
        results = json.load(f)
    base = os.path.join(os.path.dirname(__file__), "..", "results")
    scenarios = {
        label: load_scenario(os.path.join(base, fname))
        for label, fname in DEFAULT_SCENARIOS.items()
    }
    return results, scenarios


def fig_thickness_pdf(results, scenarios):
    """Figure 1: 4-scenario thickness PDF overlay."""
    fig, ax = plt.subplots(figsize=figsize_single(0.8))
    for name, data in scenarios.items():
        t = data["qois"]["thickness_km"]
        kde = gaussian_kde(t, bw_method=0.15)
        x = np.linspace(0, 120, 500)
        ax.plot(x, kde(x), color=SCENARIO_COLORS[name], label=SCENARIO_LABELS[name], lw=1.2)
        desc = results["scenarios"][name]["descriptive"]["thickness_km"]
        ax.axvline(desc["median"], color=SCENARIO_COLORS[name], ls="--", lw=0.6, alpha=0.7)
        ax.axvline(desc["cbe"], color=SCENARIO_COLORS[name], ls=":", lw=0.6, alpha=0.5)
        ax.axvspan(desc["P16"], desc["P84"], color=SCENARIO_COLORS[name], alpha=0.06)
    ax.set_xlabel("Ice shell thickness (km)")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=7, frameon=False)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    return fig


def fig_quantile_shift(results, scenarios):
    """Figure 2: P5/P50/P95 vs enhancement factor."""
    fig, ax = plt.subplots(figsize=figsize_single(0.8))
    enhancements = [1.0, 1.2, 1.5]
    eq_names = ["Eq Baseline", "Eq Moderate", "Eq Strong"]
    styles = {"P5": ("s", "--"), "P50": ("o", "-"), "P95": ("^", "--")}

    for pct, (marker, ls) in styles.items():
        vals = [
            results["scenarios"][n]["descriptive"]["thickness_km"][pct]
            for n in eq_names
        ]
        ax.plot(enhancements, vals, marker=marker, ls=ls, color=PAL.BLACK,
                label=pct, markersize=5, lw=1)

    # Annotate QR slopes if available
    trend = results.get("enhancement_trend", {}).get("thickness_km", {})
    if "qr_slope_P50" in trend:
        slope = trend["qr_slope_P50"]
        ax.text(0.95, 0.95, f"$\\Delta$P50 = {slope:.1f} km / 0.1$\\times$",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)

    ax.set_xlabel("Enhancement factor")
    ax.set_ylabel("Thickness (km)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_parameter_heatmap(results, scenarios):
    """Figure 3: |Spearman rho| heatmap for top-5 params x 4 scenarios."""
    fig, ax = plt.subplots(figsize=figsize_double(0.4))
    scenario_names = list(scenarios.keys())
    qoi = "thickness_km"

    # Find top-5 params by max |rho| across scenarios
    all_params = set()
    for name in scenario_names:
        all_params.update(results["parameter_ranking"][name][qoi].keys())
    param_max_rho = {}
    for p in all_params:
        rhos = []
        for name in scenario_names:
            entry = results["parameter_ranking"][name][qoi].get(p)
            if entry:
                rhos.append(abs(entry["rho"]))
        param_max_rho[p] = max(rhos) if rhos else 0
    top_params = sorted(param_max_rho, key=param_max_rho.get, reverse=True)[:5]

    matrix = np.zeros((len(top_params), len(scenario_names)))
    for j, name in enumerate(scenario_names):
        for i, p in enumerate(top_params):
            entry = results["parameter_ranking"][name][qoi].get(p)
            matrix[i, j] = abs(entry["rho"]) if entry else 0

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels([SCENARIO_LABELS[n] for n in scenario_names], fontsize=7)
    ax.set_yticks(range(len(top_params)))
    ax.set_yticklabels(top_params, fontsize=7)
    for i in range(len(top_params)):
        for j in range(len(scenario_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, label=r"|$\rho$|", shrink=0.8)
    ax.set_title(r"Spearman |$\rho$| $\rightarrow$ thickness", fontsize=8)
    fig.tight_layout()
    return fig


def fig_shell_structure(results, scenarios):
    """Figure 4: D_cond vs D_conv scatter, 4 panels."""
    fig, axes = plt.subplots(2, 2, figsize=figsize_double_tall(), sharex=True, sharey=True)
    for ax, (name, data) in zip(axes.flat, scenarios.items()):
        dc = data["qois"]["D_cond_km"]
        dv = data["qois"]["D_conv_km"]
        ax.scatter(dc, dv, s=0.5, alpha=0.15, color=SCENARIO_COLORS[name], rasterized=True)
        ax.set_title(SCENARIO_LABELS[name], fontsize=8)
        ax.plot([0, 100], [0, 100], "k--", lw=0.5, alpha=0.3)
    axes[1, 0].set_xlabel(r"$D_\mathrm{cond}$ (km)")
    axes[1, 1].set_xlabel(r"$D_\mathrm{cond}$ (km)")
    axes[0, 0].set_ylabel(r"$D_\mathrm{conv}$ (km)")
    axes[1, 0].set_ylabel(r"$D_\mathrm{conv}$ (km)")
    fig.tight_layout()
    return fig


def fig_bootstrap_convergence(results, scenarios):
    """Figure 5: median +/- CI vs sample size."""
    fig, ax = plt.subplots(figsize=figsize_single(0.8))
    for name in scenarios:
        conv = results["bootstrap_convergence"].get(name, {}).get("thickness_km", {})
        if not conv:
            continue
        ns = sorted(int(k) for k in conv.keys())
        meds = [conv[str(n)]["median"] for n in ns]
        lows = [conv[str(n)]["ci_median_low"] for n in ns]
        highs = [conv[str(n)]["ci_median_high"] for n in ns]
        color = SCENARIO_COLORS[name]
        ax.plot(ns, meds, "o-", color=color, label=SCENARIO_LABELS[name], markersize=3, lw=1)
        ax.fill_between(ns, lows, highs, color=color, alpha=0.15)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Median thickness (km)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def fig_enhancement_trend(results, scenarios):
    """Figure 6: median thickness bar chart with CI whiskers + JT p."""
    fig, ax = plt.subplots(figsize=figsize_single(0.8))
    names = list(scenarios.keys())
    x = np.arange(len(names))
    for i, name in enumerate(names):
        desc = results["scenarios"][name]["descriptive"]["thickness_km"]
        med = desc["median"]
        ci_lo = desc["ci_median_low"]
        ci_hi = desc["ci_median_high"]
        ax.bar(i, med, color=SCENARIO_COLORS[name], width=0.6)
        ax.errorbar(i, med, yerr=[[med - ci_lo], [ci_hi - med]],
                     fmt="none", color="black", capsize=3, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[n] for n in names], fontsize=7)
    ax.set_ylabel("Median thickness (km)")

    jt = results.get("enhancement_trend", {}).get("thickness_km", {})
    if "jt_p" in jt:
        ax.text(0.95, 0.95, f"JT p = {jt['jt_p']:.1e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.tight_layout()
    return fig


def fig_conductive_fraction(results, scenarios):
    """Figure 7: conductive fraction bar chart with bootstrap CIs."""
    fig, ax = plt.subplots(figsize=figsize_single(0.8))
    names = list(scenarios.keys())
    fracs = results.get("shell_structure", {}).get("conductive_fractions", {})
    x = np.arange(len(names))
    for i, name in enumerate(names):
        entry = fracs.get(name, {})
        f = entry.get("fraction", 0)
        ci_lo = entry.get("ci_low", f)
        ci_hi = entry.get("ci_high", f)
        ax.bar(i, 100 * f, color=SCENARIO_COLORS[name], width=0.6)
        ax.errorbar(i, 100 * f, yerr=[[100 * (f - ci_lo)], [100 * (ci_hi - f)]],
                     fmt="none", color="black", capsize=3, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[n] for n in names], fontsize=7)
    ax.set_ylabel("Conductive fraction (%)")
    fig.tight_layout()
    return fig


FIGURES = {
    "01_thickness_pdf": fig_thickness_pdf,
    "02_quantile_shift": fig_quantile_shift,
    "03_parameter_heatmap": fig_parameter_heatmap,
    "04_shell_structure": fig_shell_structure,
    "05_bootstrap_convergence": fig_bootstrap_convergence,
    "06_enhancement_trend": fig_enhancement_trend,
    "07_conductive_fraction": fig_conductive_fraction,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(
        Path(__file__).resolve().parents[1] / "results" / "thesis_stats"))
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results, scenarios = _load_results_and_data(args.results_dir)

    for name, func in FIGURES.items():
        fig = func(results, scenarios)
        for ext in ("png", "pdf"):
            path = FIGURES_DIR / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}")

    print(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs (after Task 10 smoke test has populated results)**

Run: `python EuropaProjectDJ/scripts/thesis_figures.py`
Expected: prints "Saved 01_thickness_pdf" through "Saved 07_conductive_fraction"

- [ ] **Step 3: Commit**

```bash
git add EuropaProjectDJ/scripts/thesis_figures.py
git commit -m "feat(thesis-figures): add all 7 publication figures for thesis chapter"
```

---

## Task 12: Run the full pipeline and verify outputs

**Files:**
- None (verification only)

- [ ] **Step 1: Run thesis_stats.py**

Run: `python EuropaProjectDJ/scripts/thesis_stats.py`
Expected: "Done. Results saved to results/thesis_stats/"

- [ ] **Step 2: Verify output files exist**

Run: `ls EuropaProjectDJ/results/thesis_stats/`
Expected: `comparison_results.json`, `summary_tables.csv`, `parameter_rankings.csv`

- [ ] **Step 3: Run thesis_figures.py**

Run: `python EuropaProjectDJ/scripts/thesis_figures.py`
Expected: 7 PNG + 7 PDF files in `EuropaProjectDJ/figures/thesis_stats/`

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest EuropaProjectDJ/tests/test_thesis_stats.py -v`
Expected: all PASS

- [ ] **Step 5: Final commit**

```bash
git add EuropaProjectDJ/results/thesis_stats/ EuropaProjectDJ/figures/thesis_stats/
git commit -m "feat(thesis-stats): complete MC statistical comparison pipeline"
```
