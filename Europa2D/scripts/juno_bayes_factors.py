"""
Bayesian scenario comparison against Juno MWR D_cond constraint.

Uses importance reweighting on existing MC posteriors to compute:
1. Per-scenario marginal likelihood (evidence) for Juno D_cond = 29 ± 10 km at 35°
2. Bayes factors between all scenario pairs
3. Posterior-weighted D_cond distributions at 35°

References:
    Levin et al. (2025): Juno MWR D_cond = 29 ± 10 km (pure water ice)
    Wakita et al. (2024): H_total > 15 km (multiring basin constraint)
"""
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))
sys.path.insert(0, _SCRIPT_DIR)

RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
N_ITER = 250

# Juno constraint
JUNO_DCOND_KM = 29.0
JUNO_DCOND_SIGMA_OBS = 10.0
MODEL_DISCREPANCY = 3.0  # km
SIGMA_EFF = np.sqrt(JUNO_DCOND_SIGMA_OBS**2 + MODEL_DISCREPANCY**2)
JUNO_LAT_DEG = 35.0

SCENARIOS = [
    ("uniform_transport",             "Uniform transport"),
    ("soderlund2014_equator",         "Equator-enhanced"),
    ("lemasquerier2023_polar",        "Polar-enhanced"),
    ("lemasquerier2023_polar_strong", "Strong polar-enhanced"),
]


def _load(key):
    path = os.path.join(RESULTS_DIR, f"mc_2d_{key}_{N_ITER}.npz")
    return dict(np.load(path, allow_pickle=True))


def _interp_at_lat(lat, profiles, target):
    """Interpolate (n_samples, n_lat) profiles at a single latitude."""
    return np.array([np.interp(target, lat, profiles[i])
                     for i in range(profiles.shape[0])])


def _gaussian_likelihood(d_cond, mu=JUNO_DCOND_KM, sigma=SIGMA_EFF):
    """Per-sample Gaussian likelihood for Juno D_cond."""
    return np.exp(-0.5 * ((d_cond - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main():
    print("=" * 70)
    print("JUNO BAYESIAN SCENARIO COMPARISON")
    print(f"Constraint: D_cond = {JUNO_DCOND_KM} +/- {JUNO_DCOND_SIGMA_OBS} km at {JUNO_LAT_DEG} deg")
    print(f"sigma_eff = sqrt({JUNO_DCOND_SIGMA_OBS}^2 + {MODEL_DISCREPANCY}^2) = {SIGMA_EFF:.2f} km")
    print("=" * 70)

    evidences = {}
    summaries = {}

    for key, title in SCENARIOS:
        d = _load(key)
        lat = d["latitudes_deg"]
        Dc = d["D_cond_profiles"]
        H = d["H_profiles"]
        n_valid = int(d["n_valid"])

        # D_cond at 35° for each sample
        dc_35 = _interp_at_lat(lat, Dc, JUNO_LAT_DEG)

        # Per-sample likelihood
        lk = _gaussian_likelihood(dc_35)

        # Marginal likelihood (evidence) = mean of likelihoods under the prior
        # Since MC samples ARE draws from the prior, evidence = (1/N) Σ L(d|θ_i)
        evidence = float(np.mean(lk))

        # Importance weights (normalized)
        weights = lk / lk.sum()

        # Posterior-weighted statistics
        dc_post_mean = float(np.sum(weights * dc_35))
        dc_post_std = float(np.sqrt(np.sum(weights * (dc_35 - dc_post_mean)**2)))

        # Effective sample size
        n_eff = 1.0 / np.sum(weights**2)

        # H_total at 35° posterior
        h_35 = _interp_at_lat(lat, H, JUNO_LAT_DEG)
        h_post_mean = float(np.sum(weights * h_35))

        # Fraction with H_total > 15 km (posterior)
        h_above_15 = float(np.sum(weights * (h_35 > 15.0)))

        evidences[key] = evidence
        summaries[key] = {
            'title': title,
            'n_valid': n_valid,
            'evidence': evidence,
            'dc_prior_median': float(np.median(dc_35)),
            'dc_prior_std': float(np.std(dc_35)),
            'dc_post_mean': dc_post_mean,
            'dc_post_std': dc_post_std,
            'h_post_mean': h_post_mean,
            'h_above_15_pct': h_above_15 * 100,
            'n_eff': n_eff,
        }

    # Print per-scenario summary
    print(f"\n{'Scenario':<28s} {'N':>4s} {'Evidence':>10s} {'D_cond prior':>14s} {'D_cond post':>14s} {'H(35°) post':>12s} {'N_eff':>6s}")
    print("-" * 94)
    for key, s in summaries.items():
        print(f"{s['title']:<28s} {s['n_valid']:>4d} {s['evidence']:>10.2e} "
              f"{s['dc_prior_median']:>6.1f}±{s['dc_prior_std']:<5.1f}  "
              f"{s['dc_post_mean']:>6.1f}±{s['dc_post_std']:<5.1f}  "
              f"{s['h_post_mean']:>6.1f} km   {s['n_eff']:>5.1f}")

    # Bayes factors (all pairs, relative to uniform as reference)
    ref_key = "uniform_transport"
    ref_ev = evidences[ref_key]

    print(f"\nBayes Factors (vs Uniform transport):")
    print("-" * 50)
    for key, title in SCENARIOS:
        bf = evidences[key] / ref_ev
        if bf > 1:
            interpretation = "favoured"
        elif bf > 1/3:
            interpretation = "inconclusive"
        elif bf > 1/10:
            interpretation = "moderate evidence against"
        else:
            interpretation = "strong evidence against"
        print(f"  {title:<28s}  BF = {bf:>6.3f}  ({interpretation})")

    # Full pairwise Bayes factor matrix
    keys = [k for k, _ in SCENARIOS]
    titles = [t for _, t in SCENARIOS]
    print(f"\nPairwise Bayes Factor Matrix (row / column):")
    print(f"{'':>28s}", end="")
    for t in titles:
        print(f" {t[:12]:>12s}", end="")
    print()
    for i, ki in enumerate(keys):
        print(f"{titles[i]:<28s}", end="")
        for j, kj in enumerate(keys):
            bf = evidences[ki] / evidences[kj]
            print(f" {bf:>12.3f}", end="")
        print()

    # Jeffreys scale interpretation
    print(f"\nJeffreys scale: BF > 3 substantial, > 10 strong, > 30 very strong, > 100 decisive")


if __name__ == "__main__":
    main()
