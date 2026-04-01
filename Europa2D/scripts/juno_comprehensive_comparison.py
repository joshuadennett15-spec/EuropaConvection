"""
Comprehensive Juno inference: compare prior, Juno-reweighted, and grain-shifted
results across all 4 ocean scenarios and multiple grain priors.

Every result is reported — no hypothesis is rejected.
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
    lat = data["latitudes_deg"]
    H = data["H_profiles"]
    Nu = data["Nu_profiles"]
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
    print("=" * 105)
    print("COMPREHENSIVE JUNO INFERENCE: ALL SCENARIOS x ALL GRAIN PRIORS x PRIOR/POSTERIOR")
    print(f"Juno: D_cond = {JUNO} +/- {SIGMA_EFF:.1f} km (eff) at 35 deg")
    print("Every result is reported. No hypothesis is rejected.")
    print("=" * 105)

    header = (f"{'Scenario':14s} {'View':22s} | {'Dc(35)':>7s} {'H_eq':>6s} {'H_po':>6s} "
              f"{'C%eq':>5s} {'C%po':>5s} {'Nu|eq':>6s} {'Nu|po':>6s} {'N_eff':>6s}")
    print(header)
    print("-" * len(header))

    for key, title in SCENARIOS:
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
            dc35_arr = interp_at(data["latitudes_deg"], data["D_cond_profiles"], 35.0)
            lk = gaussian_lk(dc35_arr)
            w = lk / lk.sum()
            s_j = analyze(data, w)
            print(f"{'':14s} {'  +Juno':22s} | {s_j['dc35']:6.1f}  {s_j['h_eq']:5.1f}  {s_j['h_po']:5.1f}  "
                  f"{s_j['conv_eq']:4.0%}  {s_j['conv_po']:4.0%}  {s_j['nu_eq']:5.1f}  {s_j['nu_po']:5.1f}  {s_j['n_eff']:5.0f}")

        print("-" * len(header))

    print("""
KEY FINDINGS (fill in after reviewing the table):

  H1 (Grain dominance):
    - 0.6mm -> 1.0mm -> 1.5mm: D_cond increases monotonically
    - Convective vigor decreases with grain size
    - All grain sizes are valid results

  H2 (Bayesian selection):
    - +Juno rows show the posterior shift from each prior
    - Higher Nu|conv in reweighted vs flat prior of same D_cond

  H3 (Combined):
    - 1.0mm+Juno row shows the compromise result
    - Compare D_cond and Nu against 0.6mm+Juno and 1.5mm flat

  H5 (Ocean sensitivity):
    - Compare D_cond(35) across scenarios within each grain prior
    - Compare H_eq and H_po across scenarios (latitude structure)
""")


if __name__ == "__main__":
    main()
