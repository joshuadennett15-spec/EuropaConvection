"""Verify all numbers in the complete results tables document against NPZ data."""
from pathlib import Path
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# ============================================================
# Helper functions
# ============================================================
def pct(arr, p):
    return float(np.percentile(arr, p))

def med(arr):
    return float(np.median(arr))

def flag(computed, reported, tol_km=0.5, is_pct=False):
    tol = 1.0 if is_pct else tol_km
    diff = abs(computed - reported)
    if diff > tol:
        return f" *** DISCREPANCY (diff={diff:.2f}) ***"
    return ""

def compare(label, computed, reported, is_pct=False):
    unit = "%" if is_pct else "km"
    f = flag(computed, reported, is_pct=is_pct)
    print(f"  {label:30s}: computed={computed:8.1f} {unit}  reported={reported:8.1f} {unit}{f}")

# ============================================================
# TABLE 1: 1D Global Baselines
# ============================================================
print("=" * 90)
print("TABLE 1: 1D Global Baselines (N = 15,000)")
print("=" * 90)

runs_1d = {
    'Howell (Maxwell)': {
        'file': 'EuropaProjectDJ/results/mc_15000_howell.npz',
        'reported': {
            'N': 10350, 'CBE': 56.1, 'med_H': 50.8, 'lo_H': 32.7, 'hi_H': 64.3,
            'med_Dcond': 8.9, 'lo_Dcond': 4.8, 'hi_Dcond': 25.6,
            'med_Dconv': 35.5, 'lo_Dconv': 19.4, 'hi_Dconv': 50.7,
            'conv_frac': 100.0, 'mean_lid': 28.0,
        }
    },
    'Audited (Andrade)': {
        'file': 'EuropaProjectDJ/results/mc_15000_optionA_v2_andrade.npz',
        'reported': {
            'N': 15000, 'CBE': 18.9, 'med_H': 28.9, 'lo_H': 16.3, 'hi_H': 56.7,
            'med_Dcond': 15.2, 'lo_Dcond': 6.8, 'hi_Dcond': 25.6,
            'med_Dconv': 9.3, 'lo_Dconv': 0.0, 'hi_Dconv': 40.5,
            'conv_frac': 61.5, 'mean_lid': 60.0,
        }
    }
}

for name, info in runs_1d.items():
    d = np.load(info['file'])
    r = info['reported']
    H = d['thicknesses_km']
    D_cond = d['D_cond_km']
    D_conv = d['D_conv_km']
    Ra = d['Ra_values']
    lid = d['lid_fractions']

    n_valid = len(H)
    conv_mask = Ra >= 1000
    conv_frac = 100.0 * np.sum(conv_mask) / n_valid

    # CBE: the doc header says "conductive-only best estimate = median D_cond for non-convecting"
    # For Maxwell with 100% convecting, use stored cbe_km (KDE mode of H)
    non_conv = ~conv_mask
    if np.sum(non_conv) > 0:
        cbe = med(D_cond[non_conv])
    else:
        cbe = float(d['cbe_km'])

    print(f"\n--- {name} ---")
    compare("N valid", n_valid, r['N'])
    compare("CBE", cbe, r['CBE'])
    compare("Median H", med(H), r['med_H'])
    compare("16th H", pct(H, 15.87), r['lo_H'])
    compare("84th H", pct(H, 84.13), r['hi_H'])
    compare("Median D_cond", med(D_cond), r['med_Dcond'])
    compare("16th D_cond", pct(D_cond, 15.87), r['lo_Dcond'])
    compare("84th D_cond", pct(D_cond, 84.13), r['hi_Dcond'])
    compare("Median D_conv", med(D_conv), r['med_Dconv'])
    compare("16th D_conv", pct(D_conv, 15.87), r['lo_Dconv'])
    compare("84th D_conv", pct(D_conv, 84.13), r['hi_Dconv'])
    compare("Conv. frac", conv_frac, r['conv_frac'], is_pct=True)
    compare("Mean lid frac", 100 * np.mean(lid), r['mean_lid'], is_pct=True)

# ============================================================
# TABLE 2: 1D Equatorial Proxy Suite
# ============================================================
print("\n" + "=" * 90)
print("TABLE 2: 1D Equatorial Proxy Suite (N = 15,000, Andrade)")
print("=" * 90)

eq_modes = {
    'Depleted strong (0.55x)': {
        'file': 'EuropaProjectDJ/results/eq_depleted_strong_andrade.npz',
        'reported': {
            'N': 14995, 'CBE': 60.0, 'med_H': 50.9, 'lo_H': 34.3, 'hi_H': 67.0,
            'med_Dcond': 15.6, 'lo_Dcond': 8.7, 'hi_Dcond': 31.6,
            'med_Dconv': 31.0, 'lo_Dconv': 14.0, 'hi_Dconv': 48.2,
            'conv_frac': 89.3, 'mean_lid': 41.0,
        }
    },
    'Depleted (0.67x)': {
        'file': 'EuropaProjectDJ/results/eq_depleted_andrade.npz',
        'reported': {
            'N': 14997, 'CBE': 32.6, 'med_H': 46.6, 'lo_H': 27.9, 'hi_H': 65.8,
            'med_Dcond': 15.5, 'lo_Dcond': 8.5, 'hi_Dcond': 31.8,
            'med_Dconv': 26.5, 'lo_Dconv': 6.4, 'hi_Dconv': 46.6,
            'conv_frac': 84.7, 'mean_lid': 45.0,
        }
    },
    'Baseline (1.0x)': {
        'file': 'EuropaProjectDJ/results/eq_baseline_andrade.npz',
        'reported': {
            'N': 14998, 'CBE': 21.3, 'med_H': 35.3, 'lo_H': 18.2, 'hi_H': 62.8,
            'med_Dcond': 17.0, 'lo_Dcond': 8.2, 'hi_Dcond': 29.0,
            'med_Dconv': 14.6, 'lo_Dconv': 0.0, 'hi_Dconv': 42.9,
            'conv_frac': 67.8, 'mean_lid': 58.0,
        }
    },
    'Moderate (1.2x)': {
        'file': 'EuropaProjectDJ/results/eq_moderate_andrade.npz',
        'reported': {
            'N': 14998, 'CBE': 18.9, 'med_H': 29.8, 'lo_H': 16.7, 'hi_H': 61.1,
            'med_Dcond': 17.2, 'lo_Dcond': 8.1, 'hi_Dcond': 27.3,
            'med_Dconv': 7.7, 'lo_Dconv': 0.0, 'hi_Dconv': 41.1,
            'conv_frac': 60.5, 'mean_lid': 63.0,
        }
    },
    'Strong (1.5x)': {
        'file': 'EuropaProjectDJ/results/eq_strong_andrade.npz',
        'reported': {
            'N': 14997, 'CBE': 16.8, 'med_H': 23.6, 'lo_H': 14.7, 'hi_H': 59.0,
            'med_Dcond': 16.2, 'lo_Dcond': 8.4, 'hi_Dcond': 25.4,
            'med_Dconv': 2.5, 'lo_Dconv': 0.0, 'hi_Dconv': 38.6,
            'conv_frac': 52.4, 'mean_lid': 69.0,
        }
    },
}

for name, info in eq_modes.items():
    d = np.load(info['file'])
    r = info['reported']
    H = d['thicknesses_km']
    D_cond = d['D_cond_km']
    D_conv = d['D_conv_km']
    Ra = d['Ra_values']
    lid = d['lid_fractions']

    n_valid = len(H)
    conv_mask = Ra >= 1000
    conv_frac = 100.0 * np.sum(conv_mask) / n_valid

    non_conv = ~conv_mask
    if np.sum(non_conv) > 0:
        cbe = med(D_cond[non_conv])
    else:
        cbe = float(d['cbe_km'])

    print(f"\n--- {name} ---")
    compare("N valid", n_valid, r['N'])
    compare("CBE", cbe, r['CBE'])
    compare("Median H", med(H), r['med_H'])
    compare("16th H", pct(H, 15.87), r['lo_H'])
    compare("84th H", pct(H, 84.13), r['hi_H'])
    compare("Median D_cond", med(D_cond), r['med_Dcond'])
    compare("16th D_cond", pct(D_cond, 15.87), r['lo_Dcond'])
    compare("84th D_cond", pct(D_cond, 84.13), r['hi_Dcond'])
    compare("Median D_conv", med(D_conv), r['med_Dconv'])
    compare("16th D_conv", pct(D_conv, 15.87), r['lo_Dconv'])
    compare("84th D_conv", pct(D_conv, 84.13), r['hi_Dconv'])
    compare("Conv. frac", conv_frac, r['conv_frac'], is_pct=True)
    compare("Mean lid frac", 100 * np.mean(lid), r['mean_lid'], is_pct=True)

# ============================================================
# TABLE 3: 2D Latitude-Resolved
# ============================================================
print("\n" + "=" * 90)
print("TABLE 3: 2D Latitude-Resolved Model (N = 500, Andrade, 37 lat columns)")
print("=" * 90)

scenarios_2d = {
    'Uniform transport': {
        'file': 'Europa2D/results/mc_2d_uniform_transport_500.npz',
        'reported_3a': {'N': 477, 'med_H': 28.8, 'lo_H': 18.5, 'hi_H': 39.5, 'conv_frac': 43.0},
        'reported_3b': {
            'med_H': 26.4, 'lo_H': 14.5, 'hi_H': 38.5,
            'med_Dcond': 21.3, 'lo_Dcond': 9.1, 'hi_Dcond': 31.5,
            'med_Dconv': 4.2, 'lo_Dconv': 3.0, 'hi_Dconv': 8.9,
            'lid_frac': 76.0,
        },
        'reported_3c': {
            'med_H': 45.2, 'lo_H': 34.9, 'hi_H': 65.6,
            'med_Dcond': 41.4, 'lo_Dcond': 31.7, 'hi_Dcond': 60.4,
            'med_Dconv': 3.6, 'lo_Dconv': 2.7, 'hi_Dconv': 5.6,
            'lid_frac': 92.0,
        },
        'reported_3d': {'delta_H': 18.8, 'delta_Dcond': 20.1, 'ratio': 1.71},
    },
    'Soderlund equatorial (q*=0.4)': {
        'file': 'Europa2D/results/mc_2d_soderlund2014_equator_500.npz',
        'reported_3a': {'N': 485, 'med_H': 30.5, 'lo_H': 20.5, 'hi_H': 42.3, 'conv_frac': 40.4},
        'reported_3b': {
            'med_H': 24.3, 'lo_H': 15.0, 'hi_H': 35.7,
            'med_Dcond': 20.7, 'lo_Dcond': 9.0, 'hi_Dcond': 30.5,
            'med_Dconv': 3.7, 'lo_Dconv': 2.6, 'hi_Dconv': 6.4,
            'lid_frac': 79.0,
        },
        'reported_3c': {
            'med_H': 53.4, 'lo_H': 40.9, 'hi_H': 73.4,
            'med_Dcond': 48.8, 'lo_Dcond': 36.8, 'hi_Dcond': 67.6,
            'med_Dconv': 4.5, 'lo_Dconv': 3.5, 'hi_Dconv': 6.7,
            'lid_frac': 91.0,
        },
        'reported_3d': {'delta_H': 29.1, 'delta_Dcond': 28.1, 'ratio': 2.20},
    },
    'Lemasquerier polar (q*=0.455)': {
        'file': 'Europa2D/results/mc_2d_lemasquerier2023_polar_500.npz',
        'reported_3a': {'N': 486, 'med_H': 28.0, 'lo_H': 19.3, 'hi_H': 40.6, 'conv_frac': 40.7},
        'reported_3b': {
            'med_H': 30.7, 'lo_H': 15.4, 'hi_H': 43.4,
            'med_Dcond': 22.5, 'lo_Dcond': 8.7, 'hi_Dcond': 34.4,
            'med_Dconv': 4.9, 'lo_Dconv': 3.5, 'hi_Dconv': 17.4,
            'lid_frac': 71.0,
        },
        'reported_3c': {
            'med_H': 40.3, 'lo_H': 29.5, 'hi_H': 63.0,
            'med_Dcond': 37.4, 'lo_Dcond': 27.3, 'hi_Dcond': 57.5,
            'med_Dconv': 3.1, 'lo_Dconv': 2.1, 'hi_Dconv': 5.1,
            'lid_frac': 92.0,
        },
        'reported_3d': {'delta_H': 9.6, 'delta_Dcond': 14.9, 'ratio': 1.31},
    },
    'Lemasquerier polar strong (q*=0.819)': {
        'file': 'Europa2D/results/mc_2d_lemasquerier2023_polar_strong_500.npz',
        'reported_3a': {'N': 477, 'med_H': 27.0, 'lo_H': 19.8, 'hi_H': 36.8, 'conv_frac': 34.6},
        'reported_3b': {
            'med_H': 33.3, 'lo_H': 16.9, 'hi_H': 46.3,
            'med_Dcond': 18.9, 'lo_Dcond': 10.1, 'hi_Dcond': 36.4,
            'med_Dconv': 5.4, 'lo_Dconv': 4.1, 'hi_Dconv': 18.4,
            'lid_frac': 69.0,
        },
        'reported_3c': {
            'med_H': 33.8, 'lo_H': 26.4, 'hi_H': 51.4,
            'med_Dcond': 31.4, 'lo_Dcond': 24.4, 'hi_Dcond': 47.0,
            'med_Dconv': 2.5, 'lo_Dconv': 1.9, 'hi_Dconv': 4.0,
            'lid_frac': 92.0,
        },
        'reported_3d': {'delta_H': 0.5, 'delta_Dcond': 12.5, 'ratio': 1.02},
    },
}

for name, info in scenarios_2d.items():
    d = np.load(info['file'])
    H = d['H_profiles']         # (N_valid, 37)
    D_cond = d['D_cond_profiles']
    D_conv = d['D_conv_profiles']
    Ra = d['Ra_profiles']
    lats = d['latitudes_deg']
    lid = d['lid_fraction_profiles']
    n_valid = int(d['n_valid'])

    r3a = info['reported_3a']
    r3b = info['reported_3b']
    r3c = info['reported_3c']
    r3d = info['reported_3d']

    # --- 3a: latitude-averaged statistics ---
    H_lat_avg = np.mean(H, axis=1)   # (N_valid,)

    # Conv frac: fraction of (realization, latitude) cells with Ra >= 1000
    conv_cells = np.sum(Ra >= 1000)
    total_cells = Ra.size
    conv_frac_3a = 100.0 * conv_cells / total_cells

    print(f"\n--- {name} ---")
    print(f"  Table 3a: Global (latitude-averaged)")
    compare("N valid", n_valid, r3a['N'])
    compare("Median H (lat-avg)", med(H_lat_avg), r3a['med_H'])
    compare("16th H (lat-avg)", pct(H_lat_avg, 15.87), r3a['lo_H'])
    compare("84th H (lat-avg)", pct(H_lat_avg, 84.13), r3a['hi_H'])
    compare("Conv. frac (cell-wise)", conv_frac_3a, r3a['conv_frac'], is_pct=True)

    # --- 3b: Equatorial column (lat = 0 deg, index 0) ---
    eq_idx = 0
    H_eq = H[:, eq_idx]
    D_cond_eq = D_cond[:, eq_idx]
    D_conv_eq = D_conv[:, eq_idx]
    Ra_eq = Ra[:, eq_idx]
    lid_eq = lid[:, eq_idx]

    # D_conv stats: only convecting cells (Ra >= 1000)
    conv_eq_mask = Ra_eq >= 1000
    D_conv_eq_conv = D_conv_eq[conv_eq_mask] if np.sum(conv_eq_mask) > 0 else D_conv_eq

    print(f"\n  Table 3b: Equatorial column (lat = 0 deg)")
    compare("Median H_eq", med(H_eq), r3b['med_H'])
    compare("16th H_eq", pct(H_eq, 15.87), r3b['lo_H'])
    compare("84th H_eq", pct(H_eq, 84.13), r3b['hi_H'])
    compare("Median D_cond_eq", med(D_cond_eq), r3b['med_Dcond'])
    compare("16th D_cond_eq", pct(D_cond_eq, 15.87), r3b['lo_Dcond'])
    compare("84th D_cond_eq", pct(D_cond_eq, 84.13), r3b['hi_Dcond'])
    compare("Median D_conv_eq (conv)", med(D_conv_eq_conv), r3b['med_Dconv'])
    compare("16th D_conv_eq (conv)", pct(D_conv_eq_conv, 15.87), r3b['lo_Dconv'])
    compare("84th D_conv_eq (conv)", pct(D_conv_eq_conv, 84.13), r3b['hi_Dconv'])
    compare("Lid frac_eq", 100 * np.mean(lid_eq), r3b['lid_frac'], is_pct=True)

    # --- 3c: Polar column (lat = 90 deg, index -1) ---
    pole_idx = -1
    H_pole = H[:, pole_idx]
    D_cond_pole = D_cond[:, pole_idx]
    D_conv_pole = D_conv[:, pole_idx]
    Ra_pole = Ra[:, pole_idx]
    lid_pole = lid[:, pole_idx]

    conv_pole_mask = Ra_pole >= 1000
    D_conv_pole_conv = D_conv_pole[conv_pole_mask] if np.sum(conv_pole_mask) > 0 else D_conv_pole

    print(f"\n  Table 3c: Polar column (lat = 90 deg)")
    compare("Median H_pole", med(H_pole), r3c['med_H'])
    compare("16th H_pole", pct(H_pole, 15.87), r3c['lo_H'])
    compare("84th H_pole", pct(H_pole, 84.13), r3c['hi_H'])
    compare("Median D_cond_pole", med(D_cond_pole), r3c['med_Dcond'])
    compare("16th D_cond_pole", pct(D_cond_pole, 15.87), r3c['lo_Dcond'])
    compare("84th D_cond_pole", pct(D_cond_pole, 84.13), r3c['hi_Dcond'])
    compare("Median D_conv_pole (conv)", med(D_conv_pole_conv), r3c['med_Dconv'])
    compare("16th D_conv_pole (conv)", pct(D_conv_pole_conv, 15.87), r3c['lo_Dconv'])
    compare("84th D_conv_pole (conv)", pct(D_conv_pole_conv, 84.13), r3c['hi_Dconv'])
    compare("Lid frac_pole", 100 * np.mean(lid_pole), r3c['lid_frac'], is_pct=True)

    # --- 3d: Equator-to-pole contrast ---
    delta_H = med(H_pole) - med(H_eq)
    delta_Dcond = med(D_cond_pole) - med(D_cond_eq)
    ratio = med(H_pole) / med(H_eq) if med(H_eq) != 0 else float('inf')

    print(f"\n  Table 3d: Equator-to-pole contrast")
    compare("Delta-H", delta_H, r3d['delta_H'])
    compare("Delta-D_cond", delta_Dcond, r3d['delta_Dcond'])
    computed_ratio = ratio
    reported_ratio = r3d['ratio']
    ratio_diff = abs(computed_ratio - reported_ratio)
    ratio_flag = " *** DISCREPANCY ***" if ratio_diff > 0.05 else ""
    print(f"  {'Pole/Equator H ratio':30s}: computed={computed_ratio:8.2f}       reported={reported_ratio:8.2f}      {ratio_flag}")

# ============================================================
# TABLE 4: Cross-model Juno comparison
# ============================================================
print("\n" + "=" * 90)
print("TABLE 4: Cross-Model Juno MWR Comparison")
print("=" * 90)

table4 = [
    ('1D Global Howell (Maxwell)', 'EuropaProjectDJ/results/mc_15000_howell.npz', '1d', 8.9, 25.6),
    ('1D Global Audited (Andrade)', 'EuropaProjectDJ/results/mc_15000_optionA_v2_andrade.npz', '1d', 15.2, 25.6),
    ('1D Eq Depleted strong (0.55x)', 'EuropaProjectDJ/results/eq_depleted_strong_andrade.npz', '1d', 15.6, 31.6),
    ('1D Eq Depleted (0.67x)', 'EuropaProjectDJ/results/eq_depleted_andrade.npz', '1d', 15.5, 31.8),
    ('1D Eq Baseline (1.0x)', 'EuropaProjectDJ/results/eq_baseline_andrade.npz', '1d', 17.0, 29.0),
    ('1D Eq Moderate (1.2x)', 'EuropaProjectDJ/results/eq_moderate_andrade.npz', '1d', 17.2, 27.3),
    ('1D Eq Strong (1.5x)', 'EuropaProjectDJ/results/eq_strong_andrade.npz', '1d', 16.2, 25.4),
    ('2D Uniform transport', 'Europa2D/results/mc_2d_uniform_transport_500.npz', '2d', 21.3, 31.5),
    ('2D Soderlund (q*=0.4)', 'Europa2D/results/mc_2d_soderlund2014_equator_500.npz', '2d', 20.7, 30.5),
    ('2D Lemasquerier (q*=0.455)', 'Europa2D/results/mc_2d_lemasquerier2023_polar_500.npz', '2d', 22.5, 34.4),
    ('2D Lemasquerier strong (q*=0.819)', 'Europa2D/results/mc_2d_lemasquerier2023_polar_strong_500.npz', '2d', 18.9, 36.4),
]

print(f"\n{'Model':<42s} {'Med Dcond comp':>15s} {'Med Dcond rep':>15s} {'84th comp':>10s} {'84th rep':>10s}  Flags")
print("-" * 110)

for label, fpath, dtype, rep_med, rep_hi in table4:
    d = np.load(fpath)
    if dtype == '1d':
        D_cond = d['D_cond_km']
    else:
        D_cond = d['D_cond_profiles'][:, 0]
    med_dc = med(D_cond)
    hi_dc = pct(D_cond, 84.13)
    f1 = flag(med_dc, rep_med)
    f2 = flag(hi_dc, rep_hi)
    print(f"  {label:<40s} {med_dc:>13.1f} km {rep_med:>13.1f} km {hi_dc:>8.1f} km {rep_hi:>8.1f} km  {f1}{f2}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY: All values with no flag match within 0.5 km / 1 percentage point.")
print("Values flagged with *** DISCREPANCY *** exceed the tolerance.")
print("=" * 90)
