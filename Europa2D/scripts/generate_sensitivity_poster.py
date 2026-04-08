"""
Generate poster-ready sensitivity analysis outputs:

  1. Tornado plot of Sobol total-order indices (S_T) for key QoIs (PNG + PDF)
  2. Standalone LaTeX table (booktabs) with S1 and S_T values

Reads 1D Sobol indices from EuropaProjectDJ/results/sobol/.
Uses pub_style for consistent journal formatting.

Usage:
    python scripts/generate_sensitivity_poster.py
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_1D_DIR = _PROJECT_DIR.parent / "EuropaProjectDJ"

sys.path.insert(0, str(_1D_DIR / "scripts"))
from pub_style import apply_style, PAL, figsize_double, label_panel, save_fig

FIGURES_DIR = _PROJECT_DIR / "figures"
SOBOL_DIR = _1D_DIR / "results" / "sobol"

# Run to use (global audited, individual parameters, N=512)
RUN_NAME = "global_audited_sobol_params_N512"

# QoIs to show on the poster (most physically meaningful)
POSTER_QOIS = ["thickness_km", "D_cond_km", "lid_fraction", "convective_flag"]
QOI_LABELS = {
    "thickness_km":    r"Total shell $H$ (km)",
    "D_cond_km":       r"Conductive lid $D_{\mathrm{cond}}$ (km)",
    "lid_fraction":    r"Lid fraction $D_{\mathrm{cond}}/H$",
    "convective_flag": r"Convective probability",
}
QOI_LATEX = {
    "thickness_km":    r"Total shell $H$ (km)",
    "D_cond_km":       r"Conductive lid $D_\mathrm{cond}$ (km)",
    "lid_fraction":    r"Lid fraction $D_\mathrm{cond}/H$",
    "convective_flag": r"Convective probability",
}

FACTOR_LABEL = {
    "q_basal_target_mW_m2": r"$q_{\mathrm{basal}}$",
    "d_grain_mm":            r"$d_{\mathrm{grain}}$",
    "epsilon_0":             r"$\varepsilon_0$",
    "T_surf_K":              r"$T_{\mathrm{surf}}$",
    "D_H2O_km":              r"$D_{\mathrm{H_2O}}$",
    "mu_ice_GPa":            r"$\mu_{\mathrm{ice}}$",
    "Q_v_kJ_mol":            r"$Q_v$",
    "Q_b_kJ_mol":            r"$Q_b$",
    "H_rad_pW_kg":           r"$H_{\mathrm{rad}}$",
    "f_porosity":            r"$f_{\mathrm{por}}$",
}
FACTOR_LATEX = {
    "q_basal_target_mW_m2": r"$q_\mathrm{basal}$",
    "d_grain_mm":            r"$d_\mathrm{grain}$",
    "epsilon_0":             r"$\varepsilon_0$",
    "T_surf_K":              r"$T_\mathrm{surf}$",
    "D_H2O_km":              r"$D_\mathrm{H_2O}$",
    "mu_ice_GPa":            r"$\mu_\mathrm{ice}$",
    "Q_v_kJ_mol":            r"$Q_v$",
    "Q_b_kJ_mol":            r"$Q_b$",
    "H_rad_pW_kg":           r"$H_\mathrm{rad}$",
    "f_porosity":            r"$f_\mathrm{por}$",
}
FACTOR_DESCRIPTION = {
    "q_basal_target_mW_m2": "Basal heat flux",
    "d_grain_mm":            "Grain size",
    "epsilon_0":             "Tidal strain amplitude",
    "T_surf_K":              "Surface temperature",
    "D_H2O_km":              "Ocean layer thickness",
    "mu_ice_GPa":            "Ice shear modulus",
    "Q_v_kJ_mol":            "Volume diffusion activation energy",
    "Q_b_kJ_mol":            "Boundary diffusion activation energy",
    "H_rad_pW_kg":           "Radiogenic heating",
    "f_porosity":            "Porosity fraction",
}

# Significance threshold for S_T (dashed line on plot)
ST_THRESHOLD = 0.05


def load_sobol_data(run_dir):
    """Load and parse Sobol indices CSV into nested dict."""
    csv_files = list(run_dir.glob("*_indices.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No indices CSV in {run_dir}")
    rows = []
    with open(csv_files[0], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    data = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["index_type"] != "main":
            continue
        output = row["output"]
        try:
            n = int(row["sample_size"])
        except (ValueError, TypeError):
            continue
        factor = row["factor"]
        try:
            s1 = float(row["S1"])
            s1_conf = float(row["S1_conf"])
            st = float(row["ST"])
            st_conf = float(row["ST_conf"])
        except (ValueError, TypeError):
            continue
        if np.isnan(st):
            continue
        data[output][n][factor] = {
            "S1": s1, "S1_conf": s1_conf,
            "ST": st, "ST_conf": st_conf,
        }
    return data


def get_final_indices(data, qoi):
    """Get indices at the largest sample size for a given QoI."""
    if qoi not in data:
        return {}
    final_n = max(data[qoi].keys())
    return data[qoi][final_n]


# ── Tornado plot ─────────────────────────────────────────────────────────
def plot_tornado(data):
    """4-panel horizontal bar chart of S_T, sorted by magnitude."""
    qois = [q for q in POSTER_QOIS if q in data]
    n_qois = len(qois)

    fig, axes = plt.subplots(
        1, n_qois,
        figsize=(figsize_double()[0], 2.2),
        sharey=False,
        constrained_layout=True,
    )
    if n_qois == 1:
        axes = [axes]

    # Fixed colour per parameter (consistent across panels)
    factor_colours = {
        "q_basal_target_mW_m2": PAL.BLUE,
        "d_grain_mm":            PAL.ORANGE,
        "epsilon_0":             PAL.GREEN,
        "Q_v_kJ_mol":           PAL.RED,
        "mu_ice_GPa":           PAL.PURPLE,
        "T_surf_K":              PAL.CYAN,
        "D_H2O_km":              PAL.YELLOW,
        "Q_b_kJ_mol":           PAL.BLACK,
        "H_rad_pW_kg":          "#888888",
        "f_porosity":            "#AAAAAA",
    }
    letters = "abcdefgh"

    for idx, (ax, qoi) in enumerate(zip(axes, qois)):
        factors = get_final_indices(data, qoi)
        sorted_items = sorted(factors.items(), key=lambda x: x[1]["ST"], reverse=True)[:5]

        names = [FACTOR_LABEL.get(f, f) for f, _ in sorted_items]
        st_vals = np.array([v["ST"] for _, v in sorted_items])
        st_errs = np.array([v["ST_conf"] for _, v in sorted_items])

        # Clip negative values to zero for display
        st_vals_disp = np.maximum(st_vals, 0.0)

        colours = [factor_colours.get(f, "#888888") for f, _ in sorted_items]

        y_pos = np.arange(len(names))
        bars = ax.barh(
            y_pos, st_vals_disp,
            xerr=st_errs, height=0.6,
            color=colours,
            edgecolor="none",
            capsize=2, error_kw={"lw": 0.7, "capthick": 0.7},
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlabel(r"$S_T$ (total-order index)")
        ax.set_xlim(0, min(1.05, max(st_vals_disp) * 1.35 + 0.05))

        # Significance threshold
        ax.axvline(ST_THRESHOLD, color="0.55", ls=":", lw=0.6, zorder=0)

        ax.invert_yaxis()
        label_panel(ax, letters[idx], x=-0.22)
        ax.set_title(QOI_LABELS.get(qoi, qoi), fontsize=8)

    fig.suptitle(
        r"Global Sobol sensitivity ($N = 512$, Andrade rheology, audited priors)",
        fontsize=9, fontweight="bold",
    )
    return fig


# ── LaTeX table ──────────────────────────────────────────────────────────
def generate_latex_table(data):
    """Generate a standalone booktabs LaTeX table of S1 and S_T values."""
    qois = [q for q in POSTER_QOIS if q in data]

    # Collect all factors (ordered by mean S_T across QoIs)
    all_factors = set()
    for qoi in qois:
        all_factors.update(get_final_indices(data, qoi).keys())
    all_factors = sorted(all_factors)

    # Rank by mean S_T, keep top 5
    mean_st = {}
    for f in all_factors:
        vals = []
        for qoi in qois:
            indices = get_final_indices(data, qoi)
            if f in indices:
                vals.append(max(indices[f]["ST"], 0.0))
        mean_st[f] = np.mean(vals) if vals else 0.0
    all_factors = sorted(all_factors, key=lambda f: mean_st[f], reverse=True)[:5]

    # Build LaTeX
    n_qoi = len(qois)
    col_spec = "l" + "cc" * n_qoi
    header_row_1 = "Parameter"
    header_row_2 = ""
    for qoi in qois:
        short = {
            "thickness_km": r"$H$",
            "D_cond_km": r"$D_\mathrm{cond}$",
            "lid_fraction": r"Lid frac.",
            "convective_flag": r"Conv.\ prob.",
        }.get(qoi, qoi)
        header_row_1 += f" & \\multicolumn{{2}}{{c}}{{{short}}}"
    header_row_1 += r" \\"

    for _ in qois:
        header_row_2 += r" & $S_1$ & $S_T$"
    header_row_2 += r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sobol sensitivity indices for Europa ice shell model "
        r"($N = 512$ base samples, Andrade rheology, audited priors). "
        r"Bold values indicate $S_T > 0.05$.}",
        r"\label{tab:sobol}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row_1,
    ]

    # Cmidrules under each QoI pair
    cmidrules = ""
    for i in range(n_qoi):
        col_start = 2 + 2 * i
        col_end = col_start + 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}} "
    lines.append(cmidrules.strip())
    lines.append(header_row_2)
    lines.append(r"\midrule")

    for f in all_factors:
        label = FACTOR_LATEX.get(f, f)
        desc = FACTOR_DESCRIPTION.get(f, "")
        row = f"{label}"
        for qoi in qois:
            indices = get_final_indices(data, qoi)
            if f in indices:
                s1 = indices[f]["S1"]
                st = indices[f]["ST"]
                s1_str = f"{max(s1, 0.0):.3f}" if s1 > 0.005 else r"$<$0.01"
                if st > ST_THRESHOLD:
                    st_str = rf"\textbf{{{st:.3f}}}"
                elif st > 0.005:
                    st_str = f"{st:.3f}"
                else:
                    st_str = r"$<$0.01"
            else:
                s1_str = "---"
                st_str = "---"
            row += f" & {s1_str} & {st_str}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    apply_style()
    run_dir = SOBOL_DIR / RUN_NAME
    print(f"Loading Sobol data from: {run_dir}")
    data = load_sobol_data(run_dir)

    fig_dir = str(FIGURES_DIR)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Tornado plot
    fig = plot_tornado(data)
    if fig:
        save_fig(fig, "poster_sobol_tornado", fig_dir, formats=("png", "pdf"))

    # 2. LaTeX table
    tex = generate_latex_table(data)
    tex_path = FIGURES_DIR / "poster_sobol_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Saved: {tex_path}")


if __name__ == "__main__":
    main()
