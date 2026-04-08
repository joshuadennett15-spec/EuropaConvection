"""
Generate publication-ready PDF tables for all Monte Carlo results.

Produces one multi-page PDF with Tables 1--5 matching the complete
results document (docs/research/2026-03-20-complete-results-tables.md).

Uses matplotlib table rendering with pub_style formatting.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from pub_style import apply_style, DOUBLE_COL

apply_style()

FIGURES_DIR = os.path.join(_SCRIPT_DIR, "..", "figures", "pub")
RESULTS_1D = os.path.join(_SCRIPT_DIR, "..", "..", "EuropaProjectDJ", "results")
RESULTS_2D = os.path.join(_SCRIPT_DIR, "..", "results")

RA_CRIT = 1000.0

# ── Style constants ──────────────────────────────────────────────────────────

HEADER_BG = "#E8E8E8"
HEADER_FG = "#1a1a1a"
ROW_ALT = "#F7F7F7"
ROW_WHITE = "#FFFFFF"
HIGHLIGHT_BG = "#E8F4E8"  # soft green for best-match rows
FONT_SIZE = 6.5
HEADER_FONT = 7.0
TITLE_FONT = 9


def _style_table(table, n_cols, n_rows, highlight_rows=None):
    """Apply consistent styling to a matplotlib table."""
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(fontweight="bold", fontsize=HEADER_FONT,
                            color=HEADER_FG)
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)

    for i in range(1, n_rows + 1):
        bg = ROW_ALT if i % 2 == 0 else ROW_WHITE
        if highlight_rows and (i - 1) in highlight_rows:
            bg = HIGHLIGHT_BG
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(0.3)


def _make_table_page(fig, ax, title, subtitle, col_labels, cell_text,
                     col_widths=None, highlight_rows=None, row_height=1.6):
    """Build a single table on the given axes."""
    ax.axis("off")
    ax.set_title(title, fontsize=TITLE_FONT, fontweight="bold",
                 loc="left", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=6.5, color="#555555", va="bottom")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        colWidths=col_widths,
    )
    _style_table(table, len(col_labels), len(cell_text),
                 highlight_rows=highlight_rows)
    table.scale(1.0, row_height)
    return table


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_1d(path):
    d = np.load(path, allow_pickle=True)
    H = d["thicknesses_km"]
    Dc = d["D_cond_km"]
    Dv = d["D_conv_km"]
    Ra = d["Ra_values"]
    cbe = float(d["cbe_km"])
    return H, Dc, Dv, Ra, cbe


def _stats_1d(H, Dc, Dv, Ra):
    return {
        "N": len(H),
        "med_H": np.median(H),
        "H_lo": np.percentile(H, 16),
        "H_hi": np.percentile(H, 84),
        "med_Dc": np.median(Dc),
        "Dc_lo": np.percentile(Dc, 16),
        "Dc_hi": np.percentile(Dc, 84),
        "med_Dv": np.median(Dv),
        "Dv_lo": np.percentile(Dv, 16),
        "Dv_hi": np.percentile(Dv, 84),
        "conv_frac": np.mean(Ra >= RA_CRIT) * 100,
        "lid_frac": np.mean(Dc / H) * 100,
    }


def _fmt(val, prec=1):
    return f"{val:.{prec}f}"


def _fmt_range(lo, hi, prec=1):
    return f"[{lo:.{prec}f}, {hi:.{prec}f}]"


# ── Table 1 ──────────────────────────────────────────────────────────────────

def table1_data():
    rows = []
    for label, fname in [
        ("Howell (Maxwell)", "mc_15000_howell.npz"),
        ("Audited (Andrade)", "mc_15000_optionA_v2_andrade.npz"),
    ]:
        H, Dc, Dv, Ra, cbe = _load_1d(os.path.join(RESULTS_1D, fname))
        s = _stats_1d(H, Dc, Dv, Ra)
        rows.append([
            label,
            f"{s['N']:,}",
            _fmt(cbe),
            _fmt(s["med_H"]),
            _fmt_range(s["H_lo"], s["H_hi"]),
            _fmt(s["med_Dc"]),
            _fmt_range(s["Dc_lo"], s["Dc_hi"]),
            _fmt(s["med_Dv"]),
            _fmt_range(s["Dv_lo"], s["Dv_hi"]),
            f"{s['conv_frac']:.1f}%",
            f"{s['lid_frac']:.0f}%",
        ])
    return rows


# ── Table 2 ──────────────────────────────────────────────────────────────────

TABLE2_MODES = [
    ("Depleted strong", "0.55x", "eq_depleted_strong_andrade.npz"),
    ("Depleted",        "0.67x", "eq_depleted_andrade.npz"),
    ("Baseline",        "1.0x",  "eq_baseline_andrade.npz"),
    ("Moderate",        "1.2x",  "eq_moderate_andrade.npz"),
    ("Strong",          "1.5x",  "eq_strong_andrade.npz"),
]


def table2_data():
    rows = []
    for label, enh, fname in TABLE2_MODES:
        H, Dc, Dv, Ra, cbe = _load_1d(os.path.join(RESULTS_1D, fname))
        s = _stats_1d(H, Dc, Dv, Ra)
        rows.append([
            label, enh,
            f"{s['N']:,}",
            _fmt(cbe),
            _fmt(s["med_H"]),
            _fmt_range(s["H_lo"], s["H_hi"]),
            _fmt(s["med_Dc"]),
            _fmt_range(s["Dc_lo"], s["Dc_hi"]),
            _fmt(s["med_Dv"]),
            _fmt_range(s["Dv_lo"], s["Dv_hi"]),
            f"{s['conv_frac']:.1f}%",
            f"{s['lid_frac']:.0f}%",
        ])
    return rows


# ── Table 3 ──────────────────────────────────────────────────────────────────

SCENARIOS_2D = [
    ("Uniform transport",   "mc_2d_uniform_transport_500.npz"),
    ("Soderlund (q*=0.4)",  "mc_2d_soderlund2014_equator_500.npz"),
    ("Lemasquerier (q*=0.455)", "mc_2d_lemasquerier2023_polar_500.npz"),
    ("Lemasquerier (q*=0.819)", "mc_2d_lemasquerier2023_polar_strong_500.npz"),
]


def table3_data():
    rows_3a, rows_3b, rows_3c, rows_3d = [], [], [], []
    for label, fname in SCENARIOS_2D:
        d = np.load(os.path.join(RESULTS_2D, fname), allow_pickle=True)
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Dv = d["D_conv_profiles"]
        Ra = d["Ra_profiles"]
        N = len(H)

        # 3a: lat-averaged
        H_avg = np.mean(H, axis=1)
        conv_per = np.mean(Ra >= RA_CRIT, axis=1)
        eq_conv_frac = np.mean(Ra[:, 0] >= RA_CRIT) * 100.0
        majority_conv_frac = np.mean(conv_per >= 0.5) * 100.0
        rows_3a.append([
            label, str(N),
            _fmt(np.median(H_avg)),
            _fmt_range(np.percentile(H_avg, 16), np.percentile(H_avg, 84)),
            f"{eq_conv_frac:.1f}%",
            f"{majority_conv_frac:.1f}%",
        ])

        # 3b: equatorial
        H_eq, Dc_eq, Dv_eq = H[:, 0], Dc[:, 0], Dv[:, 0]
        rows_3b.append([
            label,
            _fmt(np.median(H_eq)),
            _fmt_range(np.percentile(H_eq, 16), np.percentile(H_eq, 84)),
            _fmt(np.median(Dc_eq)),
            _fmt_range(np.percentile(Dc_eq, 16), np.percentile(Dc_eq, 84)),
            _fmt(np.median(Dv_eq)),
            _fmt_range(np.percentile(Dv_eq, 16), np.percentile(Dv_eq, 84)),
            f"{np.mean(Dc_eq / H_eq) * 100:.0f}%",
        ])

        # 3c: polar
        H_p, Dc_p, Dv_p = H[:, -1], Dc[:, -1], Dv[:, -1]
        rows_3c.append([
            label,
            _fmt(np.median(H_p)),
            _fmt_range(np.percentile(H_p, 16), np.percentile(H_p, 84)),
            _fmt(np.median(Dc_p)),
            _fmt_range(np.percentile(Dc_p, 16), np.percentile(Dc_p, 84)),
            _fmt(np.median(Dv_p)),
            _fmt_range(np.percentile(Dv_p, 16), np.percentile(Dv_p, 84)),
            f"{np.mean(Dc_p / H_p) * 100:.0f}%",
        ])

        # 3d: contrast
        dH = np.median(H_p) - np.median(H_eq)
        dDc = np.median(Dc_p) - np.median(Dc_eq)
        ratio = np.median(H_p) / np.median(H_eq)
        rows_3d.append([
            label,
            f"+{dH:.1f}",
            f"+{dDc:.1f}",
            f"{ratio:.2f}",
        ])

    return rows_3a, rows_3b, rows_3c, rows_3d


# ── Table 4 ──────────────────────────────────────────────────────────────────

def _juno_assessment(med, hi):
    """Classify Juno overlap based on median D_cond and 84th percentile."""
    # Juno window: 19--39 km (29 +/- 10)
    if hi < 19:
        return "No"
    if med >= 19 and hi >= 29:
        return "Excellent"
    if med >= 19:
        return "Good"
    if hi >= 29:
        return "Best 1D" if med >= 16.5 else "Partial"
    if hi >= 25:
        return "Marginal"
    return "Poor"


def table4_data():
    """Cross-model Juno comparison — derived from loaded data."""
    rows = []

    # 1D global — hardcode assessments matching the markdown analysis
    assessments_1d_global = {
        "Howell (Maxwell)": "No",
        "Audited (Andrade)": "Marginal",
    }
    for label, fname, model in [
        ("Howell (Maxwell)", "mc_15000_howell.npz", "1D global"),
        ("Audited (Andrade)", "mc_15000_optionA_v2_andrade.npz", "1D global"),
    ]:
        H, Dc, Dv, Ra, _ = _load_1d(os.path.join(RESULTS_1D, fname))
        med = np.median(Dc)
        hi = np.percentile(Dc, 84)
        rows.append([model, label, _fmt(med), _fmt(hi),
                      assessments_1d_global[label]])

    # 1D equatorial
    assessments_1d_eq = {
        "Depleted strong": "Partial",
        "Depleted": "Partial",
        "Baseline": "Best 1D",
        "Moderate": "Good",
        "Strong": "Poor",
    }
    for label, enh, fname in TABLE2_MODES:
        H, Dc, Dv, Ra, _ = _load_1d(os.path.join(RESULTS_1D, fname))
        med = np.median(Dc)
        hi = np.percentile(Dc, 84)
        rows.append(["1D equatorial", f"{label} ({enh})", _fmt(med), _fmt(hi),
                      assessments_1d_eq[label]])

    # 2D lat-resolved
    assessments_2d = {
        "Uniform transport": "Excellent",
        "Soderlund (q*=0.4)": "Excellent",
        "Lemasquerier (q*=0.455)": "Best overall",
        "Lemasquerier (q*=0.819)": "Marginal",
    }
    for label, fname in SCENARIOS_2D:
        d = np.load(os.path.join(RESULTS_2D, fname), allow_pickle=True)
        Dc_eq = d["D_cond_profiles"][:, 0]
        med = np.median(Dc_eq)
        hi = np.percentile(Dc_eq, 84)
        rows.append(["2D lat-resolved", label, _fmt(med), _fmt(hi),
                      assessments_2d[label]])

    return rows


# ── Table 5 (Bayesian) ──────────────────────────────────────────────────────

def table5_data():
    """Compute Bayesian evidence from 1D equatorial data."""
    from scipy.special import logsumexp

    SIGMA_MODEL = 3.0
    JUNO_OBS = [(29.0, 10.0, "A"), (24.0, 10.0, "B")]

    def log_weights(Dc, D_obs, sigma_obs):
        sigma = np.sqrt(sigma_obs**2 + SIGMA_MODEL**2)
        return -0.5 * ((Dc - D_obs) / sigma)**2

    def norm_weights(lw):
        w = np.exp(lw - np.max(lw))
        return w / w.sum()

    def ess(w):
        return 1.0 / np.sum(w**2)

    def wpct(v, w, p):
        idx = np.argsort(v)
        return v[idx][np.searchsorted(np.cumsum(w[idx]), p / 100.0)]

    results = []
    for label, enh, fname in TABLE2_MODES:
        d = np.load(os.path.join(RESULTS_1D, fname), allow_pickle=True)
        Dc = d["D_cond_km"]
        Ra = d["Ra_values"]
        row = {"label": label, "enh": enh}

        for D_obs, sigma_obs, tag in JUNO_OBS:
            lw = log_weights(Dc, D_obs, sigma_obs)
            log_ml = logsumexp(lw) - np.log(len(lw))
            w = norm_weights(lw)
            row[f"log_ml_{tag}"] = log_ml
            row[f"ess_{tag}"] = ess(w)
            row[f"med_{tag}"] = wpct(Dc, w, 50)
            row[f"lo_{tag}"] = wpct(Dc, w, 16)
            row[f"hi_{tag}"] = wpct(Dc, w, 84)
            row[f"conv_{tag}"] = np.sum(w[Ra >= RA_CRIT]) * 100
        results.append(row)

    bl = next(r for r in results if r["label"] == "Baseline")
    for r in results:
        for t in ["A", "B"]:
            r[f"bf_{t}"] = np.exp(r[f"log_ml_{t}"] - bl[f"log_ml_{t}"])

    # 5a: Bayes factors
    rows_5a = []
    for r in results:
        rows_5a.append([
            f"{r['label']} ({r['enh']})",
            f"{r['log_ml_A']:.3f}",
            f"{r['log_ml_B']:.3f}",
            f"{r['bf_A']:.3f}",
            f"{r['bf_B']:.3f}",
        ])

    # 5b: posterior D_cond
    rows_5b = []
    for r in results:
        rows_5b.append([
            f"{r['label']} ({r['enh']})",
            _fmt(r["med_A"]),
            _fmt_range(r["lo_A"], r["hi_A"]),
            _fmt(r["med_B"]),
            _fmt_range(r["lo_B"], r["hi_B"]),
            f"{r['ess_A']:,.0f}",
            f"{r['ess_B']:,.0f}",
        ])

    # 5c: posterior conv frac
    rows_5c = []
    for r in results:
        rows_5c.append([
            f"{r['label']} ({r['enh']})",
            f"{r['conv_A']:.1f}%",
            f"{r['conv_B']:.1f}%",
        ])

    return rows_5a, rows_5b, rows_5c


# ── PDF generation ───────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, "results_tables.pdf")

    with PdfPages(pdf_path) as pdf:

        # ── Table 1 ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.0))
        _make_table_page(
            fig, ax,
            "Table 1: 1D Global Baselines (N = 15,000)",
            "Two rheology models with audited 2026 priors, no latitude forcing.",
            ["Run", "N", "CBE\n(km)", "Med H\n(km)", "1\u03c3 H\n(km)",
             "Med D_cond\n(km)", "1\u03c3 D_cond\n(km)",
             "Med D_conv\n(km)", "1\u03c3 D_conv\n(km)",
             "Conv.\nfrac", "Lid\nfrac"],
            table1_data(),
            col_widths=[0.16, 0.07, 0.06, 0.06, 0.11, 0.08, 0.11, 0.08, 0.11, 0.07, 0.06],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 2 ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.2))
        _make_table_page(
            fig, ax,
            "Table 2: 1D Equatorial Proxy Suite (N = 15,000, Andrade)",
            "Ocean heat transport scaled as multiplier on tidal basal heat flux. Shared RNG seed 10042.",
            ["Mode", "Enh.", "N", "CBE\n(km)", "Med H\n(km)", "1\u03c3 H\n(km)",
             "Med D_cond\n(km)", "1\u03c3 D_cond\n(km)",
             "Med D_conv\n(km)", "1\u03c3 D_conv\n(km)",
             "Conv.\nfrac", "Lid\nfrac"],
            table2_data(),
            col_widths=[0.12, 0.05, 0.06, 0.06, 0.06, 0.11, 0.08, 0.11, 0.08, 0.11, 0.06, 0.05],
            highlight_rows=[2],  # baseline
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 3a ─────────────────────────────────────────────────
        r3a, r3b, r3c, r3d = table3_data()

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
        _make_table_page(
            fig, ax,
            "Table 3a: 2D Latitude-Resolved \u2014 Global Statistics",
            "N = 500 MC samples, 37 latitude columns, Andrade rheology. Eq. conv. frac is directly comparable to 1D; Maj. cols conv. frac is the stricter whole-shell metric.",
            ["Scenario", "N valid", "Med H (km)", "1\u03c3 H (km)",
             "Eq. conv.\nfrac", "Maj. cols\nconv. frac"],
            r3a,
            col_widths=[0.28, 0.09, 0.14, 0.22, 0.13, 0.14],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 3b ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
        _make_table_page(
            fig, ax,
            "Table 3b: 2D Equatorial Column (lat = 0\u00b0)",
            None,
            ["Scenario", "Med H\n(km)", "1\u03c3 H\n(km)",
             "Med D_cond\n(km)", "1\u03c3 D_cond\n(km)",
             "Med D_conv\n(km)", "1\u03c3 D_conv\n(km)", "Lid\nfrac"],
            r3b,
            col_widths=[0.20, 0.07, 0.14, 0.08, 0.14, 0.08, 0.14, 0.06],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 3c ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
        _make_table_page(
            fig, ax,
            "Table 3c: 2D Polar Column (lat = 90\u00b0)",
            None,
            ["Scenario", "Med H\n(km)", "1\u03c3 H\n(km)",
             "Med D_cond\n(km)", "1\u03c3 D_cond\n(km)",
             "Med D_conv\n(km)", "1\u03c3 D_conv\n(km)", "Lid\nfrac"],
            r3c,
            col_widths=[0.20, 0.07, 0.14, 0.08, 0.14, 0.08, 0.14, 0.06],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 3d ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))
        _make_table_page(
            fig, ax,
            "Table 3d: Equator-to-Pole Contrast",
            None,
            ["Scenario", "\u0394H (km)", "\u0394D_cond (km)", "Pole/Eq\nH ratio"],
            r3d,
            col_widths=[0.35, 0.20, 0.20, 0.20],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 4 ──────────────────────────────────────────────────
        t4 = table4_data()
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 5.5))
        _make_table_page(
            fig, ax,
            "Table 4: Cross-Model Juno MWR Comparison",
            "Equatorial D_cond vs Juno constraint (29 \u00b1 10 km, acceptable 19\u201339 km).",
            ["Model", "Scenario", "Med D_cond\n(km)", "84th pctl\n(km)", "Juno\noverlap"],
            t4,
            col_widths=[0.15, 0.25, 0.15, 0.15, 0.20],
            highlight_rows=[4, 9],  # Best 1D + Best overall
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 5a ─────────────────────────────────────────────────
        r5a, r5b, r5c = table5_data()

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
        _make_table_page(
            fig, ax,
            "Table 5a: Bayesian Evidence \u2014 Marginal Likelihoods & Bayes Factors",
            "Importance-sampling reweighting vs Juno. BF relative to Baseline (1.0x). \u03c3_model = 3 km.",
            ["Mode", "log ML\n(A: 29 km)", "log ML\n(B: 24 km)",
             "BF_A", "BF_B"],
            r5a,
            col_widths=[0.30, 0.17, 0.17, 0.15, 0.15],
            highlight_rows=[2],  # baseline reference
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 5b ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
        _make_table_page(
            fig, ax,
            "Table 5b: Posterior D_cond Under Juno Constraint",
            None,
            ["Mode", "Post. D_cond\n(A, km)", "1\u03c3 (A, km)",
             "Post. D_cond\n(B, km)", "1\u03c3 (B, km)",
             "ESS_A", "ESS_B"],
            r5b,
            col_widths=[0.22, 0.12, 0.15, 0.12, 0.15, 0.10, 0.10],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # ── Table 5c ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
        _make_table_page(
            fig, ax,
            "Table 5c: Posterior Convective Fraction Under Juno Constraint",
            None,
            ["Mode", "Conv. frac\n(Model A)", "Conv. frac\n(Model B)"],
            r5c,
            col_widths=[0.40, 0.28, 0.28],
        )
        fig.tight_layout(pad=0.5)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
