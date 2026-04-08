"""
Poster-ready summary table for 2D Lemasquerier latitude-resolved results.

Pairs with latitude_structure_polar.png — provides the legend context
(colour key for D_cond / D_conv / H_total) plus key statistics in
a compact table readable at poster distance.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

RA_CRIT = 1000.0

C_LID = "#4A90D9"
C_CONV = "#E8645A"
C_TOTAL = "#333333"
C_LID_DARK = "#2C5F99"
C_CONV_DARK = "#B83B31"

SCENARIOS = [
    ("mc_2d_lemasquerier2023_polar_500.npz",
     "Polar-enhanced\n(conservative, q*=0.455)"),
    ("mc_2d_lemasquerier2023_polar_strong_500.npz",
     "Polar-enhanced\n(strong, q*=0.819)"),
]


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows = []
    for fname, label in SCENARIOS:
        d = np.load(os.path.join(RESULTS_DIR, fname), allow_pickle=True)
        H = d["H_profiles"]
        Dc = d["D_cond_profiles"]
        Dv = d["D_conv_profiles"]
        Ra = d["Ra_profiles"]
        n = len(H)

        conv_frac = np.mean(Ra >= RA_CRIT, axis=1)
        n_conv = np.sum(conv_frac >= 0.5)

        # Equatorial column
        H_eq = np.median(H[:, 0])
        Dc_eq = np.median(Dc[:, 0])
        lid_eq = Dc_eq / H_eq * 100

        # Polar column
        H_pole = np.median(H[:, -1])
        Dc_pole = np.median(Dc[:, -1])
        lid_pole = Dc_pole / H_pole * 100

        dH = H_pole - H_eq
        ratio = H_pole / H_eq

        rows.append({
            "label": label,
            "n": n, "n_conv": n_conv,
            "H_eq": H_eq, "Dc_eq": Dc_eq, "lid_eq": lid_eq,
            "H_pole": H_pole, "Dc_pole": Dc_pole, "lid_pole": lid_pole,
            "dH": dH, "ratio": ratio,
        })

    # Build table figure
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.axis("off")

    col_labels = [
        "Scenario",
        "N\n(conv/total)",
        "Eq H\n(km)",
        r"Eq $D_{\rm cond}$" + "\n(km)",
        "Eq lid\n(%)",
        "Pole H\n(km)",
        r"Pole $D_{\rm cond}$" + "\n(km)",
        "Pole lid\n(%)",
        r"$\Delta H$" + "\n(km)",
        "Pole/Eq\nratio",
    ]

    cell_text = []
    for r in rows:
        cell_text.append([
            r["label"].replace("\n", " "),
            f"{r['n_conv']}/{r['n']}",
            f"{r['H_eq']:.1f}",
            f"{r['Dc_eq']:.1f}",
            f"{r['lid_eq']:.0f}",
            f"{r['H_pole']:.1f}",
            f"{r['Dc_pole']:.1f}",
            f"{r['lid_pole']:.0f}",
            f"+{r['dH']:.1f}",
            f"{r['ratio']:.2f}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(fontweight="bold", fontsize=8)

    # Colour key legend below table
    legend_elements = [
        Patch(facecolor=C_LID, alpha=0.5, edgecolor=C_LID_DARK,
              label=r"Conductive lid ($D_{\rm cond}$)"),
        Patch(facecolor=C_CONV, alpha=0.5, edgecolor=C_CONV_DARK,
              label=r"Convective sublayer ($D_{\rm conv}$)"),
        Patch(facecolor=C_TOTAL, alpha=0.3, edgecolor=C_TOTAL,
              label=r"Total shell $H$ (16$-$84th pctl)"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, fontsize=9, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "poster_table_polar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
