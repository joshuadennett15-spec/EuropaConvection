# Latitude-Band Violin Plots for 2D Monte Carlo Shell Structure

## 1. Why latitude bands, not single nodes

When comparing equatorial and polar shell structure across Monte Carlo
ensembles, it is tempting to compare the 0 deg node directly against the
90 deg node. This is statistically and physically fragile for three reasons:

1. **Surface-area representation.** A single node at 0 deg represents a much
   larger surface area than a single node at 90 deg. In axisymmetric
   coordinates the area element scales as cos(phi), so the pole node
   represents a vanishingly small area of Europa's actual surface. Reporting
   a single polar value implicitly overweights a physically tiny region.

2. **Grid sensitivity.** The exact 90 deg node sits at the coordinate pole
   where cos(phi) = 0. Although the solver handles this correctly via
   L'Hopital's rule (see `axial_solver.py`), the surface temperature
   function drops steeply near the pole (50% of the total T_s drop occurs
   in the last 15 degrees due to cos^p(phi) geometry). A single node is
   therefore sensitive to the specific n_lat grid spacing.

3. **Geological relevance.** "Equatorial" and "polar" geology on Europa refers
   to regional zones, not one-dimensional lines. Chaos terrain clusters within
   roughly +/-30 deg of the equator; polar terrain extends from roughly 60 deg
   poleward. Band averages better represent the regions a geologist would map.

## 2. Band definitions

Two band pairs are used in this project, serving different purposes:

### Narrow bands (diagnostic, used in profile_diagnostics.py)

| Band | Range | Purpose |
|------|-------|---------|
| LOW_LAT_BAND | 0-10 deg | Equatorial endpoint, closest to the Juno MWR swath (20S-50N) |
| HIGH_LAT_BAND | 80-90 deg | Polar endpoint, captures the steepest part of the thickness ramp |

These are defined in `Europa2D/src/profile_diagnostics.py` and are pre-computed
in the MC results NPZ files as `H_low_band`, `H_high_band`, `D_cond_low_band`,
`D_cond_high_band` (one value per MC sample, area-weighted).

### Wide bands (geological, used for violin plots)

| Band | Range | Surface area fraction | Purpose |
|------|-------|-----------------------|---------|
| Equatorial | 0-30 deg | 50% of hemisphere | Covers the chaos-terrain latitude range |
| Polar | 60-90 deg | 13.4% of hemisphere | Covers the polar terrain zone, dilutes pole-node sensitivity |

Wide bands are better for violin plots because they average over more grid
points per sample, giving smoother KDE estimates. They also align with the
geological zones that a reviewer would associate with "equatorial" and "polar."

### Area weighting is mandatory

All band means must be weighted by cos(phi), the hemisphere-area Jacobian.
An unweighted arithmetic mean of grid nodes overweights the pole (where nodes
are crowded in angular space but represent tiny surface area). The existing
`band_mean_samples()` in `profile_diagnostics.py` implements this correctly:

```python
weights = np.cos(np.radians(latitudes[mask]))
return np.average(profiles[:, mask], axis=1, weights=weights)
```

Using `np.mean` instead of `np.average(..., weights=cos)` is a scientific
error that biases polar bands toward the pole node and equatorial bands
toward the 30 deg edge.

## 3. Why violin plots

Standard box plots show only quartiles (median, 25th, 75th percentiles).
Europa's parameter space is highly nonlinear: the conductive-convective
transition creates bimodal or heavily skewed thickness distributions where
some MC samples produce thin convective shells and others produce thick
conductive shells from the same prior.

A violin plot combines a box plot with a kernel density estimator (KDE),
showing the full shape of the probability distribution. This reveals:

- Whether a scenario has a long tail of thick, purely conductive shells
- Whether the distribution is bimodal (split between convective and conductive
  regimes)
- How much overlap exists between equatorial and polar bands within and across
  scenarios

The `cut=0` parameter prevents the KDE from extrapolating beyond the data range,
avoiding unphysical negative thicknesses.

## 4. Quantities to plot

The vetting document (`2026-03-19-1d-vs-2d-methodology-vetting.md`) requires
four-quantity shell-structure reporting. The violin plot should show:

| Quantity | NPZ key (profiles) | Physical meaning |
|----------|-------------------|------------------|
| $H_{\text{total}}$ | `H_profiles` | Total ice shell thickness (surface to ocean) |
| $D_{\text{cond}}$ | `D_cond_profiles` | Conductive lid thickness (Juno-facing observable) |
| $D_{\text{conv}}$ | `D_conv_profiles` | Convective sublayer thickness |
| Conv. fraction | `D_conv / H` | Fraction of shell that is convecting |

For the primary thesis figure, $D_{\text{cond}}$ is the most important quantity
because it maps directly to the Juno MWR constraint (Levin et al. 2026:
29 +/- 10 km). The Juno constraint should be overlaid on the equatorial band
as a shaded horizontal region.

## 5. Baseline subtraction

The surface-temperature-dominance analysis (`2026-03-20-surface-temperature-dominance.md`)
showed that even the uniform_transport scenario has a nonzero DeltaH due to the
tidal strain gradient. The ocean transport signal should be isolated as:

$$\Delta H_{\text{ocean}} = \Delta H_{\text{scenario}} - \Delta H_{\text{uniform}}$$

For violin plots, this means the uniform_transport scenario serves as the
**null hypothesis** baseline. Visual comparison of band distributions across
scenarios should be interpreted relative to this baseline, not as absolute
ocean-driven contrasts.

## 6. Implementation

The script below uses the existing `band_mean_samples()` infrastructure and
the publication style from `pub_style.py`. It loads the NPZ files, computes
wide-band (0-30 deg, 60-90 deg) area-weighted means, and produces a split
violin plot with Juno overlay.

```python
"""
Latitude-band violin plots for 2D MC shell structure.

Produces split violin plots of D_cond (and optionally H_total) comparing
equatorial (0-30 deg) and polar (60-90 deg) bands across three ocean
transport scenarios.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(_PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "..", "EuropaProjectDJ", "src"))

from profile_diagnostics import band_mean_samples
from pub_style import apply_style, PAL, save_fig

# --- Configuration ---
RESULTS_DIR = os.path.join(_PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(_PROJECT_DIR, "figures")

# Band definitions (wide, geological)
EQ_BAND = (0.0, 30.0)
POLAR_BAND = (60.0, 90.0)

# Juno MWR constraint on D_cond (Levin et al. 2026)
JUNO_D_COND = 29.0  # km
JUNO_D_COND_ERR = 10.0  # km

SCENARIOS = [
    ("uniform_transport", "Uniform"),
    ("soderlund2014_equator", "Equator-enhanced"),
    ("lemasquerier2023_polar", "Polar-enhanced"),
]


def load_band_means(scenario_name, n_iter, quantity="D_cond_profiles"):
    """Load NPZ and compute area-weighted band means for each MC sample.

    Returns (eq_samples, polar_samples) arrays of shape (n_valid,).
    """
    path = os.path.join(RESULTS_DIR, f"mc_2d_{scenario_name}_{n_iter}.npz")
    d = np.load(path, allow_pickle=True)
    lats = d["latitudes_deg"]
    profiles = d[quantity]  # (n_valid, n_lat)

    eq = band_mean_samples(lats, profiles, EQ_BAND)
    polar = band_mean_samples(lats, profiles, POLAR_BAND)
    return eq, polar


def violin_half(ax, data, positions, side, color, alpha=0.7):
    """Draw one half of a split violin (left or right).

    Uses matplotlib's violinplot and clips the polygon bodies.
    """
    parts = ax.violinplot(
        data, positions=positions, widths=0.7,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in parts["bodies"]:
        # Get the polygon vertices
        m = np.mean(body.get_paths()[0].vertices[:, 0])
        if side == "left":
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], -np.inf, m
            )
        else:
            body.get_paths()[0].vertices[:, 0] = np.clip(
                body.get_paths()[0].vertices[:, 0], m, np.inf
            )
        body.set_facecolor(color)
        body.set_edgecolor("k")
        body.set_linewidth(0.5)
        body.set_alpha(alpha)

    # Add median lines
    for i, d in enumerate(data):
        med = np.median(d)
        lo, hi = np.percentile(d, [25, 75])
        x = positions[i]
        dx = 0.15
        if side == "left":
            ax.hlines(med, x - dx, x, color="k", lw=1.2)
            ax.hlines([lo, hi], x - dx * 0.6, x, color="k", lw=0.6, linestyle="--")
        else:
            ax.hlines(med, x, x + dx, color="k", lw=1.2)
            ax.hlines([lo, hi], x, x + dx * 0.6, color="k", lw=0.6, linestyle="--")


def plot_dcond_violins(n_iter=20):
    """Main violin plot: D_cond by scenario, split by latitude band."""
    apply_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    positions = np.arange(len(SCENARIOS))
    eq_data = []
    polar_data = []

    for name, _ in SCENARIOS:
        eq, polar = load_band_means(name, n_iter, "D_cond_profiles")
        eq_data.append(eq)
        polar_data.append(polar)

    c_eq = PAL.CYAN
    c_polar = PAL.ORANGE

    violin_half(ax, eq_data, positions, "left", c_eq)
    violin_half(ax, polar_data, positions, "right", c_polar)

    # Juno constraint (equatorial D_cond)
    ax.axhspan(
        JUNO_D_COND - JUNO_D_COND_ERR,
        JUNO_D_COND + JUNO_D_COND_ERR,
        color=PAL.GREEN, alpha=0.12, zorder=0,
    )
    ax.axhline(JUNO_D_COND, color=PAL.GREEN, lw=0.8, ls="--", zorder=0)
    ax.text(
        len(SCENARIOS) - 0.5, JUNO_D_COND + 1,
        f"Juno MWR: {JUNO_D_COND} +/- {JUNO_D_COND_ERR:.0f} km",
        fontsize=7, color=PAL.GREEN, ha="right",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in SCENARIOS])
    ax.set_ylabel(r"$D_{\mathrm{cond}}$ (km)")
    ax.set_title(
        r"Conductive lid thickness by ocean transport scenario"
        "\n"
        r"Split violin: equatorial (0-30 deg, left) vs polar (60-90 deg, right)",
        fontsize=9,
    )

    # Legend
    eq_patch = mpatches.Patch(facecolor=c_eq, alpha=0.7, edgecolor="k", lw=0.5,
                               label="Equatorial (0-30 deg)")
    polar_patch = mpatches.Patch(facecolor=c_polar, alpha=0.7, edgecolor="k", lw=0.5,
                                  label="Polar (60-90 deg)")
    ax.legend(handles=[eq_patch, polar_patch], loc="upper left", fontsize=7)

    ax.set_ylim(0, None)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, "dcond_violin_bands.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_dcond_violins()
```

## 7. Interpretation guide

When reading the violin plots, check for:

1. **Overlap between bands within a scenario.** If the equatorial and polar
   distributions overlap heavily, the ocean transport pattern is not producing
   a strong latitude contrast at the population level. This is expected for
   the uniform scenario.

2. **Shift relative to the Juno constraint.** The Juno MWR constraint
   (29 +/- 10 km) applies to equatorial D_cond only. If the equatorial band
   median falls within the Juno range, the scenario is observationally
   consistent. If it falls outside, the scenario is in tension with Juno.

3. **Distribution shape.** A bimodal distribution (two humps) indicates the
   MC ensemble spans the conductive-convective transition: some samples
   produce thin convective shells, others thick conductive shells. A unimodal
   distribution means the regime is more uniform across the prior.

4. **Baseline subtraction.** The uniform scenario's equator-to-pole contrast
   is the tidal + surface-temperature baseline. Any additional contrast in the
   equator-enhanced or polar-enhanced scenarios is attributable to the ocean
   transport pattern.
