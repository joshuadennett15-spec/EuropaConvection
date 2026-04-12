"""
Statistical synthesis of Sobol sensitivity results across physics configs.

Loads Sobol index CSVs and manifest JSONs from all three physics configs
(baseline, improved, wattmeter) and applies thesis_stats.py statistical
machinery:

  - Kendall W on parameter ST rankings across configs
  - Cliff's delta between config output distributions
  - Jonckheere-Terpstra trend test baseline -> improved -> wattmeter
  - LaTeX table export of S1/ST indices

Usage:
    python run_sobol_synthesis.py --input results/sobol/ --output results/sobol/synthesis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# Wire EuropaProjectDJ scripts onto the path for thesis_stats imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DJ_SCRIPTS = _PROJECT_ROOT / "EuropaProjectDJ" / "scripts"
sys.path.insert(0, str(_DJ_SCRIPTS))

from thesis_stats import (  # noqa: E402
    _cliffs_delta_from_u,
    _jonckheere_terpstra,
    _kendall_w,
)

# ── Constants ───────────────────────────────────────────────────────────────

CONFIG_ORDER = ("baseline", "improved", "wattmeter")

# QoIs to synthesise across configs
SYNTHESIS_QOIS = ("thickness_km", "D_cond_km", "D_conv_km", "lid_fraction", "convective_flag")

# JT-eligible QoIs (continuous, no heavy boundary mass)
JT_QOIS = ("thickness_km", "D_cond_km", "lid_fraction")

# JT alternative hypotheses per QoI
# baseline -> improved -> wattmeter: thickness expected to decrease, D_cond decrease,
# lid_fraction decrease (more convection with improved physics)
JT_ALTERNATIVES = {
    "thickness_km": "decreasing",
    "D_cond_km": "decreasing",
    "lid_fraction": "decreasing",
}


def _load_manifest(input_dir: Path, config_name: str, base_n: int) -> Dict[str, Any]:
    """Load a single config's manifest JSON."""
    run_label = f"{config_name}_N{base_n}"
    manifest_path = input_dir / run_label / f"{run_label}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Run run_sobol_analysis.py --config {config_name} --N {base_n} first."
        )
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_indices_csv(input_dir: Path, config_name: str, base_n: int) -> List[Dict[str, str]]:
    """Load a single config's Sobol indices CSV."""
    run_label = f"{config_name}_N{base_n}"
    csv_path = input_dir / run_label / f"{run_label}_indices.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Indices CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_design_outputs(
    input_dir: Path, config_name: str, base_n: int
) -> Dict[str, np.ndarray]:
    """Load output arrays from the design NPZ for one config."""
    run_label = f"{config_name}_N{base_n}"
    npz_path = input_dir / run_label / f"{run_label}_design.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Design NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    outputs = {}
    for key in data.files:
        if key.startswith("output_"):
            outputs[key[len("output_"):]] = np.asarray(data[key], dtype=float)
    return outputs


def _extract_final_st_rankings(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """
    From the indices CSV, extract the final-N ST values per (output, factor).

    Returns: {output_name: {factor: ST_value, ...}, ...}
    """
    # Find the maximum sample_size per output (that is the final checkpoint)
    max_n_by_output: Dict[str, int] = {}
    for row in rows:
        if not row.get("sample_size") or not row.get("ST"):
            continue
        output = row["output"]
        n = int(row["sample_size"])
        if output not in max_n_by_output or n > max_n_by_output[output]:
            max_n_by_output[output] = n

    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if not row.get("sample_size") or not row.get("ST"):
            continue
        output = row["output"]
        n = int(row["sample_size"])
        if n != max_n_by_output.get(output):
            continue
        result.setdefault(output, {})[row["factor"]] = float(row["ST"])

    return result


def _rank_factors(st_dict: Dict[str, float]) -> Dict[str, int]:
    """Rank factors by descending ST. Returns {factor: rank}."""
    sorted_factors = sorted(st_dict.keys(), key=lambda f: st_dict[f], reverse=True)
    return {factor: rank + 1 for rank, factor in enumerate(sorted_factors)}


def compute_kendall_w_across_configs(
    all_st: Dict[str, Dict[str, Dict[str, float]]],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """
    Kendall W on parameter ST rankings across physics configs.

    all_st: {config_name: {output: {factor: ST}}}
    Returns: {qoi: {W, chi2, p, n_factors, n_configs, rankings}}
    """
    config_names = list(all_st.keys())
    result: Dict[str, Any] = {}

    for qoi in qois:
        # Collect factors present in all configs for this QoI
        factor_sets = []
        for cfg in config_names:
            if qoi in all_st[cfg]:
                factor_sets.append(set(all_st[cfg][qoi].keys()))
        if len(factor_sets) < 2:
            continue

        common_factors = sorted(set.intersection(*factor_sets))
        if len(common_factors) < 2:
            continue

        # Build rankings matrix (k_raters x n_items)
        rankings_per_config = {}
        for cfg in config_names:
            if qoi not in all_st[cfg]:
                continue
            rankings_per_config[cfg] = _rank_factors(all_st[cfg][qoi])

        # Filter to configs that have this QoI
        valid_configs = [c for c in config_names if qoi in all_st[c]]
        if len(valid_configs) < 2:
            continue

        matrix = np.array([
            [rankings_per_config[c][f] for f in common_factors]
            for c in valid_configs
        ], dtype=float)

        W, chi2, p = _kendall_w(matrix)
        result[qoi] = {
            "W": W,
            "chi2": chi2,
            "p": p,
            "n_factors": len(common_factors),
            "n_configs": len(valid_configs),
            "factor_order": common_factors,
            "rankings": {
                c: {f: int(rankings_per_config[c][f]) for f in common_factors}
                for c in valid_configs
            },
        }

    return result


def compute_cliffs_delta_between_configs(
    all_outputs: Dict[str, Dict[str, np.ndarray]],
    qois: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Cliff's delta between each pair of physics configs for each QoI.

    all_outputs: {config_name: {qoi: array}}
    Returns: {pair_label: {qoi: {cliff_d, mw_U, mw_p, n_a, n_b}}}
    """
    config_names = list(all_outputs.keys())
    result: Dict[str, Dict[str, Any]] = {}

    for i, cfg_a in enumerate(config_names):
        for cfg_b in config_names[i + 1:]:
            pair_label = f"{cfg_a}_vs_{cfg_b}"
            pair_result: Dict[str, Any] = {}

            for qoi in qois:
                a = all_outputs[cfg_a].get(qoi)
                b = all_outputs[cfg_b].get(qoi)
                if a is None or b is None:
                    continue

                a_finite = a[np.isfinite(a)]
                b_finite = b[np.isfinite(b)]
                if len(a_finite) < 10 or len(b_finite) < 10:
                    continue

                u_stat, u_p = stats.mannwhitneyu(
                    a_finite, b_finite, alternative="two-sided"
                )
                n1, n2 = len(a_finite), len(b_finite)
                cliff_d = _cliffs_delta_from_u(u_stat, n1, n2)

                pair_result[qoi] = {
                    "cliff_d": float(cliff_d),
                    "mw_U": float(u_stat),
                    "mw_p": float(u_p),
                    "n_a": n1,
                    "n_b": n2,
                    "interpretation": _interpret_cliff(cliff_d),
                }

            result[pair_label] = pair_result

    return result


def _interpret_cliff(d: float) -> str:
    """Qualitative interpretation of |Cliff's delta|."""
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.33:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


def compute_jt_trend(
    all_outputs: Dict[str, Dict[str, np.ndarray]],
    config_order: Sequence[str],
    qois: Sequence[str],
) -> Dict[str, Any]:
    """
    Jonckheere-Terpstra trend test across ordered configs.

    Tests whether QoI distributions shift monotonically
    baseline -> improved -> wattmeter.
    """
    result: Dict[str, Any] = {}

    for qoi in qois:
        groups = []
        for cfg in config_order:
            arr = all_outputs.get(cfg, {}).get(qoi)
            if arr is None:
                break
            finite = arr[np.isfinite(arr)]
            if len(finite) < 10:
                break
            groups.append(finite)

        if len(groups) != len(config_order):
            continue

        alternative = JT_ALTERNATIVES.get(qoi, "two-sided")
        jt_J, jt_p = _jonckheere_terpstra(groups, alternative=alternative)

        result[qoi] = {
            "J": jt_J,
            "p": jt_p,
            "alternative": alternative,
            "config_order": list(config_order),
            "group_sizes": [len(g) for g in groups],
            "group_medians": [float(np.median(g)) for g in groups],
        }

    return result


def generate_latex_table(
    all_st: Dict[str, Dict[str, Dict[str, float]]],
    all_s1: Dict[str, Dict[str, Dict[str, float]]],
    qoi: str,
) -> str:
    """
    Generate a LaTeX table of S1/ST indices for one QoI across configs.

    Returns a string containing the LaTeX tabular environment.
    """
    config_names = list(all_st.keys())

    # Collect all factors
    all_factors: List[str] = []
    for cfg in config_names:
        if qoi in all_st[cfg]:
            for f in all_st[cfg][qoi]:
                if f not in all_factors:
                    all_factors.append(f)

    n_configs = len(config_names)
    col_spec = "l" + "cc" * n_configs
    header_parts = ["Factor"]
    for cfg in config_names:
        header_parts.append(f"\\multicolumn{{2}}{{c}}{{{cfg}}}")
    subheader_parts = [""]
    for _ in config_names:
        subheader_parts.extend(["$S_1$", "$S_T$"])

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_parts) + " \\\\",
        " & ".join(subheader_parts) + " \\\\",
        "\\midrule",
    ]

    for factor in all_factors:
        row_parts = [factor.replace("_", r"\_")]
        for cfg in config_names:
            s1_val = all_s1.get(cfg, {}).get(qoi, {}).get(factor)
            st_val = all_st.get(cfg, {}).get(qoi, {}).get(factor)
            s1_str = f"{s1_val:.3f}" if s1_val is not None else "--"
            st_str = f"{st_val:.3f}" if st_val is not None else "--"
            row_parts.extend([s1_str, st_str])
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _extract_final_s1(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """Mirror of _extract_final_st_rankings but for S1 values."""
    max_n_by_output: Dict[str, int] = {}
    for row in rows:
        if not row.get("sample_size") or not row.get("S1"):
            continue
        output = row["output"]
        n = int(row["sample_size"])
        if output not in max_n_by_output or n > max_n_by_output[output]:
            max_n_by_output[output] = n

    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if not row.get("sample_size") or not row.get("S1"):
            continue
        output = row["output"]
        n = int(row["sample_size"])
        if n != max_n_by_output.get(output):
            continue
        result.setdefault(output, {})[row["factor"]] = float(row["S1"])

    return result


def _to_serializable(obj: Any) -> Any:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesise Sobol results across physics configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Directory containing Sobol results (e.g., results/sobol/).",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path for synthesis JSON output (e.g., results/sobol/synthesis.json).",
    )
    parser.add_argument(
        "--N", type=int, default=128,
        help="Base Sobol sample size that was used in the analysis run. Default: 128.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=list(CONFIG_ORDER),
        help=f"Config names to synthesise. Default: {' '.join(CONFIG_ORDER)}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [c for c in args.configs if c in CONFIG_ORDER]
    if len(configs) < 2:
        raise ValueError(
            f"Need at least 2 configs for synthesis; got {args.configs}. "
            f"Valid: {CONFIG_ORDER}"
        )

    print(f"Loading Sobol results from: {input_dir}")
    print(f"Configs: {configs}")
    print(f"Base N: {args.N}")

    # ── Load data ───────────────────────────────────────────────────────────
    all_manifests: Dict[str, Any] = {}
    all_st: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_s1: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_outputs: Dict[str, Dict[str, np.ndarray]] = {}

    for cfg in configs:
        manifest = _load_manifest(input_dir, cfg, args.N)
        all_manifests[cfg] = manifest

        rows = _load_indices_csv(input_dir, cfg, args.N)
        all_st[cfg] = _extract_final_st_rankings(rows)
        all_s1[cfg] = _extract_final_s1(rows)

        all_outputs[cfg] = _load_design_outputs(input_dir, cfg, args.N)

    # ── Analysis 1: Kendall W on ST rankings ────────────────────────────────
    print("\n--- Kendall W (ST ranking concordance) ---")
    kendall_results = compute_kendall_w_across_configs(all_st, SYNTHESIS_QOIS)
    for qoi, res in kendall_results.items():
        print(
            f"  {qoi:20s}  W={res['W']:.3f}  chi2={res['chi2']:.2f}  "
            f"p={res['p']:.4f}  ({res['n_factors']} factors, {res['n_configs']} configs)"
        )

    # ── Analysis 2: Cliff's delta between config pairs ──────────────────────
    print("\n--- Cliff's delta (pairwise) ---")
    cliff_results = compute_cliffs_delta_between_configs(all_outputs, SYNTHESIS_QOIS)
    for pair, pair_data in cliff_results.items():
        print(f"  {pair}:")
        for qoi, res in pair_data.items():
            print(
                f"    {qoi:20s}  d={res['cliff_d']:+.3f} ({res['interpretation']})  "
                f"p={res['mw_p']:.4e}"
            )

    # ── Analysis 3: JT trend test ───────────────────────────────────────────
    # Only run JT on ordered configs that overlap with CONFIG_ORDER
    ordered_configs = [c for c in CONFIG_ORDER if c in configs]
    print(f"\n--- JT trend test ({' -> '.join(ordered_configs)}) ---")
    jt_results = compute_jt_trend(all_outputs, ordered_configs, JT_QOIS)
    for qoi, res in jt_results.items():
        print(
            f"  {qoi:20s}  J={res['J']:.1f}  p={res['p']:.4f}  "
            f"alt={res['alternative']}  medians={res['group_medians']}"
        )

    # ── Analysis 4: LaTeX tables ────────────────────────────────────────────
    latex_tables: Dict[str, str] = {}
    for qoi in SYNTHESIS_QOIS:
        table = generate_latex_table(all_st, all_s1, qoi)
        latex_tables[qoi] = table

    # ── Assemble synthesis result ───────────────────────────────────────────
    synthesis = {
        "configs": configs,
        "base_N": args.N,
        "input_dir": str(input_dir),
        "kendall_w": kendall_results,
        "cliffs_delta": cliff_results,
        "jt_trend": jt_results,
        "latex_tables": latex_tables,
        "per_config_summary": {
            cfg: {
                "top_total_order": manifest.get("top_total_order", {}),
                "numerical_success_rate": manifest.get("numerical_success_rate"),
                "physical_valid_rate": manifest.get("physical_valid_rate"),
            }
            for cfg, manifest in all_manifests.items()
        },
    }

    # ── Save ────────────────────────────────────────────────────────────────
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(_to_serializable(synthesis), fh, indent=2)
    print(f"\nSynthesis saved to: {output_path}")

    # Also write LaTeX tables to separate .tex files
    tex_dir = output_path.parent / "latex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    for qoi, table in latex_tables.items():
        tex_path = tex_dir / f"sobol_{qoi}.tex"
        with tex_path.open("w", encoding="utf-8") as fh:
            fh.write(table)
    print(f"LaTeX tables saved to: {tex_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
