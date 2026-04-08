"""Composite scoring functions for autoresearch experiment modes."""
import math
import numpy as np
from typing import Any, Dict, Tuple


# --- Solver mode ---

_SOLVER_WEIGHTS = {'time': 0.5, 'err': 0.35, 'iter': 0.15}
_SOLVER_ERR_THRESHOLD = 0.1  # K


def compute_solver_score(
    result: Dict[str, Any],
    reference: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Score a solver experiment against the frozen reference.

    Args:
        result: Dict with 'time', 'steps', 'T_2d' from the experiment.
        reference: Dict with 'time', 'steps', 'T_2d' from reference/solver_ref.json.

    Returns:
        (score, metrics) where lower score is better.
    """
    result_arr = np.asarray(result['T_2d'])
    ref_arr = np.asarray(reference['T_2d'])
    if result_arr.shape != ref_arr.shape:
        raise ValueError(
            f"T_2d shape mismatch: result {result_arr.shape} vs reference {ref_arr.shape}"
        )
    max_T_err = float(np.max(np.abs(result_arr - ref_arr)))
    t_ratio = result['time'] / max(reference['time'], 1e-12)
    iter_ratio = result['steps'] / max(reference['steps'], 1)

    w = _SOLVER_WEIGHTS
    score = w['time'] * t_ratio + w['err'] * (max_T_err / _SOLVER_ERR_THRESHOLD) + w['iter'] * iter_ratio

    if max_T_err > _SOLVER_ERR_THRESHOLD:
        score += 1000.0

    metrics = {
        'time': result['time'],
        'time_ref': reference['time'],
        'steps': result['steps'],
        'steps_ref': reference['steps'],
        'max_T_err': max_T_err,
    }
    return score, metrics


# --- Physics mode ---

_JUNO_D_COND_MU = 29.0
_JUNO_D_COND_SIGMA_OBS = 10.0
_MODEL_DISCREPANCY = 3.0
_JUNO_SIGMA_EFF = math.sqrt(_JUNO_D_COND_SIGMA_OBS**2 + _MODEL_DISCREPANCY**2)
_JUNO_LATITUDE_DEG = 35.0
_H_TOTAL_MIN = 15.0
_YIELD_WEIGHT = 5.0


def _find_lat_index(latitudes_deg: np.ndarray, target_deg: float) -> int:
    """Return index of the latitude bin closest to target_deg."""
    return int(np.argmin(np.abs(latitudes_deg - target_deg)))


def compute_physics_score(
    mc_results: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Score a physics-mode MC ensemble against Juno constraints."""
    lats = np.asarray(mc_results['latitudes_deg'])
    idx_35 = _find_lat_index(lats, _JUNO_LATITUDE_DEG)

    D_cond_35 = np.asarray(mc_results['D_cond_profiles'])[:, idx_35]
    D_cond_35_median = float(np.median(D_cond_35))

    nll = 0.5 * ((D_cond_35_median - _JUNO_D_COND_MU) / _JUNO_SIGMA_EFF) ** 2

    H_all = np.asarray(mc_results['H_profiles'])
    H_min_per_sample = H_all.min(axis=1)
    thin_frac = float(np.mean(H_min_per_sample < _H_TOTAL_MIN))
    thin_penalty = 100.0 * thin_frac

    valid_frac = mc_results['n_valid'] / max(mc_results['n_iterations'], 1)
    yield_penalty = _YIELD_WEIGHT * (1.0 - valid_frac)

    score = nll + thin_penalty + yield_penalty

    metrics = {
        'D_cond_35_median': D_cond_35_median,
        'D_cond_35_mean': float(np.mean(D_cond_35)),
        'D_cond_35_std': float(np.std(D_cond_35)),
        'thin_fraction': thin_frac,
        'valid_fraction': valid_frac,
        'nll': nll,
    }
    return score, metrics


# --- Latitude mode ---

_LAT_WEIGHTS = {
    'dconv': 1.0,
    'profile_disc': 2.0,
    'ra': 0.5,
    '1d2d': 5.0,
    'juno': 3.0,
    'sanity': 10.0,
}

_JS_BIN_EDGES = np.arange(5.0, 61.0, 1.0)


def _js_divergence(samples_a: np.ndarray, samples_b: np.ndarray,
                   bin_edges: np.ndarray = _JS_BIN_EDGES) -> float:
    """Jensen-Shannon divergence between two sample sets using fixed bins."""
    if len(samples_a) == 0 or len(samples_b) == 0:
        return 0.0
    hist_a, _ = np.histogram(samples_a, bins=bin_edges, density=False)
    hist_b, _ = np.histogram(samples_b, bins=bin_edges, density=False)
    if hist_a.sum() == 0 or hist_b.sum() == 0:
        return 0.0
    eps = 1e-12
    p = hist_a.astype(float) + eps
    q = hist_b.astype(float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _profile_js_discriminability(
    scenarios: Dict[str, Dict[str, Any]],
    latitudes_deg: np.ndarray,
) -> Tuple[float, np.ndarray, Dict[str, float]]:
    """Compute cos(phi)-weighted mean JS(D_cond) across all latitudes.

    For each scenario pair, computes JS(D_cond) at every latitude, then
    takes the cos(phi)-weighted mean. The profile discriminability is the
    minimum across all pairs (conservative: weakest pair sets the score).

    Returns:
        (profile_js_min, js_per_latitude_matrix, extra_metrics)
    """
    n_lat = len(latitudes_deg)
    cos_weights = np.cos(np.radians(latitudes_deg))
    cos_weights = cos_weights / cos_weights.sum()

    scenario_names = list(scenarios.keys())
    n_pairs = 0
    pair_weighted_means = []
    js_matrix = {}  # (pair_name) -> (n_lat,) array

    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            name_a, name_b = scenario_names[i], scenario_names[j]
            pair_key = f'{name_a}-{name_b}'
            js_per_lat = np.zeros(n_lat)

            for k in range(n_lat):
                d_a = np.asarray(scenarios[name_a]['D_cond_profiles'])[:, k]
                d_b = np.asarray(scenarios[name_b]['D_cond_profiles'])[:, k]
                js_per_lat[k] = _js_divergence(d_a, d_b)

            js_matrix[pair_key] = js_per_lat
            weighted_mean = float(np.sum(js_per_lat * cos_weights))
            pair_weighted_means.append(weighted_mean)
            n_pairs += 1

    profile_js_min = min(pair_weighted_means) if pair_weighted_means else 0.0

    # Aggregate JS across all pairs per latitude (for peak detection)
    all_js = np.array(list(js_matrix.values()))  # (n_pairs, n_lat)
    min_js_per_lat = np.min(all_js, axis=0) if len(all_js) > 0 else np.zeros(n_lat)
    peak_idx = int(np.argmax(min_js_per_lat))

    extra = {
        'profile_js_min': profile_js_min,
        'profile_js_mean': float(np.mean(pair_weighted_means)) if pair_weighted_means else 0.0,
        'JS_peak': float(min_js_per_lat[peak_idx]) if n_lat > 0 else 0.0,
        'phi_peak_js': float(latitudes_deg[peak_idx]) if n_lat > 0 else 0.0,
    }

    return profile_js_min, min_js_per_lat, extra


def compute_latitude_score(
    scenarios: Dict[str, Dict[str, Any]],
    consistency_error: float,
) -> Tuple[float, Dict[str, float]]:
    """Score a latitude-mode triple-scenario experiment.

    Discriminability is now profile-level: cos(phi)-weighted mean JS(D_cond)
    across all latitudes, minimum across scenario pairs.

    D_cond@35 is retained as the Juno-fit term only.
    """
    first = next(iter(scenarios.values()))
    lats = np.asarray(first['latitudes_deg'])
    idx_35 = _find_lat_index(lats, _JUNO_LATITUDE_DEG)

    # --- D_conv contrast (unchanged) ---
    d_conv_contrasts = []
    for _name, res in scenarios.items():
        D_conv_median = np.median(np.asarray(res['D_conv_profiles']), axis=0)
        d_conv_contrasts.append(float(np.max(D_conv_median) - np.min(D_conv_median)))
    d_conv_contrast = max(d_conv_contrasts)

    # --- Ra ratio (unchanged) ---
    uniform = scenarios.get('uniform', first)
    Ra_median = np.median(np.asarray(uniform['Ra_profiles']), axis=0)
    ra_eq = max(float(Ra_median[0]), 1e-6)
    ra_pole = max(float(Ra_median[-1]), 1e-6)
    ra_log_ratio = math.log(ra_eq / ra_pole) if ra_eq > ra_pole else 0.0

    # --- Profile-level discriminability (NEW: replaces JS@35) ---
    profile_js_min, js_per_lat, profile_extras = _profile_js_discriminability(
        scenarios, lats,
    )

    # --- JS@35 (reported only, not in score) ---
    scenario_names = list(scenarios.keys())
    js_35_values = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            d_a = np.asarray(scenarios[scenario_names[i]]['D_cond_profiles'])[:, idx_35]
            d_b = np.asarray(scenarios[scenario_names[j]]['D_cond_profiles'])[:, idx_35]
            js_35_values.append(_js_divergence(d_a, d_b))
    js_35 = min(js_35_values) if js_35_values else 0.0

    # --- D_conv JS at 35 (reported only) ---
    _DCONV_BIN_EDGES = np.arange(0.0, 30.0, 0.5)
    d_conv_js_values = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            d_a = np.asarray(scenarios[scenario_names[i]]['D_conv_profiles'])[:, idx_35]
            d_b = np.asarray(scenarios[scenario_names[j]]['D_conv_profiles'])[:, idx_35]
            d_conv_js_values.append(_js_divergence(d_a, d_b, _DCONV_BIN_EDGES))
    min_js_dconv = min(d_conv_js_values) if d_conv_js_values else 0.0

    # --- Juno penalty (unchanged) ---
    D_cond_35_all = []
    for res in scenarios.values():
        D_cond_35_all.append(np.median(np.asarray(res['D_cond_profiles'])[:, idx_35]))
    D_cond_35_median = float(np.mean(D_cond_35_all))
    juno_excess = max(0.0, abs(D_cond_35_median - _JUNO_D_COND_MU) - _JUNO_D_COND_SIGMA_OBS)

    # --- Sanity (unchanged) ---
    sanity = 0.0
    for res in scenarios.values():
        H = np.asarray(res['H_profiles'])
        if np.any(H <= 0):
            sanity += 1.0
        Ra = np.asarray(res['Ra_profiles'])
        if np.any(Ra < 0):
            sanity += 1.0

    # --- Composite score ---
    w = _LAT_WEIGHTS
    score = (
        -w['dconv'] * d_conv_contrast
        - w['profile_disc'] * profile_js_min
        - w['ra'] * ra_log_ratio
        + w['1d2d'] * max(0.0, consistency_error - 0.05) * 100.0
        + w['juno'] * juno_excess * 10.0
        + w['sanity'] * sanity
    )

    metrics = {
        'D_conv_contrast': d_conv_contrast,
        'profile_JS_min': profile_js_min,
        'profile_JS_mean': profile_extras['profile_js_mean'],
        'JS_35': js_35,
        'JS_peak': profile_extras['JS_peak'],
        'phi_peak_js': profile_extras['phi_peak_js'],
        'JS_discriminability_Dconv': min_js_dconv,
        'Ra_eq_median': ra_eq,
        'Ra_pole_median': ra_pole,
        'Ra_log_ratio': ra_log_ratio,
        'consistency_error': consistency_error,
        'D_cond_35_median': D_cond_35_median,
        'juno_excess': juno_excess,
        'sanity_penalty': sanity,
    }
    return score, metrics


# --- Dispatcher ---

def compute_score(mode: str, **kwargs) -> Tuple[float, Dict[str, float]]:
    """Dispatch to the appropriate scoring function by mode name."""
    if mode == 'solver':
        return compute_solver_score(kwargs['result'], kwargs['reference'])
    elif mode == 'physics':
        return compute_physics_score(kwargs['mc_results'])
    elif mode == 'latitude':
        return compute_latitude_score(kwargs['scenarios'], kwargs['consistency_error'])
    else:
        raise ValueError(f"Unknown mode: {mode}")
