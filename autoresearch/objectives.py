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
    max_T_err = float(np.max(np.abs(
        np.asarray(result['T_2d']) - np.asarray(reference['T_2d'])
    )))
    t_ratio = result['time'] / reference['time']
    iter_ratio = result['steps'] / reference['steps']

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
    'disc': 2.0,
    'ra': 0.5,
    '1d2d': 5.0,
    'juno': 3.0,
    'sanity': 10.0,
}

_JS_BIN_EDGES = np.arange(5.0, 61.0, 1.0)


def _js_divergence(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two sample sets using fixed bins."""
    hist_a, _ = np.histogram(samples_a, bins=_JS_BIN_EDGES, density=True)
    hist_b, _ = np.histogram(samples_b, bins=_JS_BIN_EDGES, density=True)
    eps = 1e-12
    p = hist_a + eps
    q = hist_b + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_latitude_score(
    scenarios: Dict[str, Dict[str, Any]],
    consistency_error: float,
) -> Tuple[float, Dict[str, float]]:
    """Score a latitude-mode triple-scenario experiment."""
    first = next(iter(scenarios.values()))
    lats = np.asarray(first['latitudes_deg'])
    idx_35 = _find_lat_index(lats, _JUNO_LATITUDE_DEG)

    d_conv_contrasts = []
    for name, res in scenarios.items():
        D_conv_median = np.median(np.asarray(res['D_conv_profiles']), axis=0)
        d_conv_contrasts.append(float(np.max(D_conv_median) - np.min(D_conv_median)))
    d_conv_contrast = max(d_conv_contrasts)

    uniform = scenarios.get('uniform', first)
    Ra_median = np.median(np.asarray(uniform['Ra_profiles']), axis=0)
    ra_eq = max(float(Ra_median[0]), 1e-6)
    ra_pole = max(float(Ra_median[-1]), 1e-6)
    ra_log_ratio = math.log(ra_eq / ra_pole) if ra_eq > ra_pole else 0.0

    scenario_names = list(scenarios.keys())
    js_values = []
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            d_a = np.asarray(scenarios[scenario_names[i]]['D_cond_profiles'])[:, idx_35]
            d_b = np.asarray(scenarios[scenario_names[j]]['D_cond_profiles'])[:, idx_35]
            js_values.append(_js_divergence(d_a, d_b))
    min_js = min(js_values) if js_values else 0.0

    D_cond_35_all = []
    for res in scenarios.values():
        D_cond_35_all.append(np.median(np.asarray(res['D_cond_profiles'])[:, idx_35]))
    D_cond_35_median = float(np.mean(D_cond_35_all))
    juno_excess = max(0.0, abs(D_cond_35_median - _JUNO_D_COND_MU) - _JUNO_D_COND_SIGMA_OBS)

    sanity = 0.0
    for res in scenarios.values():
        H = np.asarray(res['H_profiles'])
        if np.any(H <= 0):
            sanity += 1.0
        Ra = np.asarray(res['Ra_profiles'])
        if np.any(Ra < 0):
            sanity += 1.0

    w = _LAT_WEIGHTS
    score = (
        -w['dconv'] * d_conv_contrast
        - w['disc'] * min_js
        - w['ra'] * ra_log_ratio
        + w['1d2d'] * max(0.0, consistency_error - 0.05) * 100.0
        + w['juno'] * juno_excess * 10.0
        + w['sanity'] * sanity
    )

    metrics = {
        'D_conv_contrast': d_conv_contrast,
        'JS_discriminability': min_js,
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
