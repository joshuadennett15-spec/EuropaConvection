"""Integration test: full solver-mode experiment round-trip."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from harness import ExperimentHarness


@pytest.mark.slow
def test_solver_round_trip(tmp_path):
    """Init baseline, run solver experiment, verify log and scoring."""
    harness = ExperimentHarness(base_dir=str(tmp_path))

    # Mock the MC runs so init only runs the real solver
    mock_mc = MagicMock()
    mock_mc.D_cond_profiles = [[29.0] * 19] * 100
    mock_mc.D_conv_profiles = [[2.0] * 19] * 100
    mock_mc.Ra_profiles = [[30.0] * 19] * 100
    mock_mc.H_profiles = [[31.0] * 19] * 100
    mock_mc.n_valid = 100
    mock_mc.n_iterations = 100
    mock_mc.latitudes_deg = [i * 5.0 for i in range(19)]

    with patch.object(harness, '_run_mc_ensemble', return_value=mock_mc), \
         patch.object(harness, '_run_calibration_check', return_value=0.02):
        # Init baseline (runs real solver -- takes ~10s)
        harness.init()

    ref_path = tmp_path / 'reference' / 'solver_ref.json'
    best_path = tmp_path / 'best.json'
    assert ref_path.exists()
    assert best_path.exists()

    # Run an experiment (no code change -- should match baseline)
    harness.run('solver', 'no_change_control')

    log_path = tmp_path / 'experiments.jsonl'
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().split('\n')[-1])
    assert entry['status'] == 'ok'
    assert entry['mode'] == 'solver'
    # Score should be close to baseline (small timing variance)
    assert entry['score'] < 2.0
    assert entry['metrics']['max_T_err'] < 0.01  # same code -> near-zero error
