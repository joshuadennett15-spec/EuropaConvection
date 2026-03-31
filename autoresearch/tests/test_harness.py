"""Unit tests for the experiment harness."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from harness import ExperimentHarness


def test_init_creates_reference_and_best(tmp_path):
    """--init should create reference/ dir and best.json."""
    harness = ExperimentHarness(base_dir=str(tmp_path))

    # Mock the actual model runs to avoid needing Europa2D imports
    fake_solver = {
        'time': 2.0, 'steps': 200,
        'T_2d': [[150.0] * 31], 'H_profile_km': [25.0],
    }

    with patch.object(harness, '_run_single_solver', return_value=fake_solver), \
         patch.object(harness, '_run_mc_ensemble') as mock_mc, \
         patch.object(harness, '_run_calibration_check', return_value=0.02):

        # Create a mock MC result
        mock_mc_result = MagicMock()
        mock_mc_result.D_cond_profiles = [[29.0] * 19] * 100
        mock_mc_result.D_conv_profiles = [[2.0] * 19] * 100
        mock_mc_result.Ra_profiles = [[30.0] * 19] * 100
        mock_mc_result.H_profiles = [[31.0] * 19] * 100
        mock_mc_result.n_valid = 100
        mock_mc_result.n_iterations = 100
        mock_mc_result.latitudes_deg = [i * 5.0 for i in range(19)]
        mock_mc.return_value = mock_mc_result

        harness.init()

    assert (tmp_path / 'reference' / 'solver_ref.json').exists()
    assert (tmp_path / 'reference' / 'physics_ref.json').exists()
    assert (tmp_path / 'reference' / 'latitude_ref.json').exists()
    assert (tmp_path / 'best.json').exists()

    best = json.loads((tmp_path / 'best.json').read_text())
    assert 'solver' in best
    assert 'physics' in best
    assert 'latitude' in best


def test_failed_experiment_logs_infinity(tmp_path):
    """A crashing experiment should log score=Infinity and not raise."""
    harness = ExperimentHarness(base_dir=str(tmp_path))
    (tmp_path / 'best.json').write_text('{}')

    with patch.object(harness, '_run_solver_experiment', side_effect=RuntimeError("boom")):
        harness.run('solver', 'bad_change')

    log_path = tmp_path / 'experiments.jsonl'
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip())
    assert entry['status'] == 'failed'
    assert entry['score'] == float('inf')
    assert 'boom' in entry['error']


def test_successful_experiment_logs_and_updates_best(tmp_path):
    """A successful experiment should log score and update best.json if improved."""
    harness = ExperimentHarness(base_dir=str(tmp_path))

    # Set up initial best with a high score
    initial_best = {'solver': {'score': 10.0, 'metrics': {}}}
    (tmp_path / 'best.json').write_text(json.dumps(initial_best))
    # Need reference for solver mode
    ref_dir = tmp_path / 'reference'
    ref_dir.mkdir()
    ref_data = {
        'time': 2.0, 'steps': 200,
        'T_2d': [[150.0] * 31], 'H_profile_km': [25.0],
    }
    (ref_dir / 'solver_ref.json').write_text(json.dumps(ref_data))

    with patch.object(harness, '_run_single_solver', return_value=ref_data):
        harness.run('solver', 'improved_solver')

    log_path = tmp_path / 'experiments.jsonl'
    entry = json.loads(log_path.read_text().strip())
    assert entry['status'] == 'ok'
    assert entry['improved'] is True

    best = json.loads((tmp_path / 'best.json').read_text())
    assert best['solver']['score'] < 10.0
