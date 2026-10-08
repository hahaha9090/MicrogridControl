import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd

from tools.cp_pi import cts_pid
from tools.prediction_quantiles_tools import (
    build_alpha_quantiles_map, build_target_quantiles,
    compute_qra, exec_cp, exec_cqr, fix_quantile_crossing,
)
from tools.quant_proc_tools import QuantProc
from tools.results_analysis_tools import winkler_score
from tools.stat_tests_tools import kupiec_test


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class TestQuantileMethods(unittest.TestCase):
    def test_quantile_grid_and_interval_columns(self):
        quantiles = build_target_quantiles([0.2, 0.4, 0.6, 0.8])
        np.testing.assert_allclose(quantiles, np.arange(0.1, 1.0, 0.1))
        mapping = build_alpha_quantiles_map([0.2, 0.4, 0.6, 0.8], quantiles)
        self.assertEqual(mapping['med'], 4)
        self.assertEqual(mapping[0.2], {'l': 0, 'u': 8})
        self.assertEqual(mapping[0.8], {'l': 3, 'u': 5})

    def test_crossing_repair_preserves_each_rows_values(self):
        values = np.array([[3., 1., 2.], [-1., -3., -2.]])
        np.testing.assert_array_equal(
            fix_quantile_crossing(values), [[1., 2., 3.], [-3., -2., -1.]])

    def test_cp_uses_finite_sample_residual_quantile(self):
        calibration = np.zeros((5, 24, 1))
        targets = np.repeat(np.arange(5.)[:, None], 24, axis=1)
        predicted = np.full((1, 24, 1), 10.)
        result = exec_cp(calibration, targets, predicted, {'target_alpha': [0.2]})
        np.testing.assert_allclose(result, np.tile([6., 10., 14.], (24, 1)))

    def test_cqr_adjusts_tails_with_calibration_scores(self):
        calibration = np.tile([-1., 0., 1.], (5, 24, 1))
        targets = np.repeat(np.arange(-2., 3.)[:, None], 24, axis=1)
        predicted = np.tile([-1., 0., 1.], (1, 24, 1))
        settings = {'target_alpha': [0.2],
                    'q_alpha_map': {0.2: {'l': 0, 'u': 2}, 'med': 1}}
        result = exec_cqr(calibration, targets, predicted, settings)
        np.testing.assert_allclose(result, np.tile([-1.6, 0., 1.6], (24, 1)))

    def test_qra_fits_perfect_linear_predictions(self):
        targets = np.arange(5 * 24., dtype=float).reshape(5, 24)
        calibration = targets[:, :, None]
        predicted = np.full((1, 24, 1), 125.)
        result = compute_qra(calibration, targets, predicted,
                             {'target_quantiles': [0.1, 0.5, 0.9]})
        np.testing.assert_allclose(result, np.full((24, 3), 125.), atol=1e-6)

    def test_online_intervals_do_not_use_current_or_future_targets(self):
        forecasts = [np.array([-1., 1.]) for _ in range(12)]
        data = pd.DataFrame({'y': np.arange(12.), 'forecasts': forecasts})
        changed = data.copy(deep=True)
        changed.loc[7:, 'y'] += 1000.
        options = dict(alpha=0.2, lr=0.01, Csat=10., KI=0.01, T_burnin=4)
        baseline = cts_pid(data, **options)
        modified = cts_pid(changed, **options)
        np.testing.assert_allclose(baseline['sets'][:8], modified['sets'][:8])

    def test_quantproc_reads_ensemble_files_and_calibrates(self):
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            index = pd.date_range('2020-01-01', periods=8 * 24, freq='h')
            target = np.sin(np.arange(len(index)) / 24.)
            for component in (1, 2):
                frame = pd.DataFrame({'TEST': target, 0.1: target - 2.,
                                      0.5: target, 0.9: target + 2.}, index=index)
                path = Path('experiments/tasks/TEST/QR-DNN') / ('baseline_' + str(component)) / 'results'
                path.mkdir(parents=True)
                with (path / 'recalib_test_results-tuned-grid_search.p').open('wb') as handle:
                    pickle.dump(frame, handle)
            results = QuantProc('TEST', 24, ['QR-DNN', 'CQ-QR-DNN', 'OCQ-QR-DNN'],
                                'baseline', 2, 5, [0.2]).process_quantiles()
            self.assertEqual(set(results), {'QR-DNN', 'CQ-QR-DNN', 'OCQ-QR-DNN'})
            for frame in results.values():
                self.assertEqual(len(frame), 3 * 24)
                np.testing.assert_allclose(frame['TEST'], target[5 * 24:])
                quantiles = frame[[0.1, 0.5, 0.9]].to_numpy()
                self.assertTrue(np.isfinite(quantiles).all())
                self.assertTrue((np.diff(quantiles, axis=1) >= 0).all())


class TestEvaluation(unittest.TestCase):
    def test_winkler_penalizes_misses(self):
        result = winkler_score(np.array([0., 3.]), np.array([-1., -1.]),
                               np.array([1., 1.]), 0.2)
        np.testing.assert_allclose(result, [2., 22.])

    def test_kupiec_accepts_target_coverage(self):
        result = kupiec_test([1] * 8 + [0] * 2, alpha=0.2)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['LR_UC'], 0., places=10)

    def test_kupiec_rejects_zero_coverage_with_finite_statistic(self):
        result = kupiec_test([0] * 10, alpha=0.2)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['LR_UC'], -20 * np.log(0.2))

    def test_kupiec_all_hits_has_finite_statistic(self):
        result = kupiec_test([1] * 10, alpha=0.2)
        self.assertAlmostEqual(result['LR_UC'], -20 * np.log(0.8))


class TestBaselineProject(unittest.TestCase):
    def test_entrypoints_can_be_imported_without_running_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND='Agg')
            result = subprocess.run(
                [sys.executable, '-c', 'import run_recalibration, exec_qra_cp, results_analysis'],
                cwd=folder, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_all_baseline_configs_reference_existing_data(self):
        configs = list((ROOT / 'experiments/tasks').glob('*/*/*/exper_configs.json'))
        self.assertEqual(len(configs), 7 * 5 * 4)
        for path in configs:
            with self.subTest(path=path):
                config = json.loads(path.read_text(encoding='utf-8'))
                dataset = ROOT / 'data/datasets' / config['data_config']['dataset_name']
                self.assertTrue(dataset.is_file(), str(dataset))


if __name__ == '__main__':
    unittest.main()
