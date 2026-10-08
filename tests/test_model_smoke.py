import tempfile
import unittest

import numpy as np
import tensorflow as tf

from tools.models.models_tools import Ensemble, TensorflowRegressor
from tests.test_price_baseline import working_directory


class TestDNNTraining(unittest.TestCase):
    def test_all_original_models_train_and_predict(self):
        rng = np.random.default_rng(42)
        inputs = rng.normal(size=(6, 72, 3)).astype('float32')
        targets = rng.normal(size=(6, 24)).astype('float32')
        for method in ['point', 'qr', 'Normal', 'JSU', 'STU']:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder, working_directory(folder):
                tf.random.set_seed(42)
                settings = {
                    'model_class': 'DNN', 'PF_method': method,
                    'x_columns_names': ['TARG__TEST', 'PAST__DEMAND', 'FUTU__LOAD'],
                    'pred_horiz': 24, 'target_quantiles': [0.1, 0.5, 0.9],
                    'hidden_size': 8, 'n_hidden_layers': 1,
                    'activation': 'softplus', 'lr': 0.001,
                    'max_epochs': 1, 'batch_size': 2, 'patience': 1,
                }
                model = TensorflowRegressor(settings, inputs[:1])
                model.fit(inputs[:4], targets[:4], inputs[4:], targets[4:])
                predictions = model.predict(inputs[4:])
                self.assertTrue(np.isfinite(predictions).all())
                ensemble = Ensemble(settings)
                aggregated = ensemble.aggregate_preds([predictions])
                quantiles = ensemble.get_preds_test_quantiles(aggregated)
                self.assertEqual(quantiles.shape, (48, 1 if method == 'point' else 3))
                self.assertTrue(np.isfinite(quantiles).all())
                self.assertTrue((np.diff(quantiles, axis=1) >= 0).all())
                loss = model.evaluate(inputs[4:], targets[4:])
                self.assertTrue(np.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
