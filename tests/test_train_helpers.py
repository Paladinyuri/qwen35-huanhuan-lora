import unittest
from types import SimpleNamespace

import numpy as np

from train import loss_totals


class TrainHelperTest(unittest.TestCase):
    def test_loss_totals_uses_only_weighted_tokens(self):
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": np.array([-1.0, -2.0, -3.0])}])
        datum = SimpleNamespace(loss_fn_inputs={"weights": np.array([0.0, 1.0, 1.0])})
        loss_sum, token_count = loss_totals(result, [datum])
        self.assertEqual(loss_sum, 5.0)
        self.assertEqual(token_count, 2.0)

    def test_loss_totals_rejects_empty_target(self):
        result = SimpleNamespace(loss_fn_outputs=[{"logprobs": np.array([-1.0])}])
        datum = SimpleNamespace(loss_fn_inputs={"weights": np.array([0.0])})
        with self.assertRaisesRegex(ValueError, "assistant token"):
            loss_totals(result, [datum])


if __name__ == "__main__":
    unittest.main()
