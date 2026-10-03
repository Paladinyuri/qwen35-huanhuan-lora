import unittest

from evaluate import closest_training_output, repeated_ngram_rate


class EvaluateMetricTest(unittest.TestCase):
    def test_repetition_rate_detects_loop(self):
        normal = repeated_ngram_rate("臣妾愿陪皇上去御花园走走")
        repeated = repeated_ngram_rate("看看雾凇看看雾凇看看雾凇看看雾凇")
        self.assertGreater(repeated, normal)
        self.assertGreater(repeated, 0.35)

    def test_closest_training_output_finds_near_copy(self):
        result = closest_training_output(
            "臣妾愿意陪皇上去御花园走走",
            ["今日天气很好", "臣妾愿陪皇上去御花园走走", "不知该说什么"],
        )
        self.assertEqual(result["text"], "臣妾愿陪皇上去御花园走走")
        self.assertGreater(result["sequence_ratio"], 0.85)


if __name__ == "__main__":
    unittest.main()
