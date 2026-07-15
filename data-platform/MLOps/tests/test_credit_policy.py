import unittest

from MLOps.app.api.credit_policy import CreditPolicy


class CreditPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CreditPolicy(0.35, 0.65, "test-v1")

    def test_approve(self) -> None:
        self.assertEqual(self.policy.evaluate(0.20).recommendation, "approve")

    def test_manual_review(self) -> None:
        self.assertEqual(
            self.policy.evaluate(0.50).recommendation,
            "manual_review",
        )

    def test_threshold_boundaries(self) -> None:
        self.assertEqual(self.policy.evaluate(0.35).recommendation, "manual_review")
        self.assertEqual(self.policy.evaluate(0.65).recommendation, "reject")

    def test_reject(self) -> None:
        self.assertEqual(self.policy.evaluate(0.80).recommendation, "reject")

    def test_invalid_thresholds(self) -> None:
        with self.assertRaises(ValueError):
            CreditPolicy(0.70, 0.60, "invalid")

    def test_score_outside_valid_range(self) -> None:
        for score in (-0.01, 1.01):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.policy.evaluate(score)


if __name__ == "__main__":
    unittest.main()
