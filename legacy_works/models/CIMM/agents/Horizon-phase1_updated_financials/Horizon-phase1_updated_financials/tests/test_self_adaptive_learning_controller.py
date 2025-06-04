import unittest
import torch
import random  # Add this import
from self_adaptive_learning_controller import SelfAdaptiveLearningController
import math  # Add this import

class TestSelfAdaptiveLearningController(unittest.TestCase):
    def setUp(self):
        model = torch.nn.Linear(10, 1)  # Example model
        initial_entropy = 1.0
        learning_rate = 0.01
        lambda_factor = 0.1
        self.salc = SelfAdaptiveLearningController(model, initial_entropy, learning_rate, lambda_factor)

    def test_initial_learning_rate(self):
        self.assertTrue(math.isclose(self.salc.learning_rate, 0.01, rel_tol=1e-6))

    def test_adjust_learning_rate(self):
        self.salc.adjust_learning_rate(0.1)
        expected_value = max(0.005, min(0.02, self.salc.learning_rate))
        self.assertTrue(math.isclose(self.salc.learning_rate, expected_value, rel_tol=1e-6))

    def test_reset_learning_rate(self):
        self.salc.adjust_learning_rate(0.1)
        self.salc.reset_learning_rate()
        self.assertTrue(math.isclose(self.salc.learning_rate, 0.01, rel_tol=1e-6))

    def test_dynamic_learning_rate_adjustment(self):
        for rate in [0.01, 0.05, 0.1, 0.2]:
            self.salc.adjust_learning_rate(rate)
            expected_value = max(0.005, min(0.02, self.salc.learning_rate))
            self.assertTrue(math.isclose(self.salc.learning_rate, expected_value, rel_tol=1e-6))

    def test_stability_rapid_entropy_oscillations(self):
        """
        Ensure learning rate remains stable despite rapid entropy oscillations.
        """
        for _ in range(50):
            self.salc.adjust_learning_rate(0.1)
            self.salc.adjust_learning_rate(0.01)

        final_lr = self.salc.learning_rate
        expected_range = (0.0005, 0.02)  # Learning rate should stay within stable bounds

        self.assertTrue(expected_range[0] <= final_lr <= expected_range[1],
                        f"Learning rate out of expected bounds: {final_lr}")

if __name__ == "__main__":
    unittest.main()
