import unittest
import torch
from entropy_monitoring import EntropyMonitor

class TestLearningRateAdjustment(unittest.TestCase):
    def setUp(self):
        self.monitor = EntropyMonitor()

    def test_adjust_learning_rate(self):
        current_entropy = 1.5
        learning_rate = self.monitor.adjust_learning_rate(current_entropy)
        self.assertGreater(learning_rate, 0)

    def test_monitor(self):
        data = torch.randint(-10, 10, (100, 10))
        learning_rate = self.monitor.monitor(data)
        self.assertGreater(learning_rate, 0)

    def test_learning_rate_update(self):
        old_lr = 0.01
        entropy = 0.5
        new_lr = self.monitor.adjust_learning_rate(entropy)
        self.assertNotEqual(new_lr, old_lr)  # Learning rate should change

    def test_extreme_entropy_values(self):
        # Test with very low entropy
        low_entropy = 0
        low_lr = self.monitor.adjust_learning_rate(low_entropy)
        self.assertGreater(low_lr, 0)

        # Test with very high entropy
        high_entropy = 1000
        high_lr = self.monitor.adjust_learning_rate(high_entropy)
        self.assertGreater(high_lr, 0)
        self.assertLess(high_lr, 0.1)  # Ensure it doesn't explode

    def test_learning_rate_trends(self):
        entropy_values = [0.1, 1.0, 10.0, 100.0]
        learning_rates = [self.monitor.adjust_learning_rate(entropy) for entropy in entropy_values]
        self.assertTrue(all(earlier >= later for earlier, later in zip(learning_rates, learning_rates[1:])),
                        "Learning rate should decrease as entropy increases")

if __name__ == "__main__":
    unittest.main()
