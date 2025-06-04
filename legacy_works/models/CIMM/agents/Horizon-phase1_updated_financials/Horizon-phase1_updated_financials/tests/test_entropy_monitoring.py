import unittest
import torch
from entropy_monitoring import EntropyMonitor

class TestEntropyMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = EntropyMonitor()

    def test_initial_entropy(self):
        self.assertEqual(self.monitor.entropy, 1.0)

    def test_calculate_entropy(self):
        data = torch.randint(-10, 10, (100, 10))
        entropy = self.monitor.calculate_entropy(data)
        self.assertGreater(entropy, 0)

    def test_adjust_learning_rate(self):
        current_entropy = 1.5
        learning_rate = self.monitor.adjust_learning_rate(current_entropy)
        self.assertGreater(learning_rate, 0)
        self.assertLessEqual(learning_rate, 0.03, f"Learning rate exceeded bounds: {learning_rate}")
        self.assertGreaterEqual(learning_rate, 0.0005, f"Learning rate below bounds: {learning_rate}")

    def test_monitor(self):
        data = torch.randint(-10, 10, (100, 10))
        learning_rate = self.monitor.monitor(data)
        self.assertGreater(learning_rate, 0)

    def test_entropy_clipping(self):
        entropy = 10.0
        clipped_entropy = self.monitor.clip_entropy(entropy)
        self.assertEqual(clipped_entropy, 5.0)

    def test_entropy_filtering(self):
        self.monitor.update_entropy(10)
        self.monitor.update_entropy(-5)
        self.assertGreaterEqual(self.monitor.entropy, 0)  # Ensure non-negative entropy

    def test_entropy_consistency(self):
        data = torch.randn(100, 10)
        initial_entropy = self.monitor.calculate_entropy(data)
        self.monitor.update_entropy(initial_entropy)
        updated_entropy = self.monitor.smoothed_entropy
        self.assertAlmostEqual(initial_entropy, updated_entropy, places=5)

    def test_learning_rate_trends(self):
        """
        Ensure that learning rate consistently decreases as entropy increases.
        """
        entropy_values = [0.1, 1.0, 10.0, 100.0]
        learning_rates = [self.monitor.adjust_learning_rate(entropy) for entropy in entropy_values]

        self.assertTrue(all(earlier >= later for earlier, later in zip(learning_rates, learning_rates[1:])),
                        "Learning rate should decrease as entropy increases")

    def test_stability_rapid_entropy_oscillations(self):
        """
        Ensure learning rate remains stable despite rapid oscillations in entropy.
        """
        for _ in range(50):
            self.monitor.adjust_learning_rate(0.1)
            self.monitor.adjust_learning_rate(0.01)

        final_lr = self.monitor.learning_rate
        expected_range = (0.0005, 0.02)  # Learning rate should stay within stable bounds
        self.assertTrue(expected_range[0] <= final_lr <= expected_range[1],
                        f"Learning rate out of expected bounds: {final_lr}")

    def test_stability_rapid_entropy_changes(self):
        """ Ensure learning rate remains stable despite rapid entropy changes. """
        for _ in range(100):
            self.monitor.adjust_learning_rate(0.1)
            self.monitor.adjust_learning_rate(0.01)

        final_lr = self.monitor.learning_rate
        self.assertTrue(0.0001 <= final_lr <= 0.02, 
                        f"Learning rate out of expected bounds: {final_lr}")

    def test_calculate_qpl(self):
        """
        Ensure QPL remains non-negative and respects entropy-energy constraints.
        """
        entropy = -5.0
        energy = 10.0
        qpl = self.monitor.calculate_qpl(entropy, energy)
        self.assertGreaterEqual(qpl, 0, "QPL should be non-negative")

if __name__ == "__main__":
    unittest.main()
