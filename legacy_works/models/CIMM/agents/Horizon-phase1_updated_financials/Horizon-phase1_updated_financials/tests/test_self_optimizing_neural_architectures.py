import unittest
from self_optimizing_neural_architectures import SelfOptimizingNeuralArchitectures
from skopt.space import Real, Integer
import torch

class TestSelfOptimizingNeuralArchitectures(unittest.TestCase):
    def setUp(self):
        model_class = lambda hidden_size: torch.nn.Linear(10, hidden_size)  # Example model class
        param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Integer(10, 100, name='hidden_size')
        ]
        self.sona = SelfOptimizingNeuralArchitectures(model_class, param_space)

    def test_initial_architecture(self):
        self.assertIsNone(self.sona.architecture)

    def test_optimize_architecture(self):
        self.sona.optimize_architecture()
        self.assertIsNotNone(self.sona.architecture)
        self.assertIsNotNone(self.sona.entropy_monitor)  # Ensure entropy_monitor is present

    def test_reset_architecture(self):
        self.sona.optimize_architecture()
        self.sona.reset_architecture()
        self.assertIsNone(self.sona.architecture)
        self.assertIsNotNone(self.sona.entropy_monitor)  # Ensure entropy_monitor is present

    def test_optimized_learning_rate(self):
        optimized_params, _ = self.sona.optimize()
        learning_rate = optimized_params[0]

        # Ensure learning rate is within the expected bounds
        self.assertTrue(0.005 <= learning_rate <= 0.02, f"Optimized learning rate out of bounds: {learning_rate}")
        
        # Additional check for entropy-aware adjustments
        entropy_factor = 1.0 / (1 + self.sona.entropy_monitor.entropy)
        scaled_learning_rate = learning_rate * entropy_factor
        self.assertLessEqual(scaled_learning_rate, 0.02, f"Scaled learning rate out of bounds: {scaled_learning_rate}")
        
        self.assertIsNotNone(self.sona.entropy_monitor)  # Ensure entropy_monitor is present

if __name__ == "__main__":
    unittest.main()
