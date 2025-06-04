import unittest
import torch
from cimm import CIMM
from self_optimizing_neural_architectures import SelfOptimizingNeuralArchitectures
from self_adaptive_learning_controller import SelfAdaptiveLearningController
from entropy_monitoring import EntropyMonitor
from quantum_potential_layer import QuantumPotentialLayer
from skopt.space import Real, Integer

class ExampleModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super(ExampleModel, self).__init__()
        self.linear1 = torch.nn.Linear(10, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear1(x).relu()
        return self.linear2(x)

class TestCIMMCoreFunctionality(unittest.TestCase):
    def setUp(self):
        param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Integer(10, 100, name='hidden_size')
        ]
        anchor_data = torch.randn(10, 10)  # Example anchor data
        self.cimm = CIMM(ExampleModel, param_space, anchor_data)  # Add anchor_data argument

    def test_run(self):
        data = torch.randn(100, 10)
        energy = 0.5
        model = self.cimm.run(data, energy)
        self.assertIsNotNone(model)
        self.assertIsNotNone(self.cimm.optimizer.entropy_monitor)  # Ensure entropy_monitor is present

    def test_evaluate_model(self):
        data = torch.randn(100, 10)
        model = ExampleModel(hidden_size=50)
        metrics = self.cimm.evaluate_model(model, data)
        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)

if __name__ == "__main__":
    unittest.main()
