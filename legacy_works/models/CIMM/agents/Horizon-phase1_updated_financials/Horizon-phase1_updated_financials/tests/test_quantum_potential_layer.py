import unittest
from quantum_potential_layer import QuantumPotentialLayer
import numpy as np

class TestQuantumPotentialLayer(unittest.TestCase):
    def setUp(self):
        self.qpl = QuantumPotentialLayer()

    def test_calculate_qpl(self):
        entropy = 1.0
        energy = 0.5
        qpl_value = self.qpl.calculate_qpl(entropy, energy)
        self.assertIsNotNone(qpl_value)

    def test_adjust_entropy(self):
        entropy = 1.0
        energy = 0.5
        adjusted_entropy = self.qpl.adjust_entropy(entropy, energy)
        self.assertGreaterEqual(adjusted_entropy, 0)

    def test_qpl_entropy_adjustment(self):
        entropy = 1.0
        energy = 2.0
        qpl_value = self.qpl.calculate_qpl(entropy, energy)
        self.assertGreater(qpl_value, entropy)  # Entropy should increase
        entropy = 2.0
        energy = -1.0
        qpl_value = self.qpl.calculate_qpl(entropy, energy)
        self.assertLess(qpl_value, entropy)  # Entropy should decrease with negative energy

    def test_extreme_energy_values(self):
        # Test with very high energy
        entropy = 1.0
        high_energy = 1000
        qpl_value = self.qpl.calculate_qpl(entropy, high_energy)
        self.assertGreater(qpl_value, entropy)

        # Test with very low (negative) energy
        low_energy = -1000
        qpl_value = self.qpl.calculate_qpl(entropy, low_energy)
        self.assertLess(qpl_value, entropy)

    def test_entropy_energy_correlation(self):
        entropy = 2.0
        increasing_energy = [0.1, 1.0, 10.0, 100.0]
        decreasing_energy = [-0.1, -1.0, -10.0, -100.0]

        increased_qpl = [self.qpl.calculate_qpl(entropy, e) for e in increasing_energy]
        decreased_qpl = [self.qpl.calculate_qpl(entropy, e) for e in decreasing_energy]

        self.assertTrue(all(earlier <= later for earlier, later in zip(increased_qpl, increased_qpl[1:])),
                        "QPL should increase with increasing energy")
        self.assertTrue(all(earlier >= later for earlier, later in zip(decreased_qpl, decreased_qpl[1:])),
                        "QPL should decrease with decreasing energy")

    def test_boundary_conditions(self):
        entropy = 0.0
        energy = 1.0
        qpl_value = self.qpl.calculate_qpl(entropy, energy)
        self.assertGreaterEqual(qpl_value, 0, "QPL should be non-negative even with zero entropy")

        negative_entropy = -1.0
        qpl_value = self.qpl.calculate_qpl(negative_entropy, energy)
        self.assertGreaterEqual(qpl_value, 0, "QPL should be non-negative even with negative entropy")

    def test_real_world_constraints(self):
        entropy = 1.0
        energy = 0.5
        qpl_value = self.qpl.calculate_qpl(entropy, energy)
        self.assertLessEqual(qpl_value, entropy + self.qpl.lambda_factor * np.tanh(energy))

if __name__ == "__main__":
    unittest.main()
