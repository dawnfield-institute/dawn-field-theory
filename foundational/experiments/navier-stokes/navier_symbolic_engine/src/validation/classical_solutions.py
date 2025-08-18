"""Analytical solution comparisons."""


import numpy as np


class ClassicalSolutions:
    """
    Compare symbolic solutions to analytical results (e.g., Poiseuille, Couette).
    """
    @staticmethod
    def compare_to_poiseuille(computed: np.ndarray, params: dict) -> float:
        """
        Compare computed solution to analytical Poiseuille profile.
        """
        shape = computed.shape
        max_velocity = params.get('max_velocity', 1.0)
        y = np.linspace(-1, 1, shape[0])
        profile = max_velocity * (1 - y**2)
        reference = np.tile(profile[:, None], (1, shape[1]))
        error = np.linalg.norm(computed - reference) / (np.linalg.norm(reference) + 1e-12)
        return float(error)
