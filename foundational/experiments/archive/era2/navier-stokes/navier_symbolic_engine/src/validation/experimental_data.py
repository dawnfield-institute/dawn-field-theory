"""Real experimental data validation."""


import numpy as np


class ExperimentalData:
    """
    Validate symbolic solutions against real experimental data.
    """
    @staticmethod
    def compare_to_experiment(computed: np.ndarray, experiment_data: np.ndarray) -> float:
        """
        Compare computed solution to experimental data (L2 error).
        """
        return float(np.linalg.norm(computed - experiment_data) / (np.linalg.norm(experiment_data) + 1e-12))
