"""CFD simulation comparisons."""


import numpy as np


class CFDBenchmarks:
    """
    Compare symbolic solutions to CFD simulation results.
    """
    @staticmethod
    def compare_to_cfd(computed: np.ndarray, cfd_reference: np.ndarray) -> float:
        """
        Compare computed solution to CFD reference (L2 error).
        """
        return float(np.linalg.norm(computed - cfd_reference) / (np.linalg.norm(cfd_reference) + 1e-12))
