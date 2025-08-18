"""Turbulent flow pattern templates."""


import numpy as np


class TurbulentPatterns:
    """
    Provides turbulent flow pattern templates (random, fractal, or synthetic turbulence).
    """
    @staticmethod
    def random_turbulent_field(shape, seed=None, scale=1.0):
        """
        Generate a random turbulent-like velocity field (Gaussian noise, scaled).
        """
        rng = np.random.default_rng(seed)
        return scale * rng.normal(size=shape)

    @staticmethod
    def fractal_turbulence(shape, seed=None, scale=1.0):
        """
        Generate a synthetic fractal turbulence field (placeholder: sum of sines).
        """
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 2 * np.pi, shape[1])
        y = np.linspace(0, 2 * np.pi, shape[0])
        X, Y = np.meshgrid(x, y)
        field = np.zeros(shape)
        for freq in [1, 2, 4, 8]:
            phase = rng.uniform(0, 2 * np.pi)
            field += scale / freq * np.sin(freq * X + phase) * np.sin(freq * Y + phase)
        return field
