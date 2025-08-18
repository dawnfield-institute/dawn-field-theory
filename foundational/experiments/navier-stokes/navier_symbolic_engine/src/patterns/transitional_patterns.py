"""Transition regime patterns."""


import numpy as np


class TransitionalPatterns:
    """
    Provides transitional flow pattern templates (blends/interpolates laminar and turbulent fields).
    """
    @staticmethod
    def blend(laminar_field, turbulent_field, alpha=0.5):
        """
        Blend laminar and turbulent fields to create a transitional pattern.
        """
        return (1 - alpha) * laminar_field + alpha * turbulent_field

    @staticmethod
    def bifractal_transition(laminar_field, turbulent_field, threshold=0.5):
        """
        Create a bifractal transition: below threshold use laminar, above use turbulent.
        """
        mask = np.random.rand(*laminar_field.shape) > threshold
        return np.where(mask, turbulent_field, laminar_field)
