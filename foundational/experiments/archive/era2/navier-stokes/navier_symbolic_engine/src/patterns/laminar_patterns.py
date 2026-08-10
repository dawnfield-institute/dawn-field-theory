"""Laminar flow pattern templates."""


import numpy as np


class LaminarPatterns:
    """
    Provides laminar flow pattern templates (e.g., Poiseuille, Couette, Stokes).
    """
    @staticmethod
    def poiseuille_flow(shape, max_velocity=1.0):
        """
        Generate a 2D Poiseuille flow velocity profile (parabolic, for pipe/channel).
        """
        y = np.linspace(-1, 1, shape[0])
        profile = max_velocity * (1 - y**2)
        field = np.tile(profile[:, None], (1, shape[1]))
        return field

    @staticmethod
    def couette_flow(shape, wall_velocity=1.0):
        """
        Generate a 2D Couette flow velocity profile (linear shear).
        """
        y = np.linspace(0, 1, shape[0])
        profile = wall_velocity * y
        field = np.tile(profile[:, None], (1, shape[1]))
        return field

    @staticmethod
    def stokes_flow(shape, force=1.0):
        """
        Generate a simple Stokes flow (constant force, placeholder).
        """
        return np.full(shape, force)
