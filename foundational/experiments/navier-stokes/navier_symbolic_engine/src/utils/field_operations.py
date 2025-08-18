"""Velocity/pressure field utilities."""


import numpy as np

class FieldOperations:
    """
    Utility functions for velocity and pressure fields.
    Includes field initialization, normalization, and divergence computation.
    """
    @staticmethod
    def initialize_velocity_field(shape, value=0.0):
        """
        Initialize a velocity field array with the given shape and value.
        """
        return np.full(shape, value, dtype=np.float32)

    @staticmethod
    def compute_divergence(field: np.ndarray) -> np.ndarray:
        """
        Compute the divergence of a 2D velocity field using finite differences.
        """
        if len(field.shape) != 2:
            raise ValueError("Only 2D fields supported")
        
        # Simple finite difference divergence computation
        dudx = np.gradient(field, axis=1)
        dvdy = np.gradient(field, axis=0)
        return dudx + dvdy

    @staticmethod
    def normalize_field(field: np.ndarray) -> np.ndarray:
        """
        Normalize a field to unit magnitude.
        """
        norm = np.linalg.norm(field)
        return field / (norm + 1e-12) if norm > 1e-12 else field

    @staticmethod
    def apply_boundary_conditions(field: np.ndarray, bc_type: str = "no_slip") -> np.ndarray:
        """
        Apply boundary conditions to a field.
        """
        result = field.copy()
        if bc_type == "no_slip":
            # Set boundary values to zero
            result[0, :] = 0  # top
            result[-1, :] = 0  # bottom
            result[:, 0] = 0  # left
            result[:, -1] = 0  # right
        return result

    @staticmethod
    def interpolate_field(field: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Interpolate field to target shape using simple bilinear interpolation.
        """
        from scipy.ndimage import zoom
        zoom_factors = [target_shape[i] / field.shape[i] for i in range(len(field.shape))]
        return zoom(field, zoom_factors, order=1)
