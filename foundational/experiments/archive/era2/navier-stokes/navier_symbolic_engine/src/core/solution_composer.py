"""Final solution assembly."""


from typing import List, Any


import numpy as np

class SolutionComposer:
    """
    Composes the final flow solution from a navigation path through the pattern tree.
    Responsible for assembling velocity and pressure fields from symbolic patterns.
    """
    def __init__(self, pattern_library):
        self.pattern_library = pattern_library

    def compose_solution(self, navigation_path: List[Any], field_shape=(32, 32)) -> dict:
        """
        Assemble the flow solution from the given navigation path.
        Returns the composed velocity and pressure fields.
        """
        # Compose velocity field by blending pattern_data along the path
        velocity_fields = []
        for node in navigation_path:
            if node.pattern_data is not None:
                velocity_fields.append(node.pattern_data)
        if not velocity_fields:
            # Fallback: use a default laminar pattern
            velocity_fields = [self.pattern_library.laminar.poiseuille_flow(field_shape)]
        # Simple average for demonstration
        velocity = np.mean(velocity_fields, axis=0)
        # Placeholder: pressure field (could be derived from velocity)
        pressure = np.zeros(field_shape)
        return {"velocity": velocity, "pressure": pressure}
