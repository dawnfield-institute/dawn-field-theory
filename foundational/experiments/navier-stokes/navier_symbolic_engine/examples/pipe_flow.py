"""
Simple pipe flow example using symbolic engine.
Demonstrates basic usage of the Navier-Stokes Symbolic Collapse Framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.engine_interface import EngineInterface
from src.utils.visualization import Visualization
import numpy as np

def run_pipe_flow_example():
    """
    Run a symbolic pipe flow simulation and visualize results.
    """
    print("Running Pipe Flow Example...")
    
    # Initialize the symbolic engine
    engine = EngineInterface()
    
    # Define boundary conditions for pipe flow
    boundary_conditions = {
        "geometry": "pipe",
        "reynolds": 1000,
        "velocity": 1.0,
        "pressure_gradient": -0.1,
        "boundary_values": {"inlet": 1.0, "outlet": 0.0, "walls": 0.0}
    }
    
    # Run the symbolic engine
    result = engine.run(boundary_conditions)
    
    if result["status"] == "success":
        print("✅ Simulation successful!")
        
        # Extract solution
        solution = result["solution"]
        velocity_field = solution["velocity"]
        pressure_field = solution["pressure"]
        
        print(f"Velocity field shape: {velocity_field.shape}")
        print(f"Max velocity: {np.max(velocity_field):.3f}")
        print(f"Min velocity: {np.min(velocity_field):.3f}")
        
        # Get tree information
        tree_info = engine.get_tree_info()
        print(f"Pattern tree nodes: {tree_info['node_count']}")
        print(f"Pattern tree depth: {tree_info['max_depth']}")
        
        # Visualize results
        vis = Visualization()
        vis.plot_velocity_field(velocity_field, "Pipe Flow - Velocity Field")
        
        if hasattr(result, 'entropy_signature'):
            vis.plot_entropy_signature(result["entropy_signature"], "Pipe Flow - Entropy Signature")
        
        return result
    else:
        print(f"❌ Simulation failed: {result['error']}")
        return None

if __name__ == "__main__":
    run_pipe_flow_example()
