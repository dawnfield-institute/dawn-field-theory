"""Main API interface."""

from core.pattern_tree import PatternTree
from core.entropy_navigator import EntropyNavigator
from core.memory_tracker import MemoryTracker
from core.thermodynamic_validator import ThermodynamicValidator
from core.solution_composer import SolutionComposer
from utils.entropy_hasher import EntropyHasher
from patterns.pattern_library import PatternLibrary

class EngineInterface:
    """
    Main entry point for using the symbolic Navier-Stokes engine.
    Provides high-level API for problem setup, execution, and result retrieval.
    """
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.pattern_tree = PatternTree()
        self.pattern_library = PatternLibrary()
        self.entropy_hasher = EntropyHasher()
        self.navigator = EntropyNavigator(self.pattern_tree)
        self.memory_tracker = MemoryTracker()
        self.thermodynamic_validator = ThermodynamicValidator()
        self.solution_composer = SolutionComposer(self.pattern_library)
        
        # Initialize with some basic patterns
        self._initialize_basic_patterns()

    def _initialize_basic_patterns(self):
        """
        Initialize the pattern tree with basic laminar and turbulent patterns.
        """
        # Add laminar patterns
        laminar_entropy = self.entropy_hasher._hash_component("laminar_base")
        laminar_data = self.pattern_library.laminar.poiseuille_flow((32, 32))
        self.pattern_tree.add_pattern(
            self.pattern_tree.root, 
            laminar_entropy, 
            "laminar", 
            pattern_data=laminar_data
        )
        
        # Add turbulent patterns
        turbulent_entropy = self.entropy_hasher._hash_component("turbulent_base")
        turbulent_data = self.pattern_library.turbulent.random_turbulent_field((32, 32), seed=42)
        self.pattern_tree.add_pattern(
            self.pattern_tree.root, 
            turbulent_entropy, 
            "turbulent", 
            pattern_data=turbulent_data
        )

    def run(self, boundary_conditions: dict) -> dict:
        """
        Run the symbolic engine for the given boundary conditions.
        Returns the computed solution and metadata.
        """
        try:
            # Generate entropy signature
            entropy_sig = self.entropy_hasher.generate_hierarchical_entropy(boundary_conditions)
            
            # Navigate pattern tree
            navigation_path = self.navigator.navigate(entropy_sig)
            
            # Track memory
            for node in navigation_path:
                self.memory_tracker.record(node)
            
            # Validate thermodynamics (placeholder)
            # thermodynamic_results = self.thermodynamic_validator.validate_transition(...)
            
            # Compose solution
            solution = self.solution_composer.compose_solution(navigation_path)
            
            return {
                "status": "success",
                "solution": solution,
                "navigation_path": [node.pattern_id for node in navigation_path],
                "entropy_signature": entropy_sig.levels if hasattr(entropy_sig, 'levels') else entropy_sig
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "solution": None
            }

    def get_tree_info(self) -> dict:
        """
        Get information about the current pattern tree.
        """
        node_count = 0
        max_depth = 0
        
        def count_nodes(node):
            nonlocal node_count, max_depth
            node_count += 1
            if hasattr(node, 'depth') and node.depth > max_depth:
                max_depth = node.depth
        
        self.pattern_tree.traverse(action=count_nodes)
        
        return {
            "node_count": node_count,
            "max_depth": max_depth
        }
