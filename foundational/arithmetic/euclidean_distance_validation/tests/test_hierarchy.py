"""
Unit tests for PAC hierarchy data structures.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.pac_hierarchy import PACNode, PACHierarchy


class TestPACNode(unittest.TestCase):
    """Test PACNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = PACNode(id="test", value=1.0)
        self.assertEqual(node.id, "test")
        self.assertEqual(node.value, 1.0)
        self.assertEqual(node.depth, 0)
        self.assertEqual(len(node.children), 0)
    
    def test_add_child(self):
        """Test adding children."""
        parent = PACNode(id="parent", value=1.0)
        child = PACNode(id="child", value=0.5)
        
        parent.add_child(child)
        
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.depth, 1)
    
    def test_pac_residual_perfect_conservation(self):
        """Test PAC residual with perfect conservation."""
        parent = PACNode(id="parent", value=1.0)
        child1 = PACNode(id="c1", value=0.6)
        child2 = PACNode(id="c2", value=0.4)
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        residual = parent.pac_residual()
        self.assertAlmostEqual(residual, 0.0, places=10)
    
    def test_pac_residual_with_violation(self):
        """Test PAC residual with conservation violation."""
        parent = PACNode(id="parent", value=1.0)
        child1 = PACNode(id="c1", value=0.7)
        child2 = PACNode(id="c2", value=0.5)
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        residual = parent.pac_residual()
        self.assertAlmostEqual(residual, 0.2, places=10)
    
    def test_distance_computation(self):
        """Test Euclidean distance between nodes."""
        node1 = PACNode(id="n1", value=1.0, embedding=np.array([1.0, 0.0, 0.0]))
        node2 = PACNode(id="n2", value=1.0, embedding=np.array([0.0, 1.0, 0.0]))
        
        distance = node1.distance_to(node2)
        self.assertAlmostEqual(distance, np.sqrt(2), places=10)
    
    def test_distance_residual(self):
        """Test distance conservation residual."""
        parent = PACNode(id="p", value=1.0, embedding=np.array([1.0, 1.0]))
        child1 = PACNode(id="c1", value=0.5, embedding=np.array([1.0, 0.0]))
        child2 = PACNode(id="c2", value=0.5, embedding=np.array([0.0, 1.0]))
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        residual = parent.distance_residual()
        
        # ||parent||^2 = 2
        # ||child1||^2 + ||child2||^2 = 1 + 1 = 2
        # residual = |2 - 2| = 0
        self.assertAlmostEqual(residual, 0.0, places=10)


class TestPACHierarchy(unittest.TestCase):
    """Test PACHierarchy functionality."""
    
    def setUp(self):
        """Create test hierarchy."""
        self.root = PACNode(id="root", value=1.0)
        self.hierarchy = PACHierarchy(self.root)
    
    def test_hierarchy_creation(self):
        """Test hierarchy initialization."""
        self.assertEqual(self.hierarchy.root, self.root)
        self.assertEqual(len(self.hierarchy.nodes), 1)
    
    def test_add_node(self):
        """Test adding nodes to hierarchy."""
        child = PACNode(id="child", value=0.5)
        self.hierarchy.add_node(child, "root")
        
        self.assertEqual(len(self.hierarchy.nodes), 2)
        self.assertIn("child", self.hierarchy.nodes)
        self.assertEqual(child.parent, self.root)
    
    def test_get_levels(self):
        """Test level extraction."""
        # Build 3-level hierarchy
        c1 = PACNode(id="c1", value=0.5)
        c2 = PACNode(id="c2", value=0.5)
        self.hierarchy.add_node(c1, "root")
        self.hierarchy.add_node(c2, "root")
        
        gc1 = PACNode(id="gc1", value=0.25)
        self.hierarchy.add_node(gc1, "c1")
        
        levels = self.hierarchy.get_levels()
        
        self.assertEqual(len(levels), 3)
        self.assertEqual(len(levels[0]), 1)  # root
        self.assertEqual(len(levels[1]), 2)  # c1, c2
        self.assertEqual(len(levels[2]), 1)  # gc1
    
    def test_global_pac_residual(self):
        """Test global PAC residual computation."""
        c1 = PACNode(id="c1", value=0.6)
        c2 = PACNode(id="c2", value=0.4)
        self.hierarchy.add_node(c1, "root")
        self.hierarchy.add_node(c2, "root")
        
        residual = self.hierarchy.compute_global_pac_residual()
        self.assertAlmostEqual(residual, 0.0, places=10)
    
    def test_from_dict(self):
        """Test hierarchy construction from dictionary."""
        data = {
            'root': {
                'value': 1.0,
                'children': ['a', 'b']
            },
            'a': {'value': 0.6},
            'b': {'value': 0.4}
        }
        
        hierarchy = PACHierarchy.from_dict(data)
        
        self.assertEqual(len(hierarchy.nodes), 3)
        self.assertEqual(hierarchy.root.id, 'root')
        self.assertEqual(len(hierarchy.root.children), 2)


if __name__ == '__main__':
    unittest.main()
