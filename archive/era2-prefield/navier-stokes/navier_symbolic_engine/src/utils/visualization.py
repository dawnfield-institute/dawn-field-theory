"""Plotting and visualization tools."""


import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """
    Provides plotting and visualization tools for velocity fields and pattern trees.
    """
    @staticmethod
    def plot_velocity_field(field: np.ndarray, title: str = "Velocity Field", save_path: str = None):
        """
        Plot a 2D velocity field with streamlines and contours.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Contour plot
        im1 = ax1.imshow(field, cmap='viridis', origin='lower')
        ax1.set_title(f"{title} - Magnitude")
        plt.colorbar(im1, ax=ax1)
        
        # Vector field (subsampled for clarity)
        y, x = np.mgrid[0:field.shape[0]:4, 0:field.shape[1]:4]
        u = field[::4, ::4]
        v = np.zeros_like(u)  # Assuming 2D horizontal flow
        ax2.quiver(x, y, u, v, scale=20)
        ax2.set_title(f"{title} - Vectors")
        ax2.set_xlim(0, field.shape[1])
        ax2.set_ylim(0, field.shape[0])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def plot_pattern_tree(tree, title: str = "Pattern Tree", max_depth: int = 3):
        """
        Visualize pattern tree structure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        positions = {}
        def assign_positions(node, depth=0, x=0, width=1):
            if depth > max_depth:
                return
            positions[node.pattern_id] = (x, -depth)
            if node.children:
                child_width = width / len(node.children)
                for i, child in enumerate(node.children):
                    child_x = x - width/2 + (i + 0.5) * child_width
                    assign_positions(child, depth + 1, child_x, child_width)
        
        assign_positions(tree.root)
        
        # Draw nodes
        for node_id, (x, y) in positions.items():
            ax.scatter(x, y, s=100, c='blue', alpha=0.7)
            ax.text(x, y + 0.1, str(node_id), ha='center', va='bottom')
        
        # Draw edges
        def draw_edges(node):
            if node.pattern_id in positions:
                px, py = positions[node.pattern_id]
                for child in node.children:
                    if child.pattern_id in positions:
                        cx, cy = positions[child.pattern_id]
                        ax.plot([px, cx], [py, cy], 'k-', alpha=0.5)
                        draw_edges(child)
        
        draw_edges(tree.root)
        
        ax.set_title(title)
        ax.set_xlabel("Branch Position")
        ax.set_ylabel("Tree Depth")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_entropy_signature(signature: np.ndarray, title: str = "Entropy Signature"):
        """
        Plot entropy signature as a line plot.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(signature)
        plt.title(title)
        plt.xlabel("Component Index")
        plt.ylabel("Entropy Value")
        plt.grid(True, alpha=0.3)
        plt.show()
