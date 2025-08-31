"""
Quick test of the astro data fetcher and real data analysis
"""

from astro_data_fetcher import AstroDataFetcher
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Test the fetcher
print("Testing AstroDataFetcher...")
fetcher = AstroDataFetcher(device='cpu')  # Use CPU for faster testing

# Test galaxy clusters
positions, metadata = fetcher.get_comparison_dataset('clusters', limit=1000)
print(f"\nGalaxy clusters fetched:")
print(f"- Count: {metadata['count']}")
print(f"- Source: {metadata['source']}")
print(f"- Spatial extent: {metadata['spatial_extent']:.2f}")
print(f"- Redshift range: {metadata.get('redshift_range', 'N/A')}")

# Compute metrics
metrics = fetcher.compute_real_data_metrics(positions)
print(f"\nReal data metrics:")
for key, value in metrics.items():
    print(f"- {key}: {value:.3f}")

# Quick visualization
fig = plt.figure(figsize=(15, 5))

# 3D scatter
ax1 = fig.add_subplot(131, projection='3d')
pos_np = positions.cpu().numpy()
sample_indices = np.random.choice(len(pos_np), min(1000, len(pos_np)), replace=False)
sample_pos = pos_np[sample_indices]
ax1.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], s=1, alpha=0.6)
ax1.set_title(f'Real {metadata["source"]} (3D)')

# XY projection
ax2 = fig.add_subplot(132)
ax2.scatter(sample_pos[:, 0], sample_pos[:, 1], s=1, alpha=0.6)
ax2.set_title('XY Projection')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')

# Radial distribution
ax3 = fig.add_subplot(133)
radial_dists = np.sqrt(np.sum(sample_pos**2, axis=1))
ax3.hist(radial_dists, bins=30, alpha=0.7, density=True)
ax3.set_title('Radial Distribution')
ax3.set_xlabel('Distance from Origin')
ax3.set_ylabel('Density')

plt.tight_layout()
plt.savefig('astro_data_test.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nTest completed! Visualization saved as 'astro_data_test.png'")

# Test cosmic web filaments too
print("\n" + "="*50)
print("Testing cosmic web filaments...")
fil_positions, fil_metadata = fetcher.get_comparison_dataset('filaments', limit=500)
fil_metrics = fetcher.compute_real_data_metrics(fil_positions)

print(f"Filament data:")
print(f"- Count: {fil_metadata['count']}")
print(f"- Fractal dimension: {fil_metrics['fractal_dimension']:.3f}")
print(f"- Spatial entropy: {fil_metrics['spatial_entropy']:.3f}")

print(f"\nComparison:")
print(f"Galaxy clusters vs Cosmic web:")
print(f"- Fractal dimension: {metrics['fractal_dimension']:.3f} vs {fil_metrics['fractal_dimension']:.3f}")
print(f"- Spatial entropy: {metrics['spatial_entropy']:.3f} vs {fil_metrics['spatial_entropy']:.3f}")
print(f"- Density variance: {metrics['density_variance']:.1f} vs {fil_metrics['density_variance']:.1f}")
