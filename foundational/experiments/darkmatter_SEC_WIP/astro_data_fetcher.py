"""
Astroquery module for fetching and parsing dark matter distribution data.
Focused module for importing into simulation scripts.
"""

import numpy as np
import torch
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
import logging
from typing import Tuple, Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)

class AstroDataFetcher:
    """Efficient fetcher for astronomical dark matter distribution data"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.cache = {}
        
    def fetch_galaxy_clusters(self, limit=5000) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fetch galaxy cluster data (proxy for dark matter structures)
        Returns: (positions_3d, metadata)
        """
        cache_key = f"clusters_{limit}"
        if cache_key in self.cache:
            logging.info("Using cached galaxy cluster data")
            return self.cache[cache_key]
            
        logging.info(f"Fetching {limit} galaxy clusters from SDSS...")
        
        try:
            # Query for galaxies with spectroscopic redshift (better distance estimates)
            query = f"""
                SELECT TOP {limit} 
                    p.ra, p.dec, s.z, 
                    p.psfMag_r, p.psfMag_g, p.psfMag_i,
                    s.velDisp
                FROM PhotoObj AS p
                JOIN SpecObj AS s ON p.objID = s.bestObjID
                WHERE s.z BETWEEN 0.01 AND 1.0
                AND s.zWarning = 0
                AND p.type = 3
                ORDER BY s.z
            """
            
            result = SDSS.query_sql(query)
            
            if result is None or len(result) == 0:
                raise ValueError("No data returned from SDSS")
                
            # Convert to 3D coordinates (simplified comoving coordinates)
            ra = np.array(result['ra'])
            dec = np.array(result['dec']) 
            z = np.array(result['z'])
            
            # Convert to Cartesian coordinates
            # Simplified: use redshift as radial distance (not precise but good for patterns)
            x = z * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
            y = z * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
            z_coord = z * np.sin(np.radians(dec))
            
            positions = torch.tensor(np.column_stack([x, y, z_coord]), 
                                   dtype=torch.float32, device=self.device)
            
            # Scale to reasonable simulation bounds (normalize to ±20 range)
            positions = positions * 20.0 / torch.max(torch.abs(positions))
            
            metadata = {
                'source': 'SDSS_galaxy_clusters',
                'count': len(result),
                'redshift_range': (float(np.min(z)), float(np.max(z))),
                'spatial_extent': float(torch.max(torch.abs(positions))),
                'mean_position': positions.mean(dim=0).cpu().numpy().tolist()
            }
            
            self.cache[cache_key] = (positions, metadata)
            logging.info(f"Successfully fetched {len(positions)} galaxy positions")
            return positions, metadata
            
        except Exception as e:
            logging.error(f"SDSS query failed: {e}")
            return self._generate_fallback_data(limit)
    
    def fetch_temporal_gradient_data(self, total_limit=5000, z_bins=5) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Fetch galaxy cluster data in redshift bins for temporal gradient simulation.
        Returns data from high-z (young) to low-z (evolved) structures.
        
        Args:
            total_limit: Total number of galaxies to fetch
            z_bins: Number of redshift bins to create
            
        Returns:
            (list_of_position_tensors, metadata)
        """
        cache_key = f"temporal_gradient_{total_limit}_{z_bins}"
        if cache_key in self.cache:
            logging.info("Using cached temporal gradient data")
            return self.cache[cache_key]
            
        logging.info(f"Fetching {total_limit} galaxies in {z_bins} redshift bins for temporal gradient...")
        
        try:
            # Define redshift bins from high-z (young) to low-z (evolved)
            z_min, z_max = 0.01, 1.0
            z_edges = np.linspace(z_max, z_min, z_bins + 1)  # Start from high-z
            per_bin = total_limit // z_bins
            
            position_bins = []
            bin_metadata = []
            
            for i in range(z_bins):
                z_low = z_edges[i+1]  # Lower redshift (higher evolution)
                z_high = z_edges[i]   # Higher redshift (lower evolution)
                
                query = f"""
                    SELECT TOP {per_bin} 
                        p.ra, p.dec, s.z, 
                        p.psfMag_r, s.velDisp
                    FROM PhotoObj AS p
                    JOIN SpecObj AS s ON p.objID = s.bestObjID
                    WHERE s.z BETWEEN {z_low:.3f} AND {z_high:.3f}
                    AND s.zWarning = 0
                    AND p.type = 3
                    ORDER BY s.z DESC
                """
                
                result = SDSS.query_sql(query)
                
                if result is None or len(result) == 0:
                    logging.warning(f"No data for redshift bin {z_low:.3f} - {z_high:.3f}")
                    continue
                    
                # Convert to 3D coordinates
                ra = np.array(result['ra'])
                dec = np.array(result['dec']) 
                z = np.array(result['z'])
                
                # Convert to Cartesian coordinates
                x = z * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
                y = z * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
                z_coord = z * np.sin(np.radians(dec))
                
                positions = torch.tensor(np.column_stack([x, y, z_coord]), 
                                       dtype=torch.float32, device=self.device)
                
                # Scale to reasonable simulation bounds
                positions = positions * 20.0 / torch.max(torch.abs(positions))
                
                position_bins.append(positions)
                
                bin_metadata.append({
                    'redshift_range': (float(z_low), float(z_high)),
                    'mean_redshift': float(np.mean(z)),
                    'count': len(positions),
                    'age_gyr': self._redshift_to_age(np.mean(z)),  # Approximate age
                    'evolutionary_stage': i / (z_bins - 1)  # 0 = young, 1 = evolved
                })
                
                logging.info(f"Bin {i+1}/{z_bins}: z={z_low:.3f}-{z_high:.3f}, "
                           f"age≈{self._redshift_to_age(np.mean(z)):.1f} Gyr, "
                           f"{len(positions)} galaxies")
            
            metadata = {
                'source': 'SDSS_temporal_gradient',
                'total_bins': len(position_bins),
                'total_galaxies': sum(len(pos) for pos in position_bins),
                'bin_metadata': bin_metadata,
                'approach': 'high_z_to_low_z_gradient'
            }
            
            self.cache[cache_key] = (position_bins, metadata)
            logging.info(f"Successfully created {len(position_bins)} temporal gradient bins")
            return position_bins, metadata
            
        except Exception as e:
            logging.error(f"Temporal gradient query failed: {e}")
            return self._generate_temporal_fallback_data(total_limit, z_bins)
    
    def _redshift_to_age(self, z: float) -> float:
        """Approximate conversion from redshift to universe age in Gyr"""
        # Simplified: t ≈ 13.8 * (1 + z)^(-1.5) Gyr (rough approximation)
        return 13.8 / (1 + z)**1.5
    
    def _generate_temporal_fallback_data(self, total_limit: int, z_bins: int) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Generate synthetic temporal gradient data as fallback"""
        logging.info("Generating temporal gradient fallback data")
        
        position_bins = []
        bin_metadata = []
        per_bin = total_limit // z_bins
        
        for i in range(z_bins):
            # Simulate evolutionary progression from young (clustered) to evolved (dispersed)
            evolution_factor = i / (z_bins - 1)
            
            # Young structures: more clustered, less processed
            cluster_strength = 1.0 - evolution_factor * 0.7
            dispersion = 5.0 + evolution_factor * 15.0
            
            positions = torch.randn(per_bin, 3, device=self.device) * dispersion
            
            # Add clustering for younger structures
            if evolution_factor < 0.5:
                n_clusters = max(1, int(per_bin * cluster_strength / 100))
                for j in range(n_clusters):
                    center = torch.randn(3, device=self.device) * 10
                    cluster_size = int(per_bin * 0.1)
                    start_idx = j * cluster_size
                    end_idx = min(start_idx + cluster_size, per_bin)
                    positions[start_idx:end_idx] += center
            
            # Scale to bounds
            positions = positions * 20.0 / torch.max(torch.abs(positions))
            position_bins.append(positions)
            
            bin_metadata.append({
                'redshift_range': (1.0 - evolution_factor * 0.9, 1.0 - evolution_factor * 0.9 + 0.2),
                'evolutionary_stage': evolution_factor,
                'count': len(positions),
                'synthetic': True
            })
        
        metadata = {
            'source': 'synthetic_temporal_gradient',
            'total_bins': z_bins,
            'total_galaxies': total_limit,
            'bin_metadata': bin_metadata,
            'approach': 'fallback_gradient'
        }
        
        return position_bins, metadata
    
    def fetch_cosmic_web_filaments(self, limit=3000) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fetch data representing cosmic web filaments
        Returns: (positions_3d, metadata)
        """
        cache_key = f"filaments_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        logging.info(f"Fetching cosmic web filament data...")
        
        try:
            # Query for Lyman-alpha forest quasars (traces cosmic web)
            query = f"""
                SELECT TOP {limit}
                    p.ra, p.dec, s.z,
                    p.psfMag_g, p.psfMag_r
                FROM PhotoObj AS p
                JOIN SpecObj AS s ON p.objID = s.bestObjID  
                WHERE s.z BETWEEN 1.5 AND 4.0
                AND p.type = 1
                AND s.zWarning = 0
                ORDER BY NEWID()
            """
            
            result = SDSS.query_sql(query)
            
            if result is None or len(result) == 0:
                raise ValueError("No quasar data returned")
                
            # Convert to 3D positions
            ra = np.array(result['ra'])
            dec = np.array(result['dec'])
            z = np.array(result['z'])
            
            # Higher redshift objects - scale differently
            x = z * np.cos(np.radians(dec)) * np.cos(np.radians(ra)) * 0.3
            y = z * np.cos(np.radians(dec)) * np.sin(np.radians(ra)) * 0.3
            z_coord = z * np.sin(np.radians(dec)) * 0.3
            
            positions = torch.tensor(np.column_stack([x, y, z_coord]), 
                                   dtype=torch.float32, device=self.device)
            
            # Scale to simulation bounds
            positions = positions * 15.0 / torch.max(torch.abs(positions))
            
            metadata = {
                'source': 'SDSS_cosmic_web_tracers',
                'count': len(result),
                'redshift_range': (float(np.min(z)), float(np.max(z))),
                'spatial_extent': float(torch.max(torch.abs(positions))),
                'structure_type': 'cosmic_web_filaments'
            }
            
            self.cache[cache_key] = (positions, metadata)
            return positions, metadata
            
        except Exception as e:
            logging.error(f"Cosmic web query failed: {e}")
            return self._generate_fallback_data(limit)
    
    def compute_real_data_metrics(self, positions: torch.Tensor) -> Dict[str, float]:
        """
        Compute the same metrics as our simulation for comparison
        """
        n = positions.shape[0]
        
        # Fractal dimension (simplified box counting)
        center = torch.mean(positions, dim=0)
        distances = torch.norm(positions - center, dim=1)
        
        # Radial distribution for fractal analysis
        max_dist = torch.max(distances).item()
        radii = torch.logspace(-1, np.log10(max_dist), 15, device=self.device)
        counts = []
        
        for r in radii:
            count = torch.sum(distances <= r).item()
            counts.append(max(1, count))
        
        # Fit fractal dimension
        log_radii = torch.log(radii).cpu().numpy()
        log_counts = np.log(counts)
        fractal_dim = abs(np.polyfit(log_radii, log_counts, 1)[0])
        
        # Spatial entropy (3D grid)
        positions_cpu = positions.cpu().numpy()
        hist, _ = np.histogramdd(positions_cpu, bins=12)
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]
        spatial_entropy = -np.sum(hist_norm * np.log(hist_norm))
        
        # Density variance (simplified)
        chunk_size = min(1000, n)
        densities = []
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            chunk_pos = positions[i:end_i]
            
            # Count neighbors within radius
            diff = chunk_pos.unsqueeze(1) - positions.unsqueeze(0)
            dists = torch.norm(diff, dim=2)
            neighbors = torch.sum((dists < 2.0) & (dists > 0), dim=1)
            densities.extend(neighbors.cpu().numpy())
        
        density_variance = np.var(densities)
        
        return {
            'fractal_dimension': float(fractal_dim),
            'spatial_entropy': float(spatial_entropy),
            'density_variance': float(density_variance),
            'mean_density': float(np.mean(densities)),
            'radial_extent': float(torch.std(distances)),
            'n_points': n
        }
    
    def _generate_fallback_data(self, n_points: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate realistic fallback data if queries fail"""
        logging.info(f"Generating {n_points} fallback data points")
        
        # Generate clustered structure mimicking real observations
        n_clusters = max(5, n_points // 200)
        positions = []
        
        for i in range(n_clusters):
            # Cluster centers
            center = torch.randn(3, device=self.device) * 10
            cluster_size = max(10, n_points // n_clusters)
            
            # Points around cluster
            cluster_points = torch.randn(cluster_size, 3, device=self.device) * 2 + center
            positions.append(cluster_points)
        
        positions = torch.cat(positions, dim=0)[:n_points]
        
        metadata = {
            'source': 'fallback_synthetic',
            'count': len(positions),
            'spatial_extent': float(torch.max(torch.abs(positions))),
            'n_clusters': n_clusters
        }
        
        return positions, metadata
    
    def get_comparison_dataset(self, dataset_type='clusters', limit=5000):
        """
        Main interface for getting comparison data
        """
        if dataset_type == 'clusters':
            return self.fetch_galaxy_clusters(limit)
        elif dataset_type == 'filaments':
            return self.fetch_cosmic_web_filaments(limit)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
