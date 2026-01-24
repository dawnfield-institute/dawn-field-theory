"""
N² Convergence Analyzer
========================

Computes kNN neighborhood overlap (Jaccard similarity) between feature spaces
to detect shared latent structure - "entangled roots" in PAC terminology.

SEC Interpretation:
- High convergence = entropy has collapsed = structure exists
- Low convergence = high entropy = unexplored territory

Key Reference: ../arithmetic/infodynamics_arithmetic_v1.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ConvergenceResult:
    """Result of convergence analysis between two feature spaces"""
    source_space: str
    target_space: str
    convergence: float  # Mean Jaccard similarity of k-NN neighborhoods
    source_features: List[str]
    target_features: List[str]
    n_samples: int
    k: int
    
    @property
    def is_high_convergence(self) -> bool:
        """Above threshold indicates exploitable structure"""
        return self.convergence > 0.05
    
    @property
    def entropy_proxy(self) -> float:
        """SEC interpretation: 1 - convergence = entropy level"""
        return 1.0 - self.convergence


class ConvergenceAnalyzer:
    """
    Computes N² kNN convergence between all pairs of feature spaces.
    
    This measures how similarly k-nearest neighbors organize across spaces.
    High Jaccard overlap = spaces share latent structure = "entangled roots"
    
    SEC Mapping:
    - convergence → collapsed entropy (structure exists)
    - 1 - convergence → residual entropy (unexplored territory)
    """
    
    def __init__(self, k: int = 5, threshold: float = 0.05):
        """
        Args:
            k: Number of neighbors for kNN comparison
            threshold: Convergence threshold for "high convergence" classification
        """
        self.k = k
        self.threshold = threshold
        self.scalers: Dict[str, StandardScaler] = {}
        
    def compute_pairwise(self, 
                         X_source: np.ndarray, 
                         X_target: np.ndarray,
                         source_name: str = "source",
                         target_name: str = "target") -> ConvergenceResult:
        """
        Compute Jaccard convergence between two feature spaces.
        
        Args:
            X_source: Features from space A (n_samples, n_features_a)
            X_target: Features from space B (n_samples, n_features_b)
            source_name: Name identifier for source space
            target_name: Name identifier for target space
            
        Returns:
            ConvergenceResult with Jaccard similarity of k-NN neighborhoods
        """
        n_samples = len(X_source)
        if len(X_target) != n_samples:
            raise ValueError("Feature spaces must have same number of samples")
            
        # Adjust k if necessary
        k_actual = min(self.k + 1, n_samples)  # +1 because we exclude self
        
        # Normalize features
        scaler_source = StandardScaler()
        scaler_target = StandardScaler()
        X_source_norm = scaler_source.fit_transform(X_source)
        X_target_norm = scaler_target.fit_transform(X_target)
        
        # Build kNN indices
        knn_source = NearestNeighbors(n_neighbors=k_actual, algorithm='auto')
        knn_target = NearestNeighbors(n_neighbors=k_actual, algorithm='auto')
        
        knn_source.fit(X_source_norm)
        knn_target.fit(X_target_norm)
        
        # Get neighbors (exclude self - first column)
        neighbors_source = knn_source.kneighbors(X_source_norm, return_distance=False)[:, 1:]
        neighbors_target = knn_target.kneighbors(X_target_norm, return_distance=False)[:, 1:]
        
        # Compute Jaccard similarity for each sample
        jaccards = []
        for i in range(n_samples):
            set_source = set(neighbors_source[i])
            set_target = set(neighbors_target[i])
            union = set_source | set_target
            if len(union) > 0:
                jaccard = len(set_source & set_target) / len(union)
                jaccards.append(jaccard)
        
        mean_jaccard = np.mean(jaccards) if jaccards else 0.0
        
        return ConvergenceResult(
            source_space=source_name,
            target_space=target_name,
            convergence=mean_jaccard,
            source_features=[],  # Will be filled by caller
            target_features=[],
            n_samples=n_samples,
            k=self.k
        )
    
    def compute_all_pairs(self,
                          data: pd.DataFrame,
                          feature_spaces: Dict[str, List[str]],
                          symmetric: bool = True) -> pd.DataFrame:
        """
        Compute N² convergence matrix for all feature space pairs.
        
        Args:
            data: DataFrame containing all features
            feature_spaces: Dict mapping space names to feature column lists
            symmetric: If True, only compute upper triangle (A→B = B→A)
            
        Returns:
            DataFrame with convergence results for all pairs
        """
        space_names = list(feature_spaces.keys())
        results = []
        
        for i, source_name in enumerate(space_names):
            source_cols = feature_spaces[source_name]
            X_source = data[source_cols].values
            
            # Skip spaces with no variance
            if X_source.std() == 0:
                continue
                
            start_j = i + 1 if symmetric else 0
            
            for j in range(start_j, len(space_names)):
                if i == j:
                    continue
                    
                target_name = space_names[j]
                target_cols = feature_spaces[target_name]
                X_target = data[target_cols].values
                
                # Skip spaces with no variance
                if X_target.std() == 0:
                    continue
                
                result = self.compute_pairwise(
                    X_source, X_target,
                    source_name, target_name
                )
                result.source_features = source_cols
                result.target_features = target_cols
                
                results.append({
                    'source_space': result.source_space,
                    'target_space': result.target_space,
                    'convergence': result.convergence,
                    'entropy_proxy': result.entropy_proxy,
                    'is_high_convergence': result.is_high_convergence,
                    'n_source_features': len(source_cols),
                    'n_target_features': len(target_cols),
                    'n_samples': result.n_samples,
                    'k': result.k
                })
                
                # Add symmetric entry if needed
                if symmetric:
                    results.append({
                        'source_space': result.target_space,
                        'target_space': result.source_space,
                        'convergence': result.convergence,
                        'entropy_proxy': result.entropy_proxy,
                        'is_high_convergence': result.is_high_convergence,
                        'n_source_features': len(target_cols),
                        'n_target_features': len(source_cols),
                        'n_samples': result.n_samples,
                        'k': result.k
                    })
        
        return pd.DataFrame(results)
    
    def compute_entropy_field(self, convergence_df: pd.DataFrame) -> Dict:
        """
        Compute SEC entropy field from convergence matrix.
        
        SEC: ∂S/∂t = α∇I - β∇H
        
        High convergence = low entropy (collapsed)
        Low convergence = high entropy (unexplored)
        
        Returns:
            Dict with entropy field analysis
        """
        # Build convergence matrix
        spaces = sorted(set(convergence_df['source_space']) | set(convergence_df['target_space']))
        n_spaces = len(spaces)
        space_idx = {s: i for i, s in enumerate(spaces)}
        
        conv_matrix = np.zeros((n_spaces, n_spaces))
        for _, row in convergence_df.iterrows():
            i = space_idx[row['source_space']]
            j = space_idx[row['target_space']]
            conv_matrix[i, j] = row['convergence']
        
        # Entropy field = 1 - convergence
        entropy_field = 1.0 - conv_matrix
        
        # Collapse potential = where convergence gradient is steep
        # (indicates boundary between structured and unstructured)
        collapse_potential = np.abs(np.gradient(conv_matrix)[0])
        
        # Identify zones
        high_structure_zone = conv_matrix > self.threshold
        exploration_zone = conv_matrix < self.threshold / 2
        
        return {
            'spaces': spaces,
            'convergence_matrix': conv_matrix,
            'entropy_field': entropy_field,
            'collapse_potential': collapse_potential,
            'high_structure_zone': high_structure_zone,
            'exploration_zone': exploration_zone,
            'mean_entropy': entropy_field.mean(),
            'entropy_variance': entropy_field.var(),
            'n_high_convergence_pairs': high_structure_zone.sum(),
            'n_exploration_pairs': exploration_zone.sum()
        }


def analyze_convergence_distribution(convergence_df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of convergence values across a domain.
    
    This helps characterize whether a domain has exploitable structure.
    """
    convergence_values = convergence_df['convergence'].values
    
    return {
        'mean': np.mean(convergence_values),
        'std': np.std(convergence_values),
        'min': np.min(convergence_values),
        'max': np.max(convergence_values),
        'median': np.median(convergence_values),
        'q25': np.percentile(convergence_values, 25),
        'q75': np.percentile(convergence_values, 75),
        'n_above_threshold': (convergence_values > 0.05).sum(),
        'n_total': len(convergence_values),
        'fraction_structured': (convergence_values > 0.05).mean(),
        'has_exploitable_structure': np.max(convergence_values) > 0.1
    }
