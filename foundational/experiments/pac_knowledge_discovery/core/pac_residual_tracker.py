"""
PAC Residual Tracker
=====================

Computes and analyzes PAC conservation residuals to detect missing children
in informational hierarchies.

PAC Conservation: f(Parent) = Σf(Children)
Residual = observed - predicted = evidence of missing children

Key Reference: ../arithmetic/PACEngine/core/pac_kernel.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ResidualAnalysis:
    """Analysis of PAC residuals for a target space"""
    target_space: str
    residuals: np.ndarray
    mean_residual: float
    std_residual: float
    conservation_quality: float  # 0-1, higher = better conservation
    systematic_fraction: float   # Fraction of residuals that cluster
    n_clusters: int              # Number of residual clusters (potential missing children)
    cluster_labels: np.ndarray
    
    @property
    def has_missing_children(self) -> bool:
        """Indicates if residual structure suggests missing factors"""
        return self.n_clusters > 0 and self.systematic_fraction > 0.3
    
    @property
    def estimated_missing_children(self) -> int:
        """Estimate number of missing children from cluster count"""
        return max(0, self.n_clusters - 1)  # Subtract 1 for noise cluster


@dataclass
class ConservationMetrics:
    """Global PAC conservation metrics across all targets"""
    total_residual_norm: float
    mean_residual_magnitude: float
    conservation_quality: float
    n_targets_with_gaps: int
    estimated_total_missing: int
    target_analyses: Dict[str, ResidualAnalysis] = field(default_factory=dict)


class PACResidualTracker:
    """
    Track PAC conservation residuals to detect missing children.
    
    PAC says f(Parent) = Σf(Children). When our model predictions
    (based on known children) don't match observations, the residual
    encodes information about missing children.
    
    The structure of residuals tells us:
    - Random scatter → fully described (PAC conserved)
    - Systematic clusters → missing children with specific structure
    - Correlation with known spaces → decomposition targets
    """
    
    def __init__(self, 
                 cluster_eps: float = 0.5,
                 min_cluster_samples: int = 5):
        """
        Args:
            cluster_eps: DBSCAN epsilon for residual clustering
            min_cluster_samples: Minimum samples per cluster
        """
        self.cluster_eps = cluster_eps
        self.min_cluster_samples = min_cluster_samples
        self.residual_history: List[Dict] = []
        
    def compute_residual(self,
                         predicted: np.ndarray,
                         observed: np.ndarray,
                         target_name: str = "target") -> ResidualAnalysis:
        """
        Compute PAC residual and analyze its structure.
        
        Args:
            predicted: Model predictions (Σf(known_children))
            observed: Actual observations (f(parent))
            target_name: Name of the target space
            
        Returns:
            ResidualAnalysis with clustering and statistics
        """
        residuals = observed - predicted
        
        # Basic statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Conservation quality: exp(-|mean_residual|)
        # Perfect conservation = 1.0, poor = approaches 0
        conservation_quality = np.exp(-np.abs(residuals).mean())
        
        # Cluster residuals to detect missing children
        cluster_result = self._cluster_residuals(residuals)
        
        analysis = ResidualAnalysis(
            target_space=target_name,
            residuals=residuals,
            mean_residual=mean_residual,
            std_residual=std_residual,
            conservation_quality=conservation_quality,
            systematic_fraction=cluster_result['systematic_fraction'],
            n_clusters=cluster_result['n_clusters'],
            cluster_labels=cluster_result['labels']
        )
        
        # Record in history
        self.residual_history.append({
            'target': target_name,
            'mean_residual': mean_residual,
            'conservation_quality': conservation_quality,
            'n_clusters': cluster_result['n_clusters'],
            'timestamp': pd.Timestamp.now()
        })
        
        return analysis
    
    def _cluster_residuals(self, residuals: np.ndarray) -> Dict:
        """
        Cluster residuals to identify potential missing children.
        
        Systematic clusters = evidence of missing factors
        Random scatter = fully described (no missing children)
        """
        # Reshape for clustering
        X = residuals.reshape(-1, 1)
        
        # DBSCAN for arbitrary cluster shapes
        clusterer = DBSCAN(eps=self.cluster_eps, min_samples=self.min_cluster_samples)
        labels = clusterer.fit_predict(X)
        
        # Count clusters (excluding noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Systematic fraction = points in clusters / total
        systematic_fraction = (labels != -1).sum() / len(labels)
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'systematic_fraction': systematic_fraction
        }
    
    def analyze_residual_correlations(self,
                                      residuals: np.ndarray,
                                      feature_spaces: Dict[str, np.ndarray]) -> Dict:
        """
        Check if residuals correlate with known feature spaces.
        
        Strong correlation = that space should be decomposed further
        (it contains information not captured by current children)
        """
        correlations = {}
        
        for space_name, features in feature_spaces.items():
            if features.ndim == 1:
                # Single feature
                corr, p_value = stats.pearsonr(residuals, features)
                correlations[space_name] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'action': 'decompose' if abs(corr) > 0.3 else 'none'
                }
            else:
                # Multiple features - correlate with each
                for i in range(features.shape[1]):
                    feat_name = f"{space_name}_{i}"
                    corr, p_value = stats.pearsonr(residuals, features[:, i])
                    correlations[feat_name] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'action': 'decompose' if abs(corr) > 0.3 else 'none'
                    }
        
        return correlations
    
    def compute_global_conservation(self,
                                    predictions: Dict[str, np.ndarray],
                                    observations: Dict[str, np.ndarray]) -> ConservationMetrics:
        """
        Compute PAC conservation across all target spaces.
        
        Args:
            predictions: Dict of predictions by target name
            observations: Dict of observations by target name
            
        Returns:
            ConservationMetrics with global and per-target analysis
        """
        target_analyses = {}
        total_residual = 0.0
        n_gaps = 0
        total_missing = 0
        
        for target_name in predictions.keys():
            if target_name not in observations:
                continue
                
            analysis = self.compute_residual(
                predictions[target_name],
                observations[target_name],
                target_name
            )
            target_analyses[target_name] = analysis
            
            total_residual += np.abs(analysis.residuals).sum()
            if analysis.has_missing_children:
                n_gaps += 1
                total_missing += analysis.estimated_missing_children
        
        n_samples = sum(len(a.residuals) for a in target_analyses.values())
        mean_magnitude = total_residual / max(1, n_samples)
        global_quality = np.exp(-mean_magnitude)
        
        return ConservationMetrics(
            total_residual_norm=total_residual,
            mean_residual_magnitude=mean_magnitude,
            conservation_quality=global_quality,
            n_targets_with_gaps=n_gaps,
            estimated_total_missing=total_missing,
            target_analyses=target_analyses
        )
    
    def track_residual_reduction(self,
                                 before: ResidualAnalysis,
                                 after: ResidualAnalysis) -> Dict:
        """
        Track how residual changes when a new feature is added.
        
        This validates the PAC hypothesis: adding a missing child
        should reduce the residual.
        """
        residual_reduction = (
            np.abs(before.residuals).mean() - np.abs(after.residuals).mean()
        ) / np.abs(before.residuals).mean()
        
        quality_improvement = after.conservation_quality - before.conservation_quality
        cluster_reduction = before.n_clusters - after.n_clusters
        
        return {
            'residual_reduction': residual_reduction,
            'residual_reduction_pct': residual_reduction * 100,
            'quality_improvement': quality_improvement,
            'cluster_reduction': cluster_reduction,
            'validates_pac': residual_reduction > 0.05,  # 5% reduction threshold
            'interpretation': self._interpret_reduction(residual_reduction, cluster_reduction)
        }
    
    def _interpret_reduction(self, 
                            residual_reduction: float,
                            cluster_reduction: int) -> str:
        """Generate interpretation of residual reduction"""
        if residual_reduction > 0.2:
            if cluster_reduction > 0:
                return "Strong validation: significant residual reduction with cluster elimination"
            return "Strong validation: significant residual reduction"
        elif residual_reduction > 0.05:
            return "Moderate validation: measurable residual reduction"
        elif residual_reduction > 0:
            return "Weak validation: marginal improvement"
        else:
            return "No validation: residual did not decrease (feature may not be missing child)"


def estimate_missing_children_count(residuals: np.ndarray,
                                    method: str = 'silhouette') -> int:
    """
    Estimate the number of missing children from residual structure.
    
    Uses silhouette analysis to find optimal cluster count.
    Each cluster (beyond noise) represents a potential missing child.
    """
    X = residuals.reshape(-1, 1)
    
    if method == 'silhouette':
        # Try different cluster counts
        best_k = 0
        best_score = -1
        
        for k in range(2, min(10, len(residuals) // 10)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue
        
        # Subtract 1 (one cluster is "explained variance")
        return max(0, best_k - 1)
    
    else:
        # DBSCAN method
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return n_clusters
