"""
Hypothesis Generator
=====================

Uses PAC residual patterns to hypothesize missing children
in informational hierarchies.

Modes:
- Known unknowns: Residual correlates with X → measure X more granularly
- Unknown unknowns: Residual clusters → latent factor exists
- Unexplored: Low convergence zones → potential structure to discover

Key Reference: ../arithmetic/PACEngine/ and ../arithmetic/infodynamics_arithmetic_v1.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Hypothesis:
    """A hypothesis about missing children"""
    hypothesis_type: str  # 'known_unknown', 'unknown_unknown', 'unexplored'
    description: str
    action: str
    confidence: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    
    def __repr__(self):
        return f"[{self.hypothesis_type}] {self.description} (conf={self.confidence:.2f})"


@dataclass 
class DiscoveryReport:
    """Complete report of discovery opportunities"""
    hypotheses: List[Hypothesis]
    residual_summary: Dict[str, float]
    convergence_summary: Dict[str, float]
    recommended_acquisitions: List[str]
    pac_conservation_quality: float
    estimated_missing_children: int
    
    @property
    def top_hypotheses(self) -> List[Hypothesis]:
        """Return hypotheses sorted by priority and confidence"""
        return sorted(
            self.hypotheses, 
            key=lambda h: (h.priority, h.confidence),
            reverse=True
        )[:5]


class HypothesisGenerator:
    """
    Generate hypotheses about missing children from PAC residual patterns.
    
    This is the discovery engine: it analyzes residual structure to
    suggest what data to acquire next to improve PAC conservation.
    """
    
    def __init__(self,
                 correlation_threshold: float = 0.3,
                 cluster_threshold: float = 0.3,
                 convergence_threshold: float = 0.05):
        """
        Args:
            correlation_threshold: Min correlation to suggest decomposition
            cluster_threshold: Min systematic fraction to suggest latent factor
            convergence_threshold: Below this = exploration opportunity
        """
        self.correlation_threshold = correlation_threshold
        self.cluster_threshold = cluster_threshold
        self.convergence_threshold = convergence_threshold
        
    def generate_from_residuals(self,
                                residuals: np.ndarray,
                                known_spaces: Dict[str, np.ndarray],
                                space_names: List[str]) -> List[Hypothesis]:
        """
        Generate hypotheses from residual patterns.
        
        Args:
            residuals: PAC residuals (observed - predicted)
            known_spaces: Dict of known feature space values
            space_names: Names of the feature spaces
            
        Returns:
            List of hypotheses about missing children
        """
        hypotheses = []
        
        # 1. Check correlations with known spaces (known unknowns)
        for space_name, features in known_spaces.items():
            corr_result = self._check_correlation(residuals, features, space_name)
            if corr_result:
                hypotheses.append(corr_result)
        
        # 2. Cluster residuals (unknown unknowns)
        cluster_result = self._analyze_clustering(residuals)
        if cluster_result:
            hypotheses.append(cluster_result)
        
        # 3. Check residual distribution (structural hints)
        dist_result = self._analyze_distribution(residuals)
        if dist_result:
            hypotheses.append(dist_result)
        
        # Sort by confidence
        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        return hypotheses
    
    def generate_from_convergence(self,
                                  convergence_df: pd.DataFrame,
                                  space_names: List[str]) -> List[Hypothesis]:
        """
        Generate hypotheses from convergence landscape.
        
        Low convergence zones = unexplored territory = discovery opportunity
        """
        hypotheses = []
        
        # Find low-convergence pairs
        low_conv = convergence_df[
            convergence_df['convergence'] < self.convergence_threshold
        ]
        
        if len(low_conv) > 0:
            # Group by source space
            for source in low_conv['source_space'].unique():
                targets = low_conv[
                    low_conv['source_space'] == source
                ]['target_space'].tolist()
                
                hypotheses.append(Hypothesis(
                    hypothesis_type='unexplored',
                    description=f"Space '{source}' has low convergence with {len(targets)} target spaces",
                    action=f"Either unrelated OR missing bridging feature between {source} and targets",
                    confidence=0.4,  # Moderate - could be no structure or missing bridge
                    evidence={
                        'source_space': source,
                        'low_convergence_targets': targets,
                        'mean_convergence': low_conv[
                            low_conv['source_space'] == source
                        ]['convergence'].mean()
                    },
                    priority=1
                ))
        
        # Find asymmetric convergence (potential causal direction)
        for _, row in convergence_df.iterrows():
            reverse = convergence_df[
                (convergence_df['source_space'] == row['target_space']) &
                (convergence_df['target_space'] == row['source_space'])
            ]
            if len(reverse) > 0:
                asymmetry = abs(row['convergence'] - reverse.iloc[0]['convergence'])
                if asymmetry > 0.1:
                    hypotheses.append(Hypothesis(
                        hypothesis_type='causal_hint',
                        description=f"Asymmetric convergence between {row['source_space']} and {row['target_space']}",
                        action="Asymmetry suggests causal direction - investigate temporal or mechanistic ordering",
                        confidence=asymmetry,
                        evidence={
                            'space_a': row['source_space'],
                            'space_b': row['target_space'],
                            'conv_a_to_b': row['convergence'],
                            'conv_b_to_a': reverse.iloc[0]['convergence'],
                            'asymmetry': asymmetry
                        },
                        priority=2
                    ))
        
        return hypotheses
    
    def _check_correlation(self,
                           residuals: np.ndarray,
                           features: np.ndarray,
                           space_name: str) -> Optional[Hypothesis]:
        """Check if residuals correlate with a known space"""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        max_corr = 0
        max_feat = 0
        max_p = 1.0
        
        for i in range(features.shape[1]):
            try:
                corr, p_value = stats.pearsonr(residuals.flatten(), features[:, i])
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    max_feat = i
                    max_p = p_value
            except:
                continue
        
        if abs(max_corr) > self.correlation_threshold and max_p < 0.05:
            return Hypothesis(
                hypothesis_type='known_unknown',
                description=f"Residual correlates with {space_name} (r={max_corr:.3f}, p={max_p:.4f})",
                action=f"Decompose {space_name} into finer primitives - it contains unexploited structure",
                confidence=abs(max_corr),
                evidence={
                    'space_name': space_name,
                    'correlation': max_corr,
                    'p_value': max_p,
                    'feature_index': max_feat
                },
                priority=3
            )
        
        return None
    
    def _analyze_clustering(self, residuals: np.ndarray) -> Optional[Hypothesis]:
        """Analyze residual clustering to detect latent factors"""
        from sklearn.cluster import DBSCAN
        
        X = residuals.reshape(-1, 1)
        
        # DBSCAN clustering
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        labels = clusterer.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        systematic_fraction = (labels != -1).sum() / len(labels)
        
        if n_clusters > 1 and systematic_fraction > self.cluster_threshold:
            return Hypothesis(
                hypothesis_type='unknown_unknown',
                description=f"Residuals form {n_clusters} distinct clusters ({systematic_fraction:.1%} systematic)",
                action=f"Hypothesize {n_clusters} latent factors - investigate what distinguishes cluster members",
                confidence=systematic_fraction,
                evidence={
                    'n_clusters': n_clusters,
                    'systematic_fraction': systematic_fraction,
                    'cluster_labels': labels.tolist(),
                    'cluster_sizes': [int((labels == i).sum()) for i in range(n_clusters)]
                },
                priority=4
            )
        
        return None
    
    def _analyze_distribution(self, residuals: np.ndarray) -> Optional[Hypothesis]:
        """Analyze residual distribution for structural hints"""
        # Check for bimodality
        from scipy.stats import normaltest, skew, kurtosis
        
        try:
            stat, p_value = normaltest(residuals.flatten())
            residual_skew = skew(residuals.flatten())
            residual_kurtosis = kurtosis(residuals.flatten())
        except:
            return None
        
        # Non-normal distribution suggests missing structure
        if p_value < 0.01:
            if abs(residual_skew) > 1:
                return Hypothesis(
                    hypothesis_type='structural_hint',
                    description=f"Residuals are highly skewed (skew={residual_skew:.2f})",
                    action="Asymmetric residuals suggest a threshold or ceiling effect - investigate boundary conditions",
                    confidence=min(abs(residual_skew) / 2, 0.9),
                    evidence={
                        'normality_p': p_value,
                        'skew': residual_skew,
                        'kurtosis': residual_kurtosis
                    },
                    priority=2
                )
            elif residual_kurtosis > 3:
                return Hypothesis(
                    hypothesis_type='structural_hint',
                    description=f"Residuals have heavy tails (kurtosis={residual_kurtosis:.2f})",
                    action="Heavy tails suggest outlier subpopulations - investigate extreme cases",
                    confidence=min(residual_kurtosis / 10, 0.9),
                    evidence={
                        'normality_p': p_value,
                        'skew': residual_skew,
                        'kurtosis': residual_kurtosis
                    },
                    priority=2
                )
        
        return None
    
    def generate_report(self,
                        residual_analysis: Any,  # From PACResidualTracker
                        convergence_df: pd.DataFrame,
                        known_spaces: Dict[str, np.ndarray]) -> DiscoveryReport:
        """
        Generate comprehensive discovery report.
        
        Combines residual analysis and convergence landscape to
        produce actionable hypotheses about missing children.
        """
        all_hypotheses = []
        
        # Hypotheses from residuals
        if hasattr(residual_analysis, 'residuals'):
            residual_hypotheses = self.generate_from_residuals(
                residual_analysis.residuals,
                known_spaces,
                list(known_spaces.keys())
            )
            all_hypotheses.extend(residual_hypotheses)
        
        # Hypotheses from convergence
        convergence_hypotheses = self.generate_from_convergence(
            convergence_df,
            list(known_spaces.keys())
        )
        all_hypotheses.extend(convergence_hypotheses)
        
        # Sort and deduplicate
        all_hypotheses.sort(key=lambda h: (h.priority, h.confidence), reverse=True)
        
        # Generate acquisition recommendations
        recommendations = []
        for h in all_hypotheses[:5]:
            if h.hypothesis_type == 'known_unknown':
                recommendations.append(f"Decompose {h.evidence.get('space_name', 'unknown')} into finer features")
            elif h.hypothesis_type == 'unknown_unknown':
                recommendations.append(f"Investigate distinguishing factors for {h.evidence.get('n_clusters', '?')} residual clusters")
            elif h.hypothesis_type == 'unexplored':
                recommendations.append(f"Explore bridging features for {h.evidence.get('source_space', 'unknown')}")
        
        return DiscoveryReport(
            hypotheses=all_hypotheses,
            residual_summary={
                'mean': float(residual_analysis.mean_residual) if hasattr(residual_analysis, 'mean_residual') else 0,
                'std': float(residual_analysis.std_residual) if hasattr(residual_analysis, 'std_residual') else 0,
                'conservation_quality': float(residual_analysis.conservation_quality) if hasattr(residual_analysis, 'conservation_quality') else 0
            },
            convergence_summary={
                'mean': float(convergence_df['convergence'].mean()),
                'max': float(convergence_df['convergence'].max()),
                'n_high': int((convergence_df['convergence'] > self.convergence_threshold).sum())
            },
            recommended_acquisitions=recommendations,
            pac_conservation_quality=float(residual_analysis.conservation_quality) if hasattr(residual_analysis, 'conservation_quality') else 0,
            estimated_missing_children=int(residual_analysis.n_clusters) if hasattr(residual_analysis, 'n_clusters') else 0
        )
