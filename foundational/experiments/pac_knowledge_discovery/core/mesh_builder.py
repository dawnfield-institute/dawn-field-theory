"""
Model Mesh Builder
===================

Multi-architecture ensemble training for high-convergence feature space pairs.
Follows MED bounded complexity: ≤10 architectures per ensemble.

Key Reference: ../arithmetic/macro_emergence_dynamics/README.md
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import pickle
from pathlib import Path
from datetime import datetime
import json

from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Optional imports (may not be available)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model"""
    architecture: str
    r2_train: float
    r2_test: float
    mse_test: float
    cv_mean: float
    cv_std: float
    weight: float  # Ensemble weight based on performance
    
    @property
    def is_useful(self) -> bool:
        """Model contributes positively to ensemble"""
        return self.r2_test > 0


@dataclass
class SpacePairModel:
    """Ensemble model for a source→target space pair"""
    source_space: str
    target_space: str
    convergence: float
    models: Dict[str, Any] = field(default_factory=dict)
    scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    metrics: Dict[str, ModelMetrics] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    
    @property
    def ensemble_r2(self) -> float:
        """Weighted mean R² across architectures"""
        if not self.metrics:
            return 0.0
        weights = [m.weight for m in self.metrics.values()]
        r2s = [m.r2_test for m in self.metrics.values()]
        if sum(weights) == 0:
            return np.mean(r2s)
        return np.average(r2s, weights=weights)
    
    @property
    def n_models(self) -> int:
        return len(self.models)


class MeshBuilder:
    """
    Build model mesh for high-convergence feature space pairs.
    
    MED Bounded Complexity:
    - Maximum 10 architectures per ensemble
    - Balance operator Ξ ≈ 1.0571 prevents explosion
    - Universal bounds: depth(S) ≤ 2, nodes(S) ≤ 10
    """
    
    # MED-bounded architecture set (≤10)
    ARCHITECTURES = {}
    
    def __init__(self,
                 convergence_threshold: float = 0.05,
                 test_size: float = 0.2,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Args:
            convergence_threshold: Only train models for pairs above this
            test_size: Fraction for test set
            cv_folds: Cross-validation folds
            random_state: Random seed
            n_jobs: Parallel jobs (-1 for all cores)
        """
        self.convergence_threshold = convergence_threshold
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Build architecture dictionary
        self._init_architectures()
        
        # Trained models
        self.mesh: Dict[Tuple[str, str], SpacePairModel] = {}
        self.training_log: List[Dict] = []
        
    def _init_architectures(self):
        """Initialize available architectures (MED bounded: ≤10)"""
        self.ARCHITECTURES = {
            # Tree-based (4)
            'rf_shallow': lambda: RandomForestRegressor(
                n_estimators=50, max_depth=10, 
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
            'rf_deep': lambda: RandomForestRegressor(
                n_estimators=100, max_depth=20,
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
            'gbm': lambda: GradientBoostingRegressor(
                n_estimators=100, max_depth=6,
                random_state=self.random_state
            ),
            'extra_trees': lambda: ExtraTreesRegressor(
                n_estimators=100, max_depth=15,
                random_state=self.random_state, n_jobs=self.n_jobs
            ),
            
            # Linear (3)
            'ridge': lambda: Ridge(alpha=1.0),
            'lasso': lambda: Lasso(alpha=0.1, max_iter=2000),
            'elastic': lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000),
            
            # Neural (2)
            'mlp_small': lambda: MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500, random_state=self.random_state
            ),
            'mlp_deep': lambda: MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500, random_state=self.random_state
            ),
            
            # Kernel (1)
            'svr': lambda: SVR(kernel='rbf', C=1.0),
        }
        
        # Add optional architectures if available
        if HAS_XGB:
            self.ARCHITECTURES['xgb'] = lambda: XGBRegressor(
                n_estimators=100, max_depth=6,
                random_state=self.random_state, n_jobs=self.n_jobs,
                verbosity=0
            )
        
        if HAS_LGBM:
            self.ARCHITECTURES['lgbm'] = lambda: LGBMRegressor(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=self.n_jobs,
                verbose=-1
            )
        
        if HAS_CATBOOST:
            self.ARCHITECTURES['catboost'] = lambda: CatBoostRegressor(
                iterations=100, depth=6,
                random_state=self.random_state,
                verbose=False
            )
        
        # MED bound check
        if len(self.ARCHITECTURES) > 10:
            # Keep only top 10 (prioritize tree-based and boosting)
            priority = ['xgb', 'lgbm', 'rf_deep', 'gbm', 'extra_trees', 
                       'rf_shallow', 'elastic', 'ridge', 'mlp_small', 'svr']
            self.ARCHITECTURES = {
                k: self.ARCHITECTURES[k] 
                for k in priority[:10] 
                if k in self.ARCHITECTURES
            }
    
    def build_for_pair(self,
                       X_source: np.ndarray,
                       y_target: np.ndarray,
                       source_name: str,
                       target_name: str,
                       convergence: float,
                       feature_names: Optional[List[str]] = None,
                       target_names: Optional[List[str]] = None) -> SpacePairModel:
        """
        Train ensemble for a single source→target pair.
        """
        # Check convergence threshold
        if convergence < self.convergence_threshold:
            return SpacePairModel(
                source_space=source_name,
                target_space=target_name,
                convergence=convergence
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_source, y_target,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pair_model = SpacePairModel(
            source_space=source_name,
            target_space=target_name,
            convergence=convergence,
            feature_names=feature_names or [],
            target_names=target_names or []
        )
        pair_model.scalers['source'] = scaler
        
        # Train each architecture
        for arch_name, arch_factory in self.ARCHITECTURES.items():
            try:
                model = arch_factory()
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    arch_factory(), X_train_scaled, y_train,
                    cv=self.cv_folds, scoring='r2'
                )
                
                # Weight = max(0, 1 + R²) - gives more weight to better models
                weight = max(0, 1 + r2_test)
                
                metrics = ModelMetrics(
                    architecture=arch_name,
                    r2_train=r2_train,
                    r2_test=r2_test,
                    mse_test=mse_test,
                    cv_mean=cv_scores.mean(),
                    cv_std=cv_scores.std(),
                    weight=weight
                )
                
                pair_model.models[arch_name] = model
                pair_model.metrics[arch_name] = metrics
                
            except Exception as e:
                self.training_log.append({
                    'source': source_name,
                    'target': target_name,
                    'architecture': arch_name,
                    'error': str(e)
                })
        
        # Store in mesh
        self.mesh[(source_name, target_name)] = pair_model
        
        return pair_model
    
    def build_from_convergence(self,
                               data: pd.DataFrame,
                               feature_spaces: Dict[str, List[str]],
                               target_spaces: Dict[str, List[str]],
                               convergence_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build full mesh from convergence analysis results.
        
        Args:
            data: Full dataset
            feature_spaces: Source feature space definitions
            target_spaces: Target feature space definitions  
            convergence_df: Results from ConvergenceAnalyzer
            
        Returns:
            Summary statistics of built mesh
        """
        n_pairs_trained = 0
        n_models_trained = 0
        
        # Filter to high-convergence pairs
        high_conv = convergence_df[
            convergence_df['convergence'] >= self.convergence_threshold
        ]
        
        for _, row in high_conv.iterrows():
            source_name = row['source_space']
            target_name = row['target_space']
            
            if source_name not in feature_spaces:
                continue
            if target_name not in target_spaces:
                continue
                
            source_cols = feature_spaces[source_name]
            target_cols = target_spaces[target_name]
            
            X = data[source_cols].values
            
            # Train for each target column
            for target_col in target_cols:
                y = data[target_col].values
                
                pair_model = self.build_for_pair(
                    X, y,
                    source_name, f"{target_name}_{target_col}",
                    row['convergence'],
                    feature_names=source_cols,
                    target_names=[target_col]
                )
                
                if pair_model.n_models > 0:
                    n_pairs_trained += 1
                    n_models_trained += pair_model.n_models
        
        return {
            'n_pairs_trained': n_pairs_trained,
            'n_models_trained': n_models_trained,
            'n_architectures': len(self.ARCHITECTURES),
            'convergence_threshold': self.convergence_threshold,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict(self,
                X: np.ndarray,
                source_space: str,
                target_space: str,
                method: str = 'weighted') -> Tuple[np.ndarray, float]:
        """
        Make predictions using trained ensemble.
        
        Args:
            X: Input features
            source_space: Name of source feature space
            target_space: Name of target feature space
            method: 'weighted' (default), 'mean', or 'best'
            
        Returns:
            (predictions, confidence)
        """
        key = (source_space, target_space)
        if key not in self.mesh:
            raise KeyError(f"No model for {source_space} → {target_space}")
        
        pair_model = self.mesh[key]
        if pair_model.n_models == 0:
            raise ValueError(f"No trained models for {source_space} → {target_space}")
        
        # Scale input
        X_scaled = pair_model.scalers['source'].transform(X)
        
        # Collect predictions from all models
        predictions = []
        weights = []
        
        for arch_name, model in pair_model.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weights.append(pair_model.metrics[arch_name].weight)
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Combine predictions
        if method == 'weighted':
            if weights.sum() > 0:
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_pred = predictions.mean(axis=0)
        elif method == 'mean':
            ensemble_pred = predictions.mean(axis=0)
        elif method == 'best':
            best_idx = np.argmax(weights)
            ensemble_pred = predictions[best_idx]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Confidence based on model agreement
        pred_std = predictions.std(axis=0).mean()
        pred_mean = np.abs(ensemble_pred).mean()
        confidence = 1.0 / (1.0 + pred_std / max(pred_mean, 1e-6))
        
        return ensemble_pred, confidence
    
    def save(self, path: Path) -> None:
        """Save mesh to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'mesh': self.mesh,
                'architectures': list(self.ARCHITECTURES.keys()),
                'convergence_threshold': self.convergence_threshold,
                'training_log': self.training_log
            }, f)
    
    def load(self, path: Path) -> None:
        """Load mesh from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mesh = data['mesh']
            self.training_log = data.get('training_log', [])
            self.convergence_threshold = data.get('convergence_threshold', 0.05)
    
    def get_summary(self) -> Dict:
        """Get mesh summary statistics"""
        if not self.mesh:
            return {'n_pairs': 0, 'n_models': 0}
        
        r2_scores = []
        for pair_model in self.mesh.values():
            if pair_model.metrics:
                r2_scores.extend([m.r2_test for m in pair_model.metrics.values()])
        
        return {
            'n_pairs': len(self.mesh),
            'n_models': sum(pm.n_models for pm in self.mesh.values()),
            'n_architectures': len(self.ARCHITECTURES),
            'mean_r2': np.mean(r2_scores) if r2_scores else 0,
            'std_r2': np.std(r2_scores) if r2_scores else 0,
            'convergence_threshold': self.convergence_threshold
        }
