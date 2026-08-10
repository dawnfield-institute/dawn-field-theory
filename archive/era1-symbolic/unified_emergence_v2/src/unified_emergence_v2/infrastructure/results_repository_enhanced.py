"""
Results repository implementation for the Unified Emergence Framework v2.
"""

import json
import os
import shutil
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..domain.models import EmergenceResults

logger = logging.getLogger(__name__)


class ResultsRepositoryImpl:
    """
    Enhanced file-based implementation of results repository.
    
    Stores validation results in timestamped folders with comprehensive data,
    visualizations, and analysis artifacts.
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize results repository.
        
        Args:
            base_path: Base path for storing results (default: auto-detect)
        """
        if base_path is None:
            # Auto-detect results directory
            current_path = Path(__file__).resolve()
            while current_path.parent != current_path:
                if (current_path / 'theory').exists() or (current_path.name == 'dawn-field-theory'):
                    base_path = str(current_path / 'results')
                    break
                current_path = current_path.parent
            
            if base_path is None:
                base_path = str(Path.cwd() / 'results')
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib for headless operation
        plt.style.use('default')
        try:
            sns.set_palette("husl")
        except ImportError:
            pass  # Seaborn is optional
        
        logger.info(f"Results repository initialized at: {self.base_path}")
    
    def save_results(self, results: EmergenceResults) -> str:
        """
        Save validation results to timestamped folder with comprehensive artifacts.
        
        Args:
            results: Complete validation results to save
            
        Returns:
            Path where results were saved
        """
        try:
            # Create timestamped session directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_dir = self.base_path / f"{timestamp}_{results.session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving results to: {session_dir}")
            
            # Save main results
            self._save_json_results(results, session_dir)
            
            # Generate and save visualizations
            self._generate_visualizations(results, session_dir)
            
            # Save summary report
            self._generate_summary_report(results, session_dir)
            
            # Save raw data exports
            self._save_raw_data(results, session_dir)
            
            # Copy logs if available
            self._copy_logs(session_dir)
            
            logger.info(f"Results saved successfully to: {session_dir}")
            return str(session_dir)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def load_results(self, session_id: str) -> Optional[EmergenceResults]:
        """
        Load validation results by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Loaded validation results, or None if not found
        """
        # Look for directories containing the session_id
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir() and session_id in session_dir.name:
                results_file = session_dir / 'results.json'
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            data = json.load(f)
                        return self._deserialize_results(data)
                    except Exception as e:
                        logger.error(f"Failed to load results from {results_file}: {e}")
                        return None
        return None
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs.
        
        Returns:
            List of session IDs
        """
        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                # Extract session ID from directory name (format: timestamp_sessionid)
                parts = session_dir.name.split('_', 3)
                if len(parts) >= 4:  # timestamp has underscores too
                    session_id = '_'.join(parts[3:])
                    sessions.append(session_id)
                elif len(parts) >= 2:
                    session_id = '_'.join(parts[1:])
                    sessions.append(session_id)
        return sorted(sessions)
    
    def _save_json_results(self, results: EmergenceResults, session_dir: Path):
        """Save all JSON data files."""
        # Main results
        results_file = session_dir / 'results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self._serialize_results(results), f, indent=2)
        
        # Signatures separately for easier analysis
        signatures_file = session_dir / 'signatures.json'
        with open(signatures_file, 'w', encoding='utf-8') as f:
            json.dump([self._serialize_signature(sig) for sig in results.signatures], f, indent=2)
        
        # Correlation matrix
        correlation_file = session_dir / 'correlations.json'
        with open(correlation_file, 'w', encoding='utf-8') as f:
            json.dump(self._serialize_correlation_matrix(results.correlation_matrix), f, indent=2)
        
        # Metrics summary
        metrics_file = session_dir / 'metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self._serialize_metrics(results.metrics), f, indent=2)
    
    def _generate_visualizations(self, results: EmergenceResults, session_dir: Path):
        """Generate comprehensive visualizations."""
        vis_dir = session_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        try:
            # Skip visualization if matplotlib not available
            import matplotlib.pyplot as plt
            
            # 1. Metrics overview dashboard
            self._plot_metrics_dashboard(results, vis_dir / 'metrics_dashboard.png')
            
            # 2. Pattern distribution by domain
            self._plot_pattern_distribution(results, vis_dir / 'pattern_distribution.png')
            
            # 3. Correlation heatmap
            self._plot_correlation_matrix(results, vis_dir / 'correlation_heatmap.png')
            
            # 4. Emergence strength vs confidence scatter
            self._plot_emergence_analysis(results, vis_dir / 'emergence_analysis.png')
            
            logger.info(f"Generated visualizations in {vis_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping visualizations")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
    
    def _generate_summary_report(self, results: EmergenceResults, session_dir: Path):
        """Generate human-readable summary report."""
        report_file = session_dir / 'summary_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Unified Emergence Framework v2 - Validation Report\n\n")
            f.write(f"**Session ID:** {results.session_id}\n")
            f.write(f"**Timestamp:** {results.timestamp}\n")
            f.write(f"**Success:** {'✅ Yes' if results.success else '❌ No'}\n")
            f.write(f"**Phase 1 Ready:** {'🎉 YES' if results.metrics.phase1_readiness_score >= 0.8 else '⚠️ Not yet'}\n\n")
            
            # Core metrics
            f.write(f"## Core Metrics\n\n")
            f.write(f"| Metric | Score | Status |\n")
            f.write(f"|--------|-------|--------|\n")
            f.write(f"| Overall Score | {results.metrics.overall_score:.3f} | {'🟢' if results.metrics.overall_score >= 0.7 else '🟡' if results.metrics.overall_score >= 0.5 else '🔴'} |\n")
            f.write(f"| SEC Classification | {results.metrics.sec_classification_accuracy:.3f} | {'🟢' if results.metrics.sec_classification_accuracy >= 0.7 else '🟡' if results.metrics.sec_classification_accuracy >= 0.5 else '🔴'} |\n")
            f.write(f"| Pattern Assembly | {results.metrics.pattern_assembly_success_rate:.3f} | {'🟢' if results.metrics.pattern_assembly_success_rate >= 0.8 else '🟡' if results.metrics.pattern_assembly_success_rate >= 0.6 else '🔴'} |\n")
            f.write(f"| Emergence Consistency | {results.metrics.emergence_consistency_score:.3f} | {'🟢' if results.metrics.emergence_consistency_score >= 0.8 else '🟡' if results.metrics.emergence_consistency_score >= 0.6 else '🔴'} |\n")
            f.write(f"| Phase 1 Readiness | {results.metrics.phase1_readiness_score:.3f} | {'🟢' if results.metrics.phase1_readiness_score >= 0.8 else '🟡' if results.metrics.phase1_readiness_score >= 0.6 else '🔴'} |\n\n")
            
            # Pattern statistics
            f.write(f"## Pattern Analysis\n\n")
            f.write(f"- **Total Patterns:** {len(results.signatures)}\n")
            f.write(f"- **Average Confidence:** {results.metrics.average_pattern_confidence:.3f}\n")
            f.write(f"- **Average Emergence Strength:** {results.metrics.average_emergence_strength:.3f}\n")
            f.write(f"- **Pattern Diversity:** {results.metrics.pattern_diversity_score:.3f}\n\n")
            
            # Domain breakdown
            domain_patterns = {}
            for sig in results.signatures:
                domain_patterns[sig.domain] = domain_patterns.get(sig.domain, 0) + 1
            
            f.write(f"## Domain Breakdown\n\n")
            for domain, count in sorted(domain_patterns.items()):
                f.write(f"- **{domain.title()}:** {count} patterns\n")
            
            # Performance info
            f.write(f"\n## Performance\n\n")
            f.write(f"- **Execution Time:** {results.execution_time_seconds:.2f} seconds\n")
            f.write(f"- **Peak Memory:** {results.peak_memory_usage_mb:.1f} MB\n")
    
    def _save_raw_data(self, results: EmergenceResults, session_dir: Path):
        """Save raw data exports for further analysis."""
        data_dir = session_dir / 'raw_data'
        data_dir.mkdir(exist_ok=True)
        
        # Export signatures to CSV for easy analysis
        if results.signatures:
            import csv
            
            csv_file = data_dir / 'signatures.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                max_features = max(len(sig.features) for sig in results.signatures)
                header = ['domain', 'pattern_type', 'confidence', 'emergence_strength']
                header.extend([f'feature_{i}' for i in range(max_features)])
                header.extend(['extraction_timestamp', 'feature_hash'])
                writer.writerow(header)
                
                # Data rows
                for sig in results.signatures:
                    row = [sig.domain, sig.pattern_type, sig.confidence, sig.emergence_strength]
                    
                    # Pad features to max length
                    features = list(sig.features)
                    while len(features) < max_features:
                        features.append(0.0)
                    row.extend(features)
                    
                    row.extend([sig.extraction_timestamp, sig.feature_hash])
                    writer.writerow(row)
        
        # Save raw domain results if available
        if results.raw_domain_results:
            raw_file = data_dir / 'raw_domain_results.json'
            with open(raw_file, 'w', encoding='utf-8') as f:
                json.dump(results.raw_domain_results, f, indent=2)
    
    def _copy_logs(self, session_dir: Path):
        """Copy relevant log files to the session directory."""
        try:
            logs_dir = session_dir / 'logs'
            logs_dir.mkdir(exist_ok=True)
            
            # Look for framework log file
            possible_log_paths = [
                Path('unified_emergence_v2.log'),
                Path('.') / 'unified_emergence_v2.log',
                Path('..') / 'unified_emergence_v2.log'
            ]
            
            for log_path in possible_log_paths:
                if log_path.exists():
                    shutil.copy2(log_path, logs_dir / 'framework.log')
                    break
                    
        except Exception as e:
            logger.debug(f"Could not copy logs: {e}")
    
    def _plot_metrics_dashboard(self, results: EmergenceResults, output_path: Path):
        """Create metrics overview dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Unified Emergence Framework v2 - Metrics Dashboard', fontsize=16)
        
        # Core metrics bar chart (top-left)
        metrics = [
            results.metrics.sec_classification_accuracy,
            results.metrics.pattern_assembly_success_rate,
            results.metrics.emergence_consistency_score,
            results.metrics.phase1_readiness_score
        ]
        labels = ['SEC Class.', 'Pattern Asm.', 'Emergence', 'Phase 1']
        
        bars = axes[0, 0].bar(labels, metrics, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Core Metrics')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # Pattern count by domain (top-right)
        domain_counts = {}
        for sig in results.signatures:
            domain_counts[sig.domain] = domain_counts.get(sig.domain, 0) + 1
        
        if domain_counts:
            axes[0, 1].bar(domain_counts.keys(), domain_counts.values(), alpha=0.7)
            axes[0, 1].set_title('Patterns by Domain')
            axes[0, 1].set_ylabel('Pattern Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confidence vs Emergence scatter (bottom-left)
        if results.signatures:
            confidences = [sig.confidence for sig in results.signatures]
            emergence_strengths = [sig.emergence_strength for sig in results.signatures]
            
            scatter = axes[1, 0].scatter(confidences, emergence_strengths, alpha=0.7)
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_ylabel('Emergence Strength')
            axes[1, 0].set_title('Pattern Quality Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Overall score display (bottom-right)
        score = results.metrics.overall_score
        color = 'green' if score >= 0.7 else 'orange' if score >= 0.5 else 'red'
        
        axes[1, 1].text(0.5, 0.5, f'{score:.3f}', fontsize=48, ha='center', va='center',
                       color=color, weight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Overall Score')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_pattern_distribution(self, results: EmergenceResults, output_path: Path):
        """Plot pattern distribution analysis."""
        if not results.signatures:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Pattern Distribution Analysis', fontsize=16)
        
        # Confidence histogram
        confidences = [sig.confidence for sig in results.signatures]
        axes[0, 0].hist(confidences, bins=10, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Confidence Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Emergence strength histogram
        emergence = [sig.emergence_strength for sig in results.signatures]
        axes[0, 1].hist(emergence, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Emergence Strength')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Emergence Strength Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Pattern types
        pattern_types = {}
        for sig in results.signatures:
            pattern_types[sig.pattern_type] = pattern_types.get(sig.pattern_type, 0) + 1
        
        if pattern_types:
            axes[1, 0].pie(pattern_types.values(), labels=pattern_types.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Pattern Types')
        
        # Domain distribution
        domain_counts = {}
        for sig in results.signatures:
            domain_counts[sig.domain] = domain_counts.get(sig.domain, 0) + 1
        
        if domain_counts:
            axes[1, 1].pie(domain_counts.values(), labels=domain_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Domain Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrix(self, results: EmergenceResults, output_path: Path):
        """Plot correlation matrix heatmap."""
        if not results.correlation_matrix.domains:
            return
        
        correlation_data = np.array(results.correlation_matrix.correlation_values)
        
        plt.figure(figsize=(10, 8))
        
        # Use seaborn if available, otherwise basic matplotlib
        try:
            import seaborn as sns
            sns.heatmap(correlation_data, 
                       xticklabels=results.correlation_matrix.domains,
                       yticklabels=results.correlation_matrix.domains,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.3f')
        except ImportError:
            # Fallback to matplotlib
            im = plt.imshow(correlation_data, cmap='RdBu_r', aspect='auto')
            plt.colorbar(im)
            plt.xticks(range(len(results.correlation_matrix.domains)), 
                      results.correlation_matrix.domains, rotation=45)
            plt.yticks(range(len(results.correlation_matrix.domains)), 
                      results.correlation_matrix.domains)
            
            # Add text annotations
            for i in range(len(correlation_data)):
                for j in range(len(correlation_data[0])):
                    plt.text(j, i, f'{correlation_data[i][j]:.3f}', 
                           ha='center', va='center')
        
        plt.title('Cross-Domain Correlation Matrix')
        plt.xlabel('Domains')
        plt.ylabel('Domains')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_emergence_analysis(self, results: EmergenceResults, output_path: Path):
        """Plot emergence strength vs confidence analysis."""
        if not results.signatures:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        confidences = [sig.confidence for sig in results.signatures]
        emergence = [sig.emergence_strength for sig in results.signatures]
        domains = [sig.domain for sig in results.signatures]
        
        # Scatter plot with domain coloring (left)
        unique_domains = list(set(domains))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            domain_conf = [conf for conf, dom in zip(confidences, domains) if dom == domain]
            domain_emerg = [emerg for emerg, dom in zip(emergence, domains) if dom == domain]
            axes[0].scatter(domain_conf, domain_emerg, label=domain, alpha=0.7, c=[colors[i]])
        
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Emergence Strength')
        axes[0].set_title('Emergence vs Confidence by Domain')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Quality quadrants (right)
        axes[1].scatter(confidences, emergence, alpha=0.7)
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Emergence Strength')
        axes[1].set_title('Pattern Quality Quadrants')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Serialization methods (keeping existing ones)
    def _serialize_results(self, results: EmergenceResults) -> dict:
        """Serialize EmergenceResults to dictionary."""
        return {
            'session_id': results.session_id,
            'timestamp': results.timestamp,
            'configuration': self._serialize_config(results.configuration),
            'signatures': [self._serialize_signature(sig) for sig in results.signatures],
            'metrics': self._serialize_metrics(results.metrics),
            'correlation_matrix': self._serialize_correlation_matrix(results.correlation_matrix),
            'raw_domain_results': results.raw_domain_results,
            'processing_log': results.processing_log,
            'success': results.success,
            'error_messages': results.error_messages,
            'warnings': results.warnings,
            'execution_time_seconds': results.execution_time_seconds,
            'peak_memory_usage_mb': results.peak_memory_usage_mb
        }
    
    def _serialize_signature(self, signature) -> dict:
        """Serialize EmergenceSignature to dictionary."""
        return {
            'domain': signature.domain,
            'pattern_type': signature.pattern_type,
            'features': signature.features,
            'confidence': signature.confidence,
            'emergence_strength': signature.emergence_strength,
            'metadata': signature.metadata,
            'feature_hash': signature.feature_hash,
            'extraction_timestamp': signature.extraction_timestamp
        }
    
    def _serialize_correlation_matrix(self, correlation_matrix) -> dict:
        """Serialize CorrelationMatrix to dictionary."""
        return {
            'domains': correlation_matrix.domains,
            'correlation_values': correlation_matrix.correlation_values,
            'mean_correlation': correlation_matrix.mean_correlation,
            'correlation_consistency': correlation_matrix.correlation_consistency
        }
    
    def _serialize_metrics(self, metrics) -> dict:
        """Serialize ValidationMetrics to dictionary."""
        return {
            'sec_classification_accuracy': metrics.sec_classification_accuracy,
            'pattern_assembly_success_rate': metrics.pattern_assembly_success_rate,
            'emergence_consistency_score': metrics.emergence_consistency_score,
            'phase1_readiness_score': metrics.phase1_readiness_score,
            'overall_score': metrics.overall_score,
            'total_patterns_detected': metrics.total_patterns_detected,
            'patterns_per_domain': metrics.patterns_per_domain,
            'cross_domain_correlations': metrics.cross_domain_correlations,
            'correlation_consistency': metrics.correlation_consistency,
            'processing_time_seconds': metrics.processing_time_seconds,
            'memory_usage_mb': metrics.memory_usage_mb,
            'average_pattern_confidence': metrics.average_pattern_confidence,
            'average_emergence_strength': metrics.average_emergence_strength,
            'pattern_diversity_score': metrics.pattern_diversity_score
        }
    
    def _serialize_config(self, config) -> dict:
        """Serialize ValidationConfig to dictionary."""
        return {
            'session_id': config.session_id,
            'domains': config.domains,
            'field_sizes': config.field_sizes,
            'runs_per_domain': config.runs_per_domain,
            'parallel_execution': config.parallel_execution,
            'max_workers': config.max_workers,
            'timeout_seconds': config.timeout_seconds,
            'output_directory': config.output_directory,
            'save_intermediate_results': config.save_intermediate_results,
            'save_raw_domain_results': config.save_raw_domain_results,
            'sec_classification_threshold': config.sec_classification_threshold,
            'pattern_assembly_threshold': config.pattern_assembly_threshold,
            'emergence_consistency_threshold': config.emergence_consistency_threshold,
            'phase1_readiness_threshold': config.phase1_readiness_threshold,
            'min_pattern_confidence': config.min_pattern_confidence,
            'min_emergence_strength': config.min_emergence_strength,
            'max_patterns_per_domain': config.max_patterns_per_domain,
            'correlation_method': config.correlation_method,
            'min_correlation_significance': config.min_correlation_significance
        }
    
    def _deserialize_results(self, data: dict):
        """Deserialize dictionary to EmergenceResults (basic implementation)."""
        # This would need full implementation for loading
        # For now, return None as we're focusing on saving
        return None
