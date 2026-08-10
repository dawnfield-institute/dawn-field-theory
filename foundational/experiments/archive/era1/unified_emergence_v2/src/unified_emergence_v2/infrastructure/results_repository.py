"""
Results repository implementation for the Unified Emergence Framework v2.
"""

import json
import os
import shutil
import csv
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from ..domain.models import SweepAnalysis, StatisticalSummary, ValidationConfig

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
            # Auto-detect results directory within the experiment folder
            current_path = Path(__file__).resolve()
            while current_path.parent != current_path:
                if current_path.name == 'unified_emergence_v2' and (current_path / 'src').exists():
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
            f.write(f"**Success:** {'[YES]' if results.success else '[NO]'}\n")
            f.write(f"**Phase 1 Ready:** {'[YES]' if results.metrics.phase1_readiness_score >= 0.8 else '[NOT YET]'}\n\n")
            
            # Core metrics
            f.write(f"## Core Metrics\n\n")
            f.write(f"| Metric | Score | Status |\n")
            f.write(f"|--------|-------|--------|\n")
            f.write(f"| Overall Score | {results.metrics.get_overall_score():.3f} | {'[GOOD]' if results.metrics.get_overall_score() >= 0.7 else '[OK]' if results.metrics.get_overall_score() >= 0.5 else '[NEEDS WORK]'} |\n")
            f.write(f"| SEC Classification | {results.metrics.sec_classification_accuracy:.3f} | {'[GOOD]' if results.metrics.sec_classification_accuracy >= 0.7 else '[OK]' if results.metrics.sec_classification_accuracy >= 0.5 else '[NEEDS WORK]'} |\n")
            f.write(f"| Pattern Assembly | {results.metrics.pattern_assembly_success_rate:.3f} | {'[GOOD]' if results.metrics.pattern_assembly_success_rate >= 0.8 else '[OK]' if results.metrics.pattern_assembly_success_rate >= 0.6 else '[NEEDS WORK]'} |\n")
            f.write(f"| Emergence Consistency | {results.metrics.emergence_consistency_score:.3f} | {'[GOOD]' if results.metrics.emergence_consistency_score >= 0.8 else '[OK]' if results.metrics.emergence_consistency_score >= 0.6 else '[NEEDS WORK]'} |\n")
            f.write(f"| Phase 1 Readiness | {results.metrics.phase1_readiness_score:.3f} | {'[GOOD]' if results.metrics.phase1_readiness_score >= 0.8 else '[OK]' if results.metrics.phase1_readiness_score >= 0.6 else '[NEEDS WORK]'} |\n\n")
            
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
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
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
        score = results.metrics.get_overall_score()
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
            'overall_score': metrics.get_overall_score(),
            'total_patterns_detected': metrics.total_patterns_extracted,
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
    
    def save_sweep_analysis(self, analysis: 'SweepAnalysis', config: 'ValidationConfig'):
        """Save parameter sweep analysis results with comprehensive visualizations."""
        from datetime import datetime
        import json
        import csv
        
        try:
            # Create sweep-specific session directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_dir = self.base_path / f"{timestamp}_sweep_{config.session_id}"
            session_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving sweep analysis to: {session_dir}")
            
            # Save main analysis JSON
            analysis_file = session_dir / 'sweep_analysis.json'
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self._serialize_sweep_analysis(analysis), f, indent=2)
            
            # Save detailed statistics CSV
            self._save_sweep_statistics_csv(analysis, session_dir)
            
            # Generate sweep visualizations
            self._generate_sweep_visualizations(analysis, session_dir)
            
            # Generate comprehensive sweep report
            self._generate_sweep_report(analysis, session_dir)
            
            # Save configuration
            config_file = session_dir / 'sweep_config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(analysis.sweep_parameters, f, indent=2)
            
            logger.info(f"Sweep analysis saved successfully to: {session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save sweep analysis: {e}")
            raise
    
    def _serialize_sweep_analysis(self, analysis: 'SweepAnalysis') -> dict:
        """Serialize sweep analysis to dictionary."""
        return {
            'sweep_parameters': analysis.sweep_parameters,
            'summary': {
                'total_runs': analysis.total_runs,
                'successful_runs': analysis.successful_runs,
                'phase1_success_rate': analysis.phase1_success_rate
            },
            'statistical_summaries': {
                'overall_score': self._serialize_statistical_summary(analysis.overall_score_stats),
                'sec_classification': self._serialize_statistical_summary(analysis.sec_classification_stats),
                'pattern_assembly': self._serialize_statistical_summary(analysis.pattern_assembly_stats),
                'emergence_consistency': self._serialize_statistical_summary(analysis.emergence_consistency_stats),
                'phase1_readiness': self._serialize_statistical_summary(analysis.phase1_readiness_stats),
                'execution_time': self._serialize_statistical_summary(analysis.execution_time_stats),
                'memory_usage': self._serialize_statistical_summary(analysis.memory_usage_stats),
                'total_patterns': self._serialize_statistical_summary(analysis.total_patterns_stats),
                'pattern_confidence': self._serialize_statistical_summary(analysis.pattern_confidence_stats),
                'emergence_strength': self._serialize_statistical_summary(analysis.emergence_strength_stats)
            },
            'domain_performance': {
                domain: self._serialize_statistical_summary(stats)
                for domain, stats in analysis.domain_performance.items()
            },
            'field_size_performance': {
                str(size): self._serialize_statistical_summary(stats)
                for size, stats in analysis.field_size_performance.items()
            },
            'parameter_correlations': analysis.parameter_correlations,
            'optimal_parameters': analysis.optimal_parameters,
            'convergence_analysis': analysis.convergence_analysis
        }
    
    def _serialize_statistical_summary(self, stats: 'StatisticalSummary') -> dict:
        """Serialize statistical summary to dictionary."""
        def convert_numpy_types(value):
            """Convert numpy types to Python native types."""
            import numpy as np
            if isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (list, tuple)):
                return [convert_numpy_types(item) for item in value]
            return value
        
        return {
            'mean': convert_numpy_types(stats.mean),
            'std_dev': convert_numpy_types(stats.std_dev),
            'min_value': convert_numpy_types(stats.min_value),
            'max_value': convert_numpy_types(stats.max_value),
            'median': convert_numpy_types(stats.median),
            'confidence_level': convert_numpy_types(stats.confidence_level),
            'confidence_interval': convert_numpy_types(list(stats.confidence_interval)),
            'skewness': convert_numpy_types(stats.skewness),
            'kurtosis': convert_numpy_types(stats.kurtosis),
            'is_normal': convert_numpy_types(stats.is_normal),
            'sample_size': convert_numpy_types(stats.sample_size),
            'outliers_count': convert_numpy_types(stats.outliers_count),
            'outlier_indices': convert_numpy_types(stats.outlier_indices)
        }
    
    def _save_sweep_statistics_csv(self, analysis: 'SweepAnalysis', session_dir: Path):
        """Save detailed statistics in CSV format."""
        stats_file = session_dir / 'sweep_statistics.csv'
        
        with open(stats_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Metric', 'Mean', 'Std_Dev', 'Min', 'Max', 'Median',
                'CI_Lower', 'CI_Upper', 'Skewness', 'Kurtosis', 'Is_Normal',
                'Sample_Size', 'Outliers_Count'
            ])
            
            # Write statistics for each metric
            metrics = {
                'Overall_Score': analysis.overall_score_stats,
                'SEC_Classification': analysis.sec_classification_stats,
                'Pattern_Assembly': analysis.pattern_assembly_stats,
                'Emergence_Consistency': analysis.emergence_consistency_stats,
                'Phase1_Readiness': analysis.phase1_readiness_stats,
                'Execution_Time': analysis.execution_time_stats,
                'Memory_Usage': analysis.memory_usage_stats,
                'Total_Patterns': analysis.total_patterns_stats,
                'Pattern_Confidence': analysis.pattern_confidence_stats,
                'Emergence_Strength': analysis.emergence_strength_stats
            }
            
            for metric_name, stats in metrics.items():
                writer.writerow([
                    metric_name, stats.mean, stats.std_dev, stats.min_value, stats.max_value,
                    stats.median, stats.confidence_interval[0], stats.confidence_interval[1],
                    stats.skewness, stats.kurtosis, stats.is_normal,
                    stats.sample_size, stats.outliers_count
                ])
    
    def _generate_sweep_visualizations(self, analysis: 'SweepAnalysis', session_dir: Path):
        """Generate comprehensive visualizations for parameter sweep."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create visualizations directory
            viz_dir = session_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Overall performance distribution
            self._plot_sweep_distributions(analysis, viz_dir)
            
            # 2. Parameter vs performance plots
            self._plot_parameter_performance(analysis, viz_dir)
            
            # 3. Statistical summary visualization
            self._plot_statistical_summaries(analysis, viz_dir)
            
            # 4. Convergence analysis (if available)
            if analysis.convergence_analysis:
                self._plot_convergence_analysis(analysis, viz_dir)
            
            # 5. Domain and field size comparison
            self._plot_domain_field_comparison(analysis, viz_dir)
            
            logger.info(f"Generated sweep visualizations in {viz_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping sweep visualizations")
        except Exception as e:
            logger.warning(f"Error generating sweep visualizations: {e}")
    
    def _plot_sweep_distributions(self, analysis: 'SweepAnalysis', viz_dir: Path):
        """Plot distribution of key metrics."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Parameter Sweep - Metric Distributions', fontsize=16)
        
        metrics = [
            ('Overall Score', analysis.overall_score_stats),
            ('SEC Classification', analysis.sec_classification_stats),
            ('Pattern Assembly', analysis.pattern_assembly_stats),
            ('Emergence Consistency', analysis.emergence_consistency_stats),
            ('Phase 1 Readiness', analysis.phase1_readiness_stats),
            ('Execution Time (s)', analysis.execution_time_stats)
        ]
        
        for i, (name, stats) in enumerate(metrics):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Create histogram with confidence interval
            x_range = np.linspace(stats.min_value, stats.max_value, 50)
            ax.axvline(stats.mean, color='red', linestyle='--', label=f'Mean: {stats.mean:.3f}')
            ax.axvline(stats.median, color='green', linestyle='--', label=f'Median: {stats.median:.3f}')
            ax.axvspan(stats.confidence_interval[0], stats.confidence_interval[1], 
                      alpha=0.3, color='blue', label=f'{stats.confidence_level:.0%} CI')
            
            ax.set_title(name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_performance(self, analysis: 'SweepAnalysis', viz_dir: Path):
        """Plot parameter vs performance relationships."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Parameter vs Performance Analysis', fontsize=16)
        
        # Field size vs performance
        field_sizes = list(analysis.field_size_performance.keys())
        field_means = [stats.mean for stats in analysis.field_size_performance.values()]
        field_stds = [stats.std_dev for stats in analysis.field_size_performance.values()]
        
        axes[0].errorbar(field_sizes, field_means, yerr=field_stds, marker='o', capsize=5)
        axes[0].set_xlabel('Field Size')
        axes[0].set_ylabel('Overall Score')
        axes[0].set_title('Field Size vs Performance')
        axes[0].grid(True, alpha=0.3)
        
        # Domain performance comparison
        domains = list(analysis.domain_performance.keys())
        domain_means = [stats.mean for stats in analysis.domain_performance.values()]
        domain_stds = [stats.std_dev for stats in analysis.domain_performance.values()]
        
        bars = axes[1].bar(domains, domain_means, yerr=domain_stds, capsize=5)
        axes[1].set_xlabel('Domain')
        axes[1].set_ylabel('Overall Score')
        axes[1].set_title('Domain Performance Comparison')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'parameter_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_summaries(self, analysis: 'SweepAnalysis', viz_dir: Path):
        """Plot statistical summary visualization."""
        import matplotlib.pyplot as plt
        
        # Create a comprehensive statistical summary plot
        metrics_data = {
            'Overall Score': analysis.overall_score_stats,
            'SEC Classification': analysis.sec_classification_stats,
            'Pattern Assembly': analysis.pattern_assembly_stats,
            'Emergence Consistency': analysis.emergence_consistency_stats,
            'Phase 1 Readiness': analysis.phase1_readiness_stats
        }
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_names = list(metrics_data.keys())
        means = [stats.mean for stats in metrics_data.values()]
        ci_lower = [stats.confidence_interval[0] for stats in metrics_data.values()]
        ci_upper = [stats.confidence_interval[1] for stats in metrics_data.values()]
        
        y_pos = range(len(metric_names))
        
        # Plot confidence intervals as horizontal bars
        for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
            ax.barh(i, upper - lower, left=lower, alpha=0.3, color='lightblue')
        
        # Plot means as points
        ax.scatter(means, y_pos, color='red', s=100, zorder=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Score')
        ax.set_title(f'Statistical Summary - {analysis.overall_score_stats.confidence_level:.0%} Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        # Add success rate annotation
        ax.text(0.02, 0.98, f'Phase 1 Success Rate: {analysis.phase1_success_rate:.1%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_convergence_analysis(self, analysis: 'SweepAnalysis', viz_dir: Path):
        """Plot convergence analysis if available."""
        import matplotlib.pyplot as plt
        
        convergence_data = analysis.convergence_analysis
        if not convergence_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Convergence Analysis', fontsize=16)
        axes = axes.flatten()
        
        plot_count = 0
        for config_key, data in convergence_data.items():
            if plot_count >= 4:
                break
            
            ax = axes[plot_count]
            
            scores = data['scores']
            running_means = data.get('running_means', [])
            convergence_iter = data.get('convergence_iteration')
            
            # Plot individual scores
            ax.plot(scores, 'o-', alpha=0.7, label='Individual Scores')
            
            # Plot running means
            if running_means:
                ax.plot(range(len(running_means)), running_means, 'r-', linewidth=2, label='Running Mean')
            
            # Mark convergence point
            if convergence_iter is not None:
                ax.axvline(convergence_iter, color='green', linestyle='--', 
                          label=f'Convergence at iteration {convergence_iter}')
            
            ax.set_title(f'{config_key}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_count += 1
        
        # Hide unused subplots
        for i in range(plot_count, 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_domain_field_comparison(self, analysis: 'SweepAnalysis', viz_dir: Path):
        """Plot detailed domain and field size comparison."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Detailed Performance Analysis', fontsize=16)
        
        # Domain performance with error bars and sample sizes
        domains = list(analysis.domain_performance.keys())
        domain_stats = list(analysis.domain_performance.values())
        
        means = [s.mean for s in domain_stats]
        stds = [s.std_dev for s in domain_stats]
        sample_sizes = [s.sample_size for s in domain_stats]
        
        bars = axes[0].bar(domains, means, yerr=stds, capsize=5)
        axes[0].set_ylabel('Overall Score')
        axes[0].set_title('Domain Performance')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add sample size annotations
        for bar, sample_size in zip(bars, sample_sizes):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={sample_size}', ha='center', va='bottom', fontsize=8)
        
        # Field size performance with trend line
        field_sizes = sorted(analysis.field_size_performance.keys())
        field_stats = [analysis.field_size_performance[size] for size in field_sizes]
        
        field_means = [s.mean for s in field_stats]
        field_stds = [s.std_dev for s in field_stats]
        
        axes[1].errorbar(field_sizes, field_means, yerr=field_stds, marker='o', capsize=5)
        
        # Add trend line if there are enough points
        if len(field_sizes) > 2:
            z = np.polyfit(field_sizes, field_means, 1)
            p = np.poly1d(z)
            axes[1].plot(field_sizes, p(field_sizes), "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
            axes[1].legend()
        
        axes[1].set_xlabel('Field Size')
        axes[1].set_ylabel('Overall Score')
        axes[1].set_title('Field Size vs Performance')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'domain_field_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_sweep_report(self, analysis: 'SweepAnalysis', session_dir: Path):
        """Generate comprehensive sweep analysis report."""
        report_file = session_dir / 'sweep_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# Parameter Sweep Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write(f"## Executive Summary\n\n")
            f.write(f"- **Total Runs:** {analysis.total_runs}\n")
            f.write(f"- **Successful Runs:** {analysis.successful_runs}\n")
            f.write(f"- **Phase 1 Success Rate:** {analysis.phase1_success_rate:.1%}\n")
            f.write(f"- **Best Overall Score:** {analysis.optimal_parameters.get('best_overall_score', 'N/A'):.3f}\n\n")
            
            # Statistical Summary
            f.write(f"## Statistical Summary\n\n")
            f.write(f"| Metric | Mean | Std Dev | Min | Max | Median | 95% CI |\n")
            f.write(f"|--------|------|---------|-----|-----|--------|--------|\n")
            
            metrics = [
                ('Overall Score', analysis.overall_score_stats),
                ('SEC Classification', analysis.sec_classification_stats),
                ('Pattern Assembly', analysis.pattern_assembly_stats),
                ('Emergence Consistency', analysis.emergence_consistency_stats),
                ('Phase 1 Readiness', analysis.phase1_readiness_stats)
            ]
            
            for name, stats in metrics:
                ci = f"[{stats.confidence_interval[0]:.3f}, {stats.confidence_interval[1]:.3f}]"
                f.write(f"| {name} | {stats.mean:.3f} | {stats.std_dev:.3f} | {stats.min_value:.3f} | {stats.max_value:.3f} | {stats.median:.3f} | {ci} |\n")
            
            # Domain Performance
            f.write(f"\n## Domain Performance\n\n")
            f.write(f"| Domain | Mean Score | Std Dev | Sample Size | Status |\n")
            f.write(f"|--------|------------|---------|-------------|--------|\n")
            
            for domain, stats in analysis.domain_performance.items():
                status = "[GOOD]" if stats.mean >= 0.7 else "[OK]" if stats.mean >= 0.5 else "[NEEDS WORK]"
                f.write(f"| {domain} | {stats.mean:.3f} | {stats.std_dev:.3f} | {stats.sample_size} | {status} |\n")
            
            # Field Size Analysis
            f.write(f"\n## Field Size Analysis\n\n")
            f.write(f"| Field Size | Mean Score | Std Dev | Sample Size | Status |\n")
            f.write(f"|------------|------------|---------|-------------|--------|\n")
            
            for field_size, stats in analysis.field_size_performance.items():
                status = "[GOOD]" if stats.mean >= 0.7 else "[OK]" if stats.mean >= 0.5 else "[NEEDS WORK]"
                f.write(f"| {field_size} | {stats.mean:.3f} | {stats.std_dev:.3f} | {stats.sample_size} | {status} |\n")
            
            # Optimal Parameters
            f.write(f"\n## Optimal Parameters\n\n")
            optimal = analysis.optimal_parameters
            f.write(f"- **Best Configuration:** Field Size {optimal.get('best_field_size')}, Domain {optimal.get('best_domain')}\n")
            f.write(f"- **Best Score:** {optimal.get('best_overall_score', 0):.3f}\n")
            f.write(f"- **Most Reliable Domain:** {optimal.get('most_reliable_domain')}\n")
            f.write(f"- **Most Reliable Field Size:** {optimal.get('most_reliable_field_size')}\n\n")
            
            # Parameter Correlations
            if analysis.parameter_correlations:
                f.write(f"## Parameter Correlations\n\n")
                for param, correlation in analysis.parameter_correlations.items():
                    f.write(f"- **{param}:** {correlation:.3f}\n")
                f.write(f"\n")
            
            # Recommendations
            f.write(f"## Recommendations\n\n")
            
            if analysis.phase1_success_rate >= 0.8:
                f.write(f"- [EXCELLENT] High Phase 1 success rate ({analysis.phase1_success_rate:.1%}). Framework is performing very well.\n")
            elif analysis.phase1_success_rate >= 0.5:
                f.write(f"- [GOOD] Moderate Phase 1 success rate ({analysis.phase1_success_rate:.1%}). Consider optimization.\n")
            else:
                f.write(f"- [NEEDS WORK] Low Phase 1 success rate ({analysis.phase1_success_rate:.1%}). Requires significant improvement.\n")
            
            # Best performing configuration recommendation
            best_domain = optimal.get('most_reliable_domain')
            best_field_size = optimal.get('most_reliable_field_size')
            if best_domain and best_field_size:
                f.write(f"- **Recommended Configuration:** Use field size {best_field_size} with domain {best_domain} for most reliable results.\n")
            
            # Convergence recommendations
            if analysis.convergence_analysis:
                converged_count = sum(1 for data in analysis.convergence_analysis.values() if data.get('converged', False))
                total_configs = len(analysis.convergence_analysis)
                f.write(f"- **Convergence:** {converged_count}/{total_configs} configurations showed convergence.\n")
        
        logger.info(f"Generated comprehensive sweep report: {report_file}")
