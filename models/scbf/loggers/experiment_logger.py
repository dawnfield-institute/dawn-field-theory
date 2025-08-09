"""
SCBF Experiment Logger
=====================

Structured logging for SCBF experiments with automatic metric tracking,
experiment registration, and results persistence.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib

class SCBFExperimentLogger:
    """Handles logging and tracking of SCBF experiments."""
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        
        # Create datetime-based directory structure
        current_time = datetime.now()
        datetime_str = current_time.strftime("%Y%m%d_%H%M%S")
        
        self.output_dir = Path(output_dir) / datetime_str
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Initialize log storage
        self.logs: List[Dict] = []
        self.experiment_metadata = {
            'name': experiment_name,
            'id': self.experiment_id,
            'start_time': current_time.isoformat(),
            'end_time': None,
            'total_steps': 0,
            'scbf_metrics_count': 0,
            'status': 'running'
        }
        
        # Create experiment directory
        self.experiment_dir = self.output_dir / f"{self.experiment_name}_{self.experiment_id}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"🧪 SCBF Experiment Started: {self.experiment_name}")
        print(f"📁 Logging to: {self.experiment_dir}")
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = str(time.time())
        hash_input = f"{self.experiment_name}_{timestamp}".encode()
        return hashlib.md5(hash_input).hexdigest()[:8]
    
    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log a single adaptation step with SCBF metrics."""
        # Add timestamp
        step_data['timestamp'] = datetime.now().isoformat()
        step_data['step_id'] = len(self.logs)
        
        # Track SCBF metrics
        if 'scbf' in step_data:
            self.experiment_metadata['scbf_metrics_count'] += len(step_data['scbf'])
        
        # Store log entry
        self.logs.append(step_data)
        self.experiment_metadata['total_steps'] = len(self.logs)
        
        # Periodic save (every 10 steps)
        if len(self.logs) % 10 == 0:
            self._save_checkpoint()
    
    def log_scbf_metrics(self, step: int, scbf_results: Dict[str, Any]) -> None:
        """Log SCBF metrics for a specific step."""
        log_entry = {
            'step': step,
            'scbf': scbf_results,
            'timestamp': datetime.now().isoformat()
        }
        self.log_step(log_entry)
    
    def finalize_experiment(self) -> str:
        """Finalize the experiment and save all results."""
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        self.experiment_metadata['status'] = 'completed'
        
        # Calculate experiment duration
        start_time = datetime.fromisoformat(self.experiment_metadata['start_time'])
        end_time = datetime.fromisoformat(self.experiment_metadata['end_time'])
        duration = (end_time - start_time).total_seconds()
        self.experiment_metadata['duration_seconds'] = duration
        
        # Save final results
        results_file = self._save_final_results()
        
        print(f"✅ Experiment Completed: {self.experiment_name}")
        print(f"📊 Total Steps: {self.experiment_metadata['total_steps']}")
        print(f"🔬 SCBF Metrics: {self.experiment_metadata['scbf_metrics_count']}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"💾 Results saved to: {results_file}")
        
        return str(results_file)
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint of current progress."""
        checkpoint_file = self.experiment_dir / "checkpoint.json"
        checkpoint_data = {
            'metadata': self.experiment_metadata,
            'logs': self.logs[-10:]  # Only save last 10 entries in checkpoint
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _save_final_results(self) -> Path:
        """Save the complete experiment results."""
        results_file = self.experiment_dir / "experiment_results.json"
        
        final_data = {
            'metadata': self.experiment_metadata,
            'logs': self.logs,
            'summary': self._generate_summary()
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        # Also save a human-readable summary
        self._save_summary_report()
        
        return results_file
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary statistics."""
        scbf_logs = [log for log in self.logs if 'scbf' in log]
        
        if not scbf_logs:
            return {'message': 'No SCBF metrics found'}
        
        summary = {
            'total_steps': len(self.logs),
            'scbf_analyzed_steps': len(scbf_logs),
            'analysis_coverage': len(scbf_logs) / len(self.logs) * 100,
            'metrics_breakdown': {}
        }
        
        # Analyze metric types
        entropy_count = sum(1 for log in scbf_logs if 'entropy_collapse' in log['scbf'])
        ancestry_count = sum(1 for log in scbf_logs if 'ancestry' in log['scbf'])
        attractor_count = sum(1 for log in scbf_logs if 'attractors' in log['scbf'])
        lineage_count = sum(1 for log in scbf_logs if 'lineage' in log['scbf'])
        
        # Calculate actual success rate: percentage of SCBF analyses that produced meaningful metrics
        successful_analyses = sum(1 for log in scbf_logs 
                                if any(metric in log['scbf'] for metric in 
                                      ['entropy_collapse', 'ancestry', 'attractors', 'lineage']))
        scbf_success_rate = (successful_analyses / len(scbf_logs) * 100) if scbf_logs else 0
        
        summary['scbf_success_rate'] = scbf_success_rate
        summary['successful_analyses'] = successful_analyses
        
        summary['metrics_breakdown'] = {
            'entropy_collapse': entropy_count,
            'activation_ancestry': ancestry_count,
            'semantic_attractors': attractor_count,
            'bifractal_lineage': lineage_count
        }
        
        return summary
    
    def _save_summary_report(self) -> None:
        """Save a human-readable summary report."""
        report_file = self.experiment_dir / "experiment_summary.txt"
        
        summary = self._generate_summary()
        
        with open(report_file, 'w') as f:
            f.write(f"SCBF Experiment Summary Report\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Experiment: {self.experiment_metadata['name']}\n")
            f.write(f"ID: {self.experiment_metadata['id']}\n")
            f.write(f"Duration: {self.experiment_metadata.get('duration_seconds', 0):.1f} seconds\n")
            f.write(f"Start Time: {self.experiment_metadata['start_time']}\n")
            f.write(f"End Time: {self.experiment_metadata['end_time']}\n\n")
            
            f.write(f"Results:\n")
            f.write(f"- Total Steps: {summary['total_steps']}\n")
            f.write(f"- SCBF Analyzed Steps: {summary['scbf_analyzed_steps']}\n")
            f.write(f"- Analysis Coverage: {summary['analysis_coverage']:.1f}%\n")
            f.write(f"- SCBF Success Rate: {summary['scbf_success_rate']:.1f}%\n")
            f.write(f"- Successful Analyses: {summary['successful_analyses']}\n\n")
            
            f.write(f"Metrics Breakdown:\n")
            for metric, count in summary['metrics_breakdown'].items():
                f.write(f"- {metric}: {count} occurrences\n")
    
    def get_logs(self) -> List[Dict]:
        """Get all logged data."""
        return self.logs.copy()
    
    def get_scbf_logs(self) -> List[Dict]:
        """Get only logs containing SCBF metrics."""
        return [log for log in self.logs if 'scbf' in log]
    
    def get_experiment_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        return self.experiment_metadata.copy()

class SCBFExperimentRegistry:
    """Registry for managing multiple SCBF experiments."""
    
    def __init__(self, registry_dir: str = "scbf_experiments"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.registry_file = self.registry_dir / "experiment_registry.json"
        
        # Load existing registry
        self.experiments = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load experiment registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'experiments': {}, 'created': datetime.now().isoformat()}
    
    def _save_registry(self) -> None:
        """Save experiment registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def register_experiment(self, logger: SCBFExperimentLogger) -> None:
        """Register an experiment in the registry."""
        self.experiments['experiments'][logger.experiment_id] = {
            'name': logger.experiment_name,
            'id': logger.experiment_id,
            'start_time': logger.experiment_metadata['start_time'],
            'status': 'running',
            'directory': str(logger.experiment_dir)
        }
        self._save_registry()
        print(f"📝 Registered experiment: {logger.experiment_name} ({logger.experiment_id})")
    
    def complete_experiment(self, logger: SCBFExperimentLogger) -> None:
        """Mark an experiment as completed."""
        if logger.experiment_id in self.experiments['experiments']:
            self.experiments['experiments'][logger.experiment_id]['status'] = 'completed'
            self.experiments['experiments'][logger.experiment_id]['end_time'] = logger.experiment_metadata['end_time']
            self._save_registry()
    
    def list_experiments(self) -> List[Dict]:
        """List all registered experiments."""
        return list(self.experiments['experiments'].values())
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get details of a specific experiment."""
        return self.experiments['experiments'].get(experiment_id)
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict]:
        """Load full results for an experiment."""
        exp_info = self.get_experiment(experiment_id)
        if not exp_info:
            return None
        
        results_file = Path(exp_info['directory']) / "experiment_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None

# Global registry instance
_global_registry = SCBFExperimentRegistry()

def create_experiment_logger(experiment_name: str, output_dir: str = "scbf_experiments") -> SCBFExperimentLogger:
    """Create a new SCBF experiment logger."""
    logger = SCBFExperimentLogger(experiment_name, output_dir)
    _global_registry.register_experiment(logger)
    return logger

def finalize_experiment(logger: SCBFExperimentLogger) -> str:
    """Finalize an experiment and update registry."""
    results_file = logger.finalize_experiment()
    _global_registry.complete_experiment(logger)
    return results_file

def list_all_experiments() -> List[Dict]:
    """List all registered experiments."""
    return _global_registry.list_experiments()

def load_experiment_results(experiment_id: str) -> Optional[Dict]:
    """Load results for a specific experiment."""
    return _global_registry.load_experiment_results(experiment_id)
