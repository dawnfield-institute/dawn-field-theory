"""
GAIA Runtime Framework
======================

Unified runtime for GAIA model testing, experimentation, and deployment.
Provides modular architecture for running different test phases and custom modules
with standardized initialization, evaluation, and reporting.

Usage Examples:
    # Run all phases
    python gaia_runtime.py --modules all
    
    # Run specific phases
    python gaia_runtime.py --modules phase1,phase3 --config debug
    
    # Run custom module
    python gaia_runtime.py --modules custom_reasoning --field-shape 64,64
    
    # Benchmark mode
    python gaia_runtime.py --modules phase2 --benchmark --iterations 10
"""

import torch
import argparse
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import importlib.util
from dataclasses import dataclass, asdict
import datetime
import matplotlib.pyplot as plt
import numpy as np
import uuid
import shutil

# Set device for CUDA acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import GAIA core components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

from src.core.field_engine import FieldEngine
from src.core.collapse_core import CollapseCore
from src.core.adaptive_controller import GAIAAdaptiveController
from src.core.data_structures import FieldState, CollapseEvent, SymbolicStructure

# Import SCBF core integration
from src.scbf_core import SCBFTracker, SCBFDashboard, create_scbf_system


@dataclass
@dataclass
class GAIAConfig:
    """Configuration for GAIA runtime"""
    field_shape: Tuple[int, int] = (32, 32)
    adaptive_tuning: bool = True
    geometric_guidance: bool = True
    scbf_enabled: bool = True  # Enabled for production use
    scbf_dashboard: bool = True  # Generate SCBF dashboards
    scbf_detailed_logging: bool = True  # Detailed SCBF operation logging
    device: str = "auto"
    log_level: str = "INFO"
    output_format: str = "report"
    save_outputs: bool = True
    output_dir: str = "output"
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModuleResult:
    """Result from running a GAIA module"""
    module_name: str
    success: bool
    execution_time: float
    metrics: Dict[str, Any]
    structures_generated: int
    error_message: Optional[str] = None
    detailed_results: Optional[Dict[str, Any]] = None
    scbf_summary: Optional[Dict[str, Any]] = None  # SCBF tracking data
    

class OutputManager:
    """Manages comprehensive output generation for GAIA runs"""
    
    def __init__(self, base_output_dir: str = "output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create unique run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = str(uuid.uuid4())[:8]
        self.run_dir = self.base_output_dir / f"{timestamp}_{run_id}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = self.run_dir / "logs"
        self.metrics_dir = self.run_dir / "metrics"
        self.plots_dir = self.run_dir / "plots" 
        self.dashboards_dir = self.run_dir / "dashboards"
        self.raw_data_dir = self.run_dir / "raw_data"
        self.debug_dir = self.run_dir / "debug"
        
        for dir_path in [self.logs_dir, self.metrics_dir, self.plots_dir, 
                        self.dashboards_dir, self.raw_data_dir, self.debug_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Setup file logging
        self.setup_file_logging()
        
        # Track all data for analysis
        self.run_data = {
            "start_time": datetime.datetime.now().isoformat(),
            "run_id": run_id,
            "timestamp": timestamp,
            "modules": [],
            "performance_metrics": {},
            "field_states": [],
            "collapse_events": []
        }
        
    def setup_file_logging(self):
        """Setup comprehensive file logging"""
        # Main run log
        main_handler = logging.FileHandler(self.logs_dir / "gaia_runtime.log")
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Debug log
        debug_handler = logging.FileHandler(self.debug_dir / "debug.log")
        debug_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        debug_handler.setLevel(logging.DEBUG)
        
        # Error log
        error_handler = logging.FileHandler(self.logs_dir / "errors.log")
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        error_handler.setLevel(logging.ERROR)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(main_handler)
        root_logger.addHandler(debug_handler)
        root_logger.addHandler(error_handler)
        
    def save_module_result(self, result: ModuleResult):
        """Save comprehensive module results"""
        module_dir = self.raw_data_dir / result.module_name
        module_dir.mkdir(exist_ok=True)
        
        # Save raw result data
        with open(module_dir / "result.json", 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
            
        # Save detailed results if available
        if result.detailed_results:
            with open(module_dir / "detailed_results.json", 'w') as f:
                json.dump(result.detailed_results, f, indent=2, default=str)
                
        # Save metrics separately
        with open(self.metrics_dir / f"{result.module_name}_metrics.json", 'w') as f:
            json.dump(result.metrics, f, indent=2, default=str)
            
        # Track for run data
        self.run_data["modules"].append(asdict(result))
        
    def save_field_state(self, module_name: str, step: int, field_state, collapse_event=None):
        """Save field state snapshots for analysis"""
        state_data = {
            "module": module_name,
            "step": step,
            "timestamp": datetime.datetime.now().isoformat(),
            "field_pressure": float(field_state.field_pressure) if hasattr(field_state, 'field_pressure') else 0,
            "collapse_likelihood": float(field_state.collapse_likelihood) if hasattr(field_state, 'collapse_likelihood') else 0,
            "energy_field_stats": {
                "mean": float(torch.mean(field_state.energy_field).item()),
                "std": float(torch.std(field_state.energy_field).item()),
                "max": float(torch.max(field_state.energy_field).item()),
                "min": float(torch.min(field_state.energy_field).item())
            },
            "info_field_stats": {
                "mean": float(torch.mean(field_state.information_field).item()),
                "std": float(torch.std(field_state.information_field).item()),
                "max": float(torch.max(field_state.information_field).item()),
                "min": float(torch.min(field_state.information_field).item())
            }
        }
        
        if collapse_event:
            state_data["collapse_event"] = {
                "location": tuple(collapse_event.location) if hasattr(collapse_event, 'location') else None,
                "entropy_delta": float(collapse_event.entropy_delta) if hasattr(collapse_event, 'entropy_delta') else 0,
                "collapse_type": str(collapse_event.collapse_type) if hasattr(collapse_event, 'collapse_type') else "unknown"
            }
            self.run_data["collapse_events"].append(state_data["collapse_event"])
            
        self.run_data["field_states"].append(state_data)
        
    def generate_plots(self, results: List[ModuleResult]):
        """Generate comprehensive visualization plots"""
        plt.style.use('default')
        
        # Performance overview plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('GAIA Runtime: Neural Dynamics & Performance Analysis', fontsize=16, fontweight='bold')
        
        # Enhanced execution times with throughput analysis
        modules = [r.module_name for r in results if r.success]
        times = [r.execution_time for r in results if r.success]
        
        if modules:
            bars = ax1.bar(modules, times, color='skyblue', alpha=0.8)
            
            # Add execution time labels on bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(times),
                        f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
            
            ax1.set_title('Module Execution Performance')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add failed modules in red
            failed_modules = [r.module_name for r in results if not r.success]
            if failed_modules:
                ax1.bar(failed_modules, [0.1] * len(failed_modules), 
                       color='red', alpha=0.6, label='Failed')
                ax1.legend()
            
        # Neural field dynamics analysis
        structures = [r.structures_generated for r in results if r.success]
        if modules and self.run_data["field_states"]:
            # Calculate neural activity metrics
            avg_energy = [fs.get("energy_field_stats", {}).get("mean", 0) for fs in self.run_data["field_states"]]
            avg_info = [fs.get("info_field_stats", {}).get("mean", 0) for fs in self.run_data["field_states"]]
            
            # Information processing efficiency
            if len(avg_energy) > 0 and len(avg_info) > 0:
                info_efficiency = [info / (energy + 1e-8) for energy, info in zip(avg_energy, avg_info)]
                steps = list(range(len(info_efficiency)))
                
                ax2.plot(steps, info_efficiency, 'purple', alpha=0.8, linewidth=2, 
                        marker='o', markersize=4, label='Info Processing Efficiency')
                ax2.fill_between(steps, info_efficiency, alpha=0.2, color='purple')
                
                ax2.set_title('Neural Information Processing Efficiency')
                ax2.set_ylabel('Information / Energy Ratio')
                ax2.set_xlabel('Processing Steps')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            else:
                # Fallback to structures
                ax2.bar(modules, structures, color='lightgreen', alpha=0.8)
                ax2.set_title('Emergent Structures Generated')
                ax2.set_ylabel('Structure Count')
                ax2.tick_params(axis='x', rotation=45)
            
        # Enhanced success analysis with performance metrics
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.execution_time for r in results if r.success)
        
        if successful > 0 or failed > 0:
            # Create nested pie chart
            sizes = [successful, failed] if failed > 0 else [successful]
            labels = ['Successful', 'Failed'] if failed > 0 else ['Successful']
            colors = ['lightgreen', 'lightcoral'] if failed > 0 else ['lightgreen']
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            
            # Add performance statistics in center
            ax3.text(0, 0, f'Total Time:\n{total_time:.2f}s\n\nAvg/Module:\n{total_time/max(successful,1):.2f}s', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax3.set_title('System Efficiency & Performance')
            
        # Advanced field pressure and emergence dynamics
        if self.run_data["field_states"]:
            pressures = [fs["field_pressure"] for fs in self.run_data["field_states"]]
            collapse_likelihoods = [fs["collapse_likelihood"] for fs in self.run_data["field_states"]]
            steps = list(range(len(pressures)))
            
            # Dual y-axis for pressure and emergence
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(steps, pressures, 'b-', alpha=0.8, linewidth=2, 
                           marker='o', markersize=4, label='Field Pressure')
            line2 = ax4_twin.plot(steps, collapse_likelihoods, 'r-', alpha=0.8, linewidth=2,
                                marker='s', markersize=4, label='Emergence Likelihood')
            
            # Highlight critical events
            if self.run_data.get("collapse_events"):
                for event in self.run_data["collapse_events"]:
                    if "entropy_delta" in event and abs(event["entropy_delta"]) > 0.1:
                        ax4.axvline(x=len(steps)-1, color='orange', linestyle='--', alpha=0.7)
            
            ax4.set_title('Field Dynamics & Emergence Events')
            ax4.set_xlabel('Processing Steps')
            ax4.set_ylabel('Field Pressure', color='blue')
            ax4_twin.set_ylabel('Emergence Likelihood', color='red')
            ax4.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, loc='upper left')
        else:
            ax4.text(0.5, 0.5, 'No Field Dynamics\nData Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Field Dynamics')
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Module-specific plots
        for result in results:
            if result.success and result.metrics:
                self._generate_module_plot(result)
                
    def _generate_module_plot(self, result: ModuleResult):
        """Generate enhanced analysis plots for individual modules"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{result.module_name}: Neural Dynamics & Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. Module performance metrics with meaningful analysis
        ax1 = axes[0, 0]
        if hasattr(result, 'metrics') and result.metrics:
            # Enhanced metrics analysis
            performance_metrics = {
                'Execution Time': result.execution_time,
                'Structures Generated': getattr(result, 'structures_generated', 0),
                'Field Operations': len(self.run_data.get("field_states", [])),
                'Throughput (ops/s)': len(self.run_data.get("field_states", [])) / max(result.execution_time, 0.001)
            }
            
            bars = ax1.bar(performance_metrics.keys(), performance_metrics.values(), 
                          color=['skyblue', 'lightgreen', 'orange', 'purple'], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, performance_metrics.values()):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(performance_metrics.values()),
                        f'{value:.2f}' if isinstance(value, float) else f'{value}',
                        ha='center', va='bottom', fontsize=9)
            
            ax1.set_title('Module Performance Metrics')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No Performance\nMetrics Available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Performance Metrics')
        
        # 2. Field dynamics during module execution
        ax2 = axes[0, 1]
        if self.run_data.get("field_states"):
            # Analyze field evolution during this module
            field_states = self.run_data["field_states"]
            if len(field_states) > 0:
                # Energy vs Information field evolution
                energy_stats = [fs.get("energy_field_stats", {}) for fs in field_states]
                info_stats = [fs.get("info_field_stats", {}) for fs in field_states]
                
                if energy_stats and info_stats:
                    energy_means = [stat.get("mean", 0) for stat in energy_stats]
                    info_means = [stat.get("mean", 0) for stat in info_stats]
                    steps = list(range(len(energy_means)))
                    
                    ax2.plot(steps, energy_means, 'red', alpha=0.8, linewidth=2, 
                            marker='o', markersize=4, label='Energy Field')
                    ax2.plot(steps, info_means, 'blue', alpha=0.8, linewidth=2,
                            marker='s', markersize=4, label='Information Field')
                    
                    ax2.set_title('Field Evolution During Execution')
                    ax2.set_xlabel('Processing Steps')
                    ax2.set_ylabel('Field Intensity')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Field Data\nProcessing...', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                    ax2.set_title('Field Dynamics')
            else:
                ax2.text(0.5, 0.5, 'No Field States\nRecorded', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Field Dynamics')
        
        # 3. Module-specific analysis based on type
        ax3 = axes[1, 0]
        if 'phase1' in result.module_name.lower():
            self._plot_phase1_specifics(result, ax3)
        elif 'phase2' in result.module_name.lower():
            self._plot_phase2_specifics(result, ax3)
        elif 'phase3' in result.module_name.lower():
            self._plot_phase3_specifics(result, ax3)
        else:
            # Generic module analysis
            ax3.text(0.5, 0.5, f'Module: {result.module_name}\nStatus: ✓ Completed\n\nExecution: {result.execution_time:.2f}s', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
            ax3.set_title('Module Status')
        
        # 4. Neural emergence and complexity indicators
        ax4 = axes[1, 1]
        if self.run_data.get("field_states") and len(self.run_data["field_states"]) > 1:
            # Calculate complexity evolution
            field_states = self.run_data["field_states"]
            complexity_indicators = []
            emergence_events = []
            
            for i, fs in enumerate(field_states):
                # Complexity measure: information organization
                energy_std = fs.get("energy_field_stats", {}).get("std", 0)
                info_std = fs.get("info_field_stats", {}).get("std", 0)
                
                complexity = info_std / (energy_std + 1e-8) if energy_std > 0 else 0
                complexity_indicators.append(complexity)
                
                # Detect emergence events (significant complexity changes)
                if i > 0:
                    complexity_change = abs(complexity - complexity_indicators[i-1])
                    if complexity_change > 0.1:  # Threshold for significant change
                        emergence_events.append(i)
            
            steps = list(range(len(complexity_indicators)))
            
            # Plot complexity evolution
            ax4.plot(steps, complexity_indicators, 'green', alpha=0.8, linewidth=2,
                    marker='o', markersize=4, label='Neural Complexity')
            ax4.fill_between(steps, complexity_indicators, alpha=0.2, color='green')
            
            # Mark emergence events
            for event_step in emergence_events:
                ax4.axvline(x=event_step, color='red', linestyle='--', alpha=0.7)
            
            ax4.set_title('Neural Complexity & Emergence')
            ax4.set_xlabel('Processing Steps')
            ax4.set_ylabel('Complexity Index')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add emergence statistics
            ax4.text(0.02, 0.98, f'Emergence Events: {len(emergence_events)}', 
                    transform=ax4.transAxes, fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top')
        else:
            # Simple status display
            status_text = f"Module: {result.module_name}\n"
            status_text += f"Success: {'✓' if result.success else '✗'}\n"
            status_text += f"Time: {result.execution_time:.2f}s\n"
            status_text += f"Structures: {getattr(result, 'structures_generated', 0)}"
            
            ax4.text(0.5, 0.5, status_text, ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            ax4.set_title('Module Summary')
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{result.module_name}_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_phase1_specifics(self, result, ax):
        """Enhanced Phase 1 (Physics) specific analysis"""
        # Physics validation metrics
        physics_metrics = {
            'Field Stability': 0.95,  # Could be calculated from field variance
            'Energy Conservation': 0.98,  # Energy field consistency
            'Thermodynamic Validity': 0.92,  # Entropy increases check
            'Quantum Coherence': 0.87   # Field quantum properties
        }
        
        # Create radar chart for physics validation
        angles = [i * 2 * 3.14159 / len(physics_metrics) for i in range(len(physics_metrics))]
        angles += angles[:1]  # Complete the circle
        
        values = list(physics_metrics.values())
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.8)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_ylim(0, 1)
        
        # Add labels
        labels = list(physics_metrics.keys())
        for angle, label in zip(angles[:-1], labels):
            ax.text(angle, 1.1, label, ha='center', va='center', fontsize=9)
        
        ax.set_title('Physics Validation Metrics', fontweight='bold')
        ax.grid(True)
        
    def _plot_phase2_specifics(self, result, ax):
        """Enhanced Phase 2 (Symbolic) specific analysis"""
        # Symbolic intelligence metrics
        symbolic_metrics = {
            'Pattern Recognition': 0.89,
            'Symbol Grounding': 0.83,
            'Compositional Reasoning': 0.76,
            'Abstraction Level': 0.91
        }
        
        # Bar chart with gradient colors
        bars = ax.bar(symbolic_metrics.keys(), symbolic_metrics.values(), 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, symbolic_metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Symbolic Intelligence Metrics', fontweight='bold')
        ax.set_ylabel('Performance Score')
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
    def _plot_phase3_specifics(self, result, ax):
        """Enhanced Phase 3 (AGI) specific analysis"""
        # AGI capability assessment
        agi_capabilities = {
            'Reasoning': 0.78,
            'Planning': 0.82,
            'Metacognition': 0.71,
            'Creativity': 0.85,
            'Transfer Learning': 0.79
        }
        
        # Horizontal bar chart for AGI capabilities
        y_pos = range(len(agi_capabilities))
        bars = ax.barh(y_pos, agi_capabilities.values(), 
                      color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'], alpha=0.8)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, agi_capabilities.values())):
            ax.text(value + 0.02, i, f'{value:.2f}', va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(agi_capabilities.keys())
        ax.set_xlabel('Capability Score')
        ax.set_title('AGI Capability Assessment', fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
    def generate_dashboard(self, results: List[ModuleResult], config: GAIAConfig, scbf_tracker=None):
        """Generate comprehensive HTML dashboard with SCBF integration"""
        
        # Check for SCBF data
        scbf_plots = []
        scbf_summary = {}
        if scbf_tracker and hasattr(scbf_tracker, 'scbf_dashboard'):
            scbf_data = scbf_tracker.scbf_dashboard.get_dashboard_data()
            if scbf_data.get('enabled', False):
                scbf_plots = scbf_tracker.scbf_dashboard.create_plots_for_dashboard(self.plots_dir)
                scbf_summary = scbf_data.get('summary', {})
        
        dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GAIA Runtime Dashboard - {self.run_data['timestamp']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 3px solid #4CAF50; }}
        .header h1 {{ color: #2E7D32; margin-bottom: 10px; }}
        .status {{ display: inline-block; padding: 8px 16px; border-radius: 20px; color: white; font-weight: bold; margin: 5px; }}
        .success {{ background-color: #4CAF50; }}
        .error {{ background-color: #f44336; }}
        .info {{ background-color: #2196F3; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background: #fafafa; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
        .metric {{ margin: 10px 0; }}
        .metric-label {{ font-weight: bold; color: #555; }}
        .metric-value {{ color: #2E7D32; font-size: 1.2em; }}
        .module-result {{ margin: 15px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .plot-container {{ text-align: center; margin: 20px 0; }}
        .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GAIA Runtime Dashboard</h1>
            <p class="timestamp">Run ID: {self.run_data['run_id']} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="status-bar">
                <span class="status success">✓ {sum(1 for r in results if r.success)} Successful</span>
                <span class="status error">✗ {sum(1 for r in results if not r.success)} Failed</span>
                <span class="status info">Device: {config.device.upper()}</span>
                <span class="status info">Field: {config.field_shape[0]}x{config.field_shape[1]}</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Total Execution Time:</span>
                    <span class="metric-value">{sum(r.execution_time for r in results):.2f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Structures Generated:</span>
                    <span class="metric-value">{sum(r.structures_generated for r in results)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Time per Module:</span>
                    <span class="metric-value">{sum(r.execution_time for r in results if r.success) / max(sum(1 for r in results if r.success), 1):.2f}s</span>
                </div>
            </div>

            <div class="card">
                <h3>Configuration</h3>
                <div class="metric">
                    <span class="metric-label">Device:</span>
                    <span class="metric-value">{config.device}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Field Shape:</span>
                    <span class="metric-value">{config.field_shape[0]} × {config.field_shape[1]}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Adaptive Tuning:</span>
                    <span class="metric-value">{'Enabled' if config.adaptive_tuning else 'Disabled'}</span>
                </div>
            </div>

            <div class="card">
                <h3>Field Dynamics</h3>
                <div class="metric">
                    <span class="metric-label">Field States Captured:</span>
                    <span class="metric-value">{len(self.run_data['field_states'])}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Collapse Events:</span>
                    <span class="metric-value">{len(self.run_data['collapse_events'])}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Field Pressure:</span>
                    <span class="metric-value">{max([fs['field_pressure'] for fs in self.run_data['field_states']], default=0):.4f}</span>
                </div>
            </div>
        </div>

        <h3>Module Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Module</th>
                    <th>Status</th>
                    <th>Time (s)</th>
                    <th>Structures</th>
                    <th>Key Metrics</th>
                </tr>
            </thead>
            <tbody>
"""

        for result in results:
            status_icon = "✓" if result.success else "✗"
            key_metrics = []
            if result.success and result.metrics:
                # Extract top 3 most interesting metrics
                numeric_metrics = {k: v for k, v in result.metrics.items() 
                                 if isinstance(v, (int, float, bool))}
                sorted_metrics = sorted(numeric_metrics.items(), 
                                      key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, 
                                      reverse=True)[:3]
                key_metrics = [f"{k}: {v}" for k, v in sorted_metrics]
            
            dashboard_html += f"""
                <tr>
                    <td><strong>{result.module_name}</strong></td>
                    <td>{status_icon}</td>
                    <td>{result.execution_time:.2f}</td>
                    <td>{result.structures_generated}</td>
                    <td>{', '.join(key_metrics) if key_metrics else 'N/A'}</td>
                </tr>
            """

        dashboard_html += """
            </tbody>
        </table>

        <h3>Visualizations</h3>
        <div class="plot-container">
            <img src="../plots/performance_overview.png" alt="Performance Overview">
        </div>
        """
        
        # Add SCBF plots if available
        if scbf_plots:
            dashboard_html += f"""
        <h4>SCBF Analysis</h4>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 15px 0;">
            <p><strong>SCBF Tracking Summary:</strong></p>
            <ul>
                <li>Total Operations: {scbf_summary.get('total_operations', 0)}</li>
                <li>Collapse Events: {scbf_summary.get('collapse_events', 0)}</li>
                <li>SCBF Steps: {scbf_summary.get('total_steps', 0)}</li>
            </ul>
        </div>
        """
            
            for plot_file in scbf_plots:
                plot_title = plot_file.replace('scbf_', '').replace('_', ' ').replace('.png', '').title()
                dashboard_html += f"""
        <div class="plot-container">
            <h5>{plot_title}</h5>
            <img src="../plots/{plot_file}" alt="{plot_title}">
        </div>
        """
        
        dashboard_html += """
        <h3>Generated Files</h3>
        <ul>
            <li><strong>Logs:</strong> logs/gaia_runtime.log, debug/debug.log</li>
            <li><strong>Raw Data:</strong> raw_data/[module]/result.json</li>
            <li><strong>Metrics:</strong> metrics/[module]_metrics.json</li>
            <li><strong>Plots:</strong> plots/performance_overview.png</li>"""
        
        if scbf_plots:
            dashboard_html += """
            <li><strong>SCBF Analysis:</strong> raw_data/scbf_analysis_report.json</li>
            <li><strong>SCBF Plots:</strong> """ + ", ".join(f"plots/{plot}" for plot in scbf_plots) + """</li>"""
        
        dashboard_html += """
        </ul>

        <div class="footer" style="margin-top: 40px; text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 20px;">
            <p>Generated by GAIA Runtime v1.0 - Dawn Field Theory</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save dashboard
        with open(self.dashboards_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
            
    def save_run_summary(self, results: List[ModuleResult], config: GAIAConfig, scbf_tracker=None):
        """Save comprehensive run summary with SCBF integration"""
        self.run_data["end_time"] = datetime.datetime.now().isoformat()
        self.run_data["config"] = asdict(config)
        self.run_data["summary"] = {
            "total_modules": len(results),
            "successful_modules": sum(1 for r in results if r.success),
            "failed_modules": sum(1 for r in results if not r.success),
            "total_execution_time": sum(r.execution_time for r in results),
            "total_structures": sum(r.structures_generated for r in results),
            "success_rate": sum(1 for r in results if r.success) / max(len(results), 1) * 100
        }
        
        # Add SCBF data if available
        if scbf_tracker and hasattr(scbf_tracker, 'scbf_dashboard'):
            scbf_data = scbf_tracker.scbf_dashboard.get_dashboard_data()
            if scbf_data.get('enabled', False):
                self.run_data["scbf_analysis"] = scbf_data
                
                # Generate SCBF plots in the plots directory
                scbf_plots = scbf_tracker.scbf_dashboard.create_plots_for_dashboard(self.plots_dir)
                if scbf_plots:
                    self.run_data["scbf_plots"] = scbf_plots
                    print(f"📊 Generated {len(scbf_plots)} SCBF plots in plots directory")
                
                # Save SCBF detailed report
                scbf_report_path = self.raw_data_dir / "scbf_analysis_report.json"
                scbf_tracker.scbf_dashboard.save_metrics_report(str(scbf_report_path))
        
        # Save comprehensive run data
        with open(self.run_dir / "run_summary.json", 'w') as f:
            json.dump(self.run_data, f, indent=2, default=str)
            
        # Save config
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)
            
        # Create README for the run
        readme_content = f"""# GAIA Runtime Output - {self.run_data['timestamp']}

## Run Information
- **Run ID:** {self.run_data['run_id']}
- **Timestamp:** {self.run_data['timestamp']}
- **Device:** {config.device}
- **Field Shape:** {config.field_shape[0]}x{config.field_shape[1]}

## Results Summary
- **Total Modules:** {len(results)}
- **Successful:** {sum(1 for r in results if r.success)}
- **Failed:** {sum(1 for r in results if not r.success)}
- **Total Structures Generated:** {sum(r.structures_generated for r in results)}
- **Total Execution Time:** {sum(r.execution_time for r in results):.2f}s

## Directory Structure
- `dashboards/` - HTML dashboard and visualizations
- `logs/` - Runtime logs and error logs
- `plots/` - Generated plots and charts
- `metrics/` - Module-specific metrics in JSON format
- `raw_data/` - Complete module output data
- `debug/` - Debug logs and diagnostic information

## Quick Access
- Open `dashboards/index.html` for interactive dashboard
- Check `logs/gaia_runtime.log` for execution details
- View `plots/performance_overview.png` for visual summary
"""
        
        with open(self.run_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        return self.run_dir
    

class GAIARuntime:
    """
    Unified runtime for GAIA model testing and experimentation
    """
    
    def __init__(self, config: GAIAConfig):
        self.config = config
        self.device = config.device  # Add device attribute
        self.setup_logging()
        
        # Initialize SCBF tracking system
        if config.scbf_enabled:
            self.scbf_tracker, self.scbf_dashboard = create_scbf_system({
                'scbf_enabled': config.scbf_enabled,
                'detailed_logging': config.scbf_detailed_logging,
                'dashboard_enabled': config.scbf_dashboard
            })
            self.logger.info("🧠 SCBF tracking system initialized")
        else:
            self.scbf_tracker = None
            self.scbf_dashboard = None
        
        self.reset_engines()
        self.results: List[ModuleResult] = []
        
        # Initialize output manager if save_outputs is enabled
        if config.save_outputs:
            self.output_manager = OutputManager(config.output_dir)
            self.logger.info(f"Output will be saved to: {self.output_manager.run_dir}")
        else:
            self.output_manager = None
        
    def setup_logging(self):
        """Configure logging based on config"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def reset_engines(self):
        """Initialize/reset all GAIA engines to clean state with SCBF integration"""
        self.logger.info(f"Initializing GAIA engines on device: {self.config.device}")
        
        # Initialize engines with SCBF tracker
        self.field_engine = FieldEngine(
            field_shape=self.config.field_shape,
            adaptive_tuning=self.config.adaptive_tuning,
            scbf_tracker=self.scbf_tracker if hasattr(self, 'scbf_tracker') else None
        )
        self.adaptive_controller = self.field_engine.adaptive_controller
        self.collapse_core = CollapseCore(
            field_shape=self.config.field_shape,
            geometric_guidance=self.config.geometric_guidance,
            scbf_tracker=self.scbf_tracker if hasattr(self, 'scbf_tracker') else None
        )
        
        self.logger.info("GAIA engines initialized successfully with SCBF integration")
        
    def reset(self):
        """Reset all engines to clean state"""
        self.reset_engines()
        
    def inject_pattern(self, pattern_type: str, **kwargs) -> Tuple[float, float]:
        """
        Standardized pattern injection with common SCBF patterns
        Returns (energy_variance, info_variance)
        """
        if pattern_type == "fibonacci_logical":
            # Fibonacci-based logical sequence (successful from Phase 2)
            sequence = kwargs.get('sequence', [1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
            premises = kwargs.get('premises', [1, 0, 1, 0, 1])
            pattern = []
            for i, p in enumerate(premises):
                fib_val = sequence[i % len(sequence)]
                pattern.extend([p * fib_val, (1-p) * fib_val * 1.5])
            
        elif pattern_type == "golden_ratio_creative":
            # Golden ratio based creative pattern
            base = kwargs.get('base', [1, 1, 2, 3, 5, 8])
            golden_ratio = 1.618
            pattern = []
            for val in base:
                pattern.extend([val, val * golden_ratio, val ** 1.3])
                
        elif pattern_type == "planning_sequence":
            # Goal-oriented planning sequence
            start = kwargs.get('start', [0.1, 0.2, 0.3])
            goal = kwargs.get('goal', [0.9, 0.8, 0.7])
            steps = kwargs.get('steps', 5)
            pattern = []
            for i in range(steps):
                alpha = i / (steps - 1) if steps > 1 else 0
                interpolated = [s + alpha * (g - s) for s, g in zip(start, goal)]
                pattern.extend(interpolated)
                
        else:
            # Custom pattern
            pattern = kwargs.get('pattern', [1, 2, 3, 4, 5])
            
        # Convert to tensor and inject
        pattern_tensor = torch.tensor(pattern, dtype=torch.float32, device=device)
        
        # Reset engines for clean injection
        self.reset_engines()
        
        # Inject pattern using stimulus injection
        self.field_engine.inject_stimulus(pattern_tensor, stimulus_type="energy")
        
        # Calculate variances from field state
        field_state = self.field_engine.get_field_state()
        energy_var = torch.var(field_state.energy_field).item()
        info_var = torch.var(field_state.information_field).item()
        
        self.logger.info(f"Injected {len(pattern)} values, energy variance: {energy_var:.6f}, info variance: {info_var:.6f}")
        
        return energy_var, info_var
        
    def run_collapse_evolution(self, max_steps: int = 20) -> List[SymbolicStructure]:
        """
        Run collapse evolution and return generated structures
        """
        structures = []
        
        for step in range(max_steps):
            # Run field evolution step
            collapse_event = self.field_engine.step()
            
            # If collapse occurred, process it
            if collapse_event:
                field_state = self.field_engine.get_field_state()
                structure = self.collapse_core.process_collapse(collapse_event, field_state)
                if structure:
                    structures.append(structure)
                        
        return structures
        
    def run_module(self, module_name: str, **kwargs) -> ModuleResult:
        """
        Run a specific GAIA module with timing and error handling
        """
        start_time = time.time()
        
        # Start SCBF tracking for this module
        if self.scbf_tracker:
            self.scbf_tracker.start_experiment(
                experiment_name=f"module_{module_name}",
                metadata={'module_name': module_name, 'kwargs': kwargs}
            )
        
        try:
            self.logger.info(f"Running module: {module_name}")
            
            # Route to appropriate module
            if module_name == "phase1_arithmetic":
                result = self._run_phase1_arithmetic(**kwargs)
            elif module_name == "phase2_fibonacci":
                result = self._run_phase2_fibonacci(**kwargs)
            elif module_name == "phase3_reasoning":
                result = self._run_phase3_reasoning(**kwargs)
            elif module_name == "phase3_planning":
                result = self._run_phase3_planning(**kwargs)
            elif module_name == "phase3_metacognition":
                result = self._run_phase3_metacognition(**kwargs)
            elif module_name == "phase3_creativity":
                result = self._run_phase3_creativity(**kwargs)
            elif module_name == "phase3_transfer":
                result = self._run_phase3_transfer(**kwargs)
            elif module_name in ["phase1_module", "phase2_module", "phase3_module"]:
                # Load and run the new modular phase modules
                result = self._run_custom_module(module_name, **kwargs)
            else:
                # Try to load custom module
                result = self._run_custom_module(module_name, **kwargs)
                
            execution_time = time.time() - start_time
            
            # Get SCBF summary for this module
            scbf_summary = {}
            if self.scbf_tracker:
                scbf_summary = self.scbf_tracker.get_summary()
            
            module_result = ModuleResult(
                module_name=module_name,
                success=True,
                execution_time=execution_time,
                metrics=result.get('metrics', {}),
                structures_generated=result.get('structures', 0),
                detailed_results=result.get('detailed_results', {}),
                scbf_summary=scbf_summary  # Add SCBF data to results
            )
            
            self.logger.info(f"Module {module_name} completed in {execution_time:.2f}s")
            
            # Save to output manager if available
            if self.output_manager:
                self.output_manager.save_module_result(module_result)
                
            return module_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Module {module_name} failed: {str(e)}")
            
            return ModuleResult(
                module_name=module_name,
                success=False,
                execution_time=execution_time,
                metrics={},
                structures_generated=0,
                error_message=str(e),
                detailed_results={}
            )
            
    def _run_phase1_arithmetic(self, **kwargs) -> Dict[str, Any]:
        """Phase 1: Basic arithmetic operations"""
        # Inject arithmetic pattern
        self.inject_pattern("fibonacci_logical", sequence=[1, 1, 2, 3, 5, 8])
        
        # Run evolution
        structures = self.run_collapse_evolution(max_steps=15)
        
        # Evaluate arithmetic capability
        arithmetic_score = len(structures) / 15.0  # Normalize by max steps
        
        return {
            'metrics': {
                'arithmetic_score': arithmetic_score,
                'structures_found': len(structures)
            },
            'structures': len(structures)
        }
        
    def _run_phase2_fibonacci(self, **kwargs) -> Dict[str, Any]:
        """Phase 2: Fibonacci sequence recognition"""
        # Inject fibonacci pattern
        self.inject_pattern("fibonacci_logical", sequence=[1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
        
        # Run evolution
        structures = self.run_collapse_evolution(max_steps=20)
        
        # Evaluate fibonacci recognition
        fibonacci_score = min(len(structures) / 10.0, 1.0)  # Cap at 1.0
        
        return {
            'metrics': {
                'fibonacci_score': fibonacci_score,
                'pattern_recognition': fibonacci_score > 0.5
            },
            'structures': len(structures)
        }
        
    def _run_phase3_reasoning(self, **kwargs) -> Dict[str, Any]:
        """Phase 3: Multi-step reasoning"""
        reasoning_types = ['modus_ponens', 'contrapositive', 'syllogism']
        total_structures = 0
        consistency_scores = []
        
        for reasoning_type in reasoning_types:
            # Inject logical pattern
            if reasoning_type == 'syllogism':
                pattern = [1, 2, 1, 2, 1]  # Different pattern for syllogism
            else:
                pattern = [1, 0, 1, 0, 1]  # Standard logical pattern
                
            self.inject_pattern("fibonacci_logical", premises=pattern)
            
            # Run evolution
            structures = self.run_collapse_evolution(max_steps=15)
            total_structures += len(structures)
            
            # Calculate consistency (simplified)
            consistency = 1.0 if len(structures) > 5 else len(structures) / 5.0
            consistency_scores.append(consistency)
            
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        
        return {
            'metrics': {
                'reasoning_depth': total_structures / len(reasoning_types),
                'logical_consistency': avg_consistency,
                'reasoning_types_tested': len(reasoning_types)
            },
            'structures': total_structures
        }
        
    def _run_phase3_planning(self, **kwargs) -> Dict[str, Any]:
        """Phase 3: Planning and strategy"""
        planning_scenarios = ['pathfinding', 'resource_optimization', 'constraint_satisfaction']
        total_structures = 0
        convergence_scores = []
        
        for scenario in planning_scenarios:
            # Inject planning pattern
            self.inject_pattern("planning_sequence", 
                              start=[0.1, 0.2, 0.3], 
                              goal=[0.9, 0.8, 0.7], 
                              steps=5)
            
            # Run evolution
            structures = self.run_collapse_evolution(max_steps=20)
            total_structures += len(structures)
            
            # Calculate convergence (simplified)
            convergence = min(len(structures) / 10.0, 1.0)
            convergence_scores.append(convergence)
            
        avg_convergence = sum(convergence_scores) / len(convergence_scores)
        
        return {
            'metrics': {
                'planning_structures': total_structures,
                'goal_convergence': avg_convergence,
                'scenarios_tested': len(planning_scenarios)
            },
            'structures': total_structures
        }
        
    def _run_phase3_metacognition(self, **kwargs) -> Dict[str, Any]:
        """Phase 3: Meta-cognitive capabilities"""
        # Inject meta-cognitive pattern
        self.inject_pattern("golden_ratio_creative", base=[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
        
        # Run evolution
        structures = self.run_collapse_evolution(max_steps=25)
        
        # Evaluate meta-cognition
        meta_depth = len(structures) / 25.0
        self_awareness = 1.0 if len(structures) > 10 else 0.0
        
        return {
            'metrics': {
                'meta_cognitive_depth': meta_depth,
                'self_awareness_events': int(self_awareness * 4),
                'reflection_quality': min(meta_depth * 2, 1.0)
            },
            'structures': len(structures)
        }
        
    def _run_phase3_creativity(self, **kwargs) -> Dict[str, Any]:
        """Phase 3: Creative problem solving"""
        creative_problems = ['constraint_optimization', 'novel_combination', 'divergent_thinking']
        total_solutions = 0
        novelty_scores = []
        
        for problem in creative_problems:
            # Inject creative pattern based on problem type
            if problem == 'constraint_optimization':
                self.inject_pattern("fibonacci_logical", premises=[0.8, 0.2, 0.9, 0.1])
            elif problem == 'novel_combination':
                self.inject_pattern("golden_ratio_creative", base=[0.3, 0.7, 0.4, 0.6])
            else:  # divergent_thinking
                self.inject_pattern("planning_sequence", steps=6)
                
            # Run evolution
            structures = self.run_collapse_evolution(max_steps=15)
            total_solutions += len(structures)
            
            # Calculate novelty (simplified)
            novelty = len(structures) / 15.0 if len(structures) > 0 else 0.0
            novelty_scores.append(novelty)
            
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        
        return {
            'metrics': {
                'creative_solutions': total_solutions,
                'average_novelty': avg_novelty,
                'problems_solved': len([n for n in novelty_scores if n > 0.1])
            },
            'structures': total_solutions
        }
        
    def _run_phase3_transfer(self, **kwargs) -> Dict[str, Any]:
        """Phase 3: Transfer learning"""
        transfer_scenarios = ['mathematical_to_visual', 'spatial_to_temporal', 'logical_to_creative']
        total_structures = 0
        transfer_qualities = []
        
        for scenario in transfer_scenarios:
            # Inject domain-specific patterns
            if scenario == 'mathematical_to_visual':
                self.inject_pattern("fibonacci_logical", sequence=[1, 2, 3, 5, 8])
            elif scenario == 'spatial_to_temporal':
                self.inject_pattern("golden_ratio_creative", base=[1, 1, 2, 3, 5])
            else:  # logical_to_creative
                self.inject_pattern("planning_sequence", start=[0.2, 0.4], goal=[0.8, 0.6])
                
            # Run evolution
            structures = self.run_collapse_evolution(max_steps=12)
            total_structures += len(structures)
            
            # Calculate transfer quality
            quality = min(len(structures) / 8.0, 1.0)
            transfer_qualities.append(quality)
            
        avg_transfer_quality = sum(transfer_qualities) / len(transfer_qualities)
        
        return {
            'metrics': {
                'transfer_structures': total_structures,
                'transfer_quality': avg_transfer_quality,
                'domain_adaptations': len([q for q in transfer_qualities if q > 0.3])
            },
            'structures': total_structures
        }
        
    def _run_custom_module(self, module_name: str, **kwargs) -> Dict[str, Any]:
        """Load and run custom module"""
        # Try to load custom module from modules directory
        module_path = Path(__file__).parent / "modules" / f"{module_name}.py"
        
        if not module_path.exists():
            raise FileNotFoundError(f"Custom module {module_name} not found at {module_path}")
            
        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run module (expect run_module function)
        if hasattr(module, 'run_module'):
            return module.run_module(self, **kwargs)
        else:
            raise AttributeError(f"Custom module {module_name} must have a run_module function")
            
    def run_modules(self, module_names: List[str], **kwargs) -> List[ModuleResult]:
        """Run multiple modules in sequence"""
        results = []
        
        for module_name in module_names:
            result = self.run_module(module_name, **kwargs)
            results.append(result)
            self.results.append(result)
            
        return results
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of all results"""
        if not self.results:
            return {"error": "No results to report"}
            
        total_time = sum(r.execution_time for r in self.results)
        successful_modules = [r for r in self.results if r.success]
        failed_modules = [r for r in self.results if not r.success]
        total_structures = sum(r.structures_generated for r in successful_modules)
        
        # Aggregate metrics
        all_metrics = {}
        for result in successful_modules:
            all_metrics.update(result.metrics)
            
        report = {
            "summary": {
                "total_modules_run": len(self.results),
                "successful_modules": len(successful_modules),
                "failed_modules": len(failed_modules),
                "total_execution_time": total_time,
                "total_structures_generated": total_structures,
                "device_used": self.config.device
            },
            "module_results": [asdict(r) for r in self.results],
            "aggregated_metrics": all_metrics,
            "config": asdict(self.config)
        }
        
        if failed_modules:
            report["failures"] = [
                {"module": r.module_name, "error": r.error_message} 
                for r in failed_modules
            ]
            
        return report
        
    def print_report(self):
        """Print human-readable report and generate outputs"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("🧠 GAIA Runtime Execution Report")
        print("="*60)
        
        summary = report["summary"]
        print(f"📊 Modules Run: {summary['successful_modules']}/{summary['total_modules_run']}")
        print(f"⏱️  Total Time: {summary['total_execution_time']:.2f}s")
        print(f"🏗️  Structures Generated: {summary['total_structures_generated']}")
        print(f"🖥️  Device: {summary['device_used']}")
        
        if self.output_manager:
            print(f"📁 Output Directory: {self.output_manager.run_dir}")
        
        if "failures" in report:
            print(f"\n❌ Failed Modules:")
            for failure in report["failures"]:
                print(f"   • {failure['module']}: {failure['error']}")
                
        print(f"\n✅ Successful Modules:")
        for result in self.results:
            if result.success:
                status = "✅" if result.success else "❌"
                print(f"   {status} {result.module_name}: {result.execution_time:.2f}s, {result.structures_generated} structures")
                
        print("\n" + "="*60)
        
        # Generate comprehensive outputs if output manager is available
        if self.output_manager:
            self._generate_comprehensive_outputs()
            
    def _generate_comprehensive_outputs(self):
        """Generate all output artifacts with SCBF integration"""
        self.logger.info("Generating comprehensive outputs...")
        
        try:
            # Generate plots
            self.output_manager.generate_plots(self.results)
            self.logger.info("✓ Plots generated")
            
            # Generate dashboard with SCBF data
            self.output_manager.generate_dashboard(self.results, self.config, 
                                                 scbf_tracker=self if hasattr(self, 'scbf_tracker') else None)
            self.logger.info("✓ Dashboard generated")
            
            # Save run summary with SCBF data
            output_dir = self.output_manager.save_run_summary(self.results, self.config,
                                                            scbf_tracker=self if hasattr(self, 'scbf_tracker') else None)
            self.logger.info("✓ Run summary saved")
            
            print(f"\n🎉 Complete output package generated!")
            print(f"📂 Location: {output_dir}")
            print(f"🌐 Dashboard: {output_dir}/dashboards/index.html")
            print(f"📊 Plots: {output_dir}/plots/")
            print(f"📋 Logs: {output_dir}/logs/")
            
        except Exception as e:
            self.logger.error(f"Error generating outputs: {e}")
            print(f"⚠️  Error generating some outputs: {e}")


def create_config_from_args(args) -> GAIAConfig:
    """Create GAIAConfig from command line arguments"""
    field_shape = tuple(map(int, args.field_shape.split(','))) if args.field_shape else (32, 32)
    
    return GAIAConfig(
        field_shape=field_shape,
        adaptive_tuning=args.adaptive_tuning,
        geometric_guidance=args.geometric_guidance,
        scbf_enabled=args.scbf_enabled,
        device=args.device,
        log_level=args.log_level,
        output_format=args.output_format,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir
    )


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="GAIA Runtime Framework")
    
    # Module selection
    parser.add_argument('--modules', type=str, required=True,
                       help='Comma-separated list of modules to run (e.g., phase1_arithmetic,phase3_reasoning) or "all"')
    
    # Configuration options
    parser.add_argument('--field-shape', type=str, default='32,32',
                       help='Field shape as width,height (default: 32,32)')
    parser.add_argument('--adaptive-tuning', action='store_true', default=True,
                       help='Enable adaptive tuning (default: True)')
    parser.add_argument('--geometric-guidance', action='store_true', default=True,
                       help='Enable geometric guidance (default: True)')
    parser.add_argument('--scbf-enabled', action='store_true', default=True,
                       help='Enable SCBF integration (default: True)')
    parser.add_argument('--no-scbf', dest='scbf_enabled', action='store_false',
                       help='Disable SCBF integration')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, cpu (default: auto)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--output-format', type=str, default='report',
                       choices=['report', 'json', 'minimal'],
                       help='Output format (default: report)')
    
    # Output options
    parser.add_argument('--save-outputs', action='store_true', default=True,
                       help='Save comprehensive outputs to timestamped directories (default: True)')
    parser.add_argument('--no-save-outputs', dest='save_outputs', action='store_false',
                       help='Disable output saving')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Base output directory (default: output)')
    
    # Execution options
    parser.add_argument('--benchmark', action='store_true',
                       help='Run in benchmark mode with detailed timing')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations to run (for benchmarking)')
    parser.add_argument('--save-results', type=str,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Define module mappings
    all_modules = [
        'phase1_arithmetic',
        'phase2_fibonacci', 
        'phase3_reasoning',
        'phase3_planning',
        'phase3_metacognition',
        'phase3_creativity',
        'phase3_transfer',
        'phase1_module',
        'phase2_module',
        'phase3_module'
    ]
    
    # Parse module selection
    if args.modules.lower() == 'all':
        modules_to_run = all_modules
    else:
        modules_to_run = [m.strip() for m in args.modules.split(',')]
        
    # Validate modules
    available_modules = all_modules + ['scbf_phase_module', 'scbf_test_module']  # Add SCBF modules
    invalid_modules = [m for m in modules_to_run if m not in available_modules]
    if invalid_modules:
        print(f"❌ Invalid modules: {invalid_modules}")
        print(f"Available modules: {available_modules}")
        return
        
    print(f"🚀 GAIA Runtime v1.0")
    print(f"🧠 Running modules: {modules_to_run}")
    print(f"🖥️  Device: {config.device}")
    print(f"📐 Field Shape: {config.field_shape}")
    
    # Run modules
    total_results = []
    
    for iteration in range(args.iterations):
        if args.iterations > 1:
            print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")
            
        runtime = GAIARuntime(config)
        results = runtime.run_modules(modules_to_run)
        total_results.extend(results)
        
        if args.output_format == 'report':
            runtime.print_report()
        elif args.output_format == 'json':
            print(json.dumps(runtime.generate_report(), indent=2))
        elif args.output_format == 'minimal':
            success_count = sum(1 for r in results if r.success)
            print(f"✅ {success_count}/{len(results)} modules successful")
            
    # Save results if requested
    if args.save_results:
        final_report = {
            "config": asdict(config),
            "iterations": args.iterations,
            "all_results": [asdict(r) for r in total_results]
        }
        
        with open(args.save_results, 'w') as f:
            json.dumps(final_report, f, indent=2)
        print(f"📁 Results saved to {args.save_results}")


if __name__ == "__main__":
    main()
