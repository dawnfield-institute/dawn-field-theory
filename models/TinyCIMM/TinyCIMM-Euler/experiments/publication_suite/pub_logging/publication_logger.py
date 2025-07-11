#!/usr/bin/env python3
"""
Publication Logger
==================

This module provides structured logging capabilities for TinyCIMM-Euler experiments,
generating publication-quality logs, reports, and data exports. Supports multiple
output formats (CSV, JSON, TXT) with configurable verbosity levels and automatic
report generation for research documentation.

Author: Dawn Field Theory Research Team
Date: 2025-01-27
Version: 1.0
License: MIT (see LICENSE.md)

Key Features:
- Structured logging to CSV/JSON formats
- Configurable verbosity and output channels
- Automatic summary report generation
- Experiment metadata tracking
- Performance metrics logging
- Error and exception handling
- Publication-quality formatting
"""

import logging
import json
import csv
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import sys
import traceback
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class LogLevel(Enum):
    """Enumeration for log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """Enumeration for log output formats."""
    CSV = "csv"
    JSON = "json"
    TXT = "txt"
    ALL = "all"

@dataclass
class ExperimentMetadata:
    """Metadata structure for experiment tracking."""
    experiment_id: str
    experiment_type: str
    start_time: str
    end_time: Optional[str] = None
    duration: Optional[float] = None
    status: str = "running"
    config: Dict[str, Any] = None
    results_summary: Dict[str, Any] = None

@dataclass
class LogEntry:
    """Structure for individual log entries."""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line_number: int
    experiment_id: str
    additional_data: Optional[Dict[str, Any]] = None

class PublicationLogger:
    """
    Structured logging system for TinyCIMM-Euler experiments.
    
    This class provides comprehensive logging capabilities with multiple output formats,
    automatic report generation, and publication-quality formatting. It tracks experiment
    metadata, performance metrics, and generates structured data for research documentation.
    """
    
    def __init__(self, experiment_id: str, output_dir: str = "publication_outputs",
                 log_level: LogLevel = LogLevel.INFO, formats: List[LogFormat] = None):
        """
        Initialize the publication logger.
        
        Args:
            experiment_id: Unique identifier for the experiment
            output_dir: Directory for log outputs
            log_level: Minimum log level to record
            formats: List of output formats to use
        """
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.log_level = log_level
        self.formats = formats or [LogFormat.JSON, LogFormat.TXT]
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Initialize logging components
        self.start_time = datetime.now()
        self.log_entries: List[LogEntry] = []
        self.experiment_metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_type="unknown",
            start_time=self.start_time.isoformat(),
            config={}
        )
        
        # Set up file handlers
        self._setup_file_handlers()
        
        # Set up Python logging integration
        self._setup_python_logging()
        
        self.info(f"Publication Logger initialized for experiment: {experiment_id}")
    
    def _setup_file_handlers(self):
        """Set up file handlers for different log formats."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # CSV handler
        if LogFormat.CSV in self.formats or LogFormat.ALL in self.formats:
            self.csv_path = self.output_dir / "logs" / f"{self.experiment_id}_{timestamp}.csv"
            self.csv_fieldnames = ['timestamp', 'level', 'message', 'module', 'function', 
                                  'line_number', 'experiment_id', 'additional_data']
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                writer.writeheader()
        
        # JSON handler
        if LogFormat.JSON in self.formats or LogFormat.ALL in self.formats:
            self.json_path = self.output_dir / "logs" / f"{self.experiment_id}_{timestamp}.json"
            
        # TXT handler
        if LogFormat.TXT in self.formats or LogFormat.ALL in self.formats:
            self.txt_path = self.output_dir / "logs" / f"{self.experiment_id}_{timestamp}.txt"
    
    def _setup_python_logging(self):
        """Set up integration with Python's logging module."""
        # Create custom handler
        class PublicationHandler(logging.Handler):
            def __init__(self, pub_logger):
                super().__init__()
                self.pub_logger = pub_logger
            
            def emit(self, record):
                # Convert Python log record to our format
                level_map = {
                    logging.DEBUG: LogLevel.DEBUG,
                    logging.INFO: LogLevel.INFO,
                    logging.WARNING: LogLevel.WARNING,
                    logging.ERROR: LogLevel.ERROR,
                    logging.CRITICAL: LogLevel.CRITICAL
                }
                
                level = level_map.get(record.levelno, LogLevel.INFO)
                self.pub_logger._log(level, record.getMessage(), record.module, 
                                   record.funcName, record.lineno)
        
        # Add handler to root logger
        self.python_handler = PublicationHandler(self)
        logging.getLogger().addHandler(self.python_handler)
        
        # Set logging level
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        logging.getLogger().setLevel(level_map[self.log_level])
    
    def set_experiment_metadata(self, experiment_type: str, config: Dict[str, Any]):
        """Set experiment metadata."""
        self.experiment_metadata.experiment_type = experiment_type
        self.experiment_metadata.config = config
        self.info(f"Experiment metadata set: {experiment_type}")
    
    def debug(self, message: str, additional_data: Dict[str, Any] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, additional_data=additional_data)
    
    def info(self, message: str, additional_data: Dict[str, Any] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, additional_data=additional_data)
    
    def warning(self, message: str, additional_data: Dict[str, Any] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, additional_data=additional_data)
    
    def error(self, message: str, additional_data: Dict[str, Any] = None):
        """Log error message."""
        self._log(LogLevel.ERROR, message, additional_data=additional_data)
    
    def critical(self, message: str, additional_data: Dict[str, Any] = None):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, additional_data=additional_data)
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with full traceback."""
        exc_info = sys.exc_info()
        tb_str = ''.join(traceback.format_exception(*exc_info))
        
        message = f"Exception in {context}: {str(exception)}"
        additional_data = {
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'traceback': tb_str,
            'context': context
        }
        
        self.error(message, additional_data)
    
    def log_metrics(self, metrics: Dict[str, Union[int, float, str]], category: str = "metrics"):
        """Log metrics data."""
        message = f"Metrics update: {category}"
        additional_data = {
            'category': category,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(message, additional_data)
    
    def log_array_data(self, data: np.ndarray, name: str, summary_stats: bool = True):
        """Log array data with optional summary statistics."""
        message = f"Array data logged: {name}"
        additional_data = {
            'name': name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size
        }
        
        if summary_stats:
            additional_data['summary_stats'] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data))
            }
        
        self.info(message, additional_data)
    
    def log_experiment_phase(self, phase: str, details: Dict[str, Any] = None):
        """Log experiment phase transitions."""
        message = f"Experiment phase: {phase}"
        additional_data = {
            'phase': phase,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(message, additional_data)
    
    def _log(self, level: LogLevel, message: str, module: str = None, 
             function: str = None, line_number: int = None, additional_data: Dict[str, Any] = None):
        """Internal logging method."""
        # Check if we should log this level
        level_priorities = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        
        if level_priorities[level] < level_priorities[self.log_level]:
            return
        
        # Get caller info if not provided
        if module is None or function is None or line_number is None:
            import inspect
            frame = inspect.currentframe().f_back.f_back
            if frame:
                module = frame.f_globals.get('__name__', 'unknown')
                function = frame.f_code.co_name
                line_number = frame.f_lineno
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            message=message,
            module=module or 'unknown',
            function=function or 'unknown',
            line_number=line_number or 0,
            experiment_id=self.experiment_id,
            additional_data=additional_data
        )
        
        self.log_entries.append(entry)
        
        # Write to files
        self._write_to_files(entry)
        
        # Print to console if appropriate
        if level_priorities[level] >= level_priorities[LogLevel.INFO]:
            print(f"[{entry.timestamp}] {level.value}: {message}")
    
    def _write_to_files(self, entry: LogEntry):
        """Write log entry to configured file formats."""
        try:
            # Write to CSV
            if LogFormat.CSV in self.formats or LogFormat.ALL in self.formats:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                    entry_dict = asdict(entry)
                    # Convert additional_data to string for CSV
                    if entry_dict['additional_data']:
                        entry_dict['additional_data'] = json.dumps(entry_dict['additional_data'])
                    writer.writerow(entry_dict)
            
            # Write to JSON (append to list)
            if LogFormat.JSON in self.formats or LogFormat.ALL in self.formats:
                # Read existing data
                try:
                    with open(self.json_path, 'r') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = []
                
                # Append new entry
                data.append(asdict(entry))
                
                # Write back
                with open(self.json_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            # Write to TXT
            if LogFormat.TXT in self.formats or LogFormat.ALL in self.formats:
                with open(self.txt_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{entry.timestamp}] {entry.level} - {entry.module}.{entry.function}:{entry.line_number}\n")
                    f.write(f"  {entry.message}\n")
                    if entry.additional_data:
                        f.write(f"  Additional data: {json.dumps(entry.additional_data, indent=2)}\n")
                    f.write("\n")
        
        except Exception as e:
            # Fallback to console if file writing fails
            print(f"Failed to write log entry to file: {e}")
    
    def generate_summary_report(self, results: Dict[str, Any] = None) -> str:
        """Generate a comprehensive summary report."""
        self.experiment_metadata.end_time = datetime.now().isoformat()
        self.experiment_metadata.duration = (datetime.now() - self.start_time).total_seconds()
        self.experiment_metadata.status = "completed"
        
        if results:
            self.experiment_metadata.results_summary = results
        
        # Generate report
        report = self._create_summary_report()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / "reports" / f"{self.experiment_id}_summary_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.info(f"Summary report generated: {report_path}")
        return report
    
    def _create_summary_report(self) -> str:
        """Create formatted summary report."""
        # Count log entries by level
        level_counts = {}
        for entry in self.log_entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        # Calculate statistics
        total_entries = len(self.log_entries)
        error_count = level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)
        
        report = f"""
TinyCIMM-Euler Experiment Summary Report
=======================================

Experiment ID: {self.experiment_metadata.experiment_id}
Experiment Type: {self.experiment_metadata.experiment_type}
Start Time: {self.experiment_metadata.start_time}
End Time: {self.experiment_metadata.end_time}
Duration: {self.experiment_metadata.duration:.2f} seconds
Status: {self.experiment_metadata.status}

Logging Statistics
------------------
Total Log Entries: {total_entries}
Debug: {level_counts.get('DEBUG', 0)}
Info: {level_counts.get('INFO', 0)}
Warning: {level_counts.get('WARNING', 0)}
Error: {level_counts.get('ERROR', 0)}
Critical: {level_counts.get('CRITICAL', 0)}

Error Rate: {(error_count/total_entries*100):.1f}% ({error_count}/{total_entries})

Experiment Configuration
------------------------
{json.dumps(self.experiment_metadata.config, indent=2)}

Results Summary
---------------
{json.dumps(self.experiment_metadata.results_summary, indent=2) if self.experiment_metadata.results_summary else 'No results summary available'}

Recent Log Entries (Last 10)
-----------------------------
"""
        
        # Add recent log entries
        recent_entries = self.log_entries[-10:]
        for entry in recent_entries:
            report += f"[{entry.timestamp}] {entry.level}: {entry.message}\n"
        
        report += f"""

Files Generated
---------------
"""
        
        # List generated files
        if hasattr(self, 'csv_path'):
            report += f"CSV Log: {self.csv_path}\n"
        if hasattr(self, 'json_path'):
            report += f"JSON Log: {self.json_path}\n"
        if hasattr(self, 'txt_path'):
            report += f"Text Log: {self.txt_path}\n"
        
        return report
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export all log entries to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_id}_export_{timestamp}.csv"
        
        export_path = self.output_dir / "data" / filename
        
        # Convert log entries to DataFrame
        data = []
        for entry in self.log_entries:
            row = asdict(entry)
            # Flatten additional_data
            if row['additional_data']:
                for key, value in row['additional_data'].items():
                    row[f'data_{key}'] = value
            del row['additional_data']
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(export_path, index=False)
        
        self.info(f"Data exported to CSV: {export_path}")
        return str(export_path)
    
    def export_to_json(self, filename: str = None) -> str:
        """Export all data to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_id}_export_{timestamp}.json"
        
        export_path = self.output_dir / "data" / filename
        
        export_data = {
            'metadata': asdict(self.experiment_metadata),
            'log_entries': [asdict(entry) for entry in self.log_entries],
            'statistics': {
                'total_entries': len(self.log_entries),
                'level_counts': {},
                'duration': self.experiment_metadata.duration
            }
        }
        
        # Calculate level counts
        for entry in self.log_entries:
            level = entry.level
            export_data['statistics']['level_counts'][level] = \
                export_data['statistics']['level_counts'].get(level, 0) + 1
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.info(f"Data exported to JSON: {export_path}")
        return str(export_path)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        level_counts = {}
        for entry in self.log_entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        return {
            'total_entries': len(self.log_entries),
            'level_counts': level_counts,
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'error_rate': (level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)) / len(self.log_entries) if self.log_entries else 0
        }
    
    def close(self):
        """Close the logger and finalize outputs."""
        self.info("Closing publication logger")
        
        # Remove Python logging handler
        if hasattr(self, 'python_handler'):
            logging.getLogger().removeHandler(self.python_handler)
        
        # Generate final report
        self.generate_summary_report()
        
        # Export final data
        self.export_to_csv()
        self.export_to_json()
        
        self.info("Publication logger closed successfully")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.log_exception(exc_val, "Context manager exit")
        self.close()

# Convenience functions
def create_experiment_logger(experiment_id: str, experiment_type: str = "unknown",
                           config: Dict[str, Any] = None, **kwargs) -> PublicationLogger:
    """Create a configured publication logger for an experiment."""
    logger = PublicationLogger(experiment_id, **kwargs)
    logger.set_experiment_metadata(experiment_type, config or {})
    return logger

def log_experiment_results(logger: PublicationLogger, results: Dict[str, Any]):
    """Log experiment results and generate final report."""
    logger.log_metrics(results, "final_results")
    logger.generate_summary_report(results)

if __name__ == "__main__":
    # Example usage
    print("Publication Logger - Example Usage")
    
    # Create logger
    with create_experiment_logger(
        experiment_id="example_exp_001",
        experiment_type="demonstration",
        config={"param1": 42, "param2": "test"},
        output_dir="example_logs"
    ) as logger:
        
        # Log various types of messages
        logger.info("Starting example experiment")
        logger.debug("Debug information")
        logger.warning("This is a warning")
        
        # Log metrics
        logger.log_metrics({"accuracy": 0.95, "loss": 0.05}, "training")
        
        # Log experiment phase
        logger.log_experiment_phase("training", {"epoch": 1, "batch_size": 32})
        
        # Log array data
        import numpy as np
        data = np.random.rand(100, 10)
        logger.log_array_data(data, "sample_activations")
        
        # Simulate error
        try:
            raise ValueError("Example error")
        except Exception as e:
            logger.log_exception(e, "error_simulation")
        
        logger.info("Example experiment completed")
    
    print("Example logging complete!")
