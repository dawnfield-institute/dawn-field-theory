# TinyCIMM-Euler Publication Suite

## Overview

The Publication Suite provides a comprehensive framework for conducting publication-quality experiments with the TinyCIMM-Euler model. It includes structured logging, advanced visualizations, detailed analysis, and automated report generation to support research documentation and reproducibility.

## Features

### Core Components

1. **Main Runner** (`run_publication_experiments.py`)
   - Command-line interface for experiment execution
   - Configuration-driven experiment setup
   - Automated pipeline orchestration
   - Multi-experiment batch processing

2. **Visualization Modules** (`visualizations/`)
   - **Entropy Collapse Overlay**: Visualizes information collapse and role assignment
   - **Activation Overlays**: Shows field topology and structural emergence
   - **Neuron Trace Analysis**: Analyzes specialization and role dynamics
   - **Convergence Timeline**: Tracks symbolic convergence and abstraction progression

3. **Analysis Module** (`analysis/`)
   - **SCBF Summary Generator**: Comprehensive SCBF and activation trace analysis
   - Structure counting and complexity measurement
   - Bias detection and field topology mapping
   - Convergence metrics and stability analysis

4. **Logging System** (`logging/`)
   - **Publication Logger**: Structured logging to multiple formats (CSV, JSON, TXT)
   - Experiment metadata tracking
   - Automated report generation
   - Performance metrics logging

5. **Configuration System** (`config/`)
   - Pre-defined experiment configurations
   - Template-based setup for different experiment types
   - Flexible parameter customization

## Installation and Setup

### Prerequisites

```bash
# Required Python packages
pip install numpy pandas matplotlib seaborn scikit-learn scipy pyyaml
```

### Directory Structure

```
publication_suite/
├── run_publication_experiments.py    # Main runner
├── config/
│   └── experiment_configs.yaml       # Configuration templates
├── visualizations/
│   ├── entropy_collapse_overlay.py   # Entropy visualization
│   ├── activation_overlays.py        # Activation field visualization
│   ├── neuron_trace_analysis.py      # Neuron trace analysis
│   └── convergence_timeline.py       # Convergence visualization
├── analysis/
│   └── scbf_summary_generator.py     # SCBF analysis
├── logging/
│   └── publication_logger.py         # Structured logging
└── README.md                         # This file
```

## Usage

### Basic Usage

```bash
# Run a single experiment with default configuration
python run_publication_experiments.py --experiment-type prime

# Run with custom configuration
python run_publication_experiments.py --config config/experiment_configs.yaml --experiment-type fibonacci

# Run multiple experiments
python run_publication_experiments.py --experiment-type prime,fibonacci,polynomial
```

### Command Line Arguments

- `--experiment-type`: Type of experiment to run (prime, fibonacci, polynomial, recursive, algebraic)
- `--config`: Path to configuration file (default: config/experiment_configs.yaml)
- `--output-dir`: Output directory for results (default: publication_outputs)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--formats`: Output formats (json, csv, txt, all)
- `--replications`: Number of experiment replications
- `--batch-mode`: Run in batch mode (no interactive prompts)

### Configuration

The publication suite uses YAML configuration files for flexible experiment setup. Key configuration sections:

```yaml
base_config:
  experiment_id: "exp_001"
  experiment_type: "prime_sequence"
  output_dir: "publication_outputs"
  
  logging:
    level: "INFO"
    formats: ["json", "txt", "csv"]
    
  visualization:
    save_figures: true
    figure_format: "png"
    figure_dpi: 300
    style: "publication"
    
  analysis:
    scbf_analysis: true
    convergence_analysis: true
    topology_mapping: true
    
  experiment:
    domain: "prime"
    sequence_length: 100
    network_size: 10
    iterations: 1000
```

### Programmatic Usage

```python
from publication_suite.run_publication_experiments import PublicationExperimentRunner
from publication_suite.logging.publication_logger import create_experiment_logger

# Create experiment runner
runner = PublicationExperimentRunner(
    config_path="config/experiment_configs.yaml",
    output_dir="my_experiments"
)

# Run experiment
results = runner.run_single_experiment(
    experiment_type="prime",
    config_overrides={"experiment.iterations": 2000}
)

# Or use individual components
with create_experiment_logger("my_exp_001", "prime_sequence") as logger:
    logger.info("Starting custom experiment")
    # Your experiment code here
    logger.log_metrics({"accuracy": 0.95}, "results")
```

## Output Structure

The publication suite generates organized outputs in the following structure:

```
publication_outputs/
├── visualizations/
│   ├── entropy_collapse/
│   ├── activation_overlays/
│   ├── neuron_traces/
│   └── convergence_timelines/
├── analysis/
│   ├── scbf_summaries/
│   ├── topology_maps/
│   └── convergence_metrics/
├── logs/
│   ├── experiment_logs.json
│   ├── experiment_logs.csv
│   └── experiment_logs.txt
├── reports/
│   ├── summary_reports/
│   └── detailed_analysis/
└── data/
    ├── raw_data/
    └── processed_data/
```

## Experiment Types

### 1. Prime Sequence Experiments
- **Purpose**: Analyze emergence of prime number recognition
- **Key Metrics**: Pattern recognition accuracy, mathematical structure detection
- **Visualizations**: Prime detection patterns, numerical field topology

### 2. Fibonacci Sequence Experiments
- **Purpose**: Study recursive pattern learning and golden ratio emergence
- **Key Metrics**: Recursive pattern recognition, ratio convergence
- **Visualizations**: Fibonacci spiral emergence, recursive structure

### 3. Polynomial Experiments
- **Purpose**: Investigate polynomial function approximation and coefficient extraction
- **Key Metrics**: Fitting accuracy, coefficient recovery, degree detection
- **Visualizations**: Function approximation, coefficient evolution

### 4. Recursive Pattern Experiments
- **Purpose**: Analyze hierarchical pattern recognition and recursive structures
- **Key Metrics**: Recursion depth detection, pattern hierarchy
- **Visualizations**: Recursive pattern trees, hierarchy emergence

### 5. Algebraic Structure Experiments
- **Purpose**: Study algebraic rule learning and operation detection
- **Key Metrics**: Rule accuracy, operation classification, structure emergence
- **Visualizations**: Algebraic operation maps, rule evolution

## Visualization Gallery

### Entropy Collapse Overlay
Shows information collapse patterns during learning with role assignment overlays.

### Activation Overlays
Displays field topology and structural emergence patterns in the activation space.

### Neuron Trace Analysis
Analyzes individual neuron specialization and role dynamics over time.

### Convergence Timeline
Tracks symbolic convergence and abstraction progression with milestone detection.

## Analysis Capabilities

### SCBF Analysis
- **Structure Detection**: Identifies cognitive structures using clustering
- **Complexity Measurement**: Calculates information-theoretic complexity
- **Bias Detection**: Identifies systematic biases in activation patterns
- **Field Topology**: Maps cognitive field structure and connections

### Convergence Analysis
- **Stability Measurement**: Quantifies activation pattern stability
- **Convergence Rate**: Calculates speed of pattern formation
- **Quality Assessment**: Evaluates convergence quality and consistency
- **Pattern Analysis**: Identifies dominant patterns and transitions

## Logging and Reporting

### Structured Logging
- **Multiple Formats**: JSON, CSV, and text logs
- **Hierarchical Data**: Nested metrics and metadata
- **Real-time Monitoring**: Live experiment tracking
- **Error Handling**: Comprehensive exception logging

### Automated Reports
- **Summary Reports**: High-level experiment summaries
- **Detailed Analysis**: In-depth statistical analysis
- **Visualization Reports**: Integrated figure galleries
- **Reproducibility Info**: Complete configuration and environment details

## Best Practices

### For Research
1. Use descriptive experiment IDs and types
2. Document configuration changes and rationale
3. Run multiple replications for statistical significance
4. Archive complete experiment outputs
5. Include environment and dependency information

### For Publication
1. Use publication-quality visualization settings
2. Enable comprehensive analysis modules
3. Generate detailed reports with statistical measures
4. Include reproducibility information
5. Archive raw data and processing code

### For Development
1. Use debug configurations for testing
2. Enable verbose logging for troubleshooting
3. Use smaller parameter sets for quick iteration
4. Save intermediate results for analysis
5. Document code changes and experiments

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce sequence length or network size
2. **Slow Performance**: Disable debug logging, reduce visualization resolution
3. **File Permissions**: Ensure write access to output directory
4. **Missing Dependencies**: Install required packages using pip
5. **Configuration Errors**: Validate YAML syntax and parameter values

### Debug Mode
Enable debug mode for detailed troubleshooting:

```bash
python run_publication_experiments.py --experiment-type debug --log-level DEBUG
```

## Contributing

### Adding New Experiment Types
1. Create configuration template in `config/experiment_configs.yaml`
2. Add experiment logic to main runner
3. Update visualization modules if needed
4. Add analysis metrics for new experiment type
5. Update documentation

### Adding New Visualizations
1. Create module in `visualizations/` directory
2. Follow existing module structure and interfaces
3. Add configuration options
4. Update main runner integration
5. Add examples and documentation

### Adding New Analysis
1. Create analysis functions in `analysis/` directory
2. Follow existing data structures and interfaces
3. Add configuration options
4. Update report generation
5. Add validation and testing

## License

This publication suite is part of the Dawn Field Theory project and is licensed under the MIT License. See LICENSE.md for details.

## Citation

If you use this publication suite in your research, please cite:

```bibtex
@software{tinycimm_euler_publication_suite,
  title={TinyCIMM-Euler Publication Suite},
  author={Peter Groom},
  year={2025},
  url={https://github.com/dawn-field-theory/TinyCIMM-Euler}
}
```

## Contact

For questions, issues, or contributions, please contact the Dawn Field Theory Research Team or open an issue on the project repository.
