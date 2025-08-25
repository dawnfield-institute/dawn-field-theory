# Information Amplification Proof Results

This directory contains experimental results from information amplification tests.

## Directory Structure
```
results/
├── pilot/           # Pilot study results
├── real_model/      # Real AI model test results  
├── cimm/           # CIMM model test results
├── validation/     # Independent verification results
└── analysis/       # Aggregate analysis and reports
```

## Result Files
Each experiment generates a timestamped JSON file containing:
- Input measurements (compressed sizes)
- Model weight measurements
- Output measurements  
- Environment profile
- Amplification calculations
- Statistical analysis

## Key Metrics
- **Surplus Bytes**: Output compressed size - (Input + Model + Overhead)
- **Amplification Ratio**: Output size / (Input + Model + Overhead)
- **Compression Efficiency**: Compressed size / Raw size

## Interpretation
- **Positive Surplus**: Indicates information amplification
- **Ratio > 1.0**: Confirms output exceeds input+model capacity
- **Consistent Results**: Multiple experiments showing amplification

Results demonstrating consistent information amplification provide evidence for computational novelty in AI systems.
