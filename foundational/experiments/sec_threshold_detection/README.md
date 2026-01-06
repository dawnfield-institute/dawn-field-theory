# SEC Threshold Detection Experiments

Cross-domain validation of SEC threshold detection and ξ relationships.

## Key Results

- **Threshold detection** works across diverse dynamical systems
- **ξ ≈ 1.0571** appears at phase transitions
- **A/B testing**: 1.48× faster with correct threshold, 50.96× slower with wrong
- **Combined p < 0.00001** for cross-domain ξ relationships

## Experiments

| Script | Description | Status |
|--------|-------------|--------|
| exp_01_threshold_detector.py | Core detection algorithm | ✅ |
| exp_02_lorenz_analysis.py | Lorenz attractor analysis | ✅ |
| exp_03_cross_domain_suite.py | Multi-domain validation | ✅ |
| exp_04_ab_testing.py | A/B test protocol | ✅ |

## Related Paper

See `foundational/docs/preprints/sec_threshold_detection/`
