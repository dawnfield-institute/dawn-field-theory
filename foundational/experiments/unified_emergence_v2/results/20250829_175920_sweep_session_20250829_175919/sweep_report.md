# Parameter Sweep Analysis Report

**Generated:** 2025-08-29 17:59:22

## Executive Summary

- **Total Runs:** 40
- **Successful Runs:** 40
- **Phase 1 Success Rate:** 87.5%
- **Best Overall Score:** 0.986

## Statistical Summary

| Metric | Mean | Std Dev | Min | Max | Median | 95% CI |
|--------|------|---------|-----|-----|--------|--------|
| Overall Score | 0.905 | 0.085 | 0.726 | 0.986 | 0.924 | [0.879, 0.932] |
| SEC Classification | 0.882 | 0.100 | 0.620 | 0.920 | 0.920 | [0.852, 0.912] |
| Pattern Assembly | 0.963 | 0.100 | 0.700 | 1.000 | 1.000 | [0.925, 0.993] |
| Emergence Consistency | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | [1.000, 1.000] |
| Phase 1 Readiness | 0.968 | 0.068 | 0.793 | 1.000 | 1.000 | [0.946, 0.988] |

## Domain Performance

| Domain | Mean Score | Std Dev | Sample Size | Status |
|--------|------------|---------|-------------|--------|
| med | 0.834 | 0.065 | 20 | [GOOD] |
| gravity | 0.977 | 0.010 | 20 | [GOOD] |

## Field Size Analysis

| Field Size | Mean Score | Std Dev | Sample Size | Status |
|------------|------------|---------|-------------|--------|
| 16 | 0.912 | 0.055 | 10 | [GOOD] |
| 32 | 0.849 | 0.129 | 10 | [GOOD] |
| 64 | 0.935 | 0.053 | 10 | [GOOD] |
| 128 | 0.926 | 0.064 | 10 | [GOOD] |

## Optimal Parameters

- **Best Configuration:** Field Size 64, Domain gravity
- **Best Score:** 0.986
- **Most Reliable Domain:** gravity
- **Most Reliable Field Size:** 64

## Parameter Correlations

- **execution_time_vs_performance:** 0.633

## Recommendations

- [EXCELLENT] High Phase 1 success rate (87.5%). Framework is performing very well.
- **Recommended Configuration:** Use field size 64 with domain gravity for most reliable results.
