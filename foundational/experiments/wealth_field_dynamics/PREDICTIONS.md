# Wealth Field Dynamics - Prediction Registry

**Created**: 2026-01-23
**Status**: Active Tracking
**Review Schedule**: Annually (January)

---

## Hypothesis

The Top10/Next40 wealth ratio in the US follows PAC equilibrium dynamics, settling at φ-harmonic values (φ, φ², φ³...).

**Observed**: 
- 1989: ratio = 1.637 (within 1.2% of φ = 1.618)
- 2025: ratio = 2.719 (within 3.8% of φ² = 2.618)

---

## Predictions

### P1: φ² Stabilization (Short-term)

**Prediction**: The Top10/Next40 ratio will stabilize around φ² (2.618) ± 10% through 2035.

| Metric | Value |
|--------|-------|
| Predicted range | 2.36 - 2.88 |
| Current value (2025) | 2.72 |
| Observation period | 2026-2035 |
| Data source | Federal Reserve DFA |

**Falsification criteria**:
- ❌ Ratio exceeds 3.0 and continues rising (trend toward φ³ = 4.24)
- ❌ Ratio drops below 2.0 (regression toward φ)
- ✅ Ratio oscillates within 2.4-2.9 range

**Status**: 🔄 Tracking

---

### P2: φ³ Transition Timing (Long-term)

**Prediction**: If the system transitions to the next equilibrium (φ³ ≈ 4.24), it will take approximately 30-40 years at ~0.5× Ξ rate.

| Metric | Value |
|--------|-------|
| Next equilibrium | φ³ = 4.236 |
| φ→φ² transition | ~35 years (1989-2024) |
| Predicted φ²→φ³ transition | ~2055-2065 |
| Rate | 0.03-0.05 per year |

**Falsification criteria**:
- ❌ Ratio reaches 4.0+ before 2045 (too fast)
- ❌ Ratio returns to φ and stabilizes (reversal)
- ✅ Gradual climb at ~0.03/year toward φ³

**Status**: 🔄 Tracking

---

### P3: Crisis = Incomplete Reconciliation

**Prediction**: Economic crises (recessions, crashes) will cause temporary ratio drops but NOT full resets to previous equilibrium.

| Event | Pre-crisis | Post-crisis | Recovery |
|-------|------------|-------------|----------|
| 2008 GFC | 2.27 (2007) | 2.21 (2010) | Resumed climb |
| COVID-19 | 2.43 (2019) | 2.59 (2022) | Accelerated |
| Next crisis | ? | ? | Predict: partial drop, no full reset |

**Falsification criteria**:
- ❌ Major crisis resets ratio to φ (1.618) and holds
- ✅ Crisis causes <15% drop, followed by resumed trajectory

**Status**: 🔄 Tracking

---

### P4: Cross-National Universality

**Prediction**: If φ-structure is fundamental (PAC), other developed economies should show similar patterns.

**Countries to test**:
- [ ] UK (comparable inequality data available)
- [ ] Germany
- [ ] France  
- [ ] Japan
- [ ] Canada
- [ ] Australia

**Falsification criteria**:
- ❌ No other country shows φ-harmonic wealth ratios
- ⚠️ Only Anglo-Saxon economies show pattern (cultural, not universal)
- ✅ Multiple diverse economies show φ-structure

**Status**: 📋 Not yet tested

---

### P5: Bottom50 is NOT PAC-Governed

**Prediction**: The Top10/Bottom50 ratio will NOT stabilize at any φ-harmonic value. It represents a pathological/non-equilibrium state.

| Current observation | Value |
|---------------------|-------|
| Top10/Bottom50 (2025) | 37.6 |
| Nearest φ-harmonic | φ⁵ ≈ 11.1 |
| Deviation | 238% |
| Rate vs Ξ | 10.4× (far outside PAC) |

**Falsification criteria**:
- ❌ Ratio stabilizes at some φⁿ value
- ✅ Ratio continues unbounded growth or erratic behavior

**Status**: 🔄 Tracking

---

## Data Collection Schedule

| Year | Action | Source |
|------|--------|--------|
| 2027 | Annual update | Fed DFA Q4 release |
| 2028 | Annual update | Fed DFA Q4 release |
| 2029 | Annual update | Fed DFA Q4 release |
| 2030 | 5-year review | Assess P1 trajectory |
| 2035 | P1 verdict | Stabilization test complete |

---

## Measurement Protocol

1. **Data source**: Federal Reserve Distributional Financial Accounts
2. **URL**: https://www.federalreserve.gov/releases/z1/dataviz/dfa/
3. **Metrics extracted**:
   - Top 1% Net Worth Share
   - Next 9% Net Worth Share  
   - Next 40% Net Worth Share
   - Bottom 50% Net Worth Share
4. **Derived ratio**: (Top1 + Next9) / Next40 = Top10/Next40

---

## Update Log

| Date | Observation | Notes |
|------|-------------|-------|
| 2026-01-23 | Registry created | Baseline: ratio = 2.72, within 3.8% of φ² |

---

## Epistemic Notes

**Strength of evidence**: Weak-to-moderate
- Pattern observed in 1 country, 1 time period
- Post-hoc identification of equilibria
- No causal mechanism specified

**Required for upgrade to "moderate"**:
- Cross-national replication
- Pre-registered prediction confirmed

**Required for upgrade to "strong"**:
- Causal mechanism identified
- Predictive success on novel data
- Theoretical derivation from first principles

---

## Related Experiments

- [exp_16_fred_dfa_analysis.py](scripts/exp_16_fred_dfa_analysis.py) - Primary analysis
- [exp_13_non_equilibrium_pressure.py](scripts/exp_13_non_equilibrium_pressure.py) - Pressure framing
- [exp_15_emergence_rate_xi.py](scripts/exp_15_emergence_rate_xi.py) - Ξ rate comparison
