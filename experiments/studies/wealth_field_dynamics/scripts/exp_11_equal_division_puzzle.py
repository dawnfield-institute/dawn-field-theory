#!/usr/bin/env python3
"""
Experiment 16: The Equal Division Puzzle - Tension with φ-Ratio Prediction

BACKGROUND FROM LITERATURE:
─────────────────────────────────────────────────────────────────────────────
The "Equal Division Puzzle" (Bernheim & Severinov, 2003):

    Most parents divide estates EQUALLY among children (~50/50 each for 2 kids)
    This is "puzzling" because economic theory predicts unequal division in many cases
    
Key papers:
    - "Bequests as signals" (JPE 2003) - explains equal division as signaling love
    - "Why parents play favorites" (Light & McGarry, AER 2004)
    - "The intra-family division of bequests" (Hamaaki et al., J Pop Econ 2019)

EMPIRICAL FINDINGS:
    - 70-85% of parents intend to divide estates equally among children
    - Unequal division often compensates for prior inter-vivo transfers
    - Equal division persists even when children have different needs
─────────────────────────────────────────────────────────────────────────────

PAC PREDICTION:
    φ-ratio splits (61.8%/38.2%) emerge from recursive balance
    This is DIFFERENT from equal division (50%/50%)
    
THE TENSION:
    If PAC dynamics drive wealth transfer, we'd expect φ-ratio (~1.618:1)
    But empirical evidence shows ~1:1 (equal division)
    
POSSIBLE RESOLUTIONS:
    1. PAC operates at AGGREGATE level, not individual transfers
       - Individual splits can be 50/50
       - But aggregate redistribution shows φ-ratio patterns
    
    2. Equal division is culturally IMPOSED, suppressing natural dynamics
       - Legal frameworks enforce/encourage equality
       - Social norms override natural asymmetry
       - This "suppression" might create stress that releases elsewhere
    
    3. PAC describes EMERGENT structure, not conscious decisions
       - Individual bequests: conscious → equal
       - Market forces: unconscious → φ-ratio
       
    4. Time horizon matters
       - Short-term: equal division
       - Long-term accumulation: φ-ratio emerges over generations

DATA NEEDED:
    1. IRS Table 4: Bequests by Beneficiary Type (2020-2023 available)
    2. Survey of Consumer Finances (SCF) - PSID linked data
    3. Multi-generational wealth tracking (rare but valuable)
    4. Business succession data (might show different patterns)

This experiment documents the tension and outlines proper tests.
"""

import json
from datetime import datetime
import numpy as np
from constants import print_header, print_subheader, PHI

PHI_INV = 1 / PHI


def document_equal_division_puzzle():
    """Document what the literature says about inheritance patterns."""
    
    print_header("EXPERIMENT 16: THE EQUAL DIVISION PUZZLE")
    
    print_subheader("PART 1: EMPIRICAL EVIDENCE FROM LITERATURE")
    
    print(f"""
    FROM THE LITERATURE (Bernheim & Severinov 2003, Light & McGarry 2004):
    ───────────────────────────────────────────────────────────────────
    
    KEY FINDING: Most parents divide estates EQUALLY among children
    
    Empirical statistics:
        - 70-85% of parents report intention to divide equally
        - Even when children have different financial needs
        - Even when some children have been more attentive/caring
    
    This is called the "Equal Division Puzzle" because:
        - Altruistic model predicts giving more to needier child
        - Exchange model predicts giving more to more attentive child
        - Yet equal division persists
    
    EXPLANATIONS PROPOSED:
        1. Signaling: Equal division signals equal love
        2. Conflict avoidance: Unequal = family conflict
        3. Fairness norms: Equality feels "right"
        4. Legal defaults: Many legal systems default to equal
    ───────────────────────────────────────────────────────────────────
    """)
    
    print_subheader("PART 2: TENSION WITH φ-RATIO PREDICTION")
    
    phi_major = PHI_INV  # ≈ 0.618
    phi_minor = 1 - PHI_INV  # ≈ 0.382
    phi_ratio = phi_major / phi_minor  # ≈ 1.618
    
    print(f"""
    PAC PREDICTS φ-RATIO SPLITS:
    ───────────────────────────────────────────────────────────────────
    
    From PAC (Potential-Actualization Conservation):
        f(Parent) = Σf(Children)
        
    Optimal split at φ-ratio:
        Major share: φ⁻¹ = {phi_major:.4f} (61.8%)
        Minor share: 1 - φ⁻¹ = {phi_minor:.4f} (38.2%)
        Ratio: {phi_ratio:.4f}:1
    
    EMPIRICAL DATA SHOWS:
        Typical split: 50/50 (ratio 1:1)
        
    QUANTIFYING THE DISCREPANCY:
        Expected ratio (PAC): {phi_ratio:.3f}
        Observed ratio (equal): 1.000
        Difference: {(phi_ratio - 1) * 100:.1f}%
    ───────────────────────────────────────────────────────────────────
    """)
    
    return {
        'pac_major_share': phi_major,
        'pac_minor_share': phi_minor,
        'pac_ratio': phi_ratio,
        'empirical_ratio': 1.0,
        'discrepancy_pct': (phi_ratio - 1) * 100
    }


def analyze_possible_resolutions():
    """Analyze possible resolutions to the tension."""
    
    print_subheader("PART 3: POSSIBLE RESOLUTIONS")
    
    resolutions = []
    
    # Resolution 1: Scale matters
    res1 = {
        'name': 'SCALE SEPARATION',
        'hypothesis': 'PAC operates at aggregate level, not individual transfers',
        'prediction': 'Individual splits: 50/50; Aggregate wealth distribution: φ-ratio',
        'test': 'Compare individual bequest ratios vs. wealth quintile ratios',
        'data_needed': 'IRS bequests + Fed wealth distribution'
    }
    
    # Resolution 2: Suppression
    res2 = {
        'name': 'CULTURAL SUPPRESSION',
        'hypothesis': 'Equal division norms suppress natural φ dynamics',
        'prediction': 'Societies with weaker equal-division norms show φ-patterns',
        'test': 'Compare cultures with primogeniture history vs egalitarian',
        'data_needed': 'Cross-cultural inheritance data'
    }
    
    # Resolution 3: Conscious vs Emergent
    res3 = {
        'name': 'CONSCIOUS vs EMERGENT',
        'hypothesis': 'PAC describes emergent patterns, not conscious decisions',
        'prediction': 'Market forces show φ-ratio; deliberate transfers show 50/50',
        'test': 'Compare inheritance vs. market wealth accumulation patterns',
        'data_needed': 'Decomposed wealth sources (inherited vs earned)'
    }
    
    # Resolution 4: Time horizon
    res4 = {
        'name': 'TEMPORAL ACCUMULATION',
        'hypothesis': 'φ-ratio emerges over multiple generations',
        'prediction': 'Single-generation: 50/50; Multi-generation: φ-ratio',
        'test': 'Track wealth concentration over multiple inheritance cycles',
        'data_needed': 'Multi-generational wealth tracking (rare data)'
    }
    
    resolutions = [res1, res2, res3, res4]
    
    for res in resolutions:
        print(f"""
    {res['name']}:
    ─────────────────────────────────────────────────────────────
    Hypothesis: {res['hypothesis']}
    
    Prediction: {res['prediction']}
    
    Test: {res['test']}
    
    Data needed: {res['data_needed']}
    """)
    
    return resolutions


def identify_testable_predictions():
    """Identify specific testable predictions."""
    
    print_subheader("PART 4: TESTABLE PREDICTIONS")
    
    predictions = [
        {
            'id': 'P1',
            'statement': 'Business succession shows φ-ratio more than personal inheritance',
            'rationale': 'Business has functional requirements that might favor asymmetric division',
            'data': 'Family business succession patterns',
            'metric': 'Ratio of shares to different heirs'
        },
        {
            'id': 'P2',
            'statement': 'Inter-vivo gifts deviate from equality more than bequests',
            'rationale': 'Living gifts are less visible, less bound by equality norms',
            'data': 'SCF inter-vivo transfer data',
            'metric': 'Gini of gifts vs. bequests'
        },
        {
            'id': 'P3',
            'statement': 'Aggregate wealth distribution follows φ-ratio at quintile boundaries',
            'rationale': 'Emergence from many equal-split transactions',
            'data': 'Fed Distributional Financial Accounts',
            'metric': 'Ratios between adjacent quintile shares'
        },
        {
            'id': 'P4',
            'statement': 'Countries with primogeniture history have different wealth dynamics',
            'rationale': 'Historical φ-like patterns might persist in aggregate',
            'data': 'Cross-country wealth inequality (WID.world)',
            'metric': 'Top 1%/10% wealth share ratios'
        }
    ]
    
    print("""
    TESTABLE PREDICTIONS (Falsifiable):
    ═══════════════════════════════════════════════════════════════════
    """)
    
    for p in predictions:
        print(f"""
    [{p['id']}] {p['statement']}
    
        Rationale: {p['rationale']}
        Data source: {p['data']}
        Metric: {p['metric']}
    """)
    
    return predictions


def document_data_sources():
    """Document available data sources."""
    
    print_subheader("PART 5: AVAILABLE DATA SOURCES")
    
    print("""
    IRS ESTATE TAX DATA:
    ─────────────────────────────────────────────────────────────────
    URL: https://www.irs.gov/statistics/soi-tax-stats-estate-tax-filing-year-tables
    
    KEY TABLE:
        Table 4: Bequests Reported for Estate Tax Returns, by Beneficiary Type
        Available: 2020, 2021, 2022, 2023 (Excel format)
    
    LIMITATION:
        - Only estates above filing threshold (~$12M in 2023)
        - Aggregated by beneficiary TYPE (spouse, children, charity)
        - Does NOT show division among multiple children
    ─────────────────────────────────────────────────────────────────
    
    FEDERAL RESERVE DATA:
    ─────────────────────────────────────────────────────────────────
    Distributional Financial Accounts (DFA)
    URL: https://www.federalreserve.gov/releases/z1/dataviz/dfa/
    
    KEY DATA:
        - Wealth by percentile since 1989
        - Downloadable CSV
    
    USEFUL FOR:
        - Testing aggregate φ-ratio hypothesis (P3)
        - Historical wealth concentration trends
    ─────────────────────────────────────────────────────────────────
    
    WORLD INEQUALITY DATABASE:
    ─────────────────────────────────────────────────────────────────
    URL: https://wid.world/
    
    KEY DATA:
        - Cross-country wealth and income distribution
        - Historical series back to 1900s for some countries
    
    USEFUL FOR:
        - Cross-cultural comparison (P4)
        - Primogeniture vs egalitarian comparison
    ─────────────────────────────────────────────────────────────────
    
    ACADEMIC MICRODATA (Limited access):
    ─────────────────────────────────────────────────────────────────
    - Survey of Consumer Finances (SCF) - inheritance questions
    - Panel Study of Income Dynamics (PSID) - wealth transfers
    - Health and Retirement Study (HRS) - bequest expectations
    
    These contain actual inheritance amounts by household
    but require data access agreements
    ─────────────────────────────────────────────────────────────────
    """)


def compute_aggregate_phi_test():
    """Test if aggregate wealth distribution shows φ-ratio patterns."""
    
    print_subheader("PART 6: PRELIMINARY AGGREGATE TEST")
    
    # Approximate US wealth shares by quintile (2023 data from Fed)
    # These are rough approximations
    wealth_shares = {
        'bottom_50': 2.6,
        'next_40': 27.6,  # 50-90th percentile
        'top_10': 69.8,
        'top_1': 31.4
    }
    
    # Test ratios between segments
    ratios = {
        'top10_to_next40': wealth_shares['top_10'] / wealth_shares['next_40'],
        'top1_to_next9': wealth_shares['top_1'] / (wealth_shares['top_10'] - wealth_shares['top_1']),
        'next40_to_bottom50': wealth_shares['next_40'] / wealth_shares['bottom_50']
    }
    
    print(f"""
    US WEALTH DISTRIBUTION (approx 2023):
    ─────────────────────────────────────────────────────────────────
    Bottom 50%:  {wealth_shares['bottom_50']:.1f}%
    Next 40%:    {wealth_shares['next_40']:.1f}%
    Top 10%:     {wealth_shares['top_10']:.1f}%
    Top 1%:      {wealth_shares['top_1']:.1f}%
    
    RATIOS BETWEEN SEGMENTS:
    ─────────────────────────────────────────────────────────────────
    Top 10% / Next 40%:     {ratios['top10_to_next40']:.3f}  (φ = 1.618)
    Top 1% / Next 9%:       {ratios['top1_to_next9']:.3f}   (φ = 1.618)
    Next 40% / Bottom 50%:  {ratios['next40_to_bottom50']:.3f} (φ = 1.618)
    
    ANALYSIS:
    ─────────────────────────────────────────────────────────────────
    Top10/Next40 ratio {ratios['top10_to_next40']:.3f} is close to φ² = {PHI**2:.3f}
    Top1/Next9 ratio {ratios['top1_to_next9']:.3f} is less than φ
    
    These are ROUGH comparisons - need proper statistical tests.
    The question is whether these ratios are CLOSER to φ-powers
    than random or closer to integer ratios (2:1, 3:1, etc.)
    ─────────────────────────────────────────────────────────────────
    """)
    
    return {
        'wealth_shares': wealth_shares,
        'ratios': ratios,
        'phi': PHI,
        'phi_squared': PHI**2
    }


def main():
    """Run the analysis."""
    
    tension = document_equal_division_puzzle()
    resolutions = analyze_possible_resolutions()
    predictions = identify_testable_predictions()
    document_data_sources()
    aggregate_test = compute_aggregate_phi_test()
    
    print_subheader("SUMMARY")
    
    print(f"""
    ═══════════════════════════════════════════════════════════════════
    EXPERIMENT 16: THE EQUAL DIVISION PUZZLE
    ═══════════════════════════════════════════════════════════════════
    
    KEY TENSION:
        PAC predicts: φ-ratio splits (61.8%/38.2%)
        Literature shows: Equal splits (50%/50%) in individual bequests
    
    THIS IS NOT NECESSARILY FALSIFICATION:
        - PAC may operate at aggregate, not individual level
        - Cultural norms may suppress natural dynamics
        - Conscious decisions may differ from emergent patterns
        - Multi-generational effects may differ from single transfers
    
    NEXT STEPS:
        1. Download IRS Table 4 data (bequests by beneficiary type)
        2. Analyze Fed DFA data for quintile ratios
        3. Compare cross-cultural wealth patterns (WID.world)
        4. Seek business succession data (may show φ-ratio)
    
    STATUS: TENSION IDENTIFIED - REQUIRES EMPIRICAL RESOLUTION
    ═══════════════════════════════════════════════════════════════════
    """)
    
    results = {
        'experiment': 'exp_16_equal_division_puzzle',
        'timestamp': datetime.now().isoformat(),
        'tension': tension,
        'resolutions_proposed': len(resolutions),
        'predictions': len(predictions),
        'aggregate_test': aggregate_test,
        'status': 'TENSION_IDENTIFIED',
        'next_steps': [
            'Download IRS Table 4 data',
            'Analyze Fed DFA quintile ratios',
            'Cross-cultural comparison via WID.world',
            'Seek business succession data'
        ]
    }
    
    # Save results
    results_file = f'results/exp_16_equal_division_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")
    except Exception as e:
        print(f"\nCould not save results: {e}")
    
    return results


if __name__ == "__main__":
    import json
    main()
