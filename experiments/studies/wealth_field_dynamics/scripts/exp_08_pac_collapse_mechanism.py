"""
Experiment 13: PAC Collapse Mechanism in Economic Systems

PURPOSE:
    Connect the exp_12 finding (response TYPE matters, d = -6.02) to PAC collapse 
    dynamics from oscillation_attractor_dynamics/exp_24.

KEY INSIGHT:
    From exp_24, the Ξ derivation shows:
    
    1. PAC collapse = φ-splitting cascade (parent → children at 61.8/38.2)
    2. Each level accumulates "twist" = π/55
    3. At depth 55, cumulative twist = π (one Möbius half-twist)
    
    In economics:
    - Redistributive policy → PERMITS PAC collapse → wealth hierarchy reorganizes
    - Stabilizing policy → PREVENTS PAC collapse → freezes hierarchy in stressed state
    
    The question: Does the massive Cohen's d = -6.02 from exp_12 connect to the
    mathematics of permitted vs prevented collapse?

THE PAC COLLAPSE ANALOGY:
    
    Physical system:           Economic parallel:
    ─────────────────────────────────────────────────
    Parent node splits         Large fortune divides
    φ-ratio (61.8/38.2)       Inheritance/redistribution ratio
    Twist accumulation         Stress accumulation  
    Collapse permitted         Crisis/redistribution permitted
    New equilibrium            New inequality equilibrium
    
    If collapse is PREVENTED:
    - Twist continues to accumulate
    - No release of accumulated stress
    - Pressure builds beyond Ξ threshold
    
    If collapse is PERMITTED (redistributive):
    - Accumulated twist releases
    - System reorganizes at new hierarchy
    - Returns toward Ξ-balance

HYPOTHESIS:
    The Cohen's d = -6.02 for redistributive vs stabilizing responses reflects
    whether the system is allowed to execute PAC collapse dynamics.

REFERENCES:
    - oscillation_attractor_dynamics/scripts/exp_24_comprehensive_validation.py
    - milestone1/SYNTHESIS.md (Ξ derivation chain)
    - asymmetric_conservation/scripts/exp_10_xi_emergence.py
"""

import numpy as np
import json
from datetime import datetime
from constants import print_header, print_subheader, PHI, XI, F10

# PAC constants from exp_24
PHI_INV = 1 / PHI  # ≈ 0.618
WITHIN_PER_LEVEL = 2 * np.sqrt(PHI_INV * (1 - PHI_INV)) - 1  # = -0.0283
PI_OVER_55 = np.pi / 55  # ≈ 0.0571

def print_derivation_connection():
    """Show how PAC collapse mathematics connects to economic response."""
    
    print_header("EXPERIMENT 13: PAC COLLAPSE MECHANISM")
    
    print_subheader("PART 1: THE PAC COLLAPSE MATHEMATICS")
    
    print(f"""
    From oscillation_attractor_dynamics/exp_24:
    
    PAC Collapse at φ-ratio produces:
    ─────────────────────────────────────────────────────────────
    Within-level twist:    {WITHIN_PER_LEVEL:.6f} per level
                           (sibling interference REDUCES coherence)
    
    Cross-level correction: {PI_OVER_55 - WITHIN_PER_LEVEL:.6f} per level
                           (inter-branch interference AMPLIFIES)
    
    Net twist per level:    {PI_OVER_55:.6f} = π/55 = Ξ - 1
    ─────────────────────────────────────────────────────────────
    
    At depth 55 (F₁₀):
        55 × (π/55) = π = one Möbius half-twist
    
    This is the fundamental unit of PAC collapse dynamics.
    """)
    
    print_subheader("PART 2: ECONOMIC MAPPING")
    
    print("""
    Economic System State:
    ─────────────────────────────────────────────────────────────
    
    STRESS ACCUMULATION (before collapse):
        • Each year of high inequality = accumulated "twist"
        • Stress ~ (Gini/φ - 1) per year
        • Cumulative stress builds toward Ξ threshold
    
    COLLAPSE PERMITTED (redistributive response):
        • System executes PAC collapse
        • Wealth hierarchy splits at ~φ ratio
        • Accumulated twist releases
        • Returns toward equilibrium
        
    COLLAPSE PREVENTED (stabilizing response):
        • PAC collapse blocked (bailouts, monetary policy)
        • Twist continues to accumulate
        • No release mechanism
        • Eventually: larger forced collapse OR stagnation
    ─────────────────────────────────────────────────────────────
    """)

def simulate_collapse_dynamics():
    """Simulate accumulated stress under different policy regimes."""
    
    print_subheader("PART 3: COLLAPSE DYNAMICS SIMULATION")
    
    # Parameters
    years = 50
    initial_stress = 0.0
    stress_growth_rate = 0.03  # Stress accumulation per year above threshold
    
    # Three scenarios
    scenarios = {
        'no_collapse': {
            'description': 'Collapse always prevented (pure stabilization)',
            'collapse_allowed': False,
            'stress_history': [],
            'gini_history': []
        },
        'periodic_collapse': {
            'description': 'Collapse permitted when stress > Ξ (redistributive)',
            'collapse_allowed': True,
            'collapse_threshold': XI - 1,  # ~0.057
            'stress_history': [],
            'gini_history': []
        },
        'threshold_stabilize': {
            'description': 'Collapse prevented unless stress >> Ξ (delayed)',
            'collapse_allowed': True,
            'collapse_threshold': 0.15,  # Much higher threshold
            'stress_history': [],
            'gini_history': []
        }
    }
    
    # Run simulations
    np.random.seed(42)
    
    for name, scenario in scenarios.items():
        stress = initial_stress
        gini = 0.35  # Initial Gini
        
        for year in range(years):
            # Stress accumulates naturally
            stress += stress_growth_rate + np.random.normal(0, 0.01)
            
            # Gini tracks stress (rough approximation)
            gini = 0.35 + stress * 0.5
            
            # Check for collapse
            if scenario['collapse_allowed']:
                if stress > scenario['collapse_threshold']:
                    # PAC collapse: release stress, reduce Gini
                    stress = max(0, stress - PI_OVER_55)  # Release one "twist unit"
                    gini = 0.35 + stress * 0.5
            
            scenario['stress_history'].append(stress)
            scenario['gini_history'].append(gini)
    
    # Report
    print("    Scenario Outcomes after 50 years:")
    print("    " + "─" * 60)
    
    for name, scenario in scenarios.items():
        final_stress = scenario['stress_history'][-1]
        final_gini = scenario['gini_history'][-1]
        max_stress = max(scenario['stress_history'])
        
        print(f"\n    {name}:")
        print(f"      {scenario['description']}")
        print(f"      Final stress: {final_stress:.3f}")
        print(f"      Final Gini:   {final_gini:.3f}")
        print(f"      Max stress:   {max_stress:.3f}")
    
    return scenarios

def derive_response_type_prediction():
    """Derive what exp_12's Cohen's d = -6.02 means in PAC terms."""
    
    print_subheader("PART 4: INTERPRETATION OF EXP_12 RESULTS")
    
    print("""
    Exp_12 found:
        Cohen's d = -6.02 for redistributive vs stabilizing responses
        
    This massive effect size suggests:
    
    1. REDISTRIBUTIVE responses allow PAC collapse dynamics:
       - Accumulated stress released (~π/55 per collapse event)
       - Wealth hierarchy reorganizes at φ-ratio splits
       - System returns toward Ξ-balance
       
    2. STABILIZING responses block PAC collapse:
       - Stress continues to accumulate
       - No release mechanism
       - Eventually: either forced collapse or stagnation
    
    PAC INTERPRETATION:
    ─────────────────────────────────────────────────────────────
    If Ξ - 1 = π/55 is the fundamental "twist unit" of PAC collapse,
    then the Cohen's d = -6.02 might measure:
    
        d ≈ (stress_prevented - stress_released) / pooled_std
    
    A d of 6 standard deviations suggests:
        Redistributive releases ~6σ worth of accumulated stress
        Stabilizing preserves ~6σ worth of accumulated stress
    
    This is consistent with the PAC collapse model where:
    - Each prevented collapse adds ~1 twist unit to accumulated stress
    - 6σ ≈ 6 twist units ≈ 6 × (π/55) ≈ 0.34 accumulated stress
    ─────────────────────────────────────────────────────────────
    """)

def test_collapse_count_correlation():
    """Check if number of permitted collapses predicts long-term outcomes."""
    
    print_subheader("PART 5: COLLAPSE COUNT ANALYSIS")
    
    # Historical data: periods with different policy responses
    # Based on exp_12 historical_events data
    events = [
        {'period': '1945-1965', 'type': 'redistributive', 'events': 3, 'delta_gini': -0.08},
        {'period': '1965-1980', 'type': 'mixed', 'events': 2, 'delta_gini': -0.02},
        {'period': '1980-2000', 'type': 'stabilizing', 'events': 2, 'delta_gini': +0.06},
        {'period': '2000-2020', 'type': 'stabilizing', 'events': 3, 'delta_gini': +0.04},
    ]
    
    print("    Historical Periods (from exp_12 data):")
    print("    " + "─" * 60)
    
    redistrib_events = [e for e in events if e['type'] == 'redistributive']
    stabilizing_events = [e for e in events if e['type'] == 'stabilizing']
    
    avg_redistrib_delta = np.mean([e['delta_gini'] for e in redistrib_events])
    avg_stabilizing_delta = np.mean([e['delta_gini'] for e in stabilizing_events])
    
    print(f"\n    Redistributive periods:")
    for e in redistrib_events:
        print(f"      {e['period']}: {e['events']} events → ΔGini = {e['delta_gini']:+.2f}")
    print(f"      Average ΔGini: {avg_redistrib_delta:+.3f}")
    
    print(f"\n    Stabilizing periods:")
    for e in stabilizing_events:
        print(f"      {e['period']}: {e['events']} events → ΔGini = {e['delta_gini']:+.2f}")
    print(f"      Average ΔGini: {avg_stabilizing_delta:+.3f}")
    
    # Calculate effect in "twist units"
    delta_in_twist_units = (avg_stabilizing_delta - avg_redistrib_delta) / PI_OVER_55
    
    print(f"\n    PAC Interpretation:")
    print(f"      Difference in ΔGini: {avg_stabilizing_delta - avg_redistrib_delta:+.3f}")
    print(f"      In twist units (π/55): {delta_in_twist_units:.1f} twist units")
    print(f"      Expected Cohen's d at 1σ ~ 0.02: {delta_in_twist_units * 0.02 / 0.02:.1f}")

def summarize_findings():
    """Summary with proper epistemic framing."""
    
    print_subheader("PART 6: SUMMARY AND EPISTEMIC STATUS")
    
    print("""
    WHAT THIS EXPERIMENT SHOWS:
    ─────────────────────────────────────────────────────────────
    
    1. The exp_12 finding (Cohen's d = -6.02) COULD connect to PAC 
       collapse dynamics from exp_24
       
    2. The mapping is suggestive:
       - Redistributive = permits collapse = releases accumulated twist
       - Stabilizing = prevents collapse = preserves accumulated twist
       
    3. The magnitude is plausible:
       - ~6σ effect ≈ multiple twist units accumulated/released
       - Consistent with multi-decade stress accumulation
    
    EPISTEMIC STATUS: EXPLORATORY HYPOTHESIS
    ─────────────────────────────────────────────────────────────
    
    What we've done:
    ✓ Connected exp_12 response type finding to PAC mathematics
    ✓ Shown qualitative correspondence
    ✓ Identified testable mechanism (collapse permitted vs prevented)
    
    What we have NOT done:
    ✗ Proven causal connection
    ✗ Derived the Cohen's d = -6.02 from first principles
    ✗ Ruled out alternative explanations
    
    SUGGESTED NEXT STEPS:
    
    1. Test if New Deal / Great Society show φ-ratio wealth splits
       (Would confirm PAC collapse actually occurred)
       
    2. Measure stress accumulation rate empirically
       (Currently using arbitrary 0.03/year)
       
    3. Check if crisis/reform timing correlates with Ξ-crossing
       (Would validate threshold mechanism)
       
    4. Compare to standard economic models
       (Is PAC collapse just redistribution by another name?)
    ─────────────────────────────────────────────────────────────
    """)

def run_experiment():
    """Main entry point."""
    
    results = {
        'experiment': 'exp_13_pac_collapse_mechanism',
        'timestamp': datetime.now().isoformat(),
        'purpose': 'Connect exp_12 response type finding to PAC collapse dynamics',
        'constants': {
            'phi': PHI,
            'phi_inv': PHI_INV,
            'xi': XI,
            'within_per_level': WITHIN_PER_LEVEL,
            'pi_over_55': PI_OVER_55
        }
    }
    
    # Run analysis sections
    print_derivation_connection()
    
    scenarios = simulate_collapse_dynamics()
    results['simulation'] = {
        name: {
            'description': s['description'],
            'final_stress': s['stress_history'][-1],
            'final_gini': s['gini_history'][-1],
            'max_stress': max(s['stress_history'])
        }
        for name, s in scenarios.items()
    }
    
    derive_response_type_prediction()
    test_collapse_count_correlation()
    summarize_findings()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/exp_13_pac_collapse_mechanism_{timestamp}.json'
    
    import os
    os.makedirs('results', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n    Results saved to: {filename}")
    
    return results

if __name__ == '__main__':
    run_experiment()
