#!/usr/bin/env python3
"""
Experiment 12: Response Type Analysis

REFINED HYPOTHESIS FROM EXP_10:
    It's not crisis vs intervention that matters.
    It's REDISTRIBUTIVE vs STABILIZING response.
    
    Redistributive: Progressive taxes, labor protections, social spending
    Stabilizing: QE, bailouts, rate cuts (preserves existing structure)
    
PREDICTION:
    - Redistributive responses → inequality declines
    - Stabilizing responses → inequality continues
    
METHODOLOGY:
    1. Classify historical policy periods by response type
    2. Measure Gini change during each period
    3. Test if response type predicts direction
"""

import json
import os
import numpy as np
from datetime import datetime
from constants import PHI, XI, print_header, print_subheader as print_section

# Gini data
ANNUAL_GINI = {
    1920: 0.45, 1925: 0.50, 1929: 0.54, 1930: 0.53, 1935: 0.47,
    1940: 0.43, 1945: 0.42, 1950: 0.38, 1955: 0.36, 1960: 0.35,
    1965: 0.34, 1970: 0.35, 1975: 0.36, 1980: 0.37, 1985: 0.42,
    1990: 0.45, 1995: 0.47, 2000: 0.48, 2005: 0.50, 2008: 0.52,
    2010: 0.52, 2015: 0.53, 2020: 0.54, 2025: 0.55
}

# Policy periods classified by response type
POLICY_PERIODS = [
    # (start, end, name, type, notes)
    (1933, 1945, "New Deal Era", "redistributive", 
     "Progressive taxation, labor rights, social security"),
    
    (1945, 1965, "Great Compression", "redistributive",
     "High top marginal rates (90%+), strong unions, GI Bill"),
    
    (1965, 1980, "Great Society + Stagflation", "mixed",
     "Medicare/Medicaid but also inflation erosion"),
    
    (1981, 2000, "Reagan-Clinton Era", "stabilizing",
     "Tax cuts, deregulation, financial liberalization"),
    
    (2001, 2007, "Bush Tax Cuts", "stabilizing",
     "Tax cuts for high earners, housing bubble"),
    
    (2008, 2016, "QE Era", "stabilizing",
     "Massive monetary intervention, no fiscal redistribution"),
    
    (2017, 2025, "Recent Period", "stabilizing",
     "Tax cuts (2017), pandemic support favored assets"),
]


def get_gini(year):
    years = sorted(ANNUAL_GINI.keys())
    if year in ANNUAL_GINI:
        return ANNUAL_GINI[year]
    for i, y in enumerate(years[:-1]):
        if y < year < years[i+1]:
            t = (year - y) / (years[i+1] - y)
            return ANNUAL_GINI[y] + t * (ANNUAL_GINI[years[i+1]] - ANNUAL_GINI[y])
    if year < min(years):
        return ANNUAL_GINI[min(years)]
    return ANNUAL_GINI[max(years)]


def analyze_response_types():
    """Analyze how response type correlates with inequality trajectory."""
    
    print_header("EXPERIMENT 12: RESPONSE TYPE ANALYSIS")
    
    print("""
    REFINED HYPOTHESIS:
    
    Not crisis vs intervention, but RESPONSE TYPE:
    
    REDISTRIBUTIVE responses:
        - Progressive taxation
        - Labor protections
        - Social spending expansion
        → Should REDUCE or stabilize inequality
    
    STABILIZING responses:
        - QE, bailouts
        - Tax cuts
        - Financial intervention
        → Should MAINTAIN or increase inequality
        
    TEST: Does response type predict Gini trajectory?
    """)
    
    # Analyze each period
    print_section("PART 1: PERIOD-BY-PERIOD ANALYSIS")
    
    print(f"\n  Period                   | Type          | Gini Start → End | Δ/Year")
    print("  " + "-" * 75)
    
    results_by_type = {"redistributive": [], "stabilizing": [], "mixed": []}
    
    for start, end, name, ptype, notes in POLICY_PERIODS:
        gini_start = get_gini(start)
        gini_end = get_gini(end)
        delta = gini_end - gini_start
        years = end - start
        delta_per_year = delta / years * 100  # percentage points per year
        
        results_by_type[ptype].append({
            "name": name,
            "start": start,
            "end": end,
            "gini_start": gini_start,
            "gini_end": gini_end,
            "delta": delta,
            "delta_per_year": delta_per_year
        })
        
        direction = "↓" if delta < 0 else "↑" if delta > 0 else "→"
        print(f"  {name:25} | {ptype:13} | {gini_start:.2f} → {gini_end:.2f} {direction} | {delta_per_year:+.2f}%/yr")
    
    # Part 2: Aggregate by type
    print_section("PART 2: AGGREGATE BY RESPONSE TYPE")
    
    for ptype in ["redistributive", "stabilizing", "mixed"]:
        periods = results_by_type[ptype]
        if periods:
            avg_delta = np.mean([p["delta_per_year"] for p in periods])
            total_years = sum(p["end"] - p["start"] for p in periods)
            
            print(f"\n  {ptype.upper()} responses:")
            print(f"    Periods: {len(periods)}")
            print(f"    Total years: {total_years}")
            print(f"    Average Gini change: {avg_delta:+.3f}%/year")
            
            direction = "DECREASED" if avg_delta < 0 else "INCREASED" if avg_delta > 0 else "STABLE"
            print(f"    Inequality: {direction}")
    
    # Part 3: Statistical test
    print_section("PART 3: STATISTICAL COMPARISON")
    
    redist_deltas = [p["delta_per_year"] for p in results_by_type["redistributive"]]
    stabil_deltas = [p["delta_per_year"] for p in results_by_type["stabilizing"]]
    
    if redist_deltas and stabil_deltas:
        redist_mean = np.mean(redist_deltas)
        stabil_mean = np.mean(stabil_deltas)
        
        print(f"""
    Redistributive average: {redist_mean:+.3f}%/year
    Stabilizing average:    {stabil_mean:+.3f}%/year
    Difference:             {redist_mean - stabil_mean:+.3f}%/year
        """)
        
        # Effect size
        pooled_std = np.sqrt((np.var(redist_deltas) + np.var(stabil_deltas)) / 2)
        if pooled_std > 0:
            effect_size = (redist_mean - stabil_mean) / pooled_std
            print(f"    Effect size (Cohen's d): {effect_size:.2f}")
        
        if redist_mean < stabil_mean:
            print("""
    RESULT: Redistributive responses show MORE NEGATIVE Gini changes
    than stabilizing responses.
    
    This is CONSISTENT with the refined hypothesis:
    - Redistributive policy → inequality declines
    - Stabilizing policy → inequality rises
            """)
        else:
            print("""
    RESULT: Redistributive responses do NOT show more negative 
    Gini changes than stabilizing responses.
    
    This would WEAKEN the refined hypothesis.
            """)
    
    # Part 4: The mechanism
    print_section("PART 4: PROPOSED MECHANISM")
    
    print("""
    WHY might response type matter?
    
    STABILIZING responses (QE, bailouts):
        - Preserve existing asset values
        - Benefit asset holders disproportionately
        - Prevent wealth destruction that would "reset" distribution
        - Save the system AS IT IS (including inequality)
    
    REDISTRIBUTIVE responses (progressive tax, labor):
        - Transfer resources from concentrated to distributed
        - Create countervailing power (unions)
        - Provide floor that prevents extreme concentration
        - Change the system structure
    
    KEY INSIGHT:
    "Saving the economy" and "reducing inequality" may be different goals.
    Stabilizing interventions do the former, not necessarily the latter.
    
    The 1930s had BOTH a crisis AND redistribution → reset
    The 2008+ had a crisis but ONLY stabilization → no reset
    """)
    
    # Part 5: Implications
    print_section("PART 5: IMPLICATIONS (EXPLORATORY)")
    
    print("""
    IF this pattern is real (big if), it suggests:
    
    1. CRISIS ALONE IS NOT SUFFICIENT
       A crash without redistributive response may not reduce inequality.
       The 2008 crisis is evidence of this.
    
    2. STABILIZATION MAY BE INEQUALITY-PRESERVING
       QE and bailouts "save the economy" but preserve existing structure.
       This is not a criticism - it's a description of the mechanism.
    
    3. REDISTRIBUTION REQUIRES POLITICAL CHOICE
       Unlike natural corrections, redistribution requires policy will.
       It's not automatic and can be politically difficult.
    
    4. THE POST-WAR COMPRESSION WAS INTENTIONAL
       The 1945-1970 low-inequality era wasn't natural.
       It was the result of deliberate policy choices.
       It can't be recreated through monetary policy alone.
    
    CAVEATS:
    - Correlation, not causation demonstrated
    - Small sample size
    - Many confounding factors
    - Political economy not modeled
    """)
    
    # Part 6: What this means for Ξ-threshold
    print_section("PART 6: REVISITING Ξ-THRESHOLD")
    
    print(f"""
    The Ξ-threshold crossing (early 1980s) coincides with:
    
    - Shift from redistributive to stabilizing policy regime
    - Reagan tax cuts, deregulation
    - Decline of unions
    - Financialization of the economy
    
    INTERPRETATION OPTIONS:
    
    A) Ξ is causal: System crossed a threshold that changed dynamics
       → Unlikely given exp_09 (Ξ not uniquely predictive)
    
    B) Ξ is a marker: The 1980s regime change shifted both
       policy and Gini past a threshold level
       → More plausible
    
    C) Both: Threshold crossing and policy shift reinforced each other
       → Possible feedback loop
    
    The early 1980s represent a genuine regime change.
    Whether Ξ captures something fundamental or is just a convenient
    marker remains an open question.
    """)
    
    # Results
    output = {
        "experiment": "exp_12_response_type",
        "timestamp": datetime.now().isoformat(),
        "results_by_type": {
            ptype: [
                {k: float(v) if isinstance(v, (int, float)) else v 
                 for k, v in p.items()} 
                for p in periods
            ]
            for ptype, periods in results_by_type.items()
        },
        "redistributive_avg": float(np.mean(redist_deltas)) if redist_deltas else None,
        "stabilizing_avg": float(np.mean(stabil_deltas)) if stabil_deltas else None,
        "hypothesis": "Response type (redistributive vs stabilizing) predicts inequality trajectory"
    }
    
    print_section("SUMMARY")
    
    print(f"""
    REDISTRIBUTIVE responses: {redist_mean:+.3f}%/year (inequality FELL)
    STABILIZING responses:    {stabil_mean:+.3f}%/year (inequality ROSE)
    Difference:               {stabil_mean - redist_mean:.3f}%/year
    
    REFINED MODEL:
        Crises create OPPORTUNITY for reset
        Redistributive response → reset happens
        Stabilizing response → no reset, inequality preserved
    
    STATUS: Pattern observed, causation not established
    NEXT: Cross-country validation (Scandinavia vs US)
    """)
    
    return output


def main():
    results = analyze_response_types()
    
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/exp_12_response_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\n  Results saved to: {filepath}")


if __name__ == "__main__":
    main()
