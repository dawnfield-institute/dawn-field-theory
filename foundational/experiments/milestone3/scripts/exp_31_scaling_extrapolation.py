#!/usr/bin/env python3
"""
exp_31: PAC Conservation Scaling Extrapolation

HYPOTHESIS: The Xi enrichment trend observed across Pythia model sizes
(70M→1B) follows a predictable scaling law, enabling falsifiable
predictions for 7B and 70B parameter models.

SOURCE: Paper 6 §4.4 observed data; SCBF token_pac_tree experiments
TARGET: Paper 6 - adding scaling predictions to strengthen claims

BACKGROUND:
    Paper 6 §4.4 reports Xi enrichment (ratio of Xi-proximate activations
    in trained vs random weights) across Pythia model sizes:
    
        70M:  2.8× enrichment
        160M: 2.4× enrichment
        410M: 2.1× enrichment
        1B:   1.9× enrichment
    
    The trend is monotonically decreasing: larger models show less enrichment.
    Paper 6 §10 acknowledges this gap: "This paper does not test at
    production scale."
    
    This experiment fits scaling models to the 4-point data, extrapolates
    to 7B and 70B, and produces falsifiable numeric predictions.

FALSIFICATION: If future measurements at 7B or 70B fall outside the
predicted confidence intervals, the scaling model is wrong.

METHOD:
    1. Fit power law: enrichment = a × params^b
    2. Fit logarithmic: enrichment = a × log(params) + b
    3. Fit inverse square root: enrichment = a / √params + b
    4. Compare fits by AIC/BIC
    5. Extrapolate best model to 7B, 70B, 175B with CIs
    6. Additionally predict: at what scale does enrichment → 1.0 (no signal)?
"""

import sys
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.utils import save_results

# =============================================================================
# OBSERVED DATA (Paper 6 §4.4)
# =============================================================================

# Model sizes in millions of parameters
SIZES_M = np.array([70, 160, 410, 1000])
# Xi enrichment ratios (trained/random)
ENRICHMENT = np.array([2.8, 2.4, 2.1, 1.9])

# Prediction targets (in millions)
PREDICT_SIZES = [7000, 70000, 175000]
PREDICT_LABELS = ["7B", "70B", "175B"]


# =============================================================================
# SCALING MODELS
# =============================================================================

def power_law(x, a, b):
    """enrichment = a × x^b"""
    return a * np.power(x, b)


def logarithmic(x, a, b):
    """enrichment = a × log(x) + b"""
    return a * np.log(x) + b


def inv_sqrt(x, a, b):
    """enrichment = a / √x + b"""
    return a / np.sqrt(x) + b


def asymptotic(x, a, b, c):
    """enrichment = a / (x + b) + c (approaches c asymptotically)"""
    return a / (x + b) + c


MODELS = {
    "power_law": {"func": power_law, "p0": [10, -0.1], "n_params": 2},
    "logarithmic": {"func": logarithmic, "p0": [-0.3, 4.0], "n_params": 2},
    "inv_sqrt": {"func": inv_sqrt, "p0": [10, 1.0], "n_params": 2},
    "asymptotic": {"func": asymptotic, "p0": [100, 50, 1.5], "n_params": 3},
}


# =============================================================================
# FITTING AND PREDICTION
# =============================================================================

def fit_models():
    """Fit all scaling models to the observed data."""
    print("PART 1: Model Fitting")
    print("-" * 60)
    
    results = {}
    n = len(SIZES_M)
    
    for name, spec in MODELS.items():
        try:
            popt, pcov = curve_fit(spec["func"], SIZES_M, ENRICHMENT,
                                   p0=spec["p0"], maxfev=10000)
            
            # Predictions on training data
            predicted = spec["func"](SIZES_M, *popt)
            residuals = ENRICHMENT - predicted
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ENRICHMENT - ENRICHMENT.mean())**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            # AIC/BIC with small-sample correction
            k = spec["n_params"]
            mse = ss_res / n
            # Avoid log(0)
            ll = -n / 2 * np.log(2 * np.pi * max(mse, 1e-20)) - ss_res / (2 * max(mse, 1e-20))
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            
            # Parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            
            results[name] = {
                "params": [round(float(p), 6) for p in popt],
                "param_errors": [round(float(e), 6) for e in perr],
                "r_squared": round(float(r_squared), 6),
                "ss_residuals": round(float(ss_res), 6),
                "aic": round(float(aic), 2),
                "bic": round(float(bic), 2),
                "predicted_training": [round(float(p), 4) for p in predicted],
                "residuals": [round(float(r), 4) for r in residuals],
                "max_abs_residual": round(float(np.max(np.abs(residuals))), 4),
                "fit_success": True,
            }
            
            print(f"  {name:15s}: R² = {r_squared:.4f}, AIC = {aic:.1f}, "
                  f"max |resid| = {np.max(np.abs(residuals)):.4f}")
            
        except Exception as e:
            results[name] = {
                "fit_success": False,
                "error": str(e)
            }
            print(f"  {name:15s}: FAILED — {e}")
    
    return results


def make_predictions(fit_results):
    """Extrapolate all successful models to 7B, 70B, 175B."""
    print("\nPART 2: Predictions")
    print("-" * 60)
    
    predictions = {}
    
    for name, spec in MODELS.items():
        if not fit_results[name].get("fit_success", False):
            continue
        
        popt = fit_results[name]["params"]
        
        preds = []
        for size, label in zip(PREDICT_SIZES, PREDICT_LABELS):
            try:
                value = spec["func"](size, *popt)
                
                # Bootstrap CI: perturb data and refit 1000 times
                np.random.seed(42)
                bootstrap_preds = []
                for _ in range(1000):
                    # Add Gaussian noise proportional to residual scale
                    resid_scale = fit_results[name]["max_abs_residual"]
                    noise = np.random.normal(0, max(resid_scale, 0.05), len(SIZES_M))
                    boot_data = ENRICHMENT + noise
                    try:
                        boot_popt, _ = curve_fit(spec["func"], SIZES_M, boot_data,
                                                  p0=spec["p0"], maxfev=5000)
                        boot_pred = spec["func"](size, *boot_popt)
                        if np.isfinite(boot_pred) and 0 < boot_pred < 100:
                            bootstrap_preds.append(boot_pred)
                    except:
                        pass
                
                if len(bootstrap_preds) > 10:
                    ci_low = float(np.percentile(bootstrap_preds, 2.5))
                    ci_high = float(np.percentile(bootstrap_preds, 97.5))
                else:
                    ci_low = ci_high = float('nan')
                
                preds.append({
                    "size_label": label,
                    "size_M": size,
                    "predicted_enrichment": round(float(value), 4),
                    "ci_95_low": round(ci_low, 4),
                    "ci_95_high": round(ci_high, 4),
                    "n_bootstrap": len(bootstrap_preds),
                })
                
                print(f"  {name:15s} @ {label:5s}: enrichment = {value:.3f} "
                      f"[{ci_low:.3f}, {ci_high:.3f}]")
                
            except Exception as e:
                preds.append({
                    "size_label": label,
                    "size_M": size,
                    "error": str(e)
                })
        
        predictions[name] = preds
    
    return predictions


def find_null_crossing(fit_results):
    """
    At what model size does enrichment → 1.0 (no signal)?
    This predicts the scale at which PAC conservation becomes undetectable.
    """
    print("\nPART 3: Null-Crossing Predictions")
    print("-" * 60)
    
    crossings = {}
    
    for name, spec in MODELS.items():
        if not fit_results[name].get("fit_success", False):
            continue
        
        popt = fit_results[name]["params"]
        
        # Binary search for enrichment = 1.0
        lo, hi = 1000, 10_000_000  # 1B to 10T parameters
        target = 1.0
        
        try:
            f_lo = spec["func"](lo, *popt)
            f_hi = spec["func"](hi, *popt)
            
            if (f_lo > target) == (f_hi > target):
                # Doesn't cross 1.0 in range
                if f_hi > target:
                    crossings[name] = {
                        "crossing_size_M": "> 10T",
                        "note": "Enrichment stays above 1.0 even at 10T parameters"
                    }
                else:
                    crossings[name] = {
                        "crossing_size_M": "< 1B",
                        "note": "Enrichment already below 1.0 at 1B"
                    }
            else:
                # Binary search
                for _ in range(100):
                    mid = (lo + hi) / 2
                    f_mid = spec["func"](mid, *popt)
                    if (f_mid > target) == (f_lo > target):
                        lo = mid
                    else:
                        hi = mid
                
                crossing_B = round((lo + hi) / 2 / 1000, 1)
                crossings[name] = {
                    "crossing_size_M": round((lo + hi) / 2, 0),
                    "crossing_size_B": f"{crossing_B}B",
                    "note": f"Enrichment reaches 1.0 at ~{crossing_B}B parameters"
                }
                
                print(f"  {name:15s}: enrichment → 1.0 at ~{crossing_B}B parameters")
                
        except Exception as e:
            crossings[name] = {"error": str(e)}
    
    return crossings


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXP_31: PAC Conservation Scaling Extrapolation")
    print("=" * 70)
    print()
    print("Observed data (Paper 6 §4.4):")
    for s, e in zip(SIZES_M, ENRICHMENT):
        print(f"  {s/1000:.0f}B: {e:.1f}× Xi enrichment" if s >= 1000
              else f"  {s}M: {e:.1f}× Xi enrichment")
    print()
    
    results = {
        "experiment": "exp_31_scaling_extrapolation",
        "target_paper": "Paper 6 (Computational Validation of PAC Conservation)",
        "purpose": "Produce falsifiable scaling predictions for 7B-175B models",
        "observed_data": {
            "sizes_M": SIZES_M.tolist(),
            "enrichment": ENRICHMENT.tolist(),
            "source": "Paper 6 §4.4, Pythia model family"
        }
    }
    
    # Part 1: Fit models
    fit_results = fit_models()
    results["model_fits"] = fit_results
    
    # Select best model by AIC
    valid_fits = {k: v for k, v in fit_results.items() if v.get("fit_success", False)}
    best_model = min(valid_fits.items(), key=lambda x: x[1]["aic"])
    results["best_model"] = {
        "name": best_model[0],
        "aic": best_model[1]["aic"],
        "r_squared": best_model[1]["r_squared"]
    }
    print(f"\n  Best model (by AIC): {best_model[0]} (R²={best_model[1]['r_squared']:.4f})")
    
    # Part 2: Predictions
    predictions = make_predictions(fit_results)
    results["predictions"] = predictions
    
    # Part 3: Null crossings
    crossings = find_null_crossing(fit_results)
    results["null_crossings"] = crossings
    
    # --- Synthesis ---
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    
    best_name = best_model[0]
    if best_name in predictions and len(predictions[best_name]) >= 2:
        pred_7b = predictions[best_name][0]
        pred_70b = predictions[best_name][1]
        
        synthesis = {
            "best_model": best_name,
            "prediction_7B": pred_7b,
            "prediction_70B": pred_70b,
            "null_crossing": crossings.get(best_name, {}),
            "falsifiable_claims": [
                f"At 7B parameters, Xi enrichment should be {pred_7b.get('predicted_enrichment', '?')} "
                f"(95% CI: [{pred_7b.get('ci_95_low', '?')}, {pred_7b.get('ci_95_high', '?')}])",
                f"At 70B parameters, Xi enrichment should be {pred_70b.get('predicted_enrichment', '?')} "
                f"(95% CI: [{pred_70b.get('ci_95_low', '?')}, {pred_70b.get('ci_95_high', '?')}])",
                f"Enrichment reaches 1.0 (undetectable) at ~{crossings.get(best_name, {}).get('crossing_size_B', '?')} parameters"
            ],
            "caveats": [
                "Only 4 data points — extrapolation is speculative",
                "All models from same family (Pythia) — architecture effects unknown",
                "Enrichment measured by specific Xi-proximity metric — different metrics may differ",
                "Confidence intervals from bootstrap on 4 points are necessarily wide"
            ]
        }
        results["synthesis"] = synthesis
        
        print(f"\n  Best model: {best_name}")
        for claim in synthesis["falsifiable_claims"]:
            print(f"  • {claim}")
        print(f"\n  Caveats: {len(synthesis['caveats'])} noted (4-point extrapolation, single family, etc.)")
    
    # PASS/FAIL: The experiment always "passes" — it produces predictions.
    # The predictions themselves are tested by future measurement.
    results["PASS"] = True
    results["status"] = (
        "PASS — predictions generated. Falsification requires measuring Xi enrichment "
        "in 7B+ parameter models."
    )
    
    print(f"\n  PASS: Predictions generated for external validation.")
    
    save_results(results, "exp_31_scaling_extrapolation")
    return results


if __name__ == "__main__":
    main()
