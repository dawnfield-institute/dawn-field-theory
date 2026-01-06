"""
exp_04_ab_testing.py - A/B Test: Threshold Knowledge Impact on Prediction

Tests whether knowing the SEC threshold improves prediction accuracy
for system behavior near phase transitions.

Design:
- Task: Predict whether a system at parameter p will be chaotic or ordered
- Control (A): Predict using only trajectory features
- Treatment (B): Predict using trajectory features + threshold distance

If the threshold is meaningful, Treatment should outperform Control
because knowing distance-to-threshold is predictive information.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from scipy import stats

XI = 1 + np.pi / 55
FEIGENBAUM_POINT = 3.5699456


def logistic_trajectory(r, n_steps=500, transient=100):
    """Generate logistic map trajectory."""
    x = 0.5
    traj = []
    for i in range(n_steps + transient):
        x = r * x * (1 - x)
        if i >= transient:
            traj.append(x)
    return np.array(traj)


def compute_lyapunov_estimate(trajectory):
    """Estimate Lyapunov exponent from trajectory."""
    # Use finite-time approximation
    n = len(trajectory)
    if n < 100:
        return 0
    
    lyap_sum = 0
    count = 0
    
    for i in range(n - 1):
        x = trajectory[i]
        # Derivative of logistic map at x is r*(1-2x), but we don't know r
        # Estimate from trajectory: if x_{n+1} = f(x_n), then |f'(x_n)| ≈ |x_{n+1} - x_n| / small_perturbation
        # Instead, use the standard approximation for maps
        deriv = abs(1 - 2 * x)  # Normalized derivative shape
        if deriv > 1e-10:
            lyap_sum += np.log(deriv)
            count += 1
    
    return lyap_sum / count if count > 0 else 0


def compute_trajectory_features(trajectory):
    """Extract predictive features from trajectory."""
    return {
        'mean': np.mean(trajectory),
        'std': np.std(trajectory),
        'range': np.max(trajectory) - np.min(trajectory),
        'autocorr': np.corrcoef(trajectory[:-1], trajectory[1:])[0, 1] if len(trajectory) > 1 else 0,
        'lyapunov_est': compute_lyapunov_estimate(trajectory)
    }


def is_chaotic(r, threshold=FEIGENBAUM_POINT):
    """Ground truth: is the system chaotic at this r value?"""
    # Technically chaos is intermittent, but past Feigenbaum point
    # the system is predominantly chaotic
    return r > threshold


def predict_chaos_control(features):
    """
    Control predictor: uses only trajectory features.
    Predict chaotic if high variance and low autocorrelation.
    """
    # Simple heuristic: chaotic trajectories have high std, low autocorr
    chaos_score = features['std'] * (1 - abs(features['autocorr']))
    return chaos_score > 0.1


def predict_chaos_treatment(features, r, threshold):
    """
    Treatment predictor: uses trajectory features + threshold distance.
    Incorporates knowledge of where the threshold is.
    """
    # Distance to threshold (normalized)
    dist = (r - threshold) / threshold
    
    # Combine with trajectory features
    # If r > threshold, lean toward chaotic
    # If r < threshold, lean toward ordered
    feature_score = features['std'] * (1 - abs(features['autocorr']))
    
    if dist > 0:
        # Above threshold - bias toward chaotic
        combined_score = feature_score + 0.2 * dist
    else:
        # Below threshold - bias toward ordered
        combined_score = feature_score + 0.2 * dist
    
    return combined_score > 0.05


def run_ab_test(n_samples=200, noise_level=0.0):
    """
    Run A/B test comparing prediction accuracy.
    
    Samples r values near the threshold and tests prediction accuracy.
    """
    # Sample r values in the interesting region around threshold
    np.random.seed(42)
    
    # 50% below threshold, 50% above
    r_below = np.random.uniform(3.4, FEIGENBAUM_POINT, n_samples // 2)
    r_above = np.random.uniform(FEIGENBAUM_POINT, 3.8, n_samples // 2)
    r_values = np.concatenate([r_below, r_above])
    np.random.shuffle(r_values)
    
    control_correct = 0
    treatment_correct = 0
    
    control_predictions = []
    treatment_predictions = []
    ground_truth = []
    
    for r in r_values:
        # Generate trajectory
        traj = logistic_trajectory(r)
        features = compute_trajectory_features(traj)
        
        # Ground truth
        truth = is_chaotic(r)
        ground_truth.append(truth)
        
        # Control prediction (no threshold knowledge)
        pred_control = predict_chaos_control(features)
        control_predictions.append(pred_control)
        if pred_control == truth:
            control_correct += 1
        
        # Treatment prediction (with threshold knowledge)
        pred_treatment = predict_chaos_treatment(features, r, FEIGENBAUM_POINT)
        treatment_predictions.append(pred_treatment)
        if pred_treatment == truth:
            treatment_correct += 1
    
    control_accuracy = control_correct / n_samples
    treatment_accuracy = treatment_correct / n_samples
    
    return {
        'control_accuracy': control_accuracy,
        'treatment_accuracy': treatment_accuracy,
        'improvement': treatment_accuracy - control_accuracy,
        'n_samples': n_samples,
        'control_predictions': control_predictions,
        'treatment_predictions': treatment_predictions,
        'ground_truth': ground_truth,
        'r_values': r_values.tolist()
    }


def run_wrong_threshold_test(n_samples=200):
    """Test what happens when we use the WRONG threshold."""
    np.random.seed(42)
    
    r_below = np.random.uniform(3.4, FEIGENBAUM_POINT, n_samples // 2)
    r_above = np.random.uniform(FEIGENBAUM_POINT, 3.8, n_samples // 2)
    r_values = np.concatenate([r_below, r_above])
    np.random.shuffle(r_values)
    
    # Wrong thresholds
    wrong_high = FEIGENBAUM_POINT * 1.05  # 5% too high
    wrong_low = FEIGENBAUM_POINT * 0.95   # 5% too low
    
    correct_threshold_correct = 0
    wrong_high_correct = 0
    wrong_low_correct = 0
    
    for r in r_values:
        traj = logistic_trajectory(r)
        features = compute_trajectory_features(traj)
        truth = is_chaotic(r)
        
        # Correct threshold
        if predict_chaos_treatment(features, r, FEIGENBAUM_POINT) == truth:
            correct_threshold_correct += 1
        
        # Wrong thresholds
        if predict_chaos_treatment(features, r, wrong_high) == truth:
            wrong_high_correct += 1
        if predict_chaos_treatment(features, r, wrong_low) == truth:
            wrong_low_correct += 1
    
    return {
        'correct_threshold_accuracy': correct_threshold_correct / n_samples,
        'wrong_high_accuracy': wrong_high_correct / n_samples,
        'wrong_low_accuracy': wrong_low_correct / n_samples,
        'correct_threshold': FEIGENBAUM_POINT,
        'wrong_high': wrong_high,
        'wrong_low': wrong_low
    }


def statistical_significance(control_acc, treatment_acc, n_samples):
    """Compute statistical significance of accuracy difference."""
    # Use McNemar's test approximation
    # Or simple binomial proportion test
    
    p1 = control_acc
    p2 = treatment_acc
    
    # Pooled standard error
    p_pool = (p1 + p2) / 2
    se = np.sqrt(2 * p_pool * (1 - p_pool) / n_samples)
    
    if se > 0:
        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z = 0
        p_value = 1.0
    
    return z, p_value


def run_experiment():
    """Run full A/B testing experiment."""
    
    print("=" * 60)
    print("A/B Test: Threshold Knowledge Impact on Prediction")
    print("=" * 60)
    
    print(f"\nTask: Predict chaos vs order in logistic map")
    print(f"Threshold (Feigenbaum point): r* = {FEIGENBAUM_POINT:.6f}")
    print(f"Control: Predict using trajectory features only")
    print(f"Treatment: Predict using features + threshold distance")
    
    # Main A/B test
    print("\n" + "-" * 40)
    print("Main A/B Test (n=200)")
    print("-" * 40)
    
    results = run_ab_test(n_samples=200)
    
    print(f"\nControl accuracy:   {results['control_accuracy']:.1%}")
    print(f"Treatment accuracy: {results['treatment_accuracy']:.1%}")
    print(f"Improvement:        {results['improvement']:+.1%}")
    
    z, p = statistical_significance(
        results['control_accuracy'],
        results['treatment_accuracy'],
        results['n_samples']
    )
    print(f"Z-statistic: {z:.3f}, p-value: {p:.4f}")
    
    if results['improvement'] > 0 and p < 0.05:
        print("Result: Treatment significantly better ✓")
    elif results['improvement'] < 0 and p < 0.05:
        print("Result: Control significantly better ✗")
    else:
        print("Result: No significant difference")
    
    # Wrong threshold test
    print("\n" + "-" * 40)
    print("Wrong Threshold Test")
    print("-" * 40)
    
    wrong_results = run_wrong_threshold_test(n_samples=200)
    
    print(f"\nCorrect threshold accuracy: {wrong_results['correct_threshold_accuracy']:.1%}")
    print(f"Wrong (+5%) accuracy:       {wrong_results['wrong_high_accuracy']:.1%}")
    print(f"Wrong (-5%) accuracy:       {wrong_results['wrong_low_accuracy']:.1%}")
    
    # Key finding
    print("\n" + "=" * 60)
    print("Key Findings")
    print("=" * 60)
    
    if results['improvement'] > 0:
        print(f"1. Knowing threshold improves prediction by {results['improvement']:.1%}")
    else:
        print(f"1. Threshold knowledge did not improve prediction")
    
    if wrong_results['correct_threshold_accuracy'] > wrong_results['wrong_high_accuracy']:
        degradation = wrong_results['correct_threshold_accuracy'] - wrong_results['wrong_high_accuracy']
        print(f"2. Wrong threshold (+5%) degrades accuracy by {degradation:.1%}")
    
    if wrong_results['correct_threshold_accuracy'] > wrong_results['wrong_low_accuracy']:
        degradation = wrong_results['correct_threshold_accuracy'] - wrong_results['wrong_low_accuracy']
        print(f"3. Wrong threshold (-5%) degrades accuracy by {degradation:.1%}")
    
    # Save results
    output = {
        'experiment': 'ab_testing_prediction',
        'timestamp': datetime.now().isoformat(),
        'threshold': FEIGENBAUM_POINT,
        'main_test': {
            'n_samples': results['n_samples'],
            'control_accuracy': results['control_accuracy'],
            'treatment_accuracy': results['treatment_accuracy'],
            'improvement': results['improvement'],
            'z_statistic': z,
            'p_value': p
        },
        'wrong_threshold_test': wrong_results,
        'constants': {
            'xi': XI,
            'feigenbaum': FEIGENBAUM_POINT
        }
    }
    
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'exp_04_ab_testing_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == '__main__':
    run_experiment()
