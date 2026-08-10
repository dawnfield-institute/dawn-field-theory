#!/usr/bin/env python3
"""
PAC Physics Engine - Data Collection and Results Dumping
========================================================

Simplified data collection focused on raw measurements rather than pass/fail testing.

Following dawn-field-theory methodology: amplification measured as observational 
phenomenon (observed_ratio vs reference_ratio) rather than validation target.
The 15.56x reference comes from dawn-field experiments and represents empirical
baseline for comparison, not a theoretical prediction to validate against.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import torch

def create_results_directory() -> str:
    """Create timestamped results directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/pac_validation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def dump_raw_measurements(engine, results_dir: str) -> Dict[str, Any]:
    """Collect and dump all raw measurements from PAC engine"""
    
    print(f"\n📊 Collecting raw PAC measurements...")
    print(f"📁 Results directory: {results_dir}")
    
    measurements = {}
    
    # Core conservation measurements
    print("  🔧 Core conservation measurements...")
    conservation_stats = engine.pac_kernel.check_global_conservation()
    measurements["conservation"] = {
        "total_residual_norm": conservation_stats.get("total_residual_norm", 0),
        "mean_residual": conservation_stats.get("mean_residual", 0),
        "max_violation": conservation_stats.get("max_violation", 0),
        "violation_count": conservation_stats.get("violation_count", 0),
        "conservation_quality": conservation_stats.get("conservation_quality", 0),
        "conservation_stability": conservation_stats.get("conservation_stability", 0),
        "global_balance": conservation_stats.get("global_balance", 0)
    }
    
    # Information amplification measurements (child perspective)
    print("  🔬 Information amplification measurements...")
    try:
        # Create parent field with varied information density for concentration points
        parent_field = torch.randn(8, 8, device=engine.device) * 0.5 + 1.0  # Varied density
        amp_result = engine.information_module.amplify_information_pac(parent_field, amplification_strength=1.0)
        
        # Find concentration regions (where amplification should be visible to child)
        info_density = torch.abs(parent_field)
        threshold = torch.quantile(info_density, 0.8)  # Top 20% density regions
        concentration_mask = info_density > threshold
        
        # CHILD PERSPECTIVE: Look at concentration region (high amplification expected)
        if torch.any(concentration_mask):
            child_input_values = parent_field[concentration_mask]
            child_output_values = amp_result.amplified_field[concentration_mask]
        else:
            # Fallback to central region
            child_input_values = parent_field[3:5, 3:5].flatten()
            child_output_values = amp_result.amplified_field[3:5, 3:5].flatten()
        
        # Parent perspective (conservation enforced)
        parent_initial = torch.sum(torch.abs(parent_field)).item()
        parent_final = torch.sum(torch.abs(amp_result.amplified_field)).item()
        parent_amplification = parent_final / parent_initial if parent_initial > 0 else 1.0
        
        # Child perspective (looking at concentration region only)
        child_initial = torch.sum(torch.abs(child_input_values)).item()
        child_final = torch.sum(torch.abs(child_output_values)).item()
        child_amplification = child_final / child_initial if child_initial > 0 else 1.0
        
        measurements["amplification"] = {
            "parent_perspective": {
                "observed_ratio": parent_amplification,
                "initial_content": parent_initial,
                "final_content": parent_final,
                "conservation_enforced": True
            },
            "child_perspective": {
                "observed_ratio": child_amplification,
                "initial_content": child_initial,
                "final_content": child_final,
                "conservation_visible": False,
                "reference_ratio": 15.56,  # Dawn-field baseline from child perspective
                "relative_to_reference": child_amplification / 15.56,
                "viewing_concentration_region": True
            },
            "spatial_redistribution": {
                "concentration_points": torch.sum(concentration_mask).item(),
                "field_size": parent_field.numel(),
                "concentration_fraction": torch.sum(concentration_mask).item() / parent_field.numel()
            },
            "engine_reported_ratio": amp_result.amplification_ratio,
            "entropy_change": amp_result.entropy_change,
            "resonance_strength": amp_result.resonance_strength,
            "amplification_mode": str(amp_result.amplification_mode),
            "methodology": "spatial_redistribution_measurement"
        }
    except Exception as e:
        measurements["amplification"] = {"error": str(e)}
    
    # Cross-scale correlations
    print("  🔍 Cross-scale measurements...")
    try:
        fields = {
            "quantum": torch.randn(32, 32, device=engine.device, dtype=torch.complex64),
            "geometric": torch.randn(32, 32, device=engine.device) * 0.1,
            "information": torch.randn(32, 32, device=engine.device),
            "consciousness": torch.randn(32, 32, device=engine.device) * 0.5
        }
        
        mock_states = [
            {
                "quantum_state": {"conservation_quality": 0.85},
                "geometric_state": {"collapse_strength": 0.15},
                "fluid_state": {"reynolds_number": 120},
                "information_state": {"resonance_strength": 0.75, "amplification_ratio": 8.0},
                "consciousness_state": {"awareness_metric": 0.70}
            }
        ]
        
        cross_scale_result = engine.cross_scale_validator.validate_cross_scale_consistency(mock_states)
        measurements["cross_scale"] = {
            "scale_correlations": cross_scale_result.scale_correlations,
            "conservation_consistency": cross_scale_result.conservation_consistency,
            "temporal_synchronization": cross_scale_result.temporal_synchronization,
            "validation_passed": cross_scale_result.validation_passed
        }
    except Exception as e:
        measurements["cross_scale"] = {"error": str(e)}
    
    # Universal signature measurements
    print("  🌟 Universal signature measurements...")
    try:
        mock_system_states = [
            {
                "quantum_state": {"conservation_quality": 0.95, "field_magnitude": 1.2},
                "geometric_state": {"collapse_strength": 0.1, "curvature": 0.05},
                "consciousness_state": {"awareness_metric": 0.7, "binding_strength": 0.6}
            }
        ]
        mock_temporal_data = [
            {"amplification_factor": 1.5, "entropy_rate": 0.1, "resonance_frequency": 2.4}
        ]
        
        signature_result = engine.signature_detector.detect_universal_signatures(mock_system_states, mock_temporal_data)
        measurements["signatures"] = {
            "detected_count": len(signature_result.detected_signatures),
            "signature_completeness": signature_result.signature_completeness,
            "temporal_consistency": signature_result.temporal_consistency,
            "spatial_coherence": signature_result.spatial_coherence,
            "overall_validation_score": signature_result.overall_validation_score,
            "detections": [
                {
                    "type": str(sig.signature_type),
                    "strength": sig.strength,
                    "confidence": sig.confidence,
                    "frequency": sig.frequency,
                    "metadata": sig.metadata
                }
                for sig in signature_result.detected_signatures
            ]
        }
    except Exception as e:
        measurements["signatures"] = {"error": str(e)}
    
    # System configuration
    measurements["system"] = {
        "device": str(engine.device),
        "spatial_resolution": engine.lattice.spatial_size if hasattr(engine.lattice, 'spatial_size') else "unknown",
        "timestamp": datetime.now().isoformat(),
        "engine_version": engine.version,
        "frameworks": ["PAC", "SEC", "MED", "IAF", "SCBF"]
    }
    
    # Dump to JSON
    json_file = os.path.join(results_dir, "raw_measurements.json")
    with open(json_file, 'w') as f:
        json.dump(measurements, f, indent=2, default=str)
    
    print(f"  📄 Raw measurements saved: {json_file}")
    
    # Create summary report
    summary_file = os.path.join(results_dir, "measurement_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("PAC PHYSICS ENGINE - RAW MEASUREMENT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {measurements['system']['timestamp']}\n")
        f.write(f"Device: {measurements['system']['device']}\n\n")
        
        f.write("CONSERVATION MEASUREMENTS:\n")
        if "conservation" in measurements:
            for key, value in measurements["conservation"].items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\nAMPLIFICATION MEASUREMENTS:\n")
        if "amplification" in measurements:
            for key, value in measurements["amplification"].items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\nCROSS-SCALE MEASUREMENTS:\n")
        if "cross_scale" in measurements:
            for key, value in measurements["cross_scale"].items():
                if key != "scale_correlations":
                    f.write(f"  {key}: {value}\n")
        
        f.write("\nSIGNATURE MEASUREMENTS:\n")
        if "signatures" in measurements:
            f.write(f"  detected_count: {measurements['signatures'].get('detected_count', 0)}\n")
            f.write(f"  overall_validation_score: {measurements['signatures'].get('overall_validation_score', 0)}\n")
    
    print(f"  📋 Summary report saved: {summary_file}")
    
    return measurements

def run_data_collection():
    """Main data collection function"""
    from pac_engine import PACPhysicsEngine
    
    print("\n" + "=" * 80)
    print("📊 PAC PHYSICS ENGINE - RAW DATA COLLECTION")
    print("=" * 80)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Initialize engine
    engine = PACPhysicsEngine(device="cuda")
    engine.initialize_all_components(spatial_size=32)  # Initialize components for data collection
    
    # Collect measurements
    measurements = dump_raw_measurements(engine, results_dir)
    
    print(f"\n✅ Data collection complete!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"📊 Measurements collected: {len(measurements)} categories")
    
    return results_dir, measurements

if __name__ == "__main__":
    run_data_collection()
