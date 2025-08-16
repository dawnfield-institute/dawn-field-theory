"""
TinyCIMM-Navier SCBF Integration Test

Quick validation of SCBF integration with fluid dynamics interpretability.
Tests the core symbolic entropy collapse detection for flow learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from tinycimm_navier import TinyCIMMNavier, create_flow_boundary_conditions

def test_scbf_integration():
    """Test SCBF metrics integration with TinyCIMM-Navier"""
    print("Testing SCBF Integration with TinyCIMM-Navier...")
    
    # Create model with SCBF enabled
    model = TinyCIMMNavier(
        initial_reynolds=1000,
        hidden_size=32,
        enable_scbf=True
    )
    
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test simple flow sequence
    reynolds_sequence = [500, 1000, 2000, 3000, 5000]  # Laminar → Turbulent transition
    
    scbf_results = []
    
    for i, reynolds in enumerate(reynolds_sequence):
        print(f"\\nStep {i+1}: Reynolds = {reynolds}")
        
        # Create flow input
        flow_input = create_flow_boundary_conditions(reynolds, "pipe")
        
        # Forward pass
        prediction = model(flow_input.unsqueeze(0), reynolds_number=reynolds)
        
        # Get interpretability summary
        interpretability = model.get_flow_interpretability_summary()
        
        print(f"  Flow Regime: {interpretability['flow_regime']}")
        print(f"  Network Size: {interpretability['network_size']} neurons")
        print(f"  Patterns Learned: {interpretability['patterns_learned']}")
        print(f"  Turbulence Intensity: {interpretability['turbulence_intensity']:.3f}")
        
        # SCBF metrics
        if 'scbf_metrics' in interpretability:
            scbf = interpretability['scbf_metrics']
            print(f"  SCBF - Entropy Events: {scbf['entropy_events']}")
            
            # Handle regime stability which might be a dict
            regime_stability = scbf['regime_stability']
            if isinstance(regime_stability, dict):
                stability_value = regime_stability.get('pred_variance', 0.0)
            else:
                stability_value = regime_stability
            print(f"  SCBF - Regime Stability: {stability_value:.3f}")
            print(f"  SCBF - Vorticity Attractors: {scbf['vorticity_attractors']}")
        
        scbf_results.append({
            'reynolds': reynolds,
            'prediction': prediction.detach(),
            'interpretability': interpretability
        })
        
        # Test architectural adaptation
        if reynolds > 2000:  # Should trigger growth for turbulent flows
            old_size = model.hidden_size
            model.adapt_architecture()
            if model.hidden_size != old_size:
                print(f"  *** Network adapted: {old_size} → {model.hidden_size} neurons ***")
    
    print("\\n=== SCBF Integration Test Summary ===")
    print(f"Total Reynolds regimes tested: {len(reynolds_sequence)}")
    print(f"Flow patterns discovered: {model.flow_patterns_learned}")
    print(f"Final network size: {model.hidden_size} neurons")
    
    if model.enable_scbf and hasattr(model, 'scbf_tracker'):
        tracker = model.scbf_tracker
        print(f"Total entropy collapse events: {len(tracker.flow_entropy_history)}")
        print(f"Regime stability history: {len(tracker.regime_stability_history)}")
        print(f"Vorticity attractors tracked: {len(tracker.vorticity_attractors)}")
    
    print("\\n*** SCBF Integration Test Completed Successfully! ***")
    return scbf_results

def test_pattern_recognition():
    """Test flow pattern recognition capabilities"""
    print("\\nTesting Flow Pattern Recognition...")
    
    model = TinyCIMMNavier(initial_reynolds=1000, hidden_size=64)
    
    # Test different flow types
    flow_types = [
        ("pipe", 500),
        ("pipe", 2000),
        ("pipe", 10000),
        ("cylinder", 100),
        ("cylinder", 300),
        ("cylinder", 1000)
    ]
    
    for geometry, reynolds in flow_types:
        flow_input = create_flow_boundary_conditions(reynolds, geometry)
        prediction = model(flow_input.unsqueeze(0), reynolds_number=reynolds)
        
        print(f"  {geometry} flow @ Re={reynolds}:")
        print(f"    Prediction: {prediction.squeeze().detach().numpy()}")
        print(f"    Flow regime: {model.flow_controller.flow_regime}")
        print(f"    Patterns learned: {model.flow_patterns_learned}")

def run_quick_validation():
    """Run quick validation of core TinyCIMM-Navier functionality"""
    print("="*60)
    print("TinyCIMM-NAVIER QUICK VALIDATION")
    print("="*60)
    
    # Test 1: SCBF Integration
    scbf_results = test_scbf_integration()
    
    # Test 2: Pattern Recognition
    test_pattern_recognition()
    
    # Test 3: Basic Learning
    print("\\nTesting Basic Learning Capability...")
    
    model = TinyCIMMNavier(initial_reynolds=1000, enable_scbf=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simple learning test: predict constant flow
    target_flow = torch.tensor([1.0, 0.0, 0.5, 0.1])  # [u, v, p, vorticity]
    
    losses = []
    for epoch in range(100):
        flow_input = create_flow_boundary_conditions(1000, "pipe")
        prediction = model(flow_input.unsqueeze(0), 1000)
        
        loss = torch.nn.functional.mse_loss(prediction.squeeze(), target_flow)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"    Final loss: {losses[-1]:.6f}")
    print(f"    Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    print("\\n*** Quick Validation Completed Successfully! ***")
    print("TinyCIMM-Navier is ready for comprehensive fluid dynamics experiments.")
    
    return {
        'scbf_results': scbf_results,
        'learning_losses': losses,
        'final_model_state': model.get_flow_interpretability_summary()
    }

if __name__ == "__main__":
    validation_results = run_quick_validation()
