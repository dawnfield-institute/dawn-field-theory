"""
Resonance Detection for Pre-Field Recursion
===========================================

Detects and tracks natural oscillation frequencies in field evolution.
Uses FFT and zero-crossing analysis to identify dominant frequencies,
enabling resonance-locked convergence.

Key Insight:
    Oscillations in PAC evolution aren't noise - they represent the 
    pre-field searching for its natural resonance frequency. This aligns
    with Q-Socket principles: fields need to "ring" at natural frequency
    before stabilizing.

Author: Dawn Field Institute
Date: October 1, 2025
Version: 2.2.0
"""

import numpy as np
from typing import List, Optional, Dict
from scipy import signal


class ResonanceDetector:
    """
    Detect and track natural oscillation frequency in field evolution
    
    Uses FFT and zero-crossing analysis to identify dominant frequencies
    in PAC residual evolution, enabling resonance-locked convergence.
    
    Attributes:
        min_window: Minimum data points needed for analysis
        max_window: Maximum window size for recent data
        detected_frequencies: History of detected frequencies
        confidence_scores: History of detection confidence
    """
    
    def __init__(self, min_window: int = 20, max_window: int = 100):
        """
        Initialize resonance detector
        
        Args:
            min_window: Minimum data points for analysis (default: 20)
            max_window: Maximum window size (default: 100)
        """
        self.min_window = min_window
        self.max_window = max_window
        self.detected_frequencies: List[float] = []
        self.confidence_scores: List[float] = []
        
    def analyze_oscillations(self, pac_history: List[float]) -> Dict:
        """
        Analyze PAC evolution to detect resonance frequency
        
        Combines FFT spectral analysis with zero-crossing validation
        to robustly identify dominant oscillation patterns.
        
        Args:
            pac_history: List of PAC residual values over time
            
        Returns:
            Dictionary containing:
                - frequency: Dominant frequency (cycles per iteration)
                - period: Oscillation period (iterations)
                - confidence: Detection confidence (0-1)
                - amplitude: Oscillation amplitude
                - phase: Current phase position (0-2π)
                - trend_slope: Overall convergence trend
        """
        if len(pac_history) < self.min_window:
            return {
                'frequency': None,
                'period': None,
                'confidence': 0.0,
                'amplitude': 0.0,
                'phase': 0.0,
                'trend_slope': 0.0
            }
        
        # Use recent window
        window = min(len(pac_history), self.max_window)
        recent = np.array(pac_history[-window:])
        
        # Detrend to isolate oscillations from overall convergence
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        trend = np.poly1d(coeffs)(x)
        detrended = recent - trend
        
        # FFT spectral analysis
        fft = np.fft.fft(detrended)
        freqs = np.fft.fftfreq(len(detrended))
        power = np.abs(fft)**2
        
        # Find dominant frequency (exclude DC component)
        positive_freqs = freqs[1:len(freqs)//2]
        positive_power = power[1:len(power)//2]
        
        if len(positive_power) == 0:
            return {
                'frequency': None,
                'period': None,
                'confidence': 0.0,
                'amplitude': 0.0,
                'phase': 0.0,
                'trend_slope': float(coeffs[0])
            }
        
        # Identify dominant peak
        dominant_idx = np.argmax(positive_power)
        dominant_freq = positive_freqs[dominant_idx]
        dominant_power = positive_power[dominant_idx]
        
        # Calculate confidence based on peak prominence
        total_power = np.sum(positive_power)
        confidence = dominant_power / (total_power + 1e-10)
        
        # Zero-crossing validation for robustness
        zero_crossings = np.where(np.diff(np.sign(detrended)))[0]
        if len(zero_crossings) >= 2:
            # Measure periods between crossings
            periods = np.diff(zero_crossings) * 2  # Full cycle = 2 crossings
            avg_period = np.mean(periods)
            period_std = np.std(periods)
            
            # Higher confidence if periods are consistent
            if period_std < avg_period * 0.2:  # <20% variation
                confidence = min(confidence * 1.5, 1.0)
        else:
            # Estimate from FFT if no crossings
            avg_period = 1.0 / dominant_freq if dominant_freq > 0 else None
        
        # Calculate current phase position
        if avg_period and avg_period > 0:
            phase = (len(pac_history) % avg_period) / avg_period * 2 * np.pi
        else:
            phase = 0.0
        
        # Measure oscillation amplitude
        amplitude = float(np.std(detrended))
        
        result = {
            'frequency': float(dominant_freq),
            'period': float(avg_period) if avg_period else None,
            'confidence': float(confidence),
            'amplitude': amplitude,
            'phase': phase,
            'trend_slope': float(coeffs[0])
        }
        
        # Track detection history
        self.detected_frequencies.append(dominant_freq)
        self.confidence_scores.append(confidence)
        
        return result
    
    def suggest_twist_rate(self, resonance_info: Dict) -> Optional[float]:
        """
        Suggest optimal twist rate based on detected resonance
        
        Converts detected period into angular frequency for Möbius twist.
        
        Formula: twist_rate = 2π / period
        
        Args:
            resonance_info: Dictionary from analyze_oscillations()
            
        Returns:
            Suggested twist rate in radians, or None if insufficient confidence
        """
        # v2.2.1: Lowered confidence threshold from 0.3 to 0.1
        if resonance_info['period'] and resonance_info['confidence'] > 0.1:
            suggested = 2 * np.pi / resonance_info['period']
            # Clamp to reasonable range for Möbius transformations
            return float(np.clip(suggested, np.pi/16, np.pi))
        return None
    
    def get_detection_stability(self, window: int = 10) -> float:
        """
        Assess stability of recent frequency detections
        
        Args:
            window: Number of recent detections to analyze
            
        Returns:
            Stability score (0-1), higher = more stable
        """
        if len(self.detected_frequencies) < window:
            return 0.0
        
        recent_freqs = self.detected_frequencies[-window:]
        recent_conf = self.confidence_scores[-window:]
        
        # Weighted standard deviation (higher confidence = more weight)
        weights = np.array(recent_conf)
        mean_freq = np.average(recent_freqs, weights=weights)
        variance = np.average((recent_freqs - mean_freq)**2, weights=weights)
        std_freq = np.sqrt(variance)
        
        # Normalize to 0-1 (assume frequencies in range 0.01-0.5)
        stability = 1.0 - min(std_freq / 0.1, 1.0)
        
        return float(stability)
    
    def reset(self):
        """Reset detection history"""
        self.detected_frequencies.clear()
        self.confidence_scores.clear()


def visualize_resonance_analysis(pac_history: List[float], 
                                 resonance_info: Dict,
                                 save_path: Optional[str] = None) -> None:
    """
    Create diagnostic plots for resonance detection
    
    Args:
        pac_history: PAC residual evolution
        resonance_info: Results from ResonanceDetector.analyze_oscillations()
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Raw PAC evolution
    ax = axes[0, 0]
    ax.plot(pac_history, linewidth=1.5, color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PAC Residual')
    ax.set_title('PAC Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Detrended oscillations
    ax = axes[0, 1]
    x = np.arange(len(pac_history))
    coeffs = np.polyfit(x, pac_history, 1)
    trend = np.poly1d(coeffs)(x)
    detrended = np.array(pac_history) - trend
    
    ax.plot(detrended, linewidth=1.5, color='red')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Detrended PAC')
    ax.set_title(f'Isolated Oscillations (amp={resonance_info["amplitude"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: FFT spectrum
    ax = axes[1, 0]
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(len(detrended))
    power = np.abs(fft)**2
    
    positive_mask = freqs > 0
    ax.plot(freqs[positive_mask], power[positive_mask], linewidth=1.5)
    
    # Mark detected frequency
    if resonance_info['frequency']:
        ax.axvline(x=resonance_info['frequency'], color='red', 
                  linestyle='--', label=f"f={resonance_info['frequency']:.4f}")
        ax.legend()
    
    ax.set_xlabel('Frequency (cycles/iteration)')
    ax.set_ylabel('Power')
    ax.set_title(f'Frequency Spectrum (conf={resonance_info["confidence"]:.2f})')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Phase diagram
    ax = axes[1, 1]
    if len(pac_history) > 1:
        ax.plot(pac_history[:-1], pac_history[1:], 
               linewidth=0.5, alpha=0.6, color='purple')
        ax.scatter(pac_history[0], pac_history[1], 
                  color='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(pac_history[-2], pac_history[-1], 
                  color='red', s=100, marker='x', label='End', zorder=5)
    
    ax.set_xlabel('PAC(t)')
    ax.set_ylabel('PAC(t+1)')
    ax.set_title('Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Resonance analysis saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
