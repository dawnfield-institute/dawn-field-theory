# superfluid_dynamics.py
import torch


class SuperfluidDynamics:
    def compute_superfluid_velocity(self, entropy_gradient, qfi, viscosity=1e-5, device='cpu'):
        """
        Estimate fluid-like information velocity based on entropy gradient and quantum stability.
        Lower viscosity = faster information propagation.
        """
        entropy_gradient_tensor = torch.tensor(entropy_gradient, device=device)
        qfi_tensor = torch.tensor(qfi, device=device)

        flow_strength = torch.exp(-torch.abs(entropy_gradient_tensor)) * (1.0 + qfi_tensor)
        velocity_field = flow_strength / (viscosity + 1e-8)
        return torch.clamp(velocity_field, -5.0, 5.0)  # Bound to avoid extreme reactions

    def detect_vortex_regions(self, phase_shifts, threshold=0.6, device='cpu'):
        """
        Detect turbulent regions (vortex-like) where phase changes rapidly.
        Used to find zones of prediction instability or market chaos.
        """
        phase_shifts_tensor = torch.tensor(phase_shifts, device=device)
        gradients = torch.abs(torch.gradient(phase_shifts_tensor)[0])
        return torch.where(gradients > threshold)[0]

    def superfluid_energy_transfer(self, entropy_field, qpl_field, coherence_factor=0.9):
        """
        Model energy/information transfer across coherent surfaces.
        Could guide prediction field evolution or local refinement.
        """
        return entropy_field * qpl_field * coherence_factor

    def apply_superfluid_damping(self, probabilities, velocity_field, device='cpu'):
        """
        Damp or amplify collapse probabilities based on local flow stability.
        High velocity = instability = lower confidence.
        """
        probabilities_tensor = torch.tensor(probabilities, device=device)
        velocity_field_tensor = torch.tensor(velocity_field, device=device)

        damping = torch.exp(-torch.abs(velocity_field_tensor))
        return probabilities_tensor * damping

    def normalize_probabilities(self, probs, device='cpu'):
        """
        Ensure the output collapse field is normalized.
        """
        probs_tensor = torch.tensor(probs, device=device)
        probs_tensor = torch.nan_to_num(probs_tensor, nan=0.0)
        total = torch.sum(probs_tensor)
        if total <= 0:
            return torch.ones_like(probs_tensor) / len(probs_tensor)
        return probs_tensor / total

    def compute_superfluid_coherence(self, signal, device='cpu') -> torch.Tensor:
        """
        Estimates coherence of a signal using second-derivative curvature.
        Lower curvature = more coherence (laminar flow), higher = turbulence.
        Returns a scalar score between 0 (chaotic) and 1 (fully coherent).
        """
        signal_tensor = torch.tensor(signal, device=device) if not isinstance(signal, torch.Tensor) else signal.to(device)
        if signal_tensor.shape[0] < 3:
            return torch.tensor(1.0, device=device)  # not enough data to measure

        first_grad = torch.gradient(signal_tensor)[0]
        curvature = torch.gradient(first_grad)[0]
        coherence_score = torch.exp(-torch.mean(torch.abs(curvature)))
        return torch.clamp(coherence_score, 0.0, 1.0)
    
    def apply_superfluid_filter(self, signal: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        Applies a fluid-inspired smoothing filter to suppress volatility while preserving structure.
        Returns a modified signal array with turbulence damped.
        """
        if signal.ndim != 1 or signal.shape[0] < 3:
            return signal  # Not enough data to filter

        signal_tensor = torch.tensor(signal, device=device)
        first_grad = torch.gradient(signal_tensor)[0]
        curvature = torch.gradient(first_grad)[0]
        damping = torch.exp(-torch.abs(curvature))

        return signal_tensor * damping

    def energy_exchange_strength(self, prediction: float, entropy: float, coherence_factor: float = 0.9, device='cpu') -> float:
        """
        Models information transfer strength between entropy field and prediction signal.
        This simulates a coherent flow of informational energy like in a superfluid or QPL.
        
        Returns a score in [0, 1], where 1 means high energy coherence, 0 = low alignment.
        """
        prediction_tensor = torch.tensor(prediction, device=device)
        entropy_tensor = torch.tensor(entropy, device=device)

        interaction = torch.exp(-torch.abs(prediction_tensor - entropy_tensor)) * coherence_factor
        return float(torch.clamp(interaction, 0.0, 1.0).item())
    
    def superfluid_phase_variance(self, signal: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        Calculates phase variance of the signal — a proxy for turbulence.
        High second-order derivative = unstable flow; low = coherent phase.
        Returns a float [0, 1] where 1 is fully stable (low variance).
        """
        if signal.ndim != 1 or len(signal) < 3:
            return torch.tensor(1.0, device=device)  # Fully stable by default for short signals

        signal_tensor = torch.tensor(signal, device=device)
        try:
            first_grad = torch.gradient(signal_tensor)[0]
            second_deriv = torch.gradient(first_grad)[0]
        except Exception:
            return torch.tensor(1.0, device=device)  # Just in case — catch all

        variance = torch.var(second_deriv)
        phase_stability = torch.exp(-variance)
        return torch.clamp(phase_stability, 0.0, 1.0)
