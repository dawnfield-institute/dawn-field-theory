import torch
from cimm_core.learning.superfluid_dynamics import SuperfluidDynamics

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
class QuantumPotentialLayer:
    def __init__(self, lambda_qpl=0.1, damping_factor=0.98, qpl_target=0.5):
        """Initialize Quantum Potential Layer (QPL) for QBE stabilization."""
        self.qpl = 1.0  # Initial Quantum Potential
        self.lambda_qpl = lambda_qpl  # Regulation strength
        self.qpl_target = qpl_target  # Target balance point
        self.damping_factor = damping_factor  # Controls QPL adaptation rate
        self.qpl_momentum = 0.0  # Introduces a momentum term for smoother transitions
        self.prev_qpl = 1.0  # Tracks previous QPL state
        self.min_qpl = 0.05  # Prevents extreme low QPL collapse
        self.max_qpl = 2.0   # Prevents QPL from overshooting
        self.qpl_delta = 0.0  # Track QPL changes for higher-order corrections
        self.qpl_memory_buffer = []  # Buffer to store past collapse deviations
        self.memory_decay_factor = 0.9  # Decay factor for non-Markovian influence
        self.superfluid = SuperfluidDynamics()


    def compute_qpl(self, entropy_monitor, memory_module):
        """Use predicted collapse deltas to adjust QPL before deviations occur."""
        if isinstance(self.qpl, torch.Tensor):
            self.qpl = self.qpl.to(device)  # Ensure tensor is on the correct device
        if isinstance(self.qpl_target, torch.Tensor):
            self.qpl_target = self.qpl_target.to(device)  # Ensure tensor is on the correct device

        entropy_level = entropy_monitor.entropy
        collapse_deviation = entropy_monitor.track_collapse_deviation()

        # --- Entropy-Amplified Penalty Injection ---
        entropy_cost = entropy_monitor.compute_neuron_energy_cost(entropy_level)
        if not isinstance(entropy_cost, torch.Tensor):
            entropy_cost = torch.tensor(entropy_cost, dtype=torch.float32, device=device)
        entropy_weight = torch.exp(entropy_cost * 0.25)
        # To use: adjusted_loss = entropy_weight * loss
        # (You can apply this scaling in your training loop or loss computation.)

        # Store collapse deviation in memory buffer
        self.qpl_memory_buffer.append(collapse_deviation)
        if len(self.qpl_memory_buffer) > 10:  # Limit buffer size to 10
            self.qpl_memory_buffer.pop(0)

        # Compute weighted sum of past deviations
        weighted_sum = sum(
            self.memory_decay_factor ** i * dev for i, dev in enumerate(reversed(self.qpl_memory_buffer))
        )

        # Initialize refinement_delta to a default value
        refinement_delta = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Fixed-point iteration for predict_correction
        max_iterations = 5
        confidence_threshold = torch.tensor(1e-3, dtype=torch.float32, device=device)  # Convergence threshold
        for _ in range(max_iterations):
            predicted_correction = memory_module.predict_correction(entropy_level, self.qpl, collapse_deviation, refinement_delta)
            new_refinement_delta = torch.tensor(predicted_correction, dtype=torch.float32, device=device)
            if abs(new_refinement_delta - refinement_delta) < confidence_threshold:
                break
            refinement_delta = new_refinement_delta

        # Apply the predicted correction to stabilize collapse structuring
        self.qpl_momentum = 0.95 * self.qpl_momentum + 0.05 * weighted_sum  # Reduced correction rate
        refined_qpl = self.qpl + (self.qpl_target - self.qpl) * (0.94 + refinement_delta)  # 🔥 Reduce aggressive shifts
        refined_qpl -= self.qpl_momentum * (0.004 + refinement_delta)  # 🔧 Further stabilize momentum

        # Amplify response in regions with sharp entropy deltas (like high prediction zones)
        if collapse_deviation > 0.8:  # tweak threshold experimentally
            refined_qpl += 0.015 * collapse_deviation  # sharper push

        # Introduce damping factor to prevent oscillations
        damping_factor = 0.985  # Reduce oscillations further (was 0.98)
        refined_qpl *= damping_factor

        if hasattr(self, "qpl_memory"):
            self.qpl_memory = 0.85 * self.qpl_memory + 0.15 * refined_qpl  # 🔧 Smooth QPL transitions
        else:
            self.qpl_memory = refined_qpl

        self.qpl = 0.87 * self.qpl + 0.13 * self.qpl_memory  # 🔧 Reduce weight of past QPL values
        # Return both qpl and entropy_weight for use in training
        return self.qpl, entropy_weight

    def refine_qpl(self, qpl_value, deviation):
        """Refines QPL behavior using delta tracking corrections."""
        delta_refinement = torch.exp(-deviation) * 0.05
        return qpl_value * (1 + delta_refinement)

    def compute_imbalance(self, energy_change, info_change):
        """Calculate deviation from QBE balance."""
        qpl_current = self.compute_qpl(energy_change, info_change)
        self.imbalance = abs((energy_change + info_change) - (self.lambda_qpl * qpl_current))
        return self.imbalance

    def compute_energy_information_balance(self, entropy_monitor):
        """
        Computes the energy-information balance, ensuring neurons expand only when efficient.
        """
        entropy_level = entropy_monitor.entropy
        energy_cost = entropy_monitor.compute_neuron_energy_cost(entropy_level)

        # ✅ Normalize balance between entropy and energy constraints
        energy_information_ratio = torch.exp(-energy_cost) * (1 + torch.tanh(2 * entropy_level))

        return energy_information_ratio

    def adjust_parameters(self, entropy_monitor, adaptive_controller, bayesian_optimizer, iteration_step, memory_module):
        """
        QPL now self-tunes its own equilibrium state and correction intensity.
        """
        qbe_feedback = self.compute_qpl(entropy_monitor, memory_module)

        # If compute_qpl returns a tuple (e.g., (qpl, entropy_weight)), extract the first element
        if isinstance(qbe_feedback, (tuple, list)):
            qbe_feedback = qbe_feedback[0]
        # Ensure qbe_feedback is a tensor or float
        if isinstance(qbe_feedback, torch.Tensor):
            qbe_feedback_scalar = qbe_feedback
        else:
            qbe_feedback_scalar = torch.tensor(qbe_feedback, dtype=torch.float32)

        # Ensure qbe_feedback does not cause overflow
        if not torch.isfinite(qbe_feedback_scalar).all():
            qbe_feedback_scalar = torch.tensor(0.0, dtype=torch.float32)  # Reset to neutral if unstable

        # Adjust learning rate safely
        adaptive_controller.learning_rate *= max(0.9, min(1.1, 1 + 0.12 * qbe_feedback_scalar))

        # Adjust Bayesian optimizer safely
        bayesian_optimizer.qpl_constraint *= max(0.9, min(1.1, 1 + 0.10 * qbe_feedback_scalar))

        # ✅ Introduce higher-order correction term
        higher_order_correction = 0.03 * self.qpl_delta  # 🔧 Reduce high-order compensation from 0.05
        adaptive_controller.learning_rate *= (1 + higher_order_correction)  # ✅ Adjust learning rate moderately
        bayesian_optimizer.qpl_constraint *= (1 + higher_order_correction)  # ✅ Constrain QPL correction

        #print(f"QPL Regulating: qpl={self.qpl:.4f}, target={self.qpl_target:.4f}, damping={self.damping_factor:.4f}")

    def tune_parameters(self, entropy_monitor, adaptive_controller, bayesian_optimizer, rl_agent, iteration_step, memory_module):
        """
        Uses QBE feedback to adjust entropy smoothing, RL updates, and Bayesian constraints.
        """
        qbe_feedback = self.compute_qpl(entropy_monitor, memory_module)

        # ✅ Ensure qbe_feedback is a scalar float or tensor
        if isinstance(qbe_feedback, (list, tuple)):
            while isinstance(qbe_feedback, (list, tuple)):
                if len(qbe_feedback) == 0:
                    qbe_feedback = 0.0
                    break
                qbe_feedback = qbe_feedback[0]
        if isinstance(qbe_feedback, torch.Tensor):
            qbe_feedback = qbe_feedback.item()
        qbe_feedback = float(qbe_feedback)

        # ✅ Ensure qbe_feedback is on the same device as adaptive_controller.learning_rate
        if isinstance(adaptive_controller.learning_rate, torch.Tensor):
            adaptive_controller.learning_rate = adaptive_controller.learning_rate.to(device)
        else:
            adaptive_controller.learning_rate = torch.tensor(adaptive_controller.learning_rate, device=device, dtype=torch.float32)

        adaptive_controller.learning_rate *= (1 + 0.06 * qbe_feedback)  # 🔥 Lowered from 0.08

        # ✅ Dynamically adjust QPL stability
        self.lambda_qpl = max(0.06, min(0.18, self.lambda_qpl * (1 + 0.03 * qbe_feedback)))

        # ✅ Prevent RL overcorrection
        if hasattr(rl_agent, "learning_rate"):
            rl_agent.learning_rate *= (1 + 0.05 * qbe_feedback)

        # 🔥 Adjust RL Collapse Suppression Using QBE Feedback
        if hasattr(rl_agent, "collapse_suppression"):
            rl_agent.collapse_suppression = max(0.85, min(1.1, rl_agent.collapse_suppression * (1 + 0.04 * qbe_feedback)))

        # 🔥 Keep Bayesian Optimizer Adaptive Without Overreacting
        if hasattr(bayesian_optimizer, "qpl_constraint"):
            bayesian_optimizer.qpl_constraint *= (1 + 0.05 * qbe_feedback)  # 🔥 Lowered from 0.07

        # 🔥 Adjust Learning Rate with More Control
        if hasattr(adaptive_controller, "learning_rate"):
            adaptive_controller.learning_rate *= (1 + 0.06 * qbe_feedback)  # 🔥 Lowered from 0.08
            
        # 🔥 Adjust QPL Regulatory Strength for More Stability
        self.lambda_qpl = max(
            0.06, min(0.18, self.lambda_qpl * (1 + 0.03 * qbe_feedback))
        )

        # Adjust entropy decay factor dynamically (exists)
        if hasattr(entropy_monitor, "decay_factor"):
            entropy_monitor.decay_factor = max(
                0.9, min(0.99, entropy_monitor.decay_factor * (1 + 0.03 * qbe_feedback))
            )

        if hasattr(self, "superfluid"):
            history = entropy_monitor.past_entropies
            superfluid_score = self.superfluid.compute_superfluid_coherence(history[-10:] if len(history) >= 3 else history)
            adaptive_controller.learning_rate *= (1 + 0.05 * superfluid_score)


        # Adjust entropy-driven momentum dynamically (exists)
        if hasattr(entropy_monitor, "momentum"):
            entropy_monitor.momentum = max(
                0.5, min(0.95, entropy_monitor.momentum * (1 + 0.02 * qbe_feedback))
            )

        # Adjust gradient clipping dynamically based on entropy variance
        bayesian_optimizer.clip_grad = max(
            0.5, min(2.0, bayesian_optimizer.clip_grad * (1 + 0.05 * qbe_feedback))
        )

        # ✅ Introduce higher-order correction term
        higher_order_correction = 0.05 * self.qpl_delta
        adaptive_controller.learning_rate *= (1 + higher_order_correction)
        bayesian_optimizer.qpl_constraint *= (1 + higher_order_correction)

        #print(f"QPL Tuning: LR={adaptive_controller.learning_rate:.6f}, Decay={entropy_monitor.decay_factor:.4f}, "
        #      f"Momentum={entropy_monitor.momentum:.4f}, QPL Constraint={bayesian_optimizer.qpl_constraint:.4f}, "
        #      f"Lambda_QPL={self.lambda_qpl:.4f}")

    def apply_qbe_learning_correction(self, entropy_monitor, controller):
        """
        Refines entropy corrections to prevent small-scale oscillations while retaining structure.
        """
        entropy_level = entropy_monitor.entropy
        entropy_variance = torch.std(torch.tensor(entropy_monitor.past_entropies[-15:], device=device)) if len(entropy_monitor.past_entropies) >= 15 else torch.tensor(0.1, device=device)

        # 🔥 Less Aggressive Damping to Keep Structural Details
        damping_factor = 1.0 / (1.0 + entropy_variance * 0.7)  # 🔥 Adjusted from 0.8 to 0.7

        # 🔥 Allow More Variability in Entropy Correction Strength
        correction_strength = max(0.98, min(1.03, 1.0 + 0.04 * entropy_level))  # 🔧 Reduced from 0.05

        # Apply correction strength based on localized entropy trends
        for param_group in controller.optimizer_instance.param_groups:
            param_group['lr'] *= damping_factor * correction_strength  

        return damping_factor * correction_strength

    def report_entropy_change(self, entropy_value):
        """Update QPL based on entropy fluctuations."""
        self.qpl += entropy_value * self.lambda_qpl
        self.imbalance = abs(self.qpl - entropy_value)

    def enforce_quantum_constraints(self, scaling_factor):
        """
        Prevents excessive neuron expansion or contraction outside quantum-defined limits.
        """
        # ✅ Define max/min expansion limits
        max_limit = 1.15  
        min_limit = 0.85  

        # ✅ Apply quantum constraint limits
        scaling_factor = max(min_limit, min(max_limit, scaling_factor))

        return scaling_factor

    def log_status(self):
        """Log QPL, imbalance, and key parameters."""
        print(f"QPL: {self.qpl:.5f}, Imbalance: {self.imbalance:.5f}")

    def quantum_potential_adjustment(self, entropy_value, qfi_score):
        """
        Adjusts quantum potential corrections based on entropy balance.
        
        Args:
            entropy_value (float): Current entropy level.
            qfi_score (float): Quantum Fisher Information stability score.

        Returns:
            float: Adjusted potential correction factor.
        """
        # Quantum potential scaling factor
        quantum_adjustment = torch.exp(-qfi_score) * torch.sin(entropy_value)
        return quantum_adjustment
