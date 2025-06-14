import torch
import random

from decimal import Decimal, ROUND_HALF_UP
from scipy.ndimage import median_filter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Add this import
from skopt import gp_minimize  # Add Bayesian optimization import
from skopt.space import Real  # Add search space import
from cimm_core.optimization.bayesian_optimizer import BayesianOptimizer  # Update this import
from cimm_core.entropy.entropy_monitor import EntropyMonitor  # Update this import
from collections import deque
from cimm_core.learning.superfluid_dynamics import SuperfluidDynamics
from cimm_core.utils import get_device


device = get_device()

def gradient(x):
    return torch.cat([x[:1] - x[:1], x[1:] - x[:-1]])

def einstein_energy_error(error, mass=1.0, c=299792458):
    return mass * (c ** 2) * error


device = get_device()
print(f"Using device: {device}")

class MetaController:
    def __init__(self, window_size=5):
        self.entropy_history = deque(maxlen=window_size)

    def smooth_entropy(self, entropy_value):
        """Computes a smoothed version of entropy to prevent unnecessary oscillations."""
        self.entropy_history.append(entropy_value)
        return torch.mean(torch.tensor(self.entropy_history))

    def dampen_learning_rate(self, learning_rate, entropy_smooth, damping_factor=0.05):
        """Applies controlled damping to prevent learning rate oscillations."""
        if len(self.entropy_history) < 2:
            return learning_rate  # Not enough data to smooth

        entropy_delta = abs(self.entropy_history[-1] - self.entropy_history[-2])

        # Apply damping only if entropy fluctuations are too high
        if (entropy_delta > 0.1):
            return learning_rate * (1 - damping_factor)

        return learning_rate

class AdaptiveController:
    def __init__(self, model, optimizer, entropy_monitor, initial_entropy, learning_rate, lambda_factor, qpl_layer, lambda_qbe=0.1, memory_module=None):
        self.model = model.to(device)  # Move model to GPU
        self.optimizer_instance = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Initialize PyTorch optimizer
        self.entropy_monitor = entropy_monitor
        self.initial_entropy = initial_entropy
        self.learning_rate = learning_rate
        self.lambda_factor = lambda_factor
        self.qpl_layer = qpl_layer
        self.h_bar = 1.0  # Scaling constant
        self.dt = 0.01  # Time step for evolution
        lr_tensor = torch.tensor(self.learning_rate, dtype=torch.float32)  # Convert learning_rate to tensor
        self.psi = torch.tensor([torch.cos(lr_tensor), torch.sin(lr_tensor)])  # Represent as a 2D real tensor
        self.min_lr = 1e-6  # Prevent collapse
        self.max_lr = 0.1   # Prevent runaway scaling
        self.meta_controller = MetaController()  # Initialize MetaController
        self.lambda_qbe = lambda_qbe  # Scaling factor for QBE corrections
        self.smoothed_entropy_gradient = 0.0  # Initialize smoothed entropy gradient
        self.prev_phase_shift = 0.0  # Track previous phase shifts for stability
        self.prev_learning_rate = learning_rate  # Track previous LR for stability
        self.phase_momentum = 0.9  # Reduce phase oscillations
        self.min_lr = 1e-5  # Prevent collapse
        self.max_lr = 0.05  # Lowered max to prevent spikes
        self.superfluid = SuperfluidDynamics()
        self.memory_module = memory_module  # Memory module for QPL layer


    def update_model(self, data, targets):
        """Update the model based on the provided data and energy."""
        self.model.train()
        self.optimizer_instance.zero_grad()
        predictions = self.model(data)
        # Ensure predictions and targets are on the same device
        predictions = predictions.to(targets.device)

        # Ensure targets is a tensor and matches the shape of predictions
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=predictions.dtype, device=predictions.device)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)  # Add a dimension if targets is 1D

        loss = self.compute_qbe_loss(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Clip gradients
        self.optimizer_instance.step()

        # 🔥 Ensure learning rate is updated dynamically
        self.update_learning_rate()
        
        return loss.item()

    def compute_qbe_loss(self, predictions, targets):
        """
        Enhances QBE loss with stronger local entropy penalties for finer adjustments.
        """
        # Ensure targets are on the same device as predictions
        targets = targets.to(predictions.device)
        base_loss = torch.nn.MSELoss()(predictions, targets)  

        # Compute entropy values
        S_actual = self.entropy_monitor.entropy
        S_target = self.qpl_layer.compute_qpl(self.entropy_monitor, getattr(self, "memory_module", None))
        # Fix: If S_target is a tuple, extract the first element or convert to float
        if isinstance(S_target, (tuple, list)):
            S_target = S_target[0]
        if isinstance(S_target, torch.Tensor) and S_target.numel() > 1:
            S_target = S_target.mean()
        if isinstance(S_target, torch.Tensor):
            S_target = S_target.item()
        if isinstance(S_actual, torch.Tensor):
            S_actual = S_actual.item()

        # 🔥 NEW: Strengthen local entropy force term
        local_entropy_change = abs(S_actual - S_target) * 0.8  # 🔥 Increased penalty

        # Compute QBE loss with stronger local entropy correction
        qbe_loss = base_loss + self.lambda_qbe * local_entropy_change

        return qbe_loss
    


    def quantum_wave_learning_rate(self):
        """
        Adjusts the learning rate using entropy-aware quantum wave modulation.
        Applies Schrödinger-inspired phase shifts and wave amplitude corrections.
        """

        # Compute entropy-based phase shift modulation
        entropy_level = self.entropy_monitor.entropy
        entropy_mean = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-50:])) if len(self.entropy_monitor.past_entropies) >= 50 else entropy_level
        entropy_variance = torch.std(torch.tensor(self.entropy_monitor.past_entropies[-20:])) if len(self.entropy_monitor.past_entropies) >= 20 else 0.02
        entropy_variance_tensor = torch.tensor(entropy_variance, dtype=torch.float32)  # Convert to tensor
        entropy_variance = torch.clamp(entropy_variance_tensor, max=0.2).item()  # Clamp and convert back to float

        # 🔥 NEW: Apply entropy-based phase scaling
        phase_scaling = 1 + 0.01 * entropy_variance  
        phase_shift_input = torch.tensor(entropy_mean * torch.pi * phase_scaling, dtype=torch.float32)  # Convert to tensor
        phase_shift = torch.sin(phase_shift_input) * 0.3  

        # 🔥 NEW: Compute quantum wave amplitude adjustment
        wave_amplitude_input = torch.tensor(entropy_mean * torch.pi / 2 * phase_scaling, dtype=torch.float32)  # Convert to tensor
        wave_amplitude = 1 + 0.01 * torch.cos(wave_amplitude_input)  
        quantum_adjustment = wave_amplitude * torch.exp(1j * phase_shift).real

        # 🔥 NEW: Apply QBE scaling and prevent runaway effects
        # Use superfluid coherence instead of statistical damping
        history = torch.tensor(self.entropy_monitor.past_entropies)
        coherence_score = self.superfluid.compute_superfluid_coherence(history[-10:] if len(history) >= 3 else history)

        self.learning_rate *= coherence_score

        #quantum_correction = (1 + 0.2 * (quantum_adjustment - 1)) * qbe_feedback  

        # Apply learning rate modulation
        #self.learning_rate *= quantum_correction

        # 🔧 NEW: Feynman-inspired probabilistic damping
        feynman_exploration_input = torch.tensor(-entropy_variance * 5, dtype=torch.float32)  # Convert to tensor
        feynman_exploration = torch.exp(feynman_exploration_input)

        self.learning_rate *= feynman_exploration

        # 🔧 NEW: Momentum-aware recovery
        tanh_input = torch.tensor(entropy_variance * 5, dtype=torch.float32)  # Convert to tensor
        learning_rate_recovery = 1 + 0.05 * (1 - torch.tanh(tanh_input))
        self.learning_rate *= learning_rate_recovery

        log_variance_input = torch.tensor(entropy_variance, dtype=torch.float32)  # Convert to tensor
        log_variance_factor = torch.log1p(log_variance_input)  # 🔥 Log smoothing prevents overreaction
        learning_rate_adjustment = 1 / (1 + torch.exp(-5 * log_variance_factor))
        self.learning_rate *= learning_rate_adjustment

        self.learning_rate = torch.clamp(self.learning_rate, self.min_lr, self.max_lr)  # 🔧 Cap learning rate

        for param_group in self.optimizer_instance.param_groups:
            param_group['lr'] = self.learning_rate.item()  

        return self.learning_rate.item()

    def hamiltonian_operator(self, entropy_level):
        """Define H as an entropy-dependent Hamiltonian with bounded adjustments."""
        return torch.pi * torch.tanh(entropy_level * 0.1)  # 🔥 Apply tanh to prevent extreme values

    def schrodinger_learning_rate(self):
        """Refined Schrödinger learning rate with **soft phase shifts and controlled scaling**."""
        entropy_level = self.entropy_monitor.entropy
        previous_entropy = self.entropy_monitor.prev_entropy

        # ✅ **Compute Schrödinger-inspired reinforcement**
        H = self.hamiltonian_operator(entropy_level)
        lambda_qpl = self.qpl_layer.compute_qpl(self.entropy_monitor)

        # ✅ **Apply Soft Phase Shifting**
        phase_shift = torch.tanh((entropy_level - previous_entropy) * 0.1) * 0.1  # 🔥 Capped phase shift

        # ✅ **Controlled Quantum Wave Amplitude**
        wave_amplitude = 1.0 + 0.01 * torch.cos(entropy_level * torch.pi)  # 🔥 Soft correction
        quantum_adjustment = wave_amplitude * torch.exp(1j * phase_shift).real  # 🔥 Stabilized phase shift

        # ✅ **Entropy-Based Learning Rate Dampening**
        damping_factor = 1.0 / (1.0 + torch.exp(-torch.abs(entropy_level - previous_entropy) * 3))  # 🔥 Softer corrections
        quantum_correction = (1 + 0.05 * (quantum_adjustment - 1)) * damping_factor * lambda_qpl

        # ✅ **Apply Bounded Learning Rate Correction**
        new_learning_rate = self.learning_rate * quantum_correction

        entropy_variance = torch.var(torch.tensor(self.entropy_monitor.past_entropies[-20:])) if len(self.entropy_monitor.past_entropies) >= 20 else 0.05
        log_variance_factor = torch.log1p(entropy_variance)  # 🔥 Log smoothing prevents overreaction
        learning_rate_adjustment = 1 / (1 + torch.exp(-5 * log_variance_factor))
        self.learning_rate *= learning_rate_adjustment

        self.prev_learning_rate = self.learning_rate  # Track previous LR

        self.learning_rate = torch.clamp(self.learning_rate, self.min_lr, self.max_lr)  # 🔧 Cap learning rate

        # ✅ **Apply New Learning Rate**
        for param_group in self.optimizer_instance.param_groups:
            param_group['lr'] = self.learning_rate.item()

        return self.learning_rate.item()

    def update_learning_rate(self):
        """
        Uses QBE-driven entropy feedback to dynamically adjust learning rate.
        """
        entropy_gradient = self.entropy_monitor.entropy_gradient

        # ✅ Compute entropy variance & QBE stabilization factor
        entropy_variance = torch.var(torch.tensor(self.entropy_monitor.past_entropies[-20:])) if len(self.entropy_monitor.past_entropies) >= 20 else 0.05
        qbe_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor, getattr(self, "memory_module", None))

        # --- Fix: Ensure qbe_feedback is a scalar (float or tensor), not a sequence ---
        if isinstance(qbe_feedback, (tuple, list)):
            qbe_feedback = qbe_feedback[0]
        if isinstance(qbe_feedback, torch.Tensor) and qbe_feedback.numel() > 1:
            qbe_feedback = qbe_feedback.mean()
        if isinstance(qbe_feedback, torch.Tensor):
            qbe_feedback = qbe_feedback.item()
        qbe_feedback = float(qbe_feedback)

        # ✅ Dynamically modulate learning rate based on QBE feedback
        adaptive_scaling = 1 / (1 + torch.exp(-5 * torch.tensor(entropy_variance * qbe_feedback, dtype=torch.float32, device=device)))
        self.learning_rate *= adaptive_scaling

        # 🔥 NEW: Einstein energy-based adjustment
        c = 299792458
        mass = 1.0
        error = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-10:])) if len(self.entropy_monitor.past_entropies) >= 10 else 0.01
        energy_equiv = mass * (c ** 2) * error
        self.learning_rate *= 1 / (1 + energy_equiv * 1e-12)

        # ✅ Ensure learning rate remains within safe bounds
        self.learning_rate = torch.clamp(self.learning_rate, self.min_lr, self.max_lr)  # 🔧 Cap learning rate

        for param_group in self.optimizer_instance.param_groups:
            param_group['lr'] = self.learning_rate.item()

    def compute_loss(self, data):
        """Compute the loss for the given data."""
        data = data.to(device)  # Ensure data is on the correct device
        output = self.model(data)
        target = self.get_target_smoothing(output)
        loss = self.loss_fn(torch.nan_to_num(output), torch.nan_to_num(target))
        return loss

    def get_target_smoothing(self, output):
        """Apply a moving average of previous predictions as targets."""
        if not hasattr(self, 'previous_output'):
            self.previous_output = torch.zeros_like(output)
        smoothed_target = 0.9 * self.previous_output + 0.1 * output
        self.previous_output = output.detach()
        return smoothed_target

    def reset_learning_rate(self):
        """Reset the learning rate to its initial value."""
        self.learning_rate = self.entropy_monitor.learning_rate
        for param_group in self.optimizer_instance.param_groups:
            param_group['lr'] = self.learning_rate

    def compute_entropy_threshold(self):
        """
        Computes entropy thresholds dynamically to prevent excessive neuron expansion.
        """
        entropy_variance = torch.var(torch.tensor(self.entropy_monitor.past_entropies))

        # Set adaptive threshold based on entropy variance
        entropy_threshold = 0.1 + 0.05 * torch.tanh(3 * entropy_variance)

        return entropy_threshold.item()

    def _update_model(self, loss):
        loss.backward()

        # Compute entropy-based gradient penalty
        entropy_balance = abs(self.entropy_monitor.entropy - self.qpl_layer.compute_qpl(self.entropy_monitor))
        grad_penalty = torch.sigmoid(torch.tensor(entropy_balance))  # 🔥 Apply non-linear scaling

        # Apply adaptive gradient clipping based on entropy balance
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0 / (1 + grad_penalty.item()))

        if grad_norm > 1.0:
            self.learning_rate *= 0.95  # Reduce LR slightly when large gradients are detected

        with torch.no_grad():
            for param in self.model.parameters():
                # Apply QBE entropy correction directly to weight updates
                param -= self.learning_rate * param.grad * (1 - 0.1 * grad_penalty.item())

        self.model.zero_grad()

        # Track gradient history to prevent overfitting
        self.gradient_history.append(grad_norm)
        self.gradient_history = self.gradient_history[-100:]  # Keep last 100 gradient norms

    def optimize_hyperparameters(self):
        """Optimize hyperparameters using Bayesian optimization, constrained by entropy trends."""
        def objective(params):
            learning_rate, dropout, momentum = params
            self.optimizer_instance.param_groups[0]['lr'] = learning_rate
            self.model.dropout = dropout
            self.model.momentum = momentum

            # Compute entropy variance for adaptive constraints
            entropy_variance = torch.std(torch.tensor(self.entropy_monitor.past_entropies[-50:])) if len(self.entropy_monitor.past_entropies) >= 50 else 0.1
            entropy_penalty = 0.1 * entropy_variance * (1 + torch.exp(-entropy_variance))  # 🔥 Scale dynamically

            # Perform a single update step and return the loss with entropy penalty
            data = torch.randn(100, 10).to(device)
            targets = torch.randn(100, 1).to(device)
            loss = self.update_model(data, targets) + entropy_penalty

            return loss.item()

        res = gp_minimize(objective, self.hyperparam_space, n_calls=50, random_state=0)
        best_params = res.x
        print(f"Optimized Hyperparameters: Learning Rate={best_params[0]}, Dropout={best_params[1]}, Momentum={best_params[2]}")

# Example usage
if __name__ == "__main__":

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(10, 1)
            self.dropout = 0.0
            self.momentum = 0.0

        def forward(self, x):
            return self.linear(x)

    data = torch.randn(100, 10)
    targets = torch.randn(100, 1)  # Example targets
    model = DummyModel().to(device)  # Ensure model is on the correct device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    entropy_monitor = EntropyMonitor(initial_entropy=0.1, learning_rate=0.01)  # Create an instance of EntropyMonitor
    controller = AdaptiveController(model, optimizer, entropy_monitor, initial_entropy=0.1, learning_rate=0.01, lambda_factor=0.5, qpl_layer=None)  # Pass entropy monitor
    updated_model, loss = controller.update_model(data, targets)
    print(f"Model updated successfully with loss: {loss}")

    # Optimize hyperparameters
    controller.optimize_hyperparameters()
