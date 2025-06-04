import numpy as np
import torch
import random
from entropy_monitoring import EntropyMonitor
from quantum_potential_layer import QuantumPotentialLayer
from decimal import Decimal, ROUND_HALF_UP
from scipy.ndimage import median_filter
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Add this import
from skopt import gp_minimize  # Add Bayesian optimization import
from skopt.space import Real  # Add search space import

def get_device():
    """Returns the best available device, with fallback to CPU if necessary."""
    if torch.cuda.is_available():
        try:
            # Try using NCCL for distributed training
            torch.cuda.nccl_version()
            return torch.device("cuda")
        except Exception as e:
            print(f"NCCL not available, falling back to CPU: {e}")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

class SelfAdaptiveLearningController:
    def __init__(self, model, initial_entropy, learning_rate, lambda_factor, loss_fn=None, clip_grad=1.0):
        """
        Self-Adaptive Learning Controller using entropy-energy regulation.

        Parameters:
        - model: PyTorch model instance to update.
        - initial_entropy: Starting entropy value.
        - learning_rate: Base learning rate.
        - lambda_factor: Quantum potential regulation factor.
        - loss_fn: Custom loss function (default: MSELoss).
        - clip_grad: Maximum gradient norm for stabilization.
        """
        self.model = model.to(device)  # Ensure model is on the correct device
        self.entropy_monitor = EntropyMonitor(initial_entropy, learning_rate)
        self.qpl = QuantumPotentialLayer(input_dim=10, output_dim=10, scaling_factor=lambda_factor)  # Provide appropriate dimensions

        # Default to MSE loss if no custom loss function is provided
        self.loss_fn = loss_fn if loss_fn else torch.nn.MSELoss()

        # Optimizer initialized once and reused
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip_grad = clip_grad  # Gradient clipping threshold
        self.learning_rate = learning_rate  # Add learning_rate attribute
        self.entropy_history = []  # Add entropy history tracking
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=0.0001)  # Add scheduler
        self.gradient_history = []  # Track gradient history

        # Initialize Bayesian optimization for hyperparameter tuning
        self.hyperparam_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Real(0.0, 0.5, name='dropout'),
            Real(0.0, 0.9, name='momentum')
        ]

    def update_model(self, data, energy):
        """Update the model based on entropy-energy regulation."""
        model = self.model  # Define the model variable
        current_entropy = self.entropy_monitor.calculate_entropy(data)
        self.entropy_history.append(current_entropy)  # Track entropy history
        self.entropy_history = self.entropy_history[-100:]  # Keep last 100 entropy values

        # Apply rolling median filter to smooth entropy variations
        smoothed_entropy = median_filter(self.entropy_history, size=5)[-1]

        # Ensure smoothed_entropy is non-negative before applying log1p
        smoothed_entropy = max(smoothed_entropy, 0)
        adjusted_entropy = np.log1p(smoothed_entropy)

        # Ensure entropy never becomes zero
        adjusted_entropy = max(adjusted_entropy, 1e-6)

        # Dynamically adjust learning rate
        learning_rate = self.adjust_learning_rate(adjusted_entropy)

        # Print entropy and learning rate for monitoring
        print(f"Entropy: {adjusted_entropy}, Learning Rate: {learning_rate}")

        # Dynamic learning rate scaling based on entropy rate of change
        delta_entropy = current_entropy - self.entropy_history[-1] if len(self.entropy_history) > 1 else 0
        learning_rate = np.clip(learning_rate * (1 + delta_entropy * 0.05), 1e-6, 0.01)

        # Update optimizer's learning rate dynamically
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Forward pass
        data = data.to(device)  # Ensure data is on the correct device
        output = self.model(data)
        target = self.get_target_smoothing(output)  # Use smarter target handling
        loss = self.loss_fn(torch.nan_to_num(output), torch.nan_to_num(target))

        # Backward pass with gradient clipping
        self._update_model(loss)

        # Step the scheduler
        self.scheduler.step(loss)

        return self.model, loss.item()

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
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def compute_entropy_threshold(self):
        if (len(self.entropy_history) >= 50):
            sigma_entropy = np.std(self.entropy_history[-50:])
            return 2.0 * sigma_entropy  # Dynamic threshold
        return 0.1  # Default if insufficient history

    def adjust_learning_rate(self, target_entropy):
        """
        Dynamically adjusts learning rate with smooth exponential decay to prevent aggressive changes.
        Uses adjusted_entropy for smoother, entropy-aware adaptation.
        """
        current_entropy = self.entropy_monitor.entropy  # Fetch the latest entropy reading

        # Compute the rolling entropy change threshold
        threshold = self.compute_entropy_threshold()

        # Adaptive Smoothing for Entropy
        alpha = 0.95 if abs(current_entropy - target_entropy) > threshold else 0.85
        adjusted_entropy = alpha * target_entropy + (1 - alpha) * current_entropy  # ✅ Now adjusted entropy is used

        # Adjust decay factor using adjusted entropy
        decay_factor = np.exp(-abs(adjusted_entropy - current_entropy) / 10)  # More gradual decay

        # Apply adaptive decay
        self.learning_rate = Decimal(str(self.learning_rate)) * Decimal(str(0.95 + 0.05 * decay_factor))

        # Precision adjustment for stability
        precision_adjustment = Decimal("0.00001")  # Fine-grained updates
        self.learning_rate += precision_adjustment

        # Dynamic Minimum Learning Rate Floor (based on adjusted entropy)
        min_lr = max(0.01, 0.02 * (0.9 ** float(adjusted_entropy)))  # Gradual adjustment
        self.learning_rate = max(min(self.learning_rate, Decimal("0.02")), Decimal(str(min_lr)))

        # Round precisely for numerical stability
        self.learning_rate = self.learning_rate.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
        
        # Convert Decimal back to float
        self.learning_rate = float(self.learning_rate)
        
        # Ensure smooth learning rate transitions
        min_lr = max(1e-6, self.learning_rate * 0.95)
        max_lr = min(0.02, self.learning_rate * 1.05)
        self.learning_rate = np.clip(self.learning_rate, min_lr, max_lr)

        return self.learning_rate

    def _update_model(self, loss):
        loss.backward()

        # Apply adaptive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        if grad_norm > 1.0:
            self.learning_rate *= 0.95  # Reduce LR slightly when large gradients are detected

        with torch.no_grad():
            for param in self.model.parameters():
                param -= self.learning_rate * param.grad

        self.model.zero_grad()

        # Track gradient history to prevent overfitting
        self.gradient_history.append(grad_norm)
        self.gradient_history = self.gradient_history[-100:]  # Keep last 100 gradient norms

    def optimize_hyperparameters(self):
        """Optimize hyperparameters using Bayesian optimization."""
        def objective(params):
            learning_rate, dropout, momentum = params
            self.optimizer.param_groups[0]['lr'] = learning_rate
            # Assume model has dropout and momentum attributes
            self.model.dropout = dropout
            self.model.momentum = momentum
            # Perform a single update step and return the loss
            data = torch.randn(100, 10).to(device)  # Example data
            energy = 0.5  # Example energy
            _, loss = self.update_model(data, energy)
            return loss

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
    energy = 0.5
    model = DummyModel().to(device)  # Ensure model is on the correct device
    controller = SelfAdaptiveLearningController(model, initial_entropy=1.0, learning_rate=0.01, lambda_factor=0.1)
    updated_model, loss = controller.update_model(data, energy)
    print(f"Model updated successfully with loss: {loss}")

    # Optimize hyperparameters
    controller.optimize_hyperparameters()