import numpy as np
import torch
import random
import logging
from scipy.stats import skew, entropy
from scipy.ndimage import median_filter  # Add this import
import zlib
from quantum_potential_layer import QuantumPotentialLayer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class EntropyMonitor:
    def __init__(self, initial_entropy=1.0, learning_rate=0.01, min_lr=1e-6, max_lr=0.1, entropy_smoothing=0.99, decay_factor=0.98):
        """
        Monitors entropy and dynamically adjusts the learning rate.

        Parameters:
        - initial_entropy: Starting entropy value.
        - learning_rate: Initial learning rate.
        - min_lr: Minimum learning rate to prevent collapse.
        - max_lr: Maximum learning rate to prevent runaway scaling.
        - entropy_smoothing: Exponential moving average factor for entropy stabilization.
        - decay_factor: Controls the speed of exponential decay.
        """
        self.entropy = initial_entropy
        self.smoothed_entropy = initial_entropy
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.entropy_smoothing = entropy_smoothing
        self.past_entropies = []
        self.past_outputs = []
        self.alpha = 0.3  # Initial smoothing factor
        self.min_entropy = 0.01
        self.max_entropy = 5.0  # Correct max entropy value
        self.prev_entropy = None
        self.smoothing_factor = 0.9
        self.entropy_decay = 0.99
        self.MIN_LR = min_lr
        self.MAX_LR = max_lr
        self.decay_factor = decay_factor  # Control decay speed
        self.momentum = 0.9  # Momentum for learning rate smoothing
        self.qpl = QuantumPotentialLayer(input_dim=10, output_dim=10)  # Provide appropriate dimensions
        self.historical_entropies = []

    def compute_entropy_decay(self, current_entropy: float, decay_factor: float) -> float:
        """Smooth entropy decay using exponentially weighted moving averages (EWMA)"""
        smoothed_entropy = decay_factor * current_entropy + (1 - decay_factor) * self.smoothed_entropy
        return smoothed_entropy

    def adaptive_entropy_cap(self, entropy: float, threshold: float) -> float:
        """Apply adaptive capping to entropy values to maintain stability"""
        variance = np.var(self.past_entropies)
        cap = threshold * (1 + variance)
        return min(entropy, cap)

    def entropy_gradient_clipping(self, gradient: torch.Tensor, entropy_variance: float, max_clip: float = 0.1) -> torch.Tensor:
        """Apply entropy-aware gradient clipping for stability"""
        clip_value = max_clip * (1 + entropy_variance)
        return torch.clamp(gradient, -clip_value, clip_value)

    def validate_entropy_metrics(self, observed_entropy: float, expected_entropy: float) -> bool:
        """Check if entropy calculations remain within acceptable limits"""
        return abs(observed_entropy - expected_entropy) < 0.1

    def calculate_entropy(self, data):
        """Compute entropy and apply exponential smoothing to avoid fluctuations."""
        raw_entropy = self._calculate_entropy(data)
        self.smoothed_entropy = self.compute_entropy_decay(raw_entropy, self.entropy_smoothing)
        
        # Apply median filter to reduce noise
        self.past_entropies.append(self.smoothed_entropy)
        if len(self.past_entropies) > 10:
            self.past_entropies.pop(0)
        filtered_entropy = median_filter(self.past_entropies, size=5)[-1]

        logging.info(f"Raw Entropy: {raw_entropy}, Smoothed Entropy: {filtered_entropy}")
        return filtered_entropy

    def _calculate_entropy(self, data):
        """Calculate entropy of the given data with numerical stability improvements."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()

        if data.size == 0:
            return 1e-6

        data = data.flatten()
        
        # Detect skewness and adjust transformation
        data_skew = skew(data)
        if abs(data_skew) > 2:  # Threshold for high skew
            data = np.log1p(data - np.min(data) + 1)  # Log transform for stability

        min_value, max_value = np.min(data), np.max(data)
        if min_value == max_value:
            return 1e-6  # Avoid division by zero

        norm_data = (data - min_value) / (max_value - min_value + 1e-6)  # Avoid zero division
        hist, _ = np.histogram(norm_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins to prevent log errors
        entropy = -np.sum(hist * np.log(hist + 1e-6))  # Entropy calculation with small offset
        return entropy

    def adjust_learning_rate(self, current_entropy):
        """
        Adjust learning rate based on entropy while maintaining stability.
        Uses Exponentially Weighted Moving Average (EWMA) and momentum-based smoothing.
        """
        self.past_entropies.append(current_entropy)
        if len(self.past_entropies) > 10:  # Keep last 10 entropy values
            self.past_entropies.pop(0)
        
        entropy_std = np.std(self.past_entropies)  # Measure volatility
        decay_factor = np.exp(-entropy_std / 5)  # Smooth decay
        
        # Adjust learning rate with controlled variation
        self.learning_rate *= (0.98 + 0.02 * decay_factor)  # Min decay: 98%, Max: 100%
        self.learning_rate = max(self.learning_rate, self.min_lr)  # Prevent over-decay
        
        # Log entropy and learning rate for monitoring
        logging.info(f"Entropy: {current_entropy}, Learning Rate: {self.learning_rate}")
        
        return self.learning_rate

    def clip_entropy(self, entropy):
        """Clip entropy to be within a specified range."""
        return max(0.2, min(5.0, entropy))

    def update_entropy(self, current_entropy):
        """Update entropy using EWMA smoothing with adaptive smoothing factor."""
        previous_entropy = self.smoothed_entropy

        # Adjust smoothing factor based on entropy volatility
        if abs(current_entropy - previous_entropy) > 0.1:  # Threshold for high volatility
            self.alpha = 0.1  # More smoothing when entropy fluctuates
        else:
            self.alpha = 0.3  # Less smoothing when entropy is stable

        # Apply EWMA smoothing
        new_entropy = self.alpha * current_entropy + (1 - self.alpha) * previous_entropy

        # Clamp entropy to be within a stable range
        new_entropy = self.clip_entropy(new_entropy)

        self.smoothed_entropy = new_entropy

        # Apply exponential decay to smooth out entropy fluctuations
        decay_factor = 0.98  # Prevent overreaction
        self.entropy = decay_factor * self.entropy + (1 - decay_factor) * current_entropy

        logging.info(f"Updated Entropy: {self.smoothed_entropy}")

        return self.smoothed_entropy

    def monitor(self, data):
        """Monitor entropy and return an adjusted learning rate."""
        current_entropy = self.calculate_entropy(data)
        self.past_entropies.append(current_entropy)
        return self.adjust_learning_rate(current_entropy)

    def update_model_online(self, model, data, target, optimizer, loss_fn):
        """Update the model parameters with every new data point."""
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        
        # Add entropy regularization to prevent collapse into a single state
        entropy_penalty = torch.std(output)
        loss += entropy_penalty

        # Introduce controlled noise every 5 epochs
        if len(self.past_outputs) % 5 == 0:
            output += torch.normal(0, 0.01, output.shape)  # Small Gaussian noise

        loss.backward()
        entropy_variance = np.var(self.past_entropies)
        for param in model.parameters():
            param.grad = self.entropy_gradient_clipping(param.grad, entropy_variance)
        optimizer.step()

        # Reset weights if the model becomes too predictable
        if torch.std(output) < 0.001:  # If all values are nearly identical
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # Reinitialize weights slightly to break stagnation

        self.past_outputs.append(output)
        return loss.item()

    def compute_shannon_entropy(self, probabilities):
        """Compute Shannon entropy of the given probability distribution."""
        return entropy(probabilities)

    def compute_kolmogorov_complexity(self, data):
        """Compute Kolmogorov complexity using compression."""
        compressed_data = zlib.compress(data)
        return len(compressed_data)

    def detect_inconsistencies(self, prediction_confidences):
        """Detect inconsistencies in prediction confidences."""
        shannon_entropy = self.compute_shannon_entropy(prediction_confidences)
        kolmogorov_complexity = self.compute_kolmogorov_complexity(prediction_confidences.tobytes())
        
        self.historical_entropies.append(shannon_entropy)
        
        # Detect systematic model drift
        if len(self.historical_entropies) > 10:
            recent_entropies = self.historical_entropies[-10:]
            mean_entropy = np.mean(recent_entropies)
            std_entropy = np.std(recent_entropies)
            if shannon_entropy > mean_entropy + 2 * std_entropy:
                return True, shannon_entropy, kolmogorov_complexity
        return False, shannon_entropy, kolmogorov_complexity

    def adjust_gradient_smoothing(self, current_smoothing_factor, entropy_change):
        """Adjust gradient smoothing based on entropy changes."""
        if entropy_change > 0.1:
            return min(0.5, current_smoothing_factor + 0.05)
        elif entropy_change < -0.1:
            return max(0.01, current_smoothing_factor - 0.05)
        return current_smoothing_factor

# Example Usage
if __name__ == "__main__":
    data = torch.randint(-10, 10, (100, 10), device=device)
    monitor = EntropyMonitor()
    learning_rate = monitor.monitor(data)
    print(f"Final Adjusted Learning Rate: {learning_rate}")

    entropy_monitor = EntropyMonitor(initial_entropy=1.0)
    new_entropy_values = [0.8, 0.9, 1.1, 1.0, 0.95]

    for new_entropy in new_entropy_values:
        updated_entropy = entropy_monitor.update_entropy(new_entropy)
        print(f"Updated Entropy: {updated_entropy}")

    prediction_confidences = np.random.rand(100)
    inconsistencies, shannon_entropy, kolmogorov_complexity = monitor.detect_inconsistencies(prediction_confidences)
    print(f"Inconsistencies detected: {inconsistencies}")
    print(f"Shannon Entropy: {shannon_entropy}")
    print(f"Kolmogorov Complexity: {kolmogorov_complexity}")
