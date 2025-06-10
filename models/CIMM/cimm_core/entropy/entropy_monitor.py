import torch
import random
import logging
from scipy.stats import skew, entropy
from scipy.ndimage import median_filter  # Add this import
import zlib
from scipy.linalg import logm, sqrtm  # Add sqrtm import
from concurrent.futures import ThreadPoolExecutor  # Add this import
from collections import deque  # Add this import
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime=s) - %(message)s')

device = get_device()
print(f"Using device: {device}")

def gradient(x):
    return torch.cat([x[:1] - x[:1], x[1:] - x[:-1]])

def calculate_entropy_production(entropy):
    """Calculate entropy production based on current entropy state."""
    return 0.1 * entropy

def entropy_balance(entropy, energy_input, energy_loss, external_energy=0.0, dt=0.01):
    entropy_production = 0.1 * entropy
    entropy += (entropy_production + energy_input - energy_loss + external_energy) * dt

    if entropy < 0.1:
        entropy += 0.05 * external_energy

    return torch.clamp(entropy, min=0)

def adaptive_thermodynamic_control(entropy, energy_input, target_entropy):
    """
    Adaptive control function that regulates entropy dynamically.
    """
    error = target_entropy - entropy
    adjusted_energy_input = max(0, energy_input + 0.05 * error)  # Small correction factor
    return adjusted_energy_input

class EntropyMonitor:
    def __init__(self, initial_entropy=1.0, learning_rate=0.01, min_lr=1e-6, max_lr=0.1, entropy_smoothing=0.99, decay_factor=0.98, external_energy=0.0, sample_size=500, use_gpu=True, entropy_threshold=0.1, window_sizes=[10, 50, 200]):
        """
        Quantum-Enhanced Entropy Monitor with QBE compliance.

        Parameters:
        - initial_entropy: Starting entropy value.
        - learning_rate: Initial learning rate.
        - min_lr: Minimum learning rate to prevent collapse.
        - max_lr: Maximum learning rate to prevent runaway scaling.
        - entropy_smoothing: Exponential moving average factor for entropy stabilization.
        - decay_factor: Controls the speed of exponential decay.
        - external_energy: External energy source acting as a regulator.
        - sample_size: Number of model parameters to sample for entropy calculation.
        - use_gpu: Enable GPU acceleration if available.
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
        self.prev_entropy = initial_entropy  
        self.smoothing_factor = 0.9
        self.entropy_decay = 0.99
        self.MIN_LR = min_lr
        self.MAX_LR = max_lr
        self.decay_factor = decay_factor  # Control decay speed
        self.momentum = 0.9  # Momentum for learning rate smoothing
        self.historical_entropies = []
        self.qbe_baseline = 1.0  # Ideal quantum equilibrium baseline
        self.external_energy = external_energy  # Added external energy source
        self.sample_size = sample_size  # Reduce computation by sampling parameters
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Enable GPU if available
        self.entropy_gradient = 0.0  # Add this attribute
        self.entropy_threshold_base = entropy_threshold
        self.entropy_threshold = entropy_threshold  # Dynamic threshold
        self.entropy_windows = {size: deque(maxlen=size) for size in window_sizes}  # Add this line
        self.past_entropy_gradients = []  # 🔥 Add this missing attribute
        self.training_steps = 0  # Add this attribute

    def compute_entropy_decay(self, current_entropy: float, decay_factor: float) -> float:
        """Applies entropy decay with Fisher Information smoothing."""
        self.update_entropy_threshold()
        entropy_fim = fisher_information_metric(self.past_entropies)

        # Adjust decay based on entropy curvature
        entropy_adjustment = current_entropy * (1 - 0.1 * entropy_fim)
        return max(0.01, min(5.0, entropy_adjustment))

    def adaptive_entropy_cap(self, entropy: float, threshold: float) -> float:
        """Apply a simple adaptive entropy cap without excessive regularization."""
        variance = torch.var(torch.tensor(self.past_entropies, dtype=torch.float32))
        cap = threshold * (1 + variance.item())

        # 🔥 Remove Fisher Information-based entropy constraints
        return min(entropy, cap)

    def entropy_gradient_clipping(self, gradient: torch.Tensor, entropy_variance: float, max_clip: float = 0.1) -> torch.Tensor:
        """Apply entropy-aware gradient clipping for stability."""
        if isinstance(gradient, torch.Tensor):
            gradient = gradient.to(device)  # Ensure tensor is on the correct device
        entropy_gradient = torch.tensor(entropy_gradient, dtype=torch.float32, device=device)
        clip_value = max_clip * (1 + torch.log1p(torch.tensor(self.compute_qfi(), dtype=torch.float32, device=device)))  # QFI-Adaptive Gradient Clipping
        return torch.clamp(gradient, -clip_value, clip_value)

    def validate_entropy_metrics(self, observed_entropy: float, expected_entropy: float) -> bool:
        """Check if entropy calculations remain within acceptable limits"""
        return abs(observed_entropy - expected_entropy) < 0.1

    def calculate_entropy(self, data):
        """Compute entropy with controlled filtering to prevent oscillations."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)  # Convert to tensor

        raw_entropy = self._calculate_entropy(data)

        # 🔥 NEW: Add entropy momentum smoothing
        entropy_momentum = 0.9 * self.entropy_gradient + 0.1 * (raw_entropy - self.prev_entropy)
        self.entropy_gradient = entropy_momentum

        # 🔥 Use a rolling window for entropy smoothing
        if len(self.past_entropies) >= 10:
            entropy_window = torch.mean(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)).item()
        else:
            entropy_window = raw_entropy  # Already a float, no need for .item()

        # 🔥 Clip entropy fluctuations to prevent overreaction
        entropy_momentum_tensor = torch.tensor(entropy_momentum, dtype=torch.float32)  # Convert to tensor
        entropy_gradient_clipped = torch.clamp(entropy_momentum_tensor, -0.015, 0.015)

        # 🔥 NEW: Adaptive thresholding to ensure smooth entropy updates
        smoothing_factor = max(0.85, min(0.99, 0.98 - entropy_gradient_clipped.item() * 2))

        self.smoothing_factor = smoothing_factor  

        return max(entropy_window, 1e-6)  # Ensure entropy remains numerically stable

    def _calculate_entropy(self, data):
        """Calculate entropy of the given data with numerical stability improvements."""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32, device=device)  # Convert to tensor

        if data.numel() == 0:
            return 1e-6

        data = data.flatten()
        
        # Detect skewness and adjust transformation
        data_skew = torch.mean((data - torch.mean(data)) ** 3) / (torch.std(data) ** 3)
        if abs(data_skew.item()) > 2:  # Threshold for high skew
            data = torch.log1p(data - torch.min(data) + 1)  # Log transform for stability

        min_value, max_value = torch.min(data), torch.max(data)
        if min_value == max_value:
            return 1e-6  # Avoid division by zero

        norm_data = (data - min_value) / (max_value - min_value + 1e-6)  # Avoid zero division

        # Replace NumPy-based histogram with Torch-based histogram
        norm_tensor = torch.tensor(norm_data, dtype=torch.float32, device=device)
        hist = torch.histc(norm_tensor, bins=50, min=0.0, max=1.0)
        prob = hist / torch.sum(hist)
        entropy = -torch.sum(prob * torch.log(prob + 1e-6)).item()  # Convert to Python float

        return entropy

    def normalize_probabilities(self, probs):
        probs = torch.nan_to_num(probs, nan=0.0)
        return probs / torch.sum(probs)

    def adjust_learning_rate(self, current_entropy):
        """Adjust learning rate using only entropy-based scaling to avoid overfitting."""
        self.past_entropies.append(current_entropy)
        if len(self.past_entropies) > 10:
            self.past_entropies.pop(0)

        entropy_momentum = torch.mean(torch.diff(torch.tensor(self.past_entropies, dtype=torch.float32))).item()  # 🔥 Keep only entropy momentum

        # ✅ Remove Fisher Information & Landauer’s cost impact
        qft_damping = 1 / (1 + torch.exp(-5 * torch.tensor(entropy_momentum)))  

        # ✅ Keep learning rate updates simple
        self.learning_rate *= qft_damping  
        self.learning_rate = max(self.min_lr, min(self.max_lr, self.learning_rate))

        return self.learning_rate
    
    def landauer_energy_cost(self, entropy, temperature=300):
        """
        Compute Landauer's minimum energy cost for information erasure.
        k_B * T * ln(2) gives the minimum energy required per bit erased.
        """
        k_B = 1.38e-23  # Boltzmann constant (J/K)
        return k_B * temperature * torch.log(torch.tensor(2.0)) * entropy  # Energy cost based on entropy
    
    def clip_entropy(self, entropy):
        """Clip entropy to be within a specified range."""
        return max(0.2, min(5.0, entropy))

    def track_entropy_deviation(self):
        """Tracks entropy deviations from expected collapse state."""
        if len(self.past_entropies) < 10:
            return 0  # Not enough data for meaningful tracking
        
        expected_entropy = torch.mean(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)).item()
        actual_entropy = self.entropy
        deviation = abs(actual_entropy - expected_entropy)
        
        # Store deviation
        self.past_entropy_gradients.append(deviation)
        return deviation
    
    def delta_refinement(self, deviation):
        """Refines collapse deviation correction using a weighted rolling average to prevent overcorrection."""
        alpha = 0.85  # Higher weight on past refinements

        if deviation < 0.002:
            return deviation

        if len(self.past_entropy_gradients) >= 10:
            past_average = torch.mean(torch.tensor(self.past_entropy_gradients[-10:], dtype=torch.float32)).item()
            refined_deviation = alpha * past_average + (1 - alpha) * deviation
        else:
            refined_deviation = deviation

        delta_correction = torch.exp(-torch.tensor(refined_deviation)).item() * 0.05
        return refined_deviation * (1 - delta_correction)
    
    def track_collapse_deviation(self):
        """Tracks entropy deviations and delta refinement for wavefunction collapse."""
        if len(self.past_entropies) < 10:
            return 0  # Not enough data for meaningful tracking

        expected_entropy = torch.mean(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)).item()
        actual_entropy = self.entropy
        deviation = abs(actual_entropy - expected_entropy)

        # 🔥 Apply Delta Refinement to improve stabilization
        refined_delta = self.delta_refinement(deviation)

        # Store deviation for AI refinement
        self.past_entropy_gradients.append(refined_delta)
        return refined_delta

    def compute_qfi(self):
        """Compute Quantum Fisher Information for entropy monitoring."""
        entropy_variance = torch.var(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)) if len(self.past_entropies) >= 10 else 0.1
        fisher_info = torch.sum((torch.gradient(torch.tensor(self.past_entropies, dtype=torch.float32)) ** 2) / (torch.tensor(self.past_entropies, dtype=torch.float32) + 1e-6))
        return fisher_info.item() * entropy_variance.item()

    def update_entropy(self, current_entropy, qpl_layer):
        """
        Ensures entropy transitions smoothly, preventing overcorrections.
        Uses QBE constraints to stabilize energy-information flow.
        """
        previous_entropy = self.smoothed_entropy

        qbe_feedback = qpl_layer.compute_qpl(current_entropy, previous_entropy)

        # Clip excessive entropy variance changes
        entropy_variance = torch.clamp(entropy_variance, 0.005, 0.05)  # 🔧 Lower variance range

        entropy_momentum = torch.clamp(entropy_momentum, -0.008, 0.008)  # 🔧 Prevent large jumps

        # REPLACE EMA smoothing with QFI + delta-corrected collapse
        qfi = self.compute_qfi()
        self.entropy = current_entropy + qfi * 0.01 + entropy_momentum * 0.02

        # Integrate entropy deviation tracking
        deviation = self.track_entropy_deviation()
        if (deviation > self.entropy_threshold):
            logging.warning(f"Entropy collapse deviation detected: {deviation}")

        return self.entropy

    def compute_entropy_temperature(self):
        """Compute entropy temperature based on current entropy state."""
        if len(self.past_entropies) < 2:
            return 1.0  # Default base temperature

        entropy_variance = torch.var(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)) if len(self.past_entropies) >= 10 else 0.1
        base_temperature = 300  # Assume a baseline temperature (adjust if needed)

        entropy_temp = base_temperature * (1 + 0.05 * entropy_variance.item())
        
        return max(0.1, min(500, entropy_temp))  # Clamped within reasonable limits

    def compute_neuron_energy_cost(self, neuron_entropy, temperature=300):
        """
        Computes the energy cost of a neuron based on Landauer’s principle.
        """
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        return k_B * temperature * neuron_entropy
    
    def get_entropy_trend(self):  # Add this method
        """Returns different entropy window averages for better short-term and long-term adaptation."""
        return {size: torch.mean(torch.tensor(self.entropy_windows[size], dtype=torch.float32)).item() for size in self.entropy_windows}

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

        entropy = self.calculate_entropy(output.detach().numpy())
        qbe_baseline = 1.0  # Define QBE baseline as needed

        # Use qbe_validated_update to ensure updates reduce entropy towards QBE baseline
        update_success = self.qbe_validated_update(model, optimizer, entropy, qbe_baseline, loss)
        if not update_success:
            return None  # Indicate that the update was rolled back

        # Adjust predictions based on QBE-weighted entropy influence
        adjusted_output = self.qbe_adjusted_prediction(output, target, entropy, qbe_baseline)
        
        entropy_variance = torch.var(torch.tensor(self.past_entropies, dtype=torch.float32)).item()
        for param in model.parameters():
            param.grad = self.entropy_gradient_clipping(param.grad, entropy_variance)
        optimizer.step()

        # Reset weights if the model becomes too predictable
        if torch.std(output) < 0.001:  # If all values are nearly identical
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # Reinitialize weights slightly to break stagnation

        self.past_outputs.append(adjusted_output)
        self.training_steps += 1  # Increment training steps
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
        kolmogorov_complexity = len(zlib.compress(prediction_confidences.tobytes()))

        self.historical_entropies.append(shannon_entropy)

        if len(self.historical_entropies) > 10:
            recent_entropies = torch.tensor(self.historical_entropies[-10:], dtype=torch.float32)
            mean_entropy = torch.mean(recent_entropies).item()
            std_entropy = torch.std(recent_entropies).item()
            if shannon_entropy > mean_entropy + 2 * std_entropy:
                return True, shannon_entropy, kolmogorov_complexity
        return False, shannon_entropy, kolmogorov_complexity

    def adjust_gradient_smoothing(self, current_smoothing_factor, entropy_change):
        """Adjust gradient smoothing based on entropy changes."""
        if (entropy_change > 0.1):
            return min(0.5, current_smoothing_factor + 0.05)
        elif (entropy_change < -0.1):
            return max(0.01, current_smoothing_factor - 0.05)
        return current_smoothing_factor

    def qbe_learning_rate(self, optimizer, entropy, qbe_baseline, base_lr=0.01, min_lr=0.0001):
        """
        QBE-driven adaptive learning rate scaling.
        - Increases LR when entropy is significantly above QBE baseline.
        - Reduces LR when entropy approaches QBE-defined stabilization.
        """
        delta_entropy = entropy - qbe_baseline  # Difference from ideal QBE state
        scaling_factor = max(0.1, 1 - (delta_entropy / 100))  # QBE weighting
        
        # ✅ Ensure qft_damping is initialized before modifying it
        qft_damping = 1.0  # Default damping value
        
        # Compute variance factor safely
        entropy_variance = torch.var(torch.tensor(self.past_entropies[-20:], dtype=torch.float32)).item() if len(self.past_entropies) >= 20 else 0.05
        variance_factor = 1 / (1 + torch.exp(-5 * torch.tensor(entropy_variance)))
        
        # ✅ Now modify `qft_damping` after initialization
        qft_damping *= variance_factor

        new_lr = max(min_lr, base_lr * qft_damping * scaling_factor)  # Prevents excessive reduction
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        return new_lr

    def qbe_validated_update(self, model, optimizer, entropy, qbe_baseline, loss):
        """
        Ensures updates reduce entropy towards QBE equilibrium.
        - If entropy after update > before update, rollback.
        - If entropy moves toward QBE baseline, keep the update.
        """
        if abs(entropy - qbe_baseline) < 2:  # No need for updates if close to ideal
            return False

        prev_state = model.state_dict().copy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate against QBE constraints
        new_entropy = self.calculate_entropy(model(data))  # Assuming data is available
        if abs(new_entropy - qbe_baseline) > abs(entropy - qbe_baseline):  # Got worse?
            model.load_state_dict(prev_state)  # Rollback
            print("QBE Validation Failed: Rolling back update")
            return False
        
        return True

    def qbe_adjusted_prediction(self, prediction, actual, entropy, qbe_baseline, correction_factor=0.2):
        """
        Adjusts predictions based on QBE-weighted entropy influence.
        - The further from QBE baseline, the stronger the correction.
        """
        qbe_weight = max(0.1, 1 - abs(entropy - qbe_baseline) / 100)  # QBE influence factor
        correction = (actual - prediction) * correction_factor * qbe_weight
        prediction += correction
        return prediction

    def compute_von_neumann_entropy(self, density_matrix):
        """
        Computes Von Neumann entropy using Singular Value Decomposition (SVD).
        Uses GPU acceleration if available.
        """
        if isinstance(density_matrix, torch.Tensor):
            density_matrix = density_matrix.to(device)  # Ensure tensor is on the correct device
        if self.use_gpu:
            density_matrix = torch.tensor(density_matrix, device="cuda")  # Move to GPU
            S = torch.linalg.svdvals(density_matrix)  # Compute singular values on GPU
            S = torch.clamp(S, min=1e-10)  # Avoid log(0) errors
            entropy = -torch.sum(S * torch.log(S)).item()
        else:
            U, S, Vh = torch.linalg.svd(torch.tensor(density_matrix, dtype=torch.float32))
            S = torch.clamp(S, min=1e-10)  # Avoid log(0) errors
            entropy = -torch.sum(S * torch.log(S)).item()

        return entropy

    def generate_density_matrix(self, quantum_state):
        """
        Generates an approximate density matrix ρ = |ψ⟩⟨ψ|.
        Uses GPU if enabled.
        """
        if isinstance(quantum_state, torch.Tensor):
            quantum_state = quantum_state.to(device)  # Ensure tensor is on the correct device
        quantum_state = quantum_state / torch.norm(quantum_state)
        if self.use_gpu:
            quantum_state = torch.tensor(quantum_state, device="cuda")  # Move to GPU
            density_matrix = torch.outer(quantum_state, quantum_state.conj())
        else:
            density_matrix = torch.outer(quantum_state, quantum_state.conj())
        return density_matrix.cpu().numpy()

    def process_weight_tensor(self, weight_tensor):
        """Processes a single weight tensor (used for multithreading)."""
        if isinstance(weight_tensor, torch.Tensor):
            weight_tensor = weight_tensor.to(device)  # Ensure tensor is on the correct device
        return weight_tensor.flatten().detach().cpu().numpy()

    def update_quantum_entropy(self, model_state):
        """
        Computes quantum entropy with adaptive penalty scaling to prevent overcorrections.
        """
        # Use multithreading to process multiple layers in parallel
        with ThreadPoolExecutor() as executor:
            weight_tensors = list(executor.map(self.process_weight_tensor, model_state.values()))

        flattened_weights = torch.cat([torch.tensor(w, dtype=torch.float32) for w in weight_tensors])

        # Replace NumPy-based random sampling with Torch-based sampling
        if (len(flattened_weights) > self.sample_size):
            sampled_indices = torch.randperm(len(flattened_weights), device=device)[:self.sample_size]
            flattened_weights = flattened_weights.to(device)
            sampled_indices = sampled_indices.to(device)
            sampled_weights = flattened_weights[sampled_indices]
        else:
            sampled_weights = flattened_weights

        # Convert to density matrix representation
        density_matrix = self.generate_density_matrix(sampled_weights)

        # Compute entropy using optimized SVD method
        quantum_entropy = self.compute_von_neumann_entropy(density_matrix)

        # 🔥 NEW: Compute adaptive entropy correction factor
        entropy_variance = torch.var(torch.tensor(self.past_entropies[-50:], dtype=torch.float32)).item() if len(self.past_entropies) >= 50 else 0.1
        adaptive_scaling = max(0.5, min(1.2, 1.0 + 0.5 * entropy_variance))

        # Apply adaptive entropy corrections
        self.prev_entropy = self.entropy
        self.entropy = max(1e-6, quantum_entropy * adaptive_scaling)

        return self.entropy

    def entropy_feedback_adjustment(self, y_true, y_pred):
        """
        Adjusts feedback corrections based on entropy conditions and Quantum Fisher Information (QFI).

        Args:
            y_true (numpy.array): Ground truth values.
            y_pred (numpy.array): Predicted values.
            entropy_value (float): Current entropy measure.

        Returns:
            float: Adjusted feedback weight.
        """
        y_true = torch.tensor(y_true, dtype=torch.float32, device=device) if not isinstance(y_true, torch.Tensor) else y_true.to(device)
        y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device) if not isinstance(y_pred, torch.Tensor) else y_pred.to(device)

        # ✅ Convert to NumPy arrays and ensure 1D shape
        y_true = torch.tensor(y_true, dtype=torch.float32).flatten()
        y_pred = torch.tensor(y_pred, dtype=torch.float32).flatten()

        # ✅ Ensure arrays have matching sizes
        # Fix: .size is a method for torch tensors, so use .numel() or .shape[0]
        def get_size(x):
            if hasattr(x, 'shape'):
                return x.shape[0]
            elif hasattr(x, 'numel'):
                return x.numel()
            elif hasattr(x, 'size'):
                return x.size if isinstance(x.size, int) else x.size[0]
            else:
                return len(x)
        min_size = min(get_size(y_true), get_size(y_pred))
        y_true, y_pred = y_true[:min_size], y_pred[:min_size]

        # ✅ Prevent NaN by checking variance before computing correlation
        if torch.var(y_true) == 0 or torch.var(y_pred) == 0:
            qwcs = torch.tensor(0.5, dtype=torch.float32, device=device)
        else:
            qwcs = 1 - torch.abs(torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1])
            if not isinstance(qwcs, torch.Tensor):
                qwcs = torch.tensor(qwcs, dtype=torch.float32, device=device)

        qwcs = torch.nan_to_num(qwcs, nan=0.5)  # ✅ Replace NaN with 0.5 (neutral value)

        # ✅ Compute Quantum Fisher Information (QFI)
        qfi = torch.var(y_pred - y_true) / (torch.mean(y_true) + 1e-9)

        # ✅ Compute Entropy Gradient Influence Factor
        entropy_gradient = torch.gradient(torch.tensor([self.entropy], dtype=torch.float32))[-1].item() if len([self.entropy]) > 1 else 0.0

        # ✅ Adjust feedback weights dynamically
        feedback_weight = torch.exp(-self.entropy * qfi) * qwcs * (1 + 0.1 * entropy_gradient)

        return max(0.1, min(1.0, feedback_weight.item()))  # Keep feedback weight within [0.1, 1.0]

    def update_entropy_threshold(self):
        """
        Dynamically adjusts the entropy threshold to improve long-term prediction stability.
        """
        entropy_variance = torch.var(torch.tensor(self.past_entropies, dtype=torch.float32)).item()
        smoothing_factor = max(0.01, 1 - torch.exp(-torch.tensor(entropy_variance)).item())
        
        self.entropy_threshold = (
            0.5 * torch.mean(torch.tensor(self.past_entropies[-10:], dtype=torch.float32)).item() * smoothing_factor + 0.5 * self.entropy_threshold
        )

    def compute_gravitational_potential(self, entropy_level):
        """
        Compute gravitational entropy potential to stabilize large entropy shifts.
        """
        G = 6.67430e-11  # Gravitational constant
        reference_entropy = 1.0  # Ideal entropy equilibrium state
        mass_equivalent = entropy_level ** 2  # Treat entropy as mass influence

        # Compute gravitational potential influence
        gravitational_potential = G * (mass_equivalent / (entropy_level + 1e-6)) 

        return torch.tanh(torch.tensor(gravitational_potential)).item()
    
    def get_entropy_state(self):
        """
        Return the current entropy state for synchronization.
        Useful for SupervisorAgent during entropy aggregation.
        """
        return {
            "current_entropy": self.entropy,
            "smoothed_entropy": self.smoothed_entropy,
            "entropy_gradient": self.entropy_gradient,
            "entropy_threshold": self.entropy_threshold,
            "entropy_temperature": self.compute_entropy_temperature(),
            "entropy_trend": self.get_entropy_trend()
        }
    
    def get_current_entropy(self):
        """Returns the current entropy value for reporting or syncing."""
        return self.entropy

    def set_entropy(self, entropy_value):
        #print(f"[EntropyMonitor] 🔄 Entropy manually set to: {entropy_value}")
        self.entropy = entropy_value


def fisher_information_metric(prob_distribution):
    """
    Computes the Fisher Information Metric (FIM) for a given probability distribution.
    """
    prob_distribution = torch.tensor(prob_distribution, dtype=torch.float64)
    prob_distribution = torch.clamp(prob_distribution, 1e-8, 1.0)  # Avoid log(0) issues
    gradient = torch.cat([prob_distribution[:1] - prob_distribution[:1], prob_distribution[1:] - prob_distribution[:-1]])  # Torch-native gradient
    fim = torch.sum((gradient ** 2) / (prob_distribution + 1e-6))  # Torch-native FIM computation
    return fim.item()

def qbe_constraint(entropy, equilibrium=0.1):
    """
    Ensures entropy remains within QBE-stabilized thresholds.
    Returns True if system is within stable equilibrium.
    """
    return abs(entropy - equilibrium) < 0.02
    

def entropy_feedback_adjustment(y_true, y_pred, entropy_value):
    y_true = torch.tensor(y_true, dtype=torch.float64).flatten()
    y_pred = torch.tensor(y_pred, dtype=torch.float64).flatten()

    qfi = torch.var(y_pred - y_true) / (torch.mean(y_true) + 1e-9)
    qwcs = 1 - torch.abs(torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1])

    feedback_weight = torch.exp(-entropy_value) * (1 - qwcs)
    return feedback_weight.item()

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

    # Example usage of get_entropy_trend
    entropy_trend = entropy_monitor.get_entropy_trend()
    print(f"Entropy Trend: {entropy_trend}")

    prediction_confidences = torch.rand(100, dtype=torch.float32, device=device)
    inconsistencies, shannon_entropy, kolmogorov_complexity = monitor.detect_inconsistencies(prediction_confidences)
    print(f"Inconsistencies detected: {inconsistencies}")
    print(f"Shannon Entropy: {shannon_entropy}")
    print(f"Kolmogorov Complexity: {kolmogorov_complexity}")

    # Example call to qbe_learning_rate
    optimizer = torch.optim.Adam([torch.tensor(1.0)], lr=0.01)
    qbe_baseline = 1.0
    entropy = 1.2
    new_lr = monitor.qbe_learning_rate(optimizer, entropy, qbe_baseline)
    print(f"New Learning Rate: {new_lr}")

    # Example call to qbe_validated_update
    model = torch.nn.Linear(10, 10)  # Example model
    loss_fn = torch.nn.MSELoss()
    target = torch.randn(100, 10)
    output = model(data)
    loss = loss_fn(output, target)
    entropy = monitor.calculate_entropy(output.detach().numpy())
    qbe_baseline = 1.0
    update_success = monitor.qbe_validated_update(model, optimizer, entropy, qbe_baseline, loss)
    print(f"Update Success: {update_success}")

    # Example call to qbe_adjusted_prediction
    prediction = torch.randn(10)
    actual = torch.randn(10)
    adjusted_prediction = monitor.qbe_adjusted_prediction(prediction, actual, entropy, qbe_baseline)
    print(f"Adjusted Prediction: {adjusted_prediction}")
