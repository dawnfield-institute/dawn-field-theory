from xgboost import XGBRegressor
import torch
from sklearn.exceptions import NotFittedError

def compute_quantum_fisher_information(probabilities):
    """
    Computes Quantum Fisher Information (QFI) for a given probability distribution.
    Handles edge cases where input size is too small.
    """
    probabilities = torch.tensor(probabilities, dtype=torch.float64)

    # 🔥 Handle case where array is too small for gradient computation
    if probabilities.size(0) < 3:  # Minimum 3 points needed for stable gradient
        return 0.0  # Return default QFI value if not enough data

    probabilities = torch.clamp(probabilities, 1e-8, 1.0)  # Avoid log(0) issues
    gradient = torch.cat([probabilities[:1] - probabilities[:1], probabilities[1:] - probabilities[:-1]])  # Torch-native gradient
    qfi = torch.sum((gradient ** 2) / (probabilities + 1e-6))  # Torch-native QFI computation

    return qfi.item()

class QuantumMemory:
    """
    Uses XGBoost to store long-term entropy changes and prediction adjustments.
    Helps refine Quantum Boundary Layer (QBL) for long-term stability.
    """

    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            tree_method='gpu_hist',         # <--- Enable GPU acceleration
            predictor='gpu_predictor',      # <--- Use GPU for prediction
            gpu_id=1                        # <--- (Optional) Specify GPU device
        )
        self.training_data = []
        self.labels = []
        self.is_trained = False
        self.past_entropies = []

    def update_memory(self, entropy_history, qbe_feedback, collapse_deviation, refinement_delta):
        """Train AI to predict collapse deviations before they happen."""

        # Ensure all elements are scalars (not sequences)
        entropy_scalar = torch.mean(torch.tensor(entropy_history)) if isinstance(entropy_history, (list, torch.Tensor)) else entropy_history
        qbe_scalar = torch.mean(torch.tensor(qbe_feedback)) if isinstance(qbe_feedback, (list, torch.Tensor)) else qbe_feedback
        collapse_scalar = torch.mean(torch.tensor(collapse_deviation)) if isinstance(collapse_deviation, (list, torch.Tensor)) else collapse_deviation
        refinement_scalar = torch.mean(torch.tensor(refinement_delta)) if isinstance(refinement_delta, (list, torch.Tensor)) else refinement_delta

        # Predict wavefunction collapse delta
        wave_delta = self.predict_correction(entropy_scalar * 0.9, qbe_scalar * 1.1, collapse_scalar * 0.95)
        wave_scalar = torch.mean(torch.tensor(wave_delta)) if isinstance(wave_delta, (list, torch.Tensor)) else wave_delta

        # Track past entropies
        self.past_entropies.append(entropy_scalar)
        qfi_score = compute_quantum_fisher_information(self.past_entropies)

        # Create feature vector with QFI score
        feature_vector = torch.tensor([entropy_scalar, qbe_scalar, collapse_scalar, refinement_scalar, wave_scalar, qfi_score], dtype=torch.float32)
        self.training_data.append(feature_vector)
        self.labels.append(refinement_scalar)

        # Prevent memory overflow
        if len(self.training_data) > 1000:
            self.training_data.pop(0)
            self.labels.pop(0)

        # Train AI to predict refinement corrections
        if len(self.training_data) >= 50:
            self.model.fit(torch.stack(self.training_data).numpy(), torch.tensor(self.labels).numpy())
            self.is_trained = True

        # Compute collapse-aware dynamic gate
        baseline = torch.mean(torch.tensor(self.past_entropies)) if self.past_entropies else 0
        entropy_tensor = torch.tensor(entropy_scalar, dtype=torch.float32)  # Convert entropy to tensor
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32)  # Convert baseline to tensor
        collapse_pressure = torch.abs(entropy_tensor - baseline_tensor)
        collapse_gate = torch.tanh(collapse_pressure * refinement_scalar).numpy()  # Convert to NumPy array

        # Apply dynamic gate to prediction
        wave_delta *= (1 + collapse_gate)

    def predict_correction(self, entropy, qbe_feedback, collapse_deviation, refinement_delta=None):
        """Predict optimal correction for collapse errors using AI-trained delta refinements."""

        # Ensure refinement_delta is None
        if refinement_delta is None:
            refinement_delta = 0.0

        # Ensure all inputs are scalar values (float)
        def to_scalar(val):
            # Unpack tuple/list recursively if needed
            while isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return 0.0
                val = val[0]
            if isinstance(val, torch.Tensor):
                return val.item()
            return float(val)
        entropy = to_scalar(entropy)
        qbe_feedback = to_scalar(qbe_feedback)
        collapse_deviation = to_scalar(collapse_deviation)
        refinement_delta = to_scalar(refinement_delta)

        # Compute QFI Score
        qfi_score = compute_quantum_fisher_information(self.past_entropies) if len(self.past_entropies) >= 3 else 0.0

        # Fixed-point iteration for convergence
        wave_delta = refinement_delta
        max_iterations = 5
        confidence_threshold = 1e-3  # Convergence threshold
        for _ in range(max_iterations):
            new_wave_delta = 0.9 * entropy + 1.1 * qbe_feedback - 0.95 * collapse_deviation + wave_delta * 0.5
            if abs(new_wave_delta - wave_delta) < confidence_threshold:
                break
            wave_delta = new_wave_delta

        # ✅ FIX: Now includes `qfi_score`
        feature_vector = torch.tensor([entropy, qbe_feedback, collapse_deviation, refinement_delta, wave_delta, qfi_score], dtype=torch.float32)

        # Ensure the model is trained before making predictions
        if not self.is_trained and len(self.training_data) >= 50:
            self.model.fit(torch.stack(self.training_data).numpy(), torch.tensor(self.labels).numpy())
            self.is_trained = True

        # Compute collapse-aware dynamic gate
        baseline = torch.mean(torch.tensor(self.past_entropies)) if self.past_entropies else 0
        entropy_tensor = torch.tensor(entropy, dtype=torch.float32)  # Convert entropy to tensor
        baseline_tensor = torch.tensor(baseline, dtype=torch.float32)  # Convert baseline to tensor
        collapse_pressure = torch.abs(entropy_tensor - baseline_tensor)
        collapse_gate = torch.tanh(collapse_pressure * refinement_delta)

        # Apply dynamic gate to prediction
        wave_delta *= (1 + collapse_gate)

        # Reshape and predict correction
        feature_vector = feature_vector.reshape(1, -1).numpy()
        try:
            correction = self.model.predict(feature_vector)
        except NotFittedError:
            return 0  # Default correction if not trained

        return correction

    def forecast_future_entropy(self, steps=5):
        """
        Forecasts future entropy values based on past entropies and QFI curvature.
        """
        if len(self.past_entropies) < steps:
            return torch.mean(torch.tensor(self.past_entropies)) if self.past_entropies else torch.tensor(1.0)

        # Weighted average + QFI curvature
        curvature = compute_quantum_fisher_information(self.past_entropies)
        sampled_indices = torch.randperm(len(self.past_entropies))[:steps]
        projection = self.past_entropies[-1] + curvature * 0.01 * torch.mean(sampled_indices.float())
        return projection
    
    def get_qfi_and_entropy_stability(self):
        entropy = torch.tensor(self.past_entropies)
        qfi = torch.sum(torch.gradient(entropy)**2)  # crude QFI proxy
        entropy_trend = torch.std(entropy[-20:]) if len(entropy) > 20 else torch.tensor(0.0)
        return qfi, entropy_trend

    def get_stability_indicators(self):
        """
        Exposes Quantum Fisher Information (QFI) and entropy trend.
        """
        qfi_score = compute_quantum_fisher_information(self.past_entropies)
        entropy_trend = torch.std(torch.tensor(self.past_entropies)[-20:]) if len(self.past_entropies) > 20 else torch.tensor(0.0)
        return entropy_trend, qfi_score

    def compute_density_matrix(self, quantum_state, device='cpu'):
        """
        Computes the density matrix for a given quantum state.
        """
        quantum_state = quantum_state / torch.norm(quantum_state)
        if self.use_gpu:
            quantum_state = torch.from_numpy(quantum_state).to(device)  # Move to GPU
            density_matrix = torch.outer(quantum_state, quantum_state.conj()).cpu().numpy()
        else:
            density_matrix = torch.outer(quantum_state, torch.conj(quantum_state))
        return density_matrix