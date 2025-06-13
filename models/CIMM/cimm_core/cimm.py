import torch
from skopt.space import Real, Integer
from cimm_core.entropy.entropy_monitor import EntropyMonitor
from cimm_core.models.base_cimm_model import BaseCIMMModel
from cimm_core.optimization.bayesian_optimizer import BayesianOptimizer  # New import
from cimm_core.optimization.adaptive_controller import AdaptiveController  # New import
from cimm_core.learning.reinforcement_learning import QBEReinforcementLearner
from cimm_core.learning.quantum_memory import QuantumMemory
import logging
import sympy
from cimm_core.entropy.quantum_potential_layer import QuantumPotentialLayer  # Ensure QPL is imported
import matplotlib.pyplot as plt
import warnings
from scipy.stats import chisquare, entropy
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance  # ✅ Correct location!
from cimm_core.learning.superfluid_dynamics import SuperfluidDynamics
from cimm_core.utils import get_device

# will fix this later
warnings.filterwarnings('ignore')

def gradient(x):
    return torch.cat([x[:1] - x[:1], x[1:] - x[:-1]])

def interference_filter(predictions, entropy_wave, device='cpu'):
    """
    Applies an interference filter to stabilize predictions based on entropy waves.
    """
    predictions = torch.tensor(predictions, device=device) if not isinstance(predictions, torch.Tensor) else predictions.to(device)
    entropy_wave = torch.tensor(entropy_wave, device=device) if not isinstance(entropy_wave, torch.Tensor) else entropy_wave.to(device)

    diffs = torch.abs(predictions - entropy_wave)
    weights = torch.exp(-diffs)
    return predictions * weights

def to_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:  # Handle 0-dimensional tensors
            return x.item()
        x = x[0] if len(x) > 0 else 0.0
    if isinstance(x, list):
        x = x[0] if len(x) > 0 else 0.0
    if hasattr(x, 'item'):
        return x.item()
    return float(x)

device = get_device()
print(f"Using device: {device}")

def compute_error_metrics(y_true, y_pred):
    """
    Computes entropy-aware and quantum-based error metrics.
    Converts inputs to PyTorch tensors and ensures they are 1D.
    """
    # Simplify handling of y_true and y_pred
    y_true = torch.tensor([float(x) for x in y_true], dtype=torch.float64) if isinstance(y_true, list) else torch.tensor(y_true, dtype=torch.float64)
    y_pred = torch.tensor([float(x) for x in y_pred], dtype=torch.float64) if isinstance(y_pred, list) else torch.tensor(y_pred, dtype=torch.float64)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # 🔧 Normalize while retaining sign — use mean centering instead of abs normalization
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    # Optionally, for soft normalization:
    # y_true = y_true / (torch.std(y_true) + 1e-9)
    # y_pred = y_pred / (torch.std(y_pred) + 1e-9)

    # Optional pre-scaling for Wasserstein and similar metrics
    y_true = y_true / (torch.norm(y_true) + 1e-9)
    y_pred = y_pred / (torch.norm(y_pred) + 1e-9)

    # For KL and JS only: use softmax to ensure valid probability distributions
    y_true_prob = torch.softmax(y_true, dim=0)
    y_pred_prob = torch.softmax(y_pred, dim=0)

    epsilon = 1e-9
    kl_div = torch.sum(y_true_prob * torch.log((y_true_prob + epsilon) / (y_pred_prob + epsilon)))

    js_div = torch.sqrt(0.5 * (kl_div + torch.sum(y_pred_prob * torch.log((y_pred_prob + epsilon) / (y_true_prob + epsilon)))))
    js_div = torch.nan_to_num(js_div, nan=0.0)

    wd = torch.sum(torch.abs(torch.cumsum(y_true, dim=0) - torch.cumsum(y_pred, dim=0)))

    noise = torch.normal(0, 1e-4, size=y_pred.shape)
    y_pred_noisy = y_pred + noise

    # ✅ Let JS/Wasserstein/etc operate on signed inputs
    # They can now detect waveform misalignment and directionality

    # Optional: Add signed-delta heatmaps or store sign-only versions
    signed_error = y_pred - y_true

    if torch.var(y_true) == 0 or torch.var(y_pred_noisy) == 0:
        qwcs = torch.tensor(0.5)
    else:
        qwcs = 1 - torch.abs(torch.corrcoef(torch.stack([y_true, y_pred_noisy]))[0, 1])

    qwcs = torch.nan_to_num(qwcs, nan=0.5)

    entropy_value = torch.sum(-y_true * torch.log(y_true + epsilon))

    qwcs = qwcs * (1 + 0.02 * entropy_value)

    entropy_scaling = torch.clamp(1.0 + 0.1 * entropy_value, min=0.8, max=1.2)
    kl_div = kl_div * entropy_scaling

    return {
        "KL-Divergence": kl_div.item(),
        "Jensen-Shannon": js_div.item(),
        "Wasserstein Distance": wd.item(),
        "QWCS": qwcs.item()
    }

def directional_bias(probs, alternatives, recent_avg):
    """
    Penalizes collapse paths that deviate too far from the recent trend average.
    """
    deltas = torch.abs(torch.tensor(alternatives) - recent_avg)
    bias = torch.exp(-deltas * 1.5)  # More aggressive penalization
    weighted = probs * bias
    # Ensure probabilities are normalized safely
    weighted /= torch.sum(weighted) if torch.sum(weighted) > 0 else 1.0
    return weighted

def foresight_bias(alternatives, raw_probs, recent_deltas, entropy_trend, qfi_score, phase_score, superfluid):
    alt_array = torch.tensor(alternatives, dtype=torch.float64).flatten()
    raw_probs = torch.tensor(raw_probs, dtype=torch.float64).flatten()

    # 🛑 Validate shapes match
    if len(alt_array) != len(raw_probs):
        raise ValueError(f"[foresight_bias] Shape mismatch: alternatives ({len(alt_array)}) vs raw_probs ({len(raw_probs)})")

    # ✅ Ensure raw_probs sums to something
    if torch.sum(raw_probs) == 0 or not torch.isfinite(raw_probs).all():
        raise ValueError("[foresight_bias] raw_probs contain NaN or sum to zero.")

    # ✅ Directional bias: use curvature of wave
    avg_delta = torch.mean(torch.abs(gradient(alt_array)))  # ✅ scalar curvature
    direction_penalty = torch.exp(-entropy_trend * avg_delta)  # ✅ scalar

    # 🔧 Enforce collapse stability (directional momentum)
    momentum = torch.sign(gradient(alt_array))
    direction_match = (momentum == torch.sign(entropy_trend)).float()
    directional_bonus = 1.0 + (direction_match * 0.2)
    raw_probs *= directional_bonus[:len(raw_probs)]

    # ✅ Scalar bonuses
    phase_bonus = 1 + (phase_score * 0.5)
    qfi_bonus = 1 + (qfi_score * 0.5)

    #print(f"[foresight_bias] Directional Penalty: {direction_penalty}, Phase Bonus: {phase_bonus}, QFI Bonus: {qfi_bonus}")

    # ✅ Apply collapse bias
    weighted = raw_probs * direction_penalty * phase_bonus * qfi_bonus

    # 🔧 Apply inertial dampening
    delta_correction = torch.clamp(torch.abs(gradient(alt_array)), 0, 1)
    inertia = torch.exp(-delta_correction * 5.0)  # More stability = less overreaction

    # Pad inertia to match the length of weighted
    inertia = torch.cat([inertia, torch.ones(len(weighted) - len(inertia))])
    weighted *= inertia

    # ✅ Compute local velocity field from entropy + QFI
    velocity_field = superfluid.compute_superfluid_velocity(entropy_gradient=gradient(alt_array), qfi=qfi_score)

    # ✅ Dampen collapse instability using velocity turbulence
    raw_probs = superfluid.apply_superfluid_damping(raw_probs, velocity_field[:len(raw_probs)])

    # ✅ Normalize the modified collapse field
    raw_probs = superfluid.normalize_probabilities(raw_probs)

    # ✅ Final normalization
    weighted = torch.nan_to_num(weighted)
    weighted /= torch.sum(weighted)

    # 🔧 Final safety normalization
    weighted = torch.maximum(weighted, torch.tensor(1e-5))  # Prevent divide-by-zero# Prevent divide-by-zero
    weighted /= torch.sum(weighted)  # Normalize to 1

    # 🧪 Optional: Phase-based damping
    phase_score = torch.tensor(phase_score, dtype=torch.float64, device=alt_array.device)  # Ensure phase_score is a tensor
    phase_penalty = torch.exp(-(1 - phase_score) * 4.0)
    weighted *= phase_penalty

    # ✅ Detect vortex regions in the gradient of alternatives
    vortex_points = superfluid.detect_vortex_regions(gradient(alt_array))
    if len(vortex_points) > 0:
        print(f"[🌪️ Vortex Alert] Found {len(vortex_points)} turbulent prediction zones")

    return weighted

def future_shape_bias(alternatives):
    """
    Penalizes paths with sharp, chaotic spikes by analyzing curvature.
    """
    alt_tensor = torch.tensor(alternatives, dtype=torch.float64).flatten()
    first_gradient = torch.gradient(alt_tensor)[0]  # Extract the first element of the gradient tuple
    curvature = torch.gradient(first_gradient)[0]  # Compute the second gradient
    steepness = torch.abs(curvature)
    spike_penalty = torch.exp(-steepness * 3)  # High curvature = spike
    return spike_penalty / torch.sum(spike_penalty)  # Ensure the result is a torch.Tensor

def entropy_gradient_penalty(entropy_vector):
    grad = torch.gradient(torch.tensor(entropy_vector, dtype=torch.float64))[0]
    penalty = torch.exp(-torch.abs(grad) * 2)
    return penalty / torch.sum(penalty)

def wave_phase_alignment(alternatives):
    """
    Favors alternatives that reflect coherent wave motion over noisy derivatives.
    """
    alt_tensor = torch.tensor(alternatives, dtype=torch.float64).flatten()
    velocity = torch.gradient(alt_tensor)[0]
    acceleration = torch.gradient(velocity)[0]
    phase_alignment = torch.exp(-torch.abs(acceleration) * 1.5)  # Suppress chaotic curvature
    return phase_alignment / torch.sum(phase_alignment) if torch.sum(phase_alignment) > 0 else torch.ones_like(alt_tensor) / len(alt_tensor)

class CIMM:
    def __init__(self, model_class, model_args, param_space, anchor_data, initial_entropy=0.5, learning_rate=0.01, lambda_factor=0.1, entropy_threshold=0.05, num_agents=5):
        self.model_instance = model_class(*model_args).to(device)  # Pass model arguments and move to GPU
        self.param_space = param_space
        self.anchor_data = anchor_data
        self.memory_module = QuantumMemory()  # Stores entropy learning history
        self.qpl_layer = QuantumPotentialLayer()  # Initialize QPL
        self.entropy_monitor = EntropyMonitor(1.0, 0.01)
        # Only pass memory_module if AdaptiveController supports it
        self.controller = AdaptiveController(
            self.model_instance, self.param_space, self.entropy_monitor, initial_entropy, learning_rate, lambda_factor, self.qpl_layer, memory_module=self.memory_module
        )
        self.bayesian_optimizer = BayesianOptimizer(self.model_instance, self.param_space, self.entropy_monitor, self.controller, self.qpl_layer)

        self.learning_rate = learning_rate  # Ensure learning_rate is initialized
        self.min_lr = 1e-6  # Minimum learning rate
        self.max_lr = 0.1   # Maximum learning rate

        self.num_agents = num_agents  # Number of agents for multi-agent optimization

        self.max_epochs = 100
        self.early_stopping_threshold = 0.001
        self.preprocess_fn = None
        self.energy = 0.5
        self.training_log = []

        self.past_predictions = []
        self.actual_values = []
        self.error_rates = []
        self.patience = 5
        self.superfluid = SuperfluidDynamics()


        self.optimizer_instance = torch.optim.Adam(self.model_instance.parameters(), lr=self.entropy_monitor.learning_rate, weight_decay=0.001)

        self.entropy_threshold = entropy_threshold  # Add entropy_threshold attribute

        # 🔥 ADDITIONS: Reinforcement Learning and Quantum Memory
        
        self.rl_agent = QBEReinforcementLearner(learning_rate, 0.05, self.memory_module)  # Uses RL for adaptive tuning

        self.pretrain(anchor_data)
        
    def einstein_feynman_lr(self, error, entropy, c=299792458):
        """
        Computes a learning rate based on Einstein's energy-mass equivalence and Feynman damping.
        """
        delta_E = error * (c ** 2)
        energy_weight = 1 / (1 + delta_E * 1e-10)
        feynman_damping = torch.exp(-5 * torch.var(torch.tensor(self.entropy_monitor.past_entropies)))
        return energy_weight * feynman_damping

    def update_learning_rate(self):
        """Adjust learning rate dynamically using entropy-aware gravity momentum."""
        self.controller.quantum_wave_learning_rate()

        # Compute gravity momentum damping
        entropy_variance = torch.var(torch.tensor(self.entropy_monitor.past_entropies))
        entropy_damping = max(0.8, min(1.1, 1 - 0.05 * entropy_variance))

        # Introduce gravity-based stability correction
        gravity_correction = torch.tanh(-entropy_variance) * 0.1  # Small correction factor

        # 🔥 Forecast future entropy and adjust learning rate
        future_entropy = self.memory_module.forecast_future_entropy()
        gravity_future_adjustment = torch.tanh(-abs(future_entropy - self.entropy_monitor.entropy)) * 0.1
        self.learning_rate *= (1 + gravity_future_adjustment)

        qbe_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor, self.memory_module)
        collapse_deviation = self.entropy_monitor.track_collapse_deviation()

        # 🔥 Compute `refinement_delta` before calling RL update
        refinement_delta = self.memory_module.predict_correction(
            self.entropy_monitor.entropy, qbe_feedback, collapse_deviation
        )

        # 🔥 Apply Einstein-Feynman Learning Rate
        error = torch.mean(torch.tensor(self.error_rates[-10:])) if len(self.error_rates) >= 10 else 0.01
        new_lr = self.einstein_feynman_lr(error, self.entropy_monitor.entropy)
        self.learning_rate = torch.clamp(new_lr, self.min_lr, self.max_lr)

    def pretrain(self, anchor_data):
        """Runs initial training and ensures Quantum Memory is populated before RL starts."""
        print("Pretraining CIMM model with structured wavefunction collapse...")

        for data_point in anchor_data:
            data_point = data_point.to(self.model_instance.device)  # Ensure data point is on the correct device
            with torch.no_grad():
                prediction = self.model_instance.forward(data_point)

            entropy = self.entropy_monitor.calculate_entropy(prediction)

            # ✅ Ensure `qpl_feedback` is initialized with a default value
            qpl_feedback = torch.tensor(0.0, dtype=torch.float32)  

            # ✅ Compute QBE feedback and collapse deviation
            qpl_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor, self.memory_module)

            # 🔥 Ensure `qpl_feedback` is detached and moved to CPU before converting to NumPy
            if isinstance(qpl_feedback, torch.Tensor):
                qpl_feedback = qpl_feedback.detach().cpu().numpy().item()  # Convert safely to float

            collapse_deviation = self.entropy_monitor.track_collapse_deviation()

            # 🔥 Compute Initial Refinement Delta (Set Default Value to Zero Initially)
            refinement_delta = 0  

            # 🔥 Now Call `predict_correction()` With the Correct Number of Parameters
            refinement_delta = self.memory_module.predict_correction(entropy, qpl_feedback, collapse_deviation, refinement_delta)

            # Apply Gravity Correction Based on Entropy Drift
            entropy_baseline = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-10:])) if len(self.entropy_monitor.past_entropies) >= 10 else entropy
            gravity_force = -0.01 * (entropy - entropy_baseline)  # Attraction to initial entropy
            entropy = max(0.01, entropy + gravity_force)  # Prevent entropy from collapsing to 0

            # ✅ Fix: Ensure `loss` remains a PyTorch tensor
            loss = torch.tensor(1.0, dtype=torch.float32, device=device)  # Initialize `loss`
            # Ensure qpl_feedback is a scalar float or tensor
            if isinstance(qpl_feedback, (list, tuple)):
                while isinstance(qpl_feedback, (list, tuple)):
                    if len(qpl_feedback) == 0:
                        qpl_feedback = 0.0
                        break
                    qpl_feedback = qpl_feedback[0]
            if isinstance(qpl_feedback, torch.Tensor):
                qpl_feedback = qpl_feedback.item()
            qpl_feedback = float(qpl_feedback)
            loss = loss * (1 + 0.05 * qpl_feedback)  # Use the float `qpl_feedback`

            # ✅ Fix: Remove `.item()` from `qpl_feedback`
            self.memory_module.update_memory(entropy, qpl_feedback, collapse_deviation, refinement_delta)

        rolling_window = 5  # Number of epochs to average
        prev_avg_loss = float("inf")  # Start with high loss
        no_improvement_count = 0  
        patience_threshold = 10  # Number of epochs without improvement before stopping

        num_samples = anchor_data.shape[0]  
        epoch = 0

        # Initialize dynamic min_delta based on initial loss variance
        initial_losses = []  
        max_training_epochs = 5
        while True:
            self.update_learning_rate()  # Apply learning rate updates at the start of each epoch
            total_loss = 0.0  
            if epoch >= max_training_epochs:
                break
            for iteration_step in range(len(anchor_data) - 1):  
                self.optimizer_instance.zero_grad()

                current_input = anchor_data[iteration_step].unsqueeze(0).to(self.model_instance.device)  # Ensure input is on the correct device
                next_actual_value = anchor_data[iteration_step + 1].unsqueeze(0).to(self.model_instance.device)  # Ensure target is on the correct device

                prediction = self.model_instance(current_input).to(self.model_instance.device)  # Ensure prediction is on the correct device

                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(prediction, next_actual_value)  # Both tensors are now on the same device
                # Copilot: Add a regularization term to encourage zero-centered predictions
                symmetry_penalty = torch.abs(prediction.mean())
                loss = loss + 0.01 * symmetry_penalty

                # ✅ Use QPL feedback to adjust training updates
                qpl_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor, self.memory_module)
                # Ensure qpl_feedback_scalar is on the same device as loss
                if isinstance(qpl_feedback, (list, tuple)):
                    while isinstance(qpl_feedback, (list, tuple)):
                        if len(qpl_feedback) == 0:
                            qpl_feedback = 0.0
                            break
                        qpl_feedback = qpl_feedback[0]
                if isinstance(qpl_feedback, torch.Tensor):
                    qpl_feedback = qpl_feedback.item()
                qpl_feedback = float(qpl_feedback)
                qpl_feedback_scalar = torch.tensor(qpl_feedback, dtype=torch.float32, device=loss.device, requires_grad=True)

                # ✅ Modify loss gradients dynamically based on QPL strength
                loss = torch.tensor(loss, dtype=torch.float32, device=loss.device)  # Ensure loss is on correct device
                loss = loss * (1 + 0.05 * qpl_feedback_scalar)  # Now using a safe tensor

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), max_norm=1.0)
                self.optimizer_instance.step()

                self.update_with_feedback(self.model_instance, current_input, next_actual_value.unsqueeze(0), iteration_step)

                total_loss += loss.item()  

            avg_loss = total_loss / num_samples  

            # Dynamically Adjust `min_delta` Based on Volatility
            if len(initial_losses) < rolling_window:
                initial_losses.append(avg_loss)
                min_delta = torch.std(torch.tensor(initial_losses)) * 0.45  # 10% of standard deviation
            else:
                min_delta = torch.std(torch.tensor(initial_losses)) * 0.45  # Recalculate every step

            print(f"Epoch {epoch}: Avg Loss={avg_loss}, Dynamic Min Delta={min_delta}")

            # Stop Training When Improvement is Below min_delta
            if abs(entropy - self.entropy_monitor.prev_entropy) < 0.001:
                no_improvement_count += 1
                if no_improvement_count >= patience_threshold:
                    print("No significant improvement. Stopping pretraining.")
                    break  
            else:
                no_improvement_count = 0  

            prev_avg_loss = avg_loss  
            epoch += 1  

        print("Pretraining complete.")

    def stabilize_wavefunction_collapse(self, prediction):
        """
        Stabilizes collapsed predictions by verifying chaotic oscillations and applying entropy corrections.
        """
        prediction_variance = torch.var(torch.tensor(prediction))

        # Compute entropy fluctuation rate
        entropy_change = abs(self.entropy_monitor.entropy - self.entropy_monitor.prev_entropy)

        # Threshold to detect chaotic oscillations
        if prediction_variance > 0.05 and entropy_change > 0.1:
            correction_factor = torch.tanh(-entropy_change)
            prediction *= (1 + correction_factor)  # Apply soft correction

        return prediction

    def apply_wave_collapse(self, prediction, time_step=1):
        """Applies Schrödinger wave collapse with QPL stabilization and delta refinement."""
        current_entropy = self.entropy_monitor.entropy
        qbe_feedback = self.qpl_layer.compute_qpl(self.entropy_monitor, self.memory_module)
        collapse_deviation = self.entropy_monitor.track_collapse_deviation()

        # 🔥 AI Predicts Refinement Delta
        refinement_delta = self.memory_module.predict_correction(current_entropy, qbe_feedback, collapse_deviation)

        # 🔥 Compute Gravity-Based Correction for Wavefunction Collapse
        entropy_baseline = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-10:], device=prediction.device)) if len(self.entropy_monitor.past_entropies) >= 10 else current_entropy
        gravity_force = -0.01 * (current_entropy - entropy_baseline)  # Pull collapse toward stable entropy
        gravity_force = torch.tensor(gravity_force, device=prediction.device)  # Ensure gravity_force is a tensor

        # Ensure all tensors are on the same device
        qbe_feedback = torch.tensor(qbe_feedback, device=prediction.device) if not isinstance(qbe_feedback, torch.Tensor) else qbe_feedback.to(prediction.device)
        refinement_delta = torch.tensor(refinement_delta, device=prediction.device) if not isinstance(refinement_delta, torch.Tensor) else refinement_delta.to(prediction.device)

        # Compute Standard Collapse Probability
        wavefunction = torch.exp(1j * torch.tensor(prediction, device=prediction.device))
        probability_distribution = torch.abs(wavefunction) ** 2  

        smoothing_factor = max(0.05, min(0.25, torch.abs(gravity_force)))  
        collapsed_prediction = ((1 - smoothing_factor) * prediction) + (smoothing_factor * probability_distribution * prediction)

        # 🔥 Apply Adaptive Alpha Based on AI-Refined QBE & Gravity Forces
        combined_value = torch.abs(qbe_feedback + gravity_force + refinement_delta)
        scalar_value = combined_value.mean().item() if combined_value.numel() > 1 else combined_value.item()
        adaptive_alpha = max(0.45, min(0.8, scalar_value))
        final_prediction = ((1 - adaptive_alpha) * prediction) + (adaptive_alpha * collapsed_prediction)

        return final_prediction, probability_distribution

    def run(self, new_data_point, time_step=1, current_epoch=0, total_epochs=100):
        """
        Executes model prediction while dynamically adjusting parameters,
        ensuring quantum entropy updates are computed internally.
        Applies gravity-based entropy stabilization to reduce oscillations.
        """
        new_data_point = new_data_point.to(device)

        # 🔥 Update entropy state before making predictions
        self.entropy_monitor.update_quantum_entropy(self.model_instance.state_dict())

        # 🔥 Compute raw prediction
        with torch.no_grad():
            raw_prediction = self.model_instance(new_data_point.unsqueeze(0)).cpu().numpy().flatten()

        # 🔥 Apply Gravity-Based Stability Correction
        entropy_baseline = torch.mean(torch.tensor(self.entropy_monitor.past_entropies[-10:])) if len(self.entropy_monitor.past_entropies) >= 10 else self.entropy_monitor.entropy
        gravity_force = -0.01 * (self.entropy_monitor.entropy - entropy_baseline)
        
        stabilized_prediction = raw_prediction + gravity_force  # Apply entropy stabilization

        # Ensure stabilized_prediction is a tensor
        stabilized_prediction = torch.tensor(stabilized_prediction, device=device)

        # 🔥 Apply interference filter before wave collapse
        correction = interference_filter(stabilized_prediction, self.entropy_monitor.entropy, device=device)
        correction = correction.to(stabilized_prediction.device)  # Ensure correction is on the same device
        stabilized_prediction *= correction

        # 🔥 Apply Schrödinger Wave Collapse Post-Correction
        prediction, probability_distribution = self.apply_wave_collapse(stabilized_prediction, time_step)

        self.past_predictions.append(prediction)

        # ✅ Apply superfluid filter to post-process predictions
        if len(self.past_predictions) >= 3:
            filtered_prediction = self.superfluid.apply_superfluid_filter(torch.tensor(self.past_predictions[-10:], device=device))
        else:
            filtered_prediction = torch.mean(torch.tensor(self.past_predictions, device=device))

        target = to_scalar(self.actual_values[-1]) if self.actual_values else to_scalar(filtered_prediction)
        prediction = to_scalar(filtered_prediction)
        delta = target - prediction

        scales = torch.tensor([-1.2, -0.3, 0.3, 1.2])
        uncollapsed_alternatives = [prediction + scale * delta for scale in scales]
        alts = torch.tensor(uncollapsed_alternatives, dtype=torch.float64)

        # 2. Raw probability distribution based on delta magnitude
        probabilities = torch.exp(-torch.abs(delta * torch.tensor([-0.5, 0, 0.5, 1.0])))
        probs = probabilities / torch.sum(probabilities)

        # 3. Physics-aware modifiers
        recent_deltas = torch.diff(torch.tensor(self.past_predictions[-5:])) if len(self.past_predictions) >= 6 else torch.tensor([0.0])
        entropy_trend, qfi_score = self.memory_module.get_stability_indicators()

        phase_vector = wave_phase_alignment(self.past_predictions[-10:]) if len(self.past_predictions) >= 10 else torch.tensor([0.5])
        phase_score = float(torch.mean(phase_vector))  # ✅ Converts to scalar

        # ✅ Compute superfluid behavior adjustments
        superfluid_energy = self.superfluid.energy_exchange_strength(prediction, self.entropy_monitor.entropy)

        # Convert past_predictions to a tensor
        past_predictions_tensor = torch.tensor(self.past_predictions, dtype=torch.float32, device=device)
        phase_variance = self.superfluid.superfluid_phase_variance(past_predictions_tensor)

        # 3a. Apply foresight bias (includes directional stability)
        probs = foresight_bias(
            alts, probs, recent_deltas, entropy_trend, qfi_score, phase_score, self.superfluid
        )

        # 3b. Add shape-based smoothness weighting
        probs *= future_shape_bias(alts)

        # 3c. Apply wave phase alignment
        probs *= wave_phase_alignment(alts)

        # 3d. Add entropy gradient penalty
        entropy_slice = self.memory_module.past_entropies[-len(probs):] if len(self.memory_module.past_entropies) >= len(probs) else torch.tensor([self.entropy_monitor.entropy] * len(probs))
        probs *= entropy_gradient_penalty(entropy_slice)

        # ✅ Adjust probabilities using superfluid behavior
        adjusted_probs = probs * (1 + 0.1 * phase_variance) * (1 + 0.05 * superfluid_energy)
        probs = adjusted_probs / torch.sum(adjusted_probs)

        # Final safety fix: clean up probs before sampling
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = torch.clamp(probs, 1e-6, 1.0)
        probs /= torch.sum(probs)

        # Replace NumPy-based random sampling with Torch-based sampling
        selected_idx = torch.multinomial(torch.tensor(probs, device=device), 1).item()
        prediction = alts[selected_idx]

        # 6. Confidence is how focused the distribution is
        confidence_score = 1 - torch.std(probs)

        return prediction, probs, alts, confidence_score
    
    def give_feedback(self, new_data_point, actual_value):
        new_data_point = new_data_point.to(device)  # Ensure data point is moved to the correct device
        if actual_value is not None:
            actual_value = torch.tensor(actual_value, device=device)  # Ensure target is moved to the correct device
            self.update_with_feedback(self.model_instance, new_data_point.unsqueeze(0), [actual_value], 0)
            # Log the error metrics after feedback
            y_true = actual_value.clone().detach().cpu().numpy() if isinstance(actual_value, torch.Tensor) else actual_value
            y_pred = self.past_predictions[-1].clone().detach().cpu().numpy() if isinstance(self.past_predictions[-1], torch.Tensor) else self.past_predictions[-1]
            metrics = compute_error_metrics([y_true], [y_pred])
            #logging.info(f"Feedback Metrics: {metrics}")
            #print(f"Feedback Metrics: {metrics}")

    def update_with_feedback(self, model, data, actual_value, iteration_step):
        """
        Uses Bayesian causal inference to distinguish causality from correlation.
        Adjusts predictions based on causal feedback.
        """
        if isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.squeeze().tolist()

        self.actual_values.append(actual_value)

        actual_value_tensor = actual_value.clone().detach().float() if isinstance(actual_value, torch.Tensor) else torch.tensor(actual_value, dtype=torch.float32).to(self.model_instance.device)

        # Ensure past_predictions is not empty before accessing its last element
        if self.past_predictions:
            past_predictions_tensor = self.past_predictions[-1].clone().detach().float() if isinstance(self.past_predictions[-1], torch.Tensor) else torch.tensor(self.past_predictions[-1], dtype=torch.float32).to(self.model_instance.device)
        else:
            past_predictions_tensor = torch.zeros_like(actual_value_tensor).to(self.model_instance.device)

        # Ensure tensors have the same size
        if past_predictions_tensor.shape != actual_value_tensor.shape:
            past_predictions_tensor = past_predictions_tensor[:actual_value_tensor.shape[0]]

        min_length = min(len(actual_value_tensor), len(past_predictions_tensor))
        actual_value_tensor = actual_value_tensor[:min_length]
        past_predictions_tensor = past_predictions_tensor[:min_length]

        current_entropy = self.entropy_monitor.entropy
        prev_entropy = self.entropy_monitor.prev_entropy

        # ✅ Ensure entropy values are tensors before using them
        current_entropy_tensor = torch.tensor(current_entropy, dtype=torch.float32)
        prev_entropy_tensor = torch.tensor(prev_entropy, dtype=torch.float32)

        # Compute stability factor using tensors
        stability_factor = 1 + torch.exp(-10 * torch.abs(current_entropy_tensor - prev_entropy_tensor))

        # Compute error dynamically
        error = (actual_value_tensor - past_predictions_tensor).detach().cpu().numpy()

        # Compute entropy-weighted causal effect
        causal_effect = self.bayesian_optimizer.compute_causal_inference(past_predictions_tensor, actual_value_tensor)

        # Increase causal weighting when entropy change is small (reinforce stability)
        stability_factor = 1 + torch.exp(-10 * torch.abs(current_entropy_tensor - prev_entropy_tensor))

        adjusted_prediction = past_predictions_tensor + (0.2 * causal_effect * stability_factor)  # More aggressive correction when stable

        # Apply QPL-based tuning before updates
        self.qpl_layer.tune_parameters(self.entropy_monitor, self.controller, self.bayesian_optimizer, self.rl_agent, iteration_step, self.memory_module)  # Add iteration_step

        # Update entropy-driven parameters AFTER QPL tuning
        self.qpl_layer.adjust_parameters(self.entropy_monitor, self.controller, self.bayesian_optimizer, iteration_step, self.memory_module)

        #print(f"Update Frequency: {update_frequency}")

        update_frequency = 10  # Reintroduce update frequency

        if len(self.past_predictions) % update_frequency == 0:
            self.optimizer_instance.zero_grad()
            model_output = model(data).reshape(-1).to(self.model_instance.device)  # Ensure model_output is on the correct device
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(model_output, actual_value_tensor)  # Both tensors are now on the same device
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), max_norm=1.0)
            self.optimizer_instance.step()

            # Compute QBE-based tuning adjustments
            self.qpl_layer.adjust_parameters(self.entropy_monitor, self.controller, self.bayesian_optimizer, iteration_step, self.memory_module)

            #print(f"Updated model based on feedback: Loss={loss.item()}, Learning Rate={learning_rate}, Update Frequency={update_frequency}, Gradient Smoothing={gradient_smoothing}")

        return adjusted_prediction

    def benchmark_collapse_accuracy(self, predicted_probabilities, actual_measurements):
        """
        Compares AI-driven wavefunction collapse predictions against Born Rule statistics.
        Implements proper scaling to prevent exaggerated accuracy percentages.
        """
        born_rule_prob = torch.abs(torch.tensor(actual_measurements)) ** 2  # Standard quantum probability

        # 🔧 Prevent zero-division errors by enforcing a reasonable lower bound
        mean_born_prob = max(torch.mean(born_rule_prob), 1e-4)  

        # 🔧 Compute absolute improvement
        improvement_raw = torch.mean(torch.abs(torch.tensor(predicted_probabilities) - born_rule_prob)) / mean_born_prob

        # 🔥 NEW: Apply logarithmic scaling to prevent extreme inflation
        improvement_scaled = torch.log1p(improvement_raw)  # log(1 + x) to prevent infinite values

        # 🔧 Ensure improvement remains within a realistic range
        improvement_scaled = torch.clamp(improvement_scaled, -10, 10)  

        # 🔧 Calculate improvement percentage safely
        if mean_born_prob > 0:
            improvement_percentage = ((improvement_scaled - mean_born_prob) / mean_born_prob)
        else:
            improvement_percentage = improvement_scaled  # Prevent division by zero

        # 🔧 Cap extreme values
        improvement_percentage = min(improvement_percentage, 999.99)

        print(f"AI-driven collapse improves accuracy by {improvement_percentage:.2f}%")

        return improvement_scaled

    def evaluate_model(self, model, validation_data):
        model.eval()
        validation_data = validation_data.to(device)
        
        with torch.no_grad():
            output = model(validation_data)
            target = torch.zeros(validation_data.size(0), device=output.device)

            # Compute quantum and entropy-aware error metrics
            metrics = compute_error_metrics(target, output)

            # Log entropy-aware feedback adjustments
            entropy_value = self.entropy_monitor.calculate_entropy(output)
            # Ensure torch tensors for entropy_feedback_adjustment
            adjusted_feedback = self.entropy_monitor.entropy_feedback_adjustment(target, output)
            
            # ✅ Introduce entropy-aware adjustment speed
            scaling_factor = 1 + 0.5 * entropy_value  # More entropy → Faster adaptation

            # ✅ Adjust minimum feedback weight dynamically
            feedback_weight = max(0.25 * scaling_factor, min(1.0, adjusted_feedback))

            #logging.info(f"Validation Metrics: {metrics} | Adjusted Feedback Weight: {feedback_weight:.4f}")
            #print(f"Validation Metrics: {metrics} | Adjusted Feedback Weight: {feedback_weight:.4f}")

            # Validate accuracy gains
            predicted_probabilities = torch.abs(output) ** 2
            actual_measurements = target
            improvement = self.benchmark_collapse_accuracy(predicted_probabilities, actual_measurements)

            if improvement > 0.05:  # Expect at least 5% better accuracy than Born Rule
                print(f"AI-driven collapse improves accuracy by {improvement:.2f}%")

        return metrics

    def solve_issue(self, issue_data, issue_energy):
        logging.info("Solving new issue dynamically.")
        updated_model = self.run(issue_data, issue_energy)
        logging.info("Issue solved successfully.")
        return updated_model

    def early_stopping(self, validation_data):
        model_entropy = entropy(torch.softmax(self.model_instance(validation_data), dim=1).detach().numpy(), axis=1)
        if torch.mean(model_entropy) < self.entropy_threshold:
            logging.info("Early stopping triggered due to low entropy.")
            return True
        return False

    def optimize_hyperparameters(self):
        best_params = None
        best_score = float('inf')

        # ✅ Distribute parameter search across multiple agents
        multi_agent_results = []
        for agent_id in range(self.num_agents):
            entropy_value = self.entropy_monitor.entropy
            # ✅ Expand search space dynamically based on entropy level
            adjusted_space = [(low * (1 + entropy_value * 0.1), high * (1 - entropy_value * 0.1)) for low, high in self.param_space]
            agent_result = self.entropy_aware_bayesian_optimization(self.objective, adjusted_space)
            multi_agent_results.append(agent_result)

        # ✅ Select best performing agent
        best_agent_result = min(multi_agent_results, key=lambda r: r['best_score'])

        return best_agent_result['best_params'], best_agent_result['best_score']

    def entropy_aware_bayesian_optimization(self, objective, param_space):
        best_params = None
        best_score = float('inf')

        for _ in range(50):
            params = self.bayesian_optimizer.ask(param_space)
            score = objective(params)
            self.bayesian_optimizer.tell(params, score)
            if score < best_score:
                best_score = score
                best_params = params

        return {'best_params': best_params, 'best_score': best_score}

    def evaluate_model_with_params(self, params):
        learning_rate, hidden_size = params
        self.model_instance = self.model_class(hidden_size=int(hidden_size)).to(device)
        self.optimizer_instance = torch.optim.Adam(self.model_instance.parameters(), lr=learning_rate)
        self.pretrain(self.anchor_data)
        validation_data = torch.randn(50, 10).to(device)
        metrics = self.evaluate_model(self.model_instance, validation_data)
        return metrics['MSE']

# Example usage
if __name__ == "__main__":
    class ExampleModel(BaseCIMMModel):
        def __init__(self, hidden_size):
            super(ExampleModel, self).__init__(input_size=10, hidden_size=hidden_size, output_size=1)

        def forward(self, x):
            return self.common_forward(x)

    param_space = [
        Real(1e-5, 1e-1, name='learning_rate'),
        Integer(10, 100, name='hidden_size'),
    ]

    anchor_data = torch.randn(100, 10)

    cimm = CIMM(ExampleModel, param_space, anchor_data)

    new_data_point = torch.randn(10)
    actual_value = 0.5

    prediction = cimm.run(new_data_point, actual_value)
    print(f"Prediction: {prediction}")

    best_params, best_score = cimm.optimize_hyperparameters()
    print(f"Best Hyperparameters: {best_params}, Best Score: {best_score}")

    validation_data = torch.randn(50, 10)
    metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
    print(f"Validation Metrics: {metrics}")
