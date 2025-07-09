import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- TinyCIMM-Euler: Higher-Order Mathematical Reasoning Utilities ---
class HigherOrderEntropyMonitor:
    """Monitor for higher-order mathematical patterns and complexity"""
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.complexity_metric = 0.0
        self.past_metrics = []
        self.order_history = []

    def update(self, signal):
        """Compute higher-order mathematical complexity from signal patterns"""
        # Compute multiple orders of derivatives for complexity analysis
        if signal.numel() < 3:
            complexity = 0.5
        else:
            signal_flat = signal.flatten()
            
            # First and second order differences (mathematical derivatives)
            first_diff = torch.diff(signal_flat)
            if len(first_diff) > 1:
                second_diff = torch.diff(first_diff)
                # Higher-order complexity based on curvature and variation
                complexity = (torch.var(first_diff) + torch.var(second_diff)).item()
            else:
                complexity = torch.var(signal_flat).item()
        
        self.complexity_metric = self.momentum * self.complexity_metric + (1 - self.momentum) * complexity
        self.past_metrics.append(self.complexity_metric)
        self.order_history.append(len(torch.where(torch.abs(signal.flatten()) > 0.1)[0]))
        return self.complexity_metric

    def get_variance(self):
        if len(self.past_metrics) < 2:
            return 0.0
        return torch.var(torch.tensor(self.past_metrics)).item()

def higher_order_transform(pred, target, complexity_factor):
    """Apply higher-order mathematical transformation for enhanced prediction"""
    delta = torch.abs(target - pred)
    # Use mathematical complexity to guide correction strength
    correction_strength = 0.05 * (1 + complexity_factor / 10)
    correction = correction_strength * (target - pred) / (1 + delta)
    return pred + correction

def compute_mathematical_coherence(signal):
    """Compute mathematical coherence using higher-order derivatives"""
    if signal.numel() < 3:
        return torch.tensor(0.5)
    
    signal_flat = signal.flatten()
    # Compute mathematical smoothness via higher-order derivatives
    first_grad = torch.gradient(signal_flat)[0]
    if len(first_grad) > 1:
        second_grad = torch.gradient(first_grad)[0]
        # Coherence based on smoothness of higher-order derivatives
        coherence = torch.exp(-torch.mean(torch.abs(second_grad)))
    else:
        coherence = torch.exp(-torch.mean(torch.abs(first_grad)))
    
    return coherence

class CIMMInspiredController:
    """CIMM-inspired adaptive controller for stability and performance"""
    def __init__(self, min_lr=1e-4, max_lr=0.05, entropy_window=20, damping_factor=0.9):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.entropy_window = entropy_window
        self.damping_factor = damping_factor
        self.entropy_history = []
        self.lr_history = []
        self.performance_history = []
        
    def adaptive_learning_rate(self, current_lr, entropy, performance, adaptation_signal):
        """CIMM-inspired adaptive learning rate with dynamic field-aware entropy feedback"""
        self.entropy_history.append(entropy)
        self.performance_history.append(performance)
        
        # Keep history within window
        if len(self.entropy_history) > self.entropy_window:
            self.entropy_history.pop(0)
            self.performance_history.pop(0)
        
        if len(self.entropy_history) < 3:
            return current_lr
            
        # Compute entropy variance and trends
        entropy_tensor = torch.tensor(self.entropy_history)
        performance_tensor = torch.tensor(self.performance_history)
        
        entropy_variance = torch.var(entropy_tensor).item()
        entropy_trend = entropy_tensor[-1] - entropy_tensor[0]
        performance_variance = torch.var(performance_tensor).item()
        performance_trend = performance_tensor[-1] - performance_tensor[0]
        
        # Dynamic base adjustment factors (CIMM-inspired balance) - more aggressive for faster learning
        base_adjustment = 1.0
        
        # Entropy-based adjustment with dynamic scaling - increased responsiveness
        entropy_factor = 1.0 + torch.tanh(entropy_trend * 0.5).item() * 0.4
        
        # Performance-based adjustment with balance consideration - more aggressive
        performance_factor = 1.0 + (performance_trend * 0.3) - (performance_variance * 0.1)
        
        # CIMM-inspired quantum wave learning rate adjustment with dynamic amplitude - increased amplitude
        phase_shift = torch.tanh(entropy_trend * 0.2) * 0.3
        wave_amplitude = 1.0 + 0.1 * torch.cos(torch.tensor(entropy) * torch.pi)
        quantum_adjustment = wave_amplitude * torch.exp(1j * phase_shift).real
        
        # Dynamic entropy-based damping (inspired by CIMM's superfluid dynamics) - reduced damping for faster learning
        damping_strength = 2 + entropy_variance * 3  # Reduced damping strength for faster adaptation
        damping = 1.0 / (1.0 + torch.exp(-torch.abs(torch.tensor(entropy_variance)) * damping_strength))
        
        # Apply all dynamic adjustments
        new_lr = current_lr * entropy_factor * performance_factor * quantum_adjustment.item() * damping.item()
        
        # Dynamic stability constraints based on field balance - more aggressive ranges
        min_lr_factor = 0.3 + entropy_variance * 1.0  # Higher minimum when more volatile for faster adaptation
        max_lr_factor = 3.0 - performance_variance * 0.5  # Allow higher maximum, less sensitive to variance
        
        dynamic_min_lr = self.min_lr * min_lr_factor
        dynamic_max_lr = self.max_lr * max_lr_factor
        
        new_lr = torch.clamp(torch.tensor(new_lr), dynamic_min_lr, dynamic_max_lr).item()
        
        self.lr_history.append(new_lr)
        return new_lr
        
    def compute_coherence(self, signal):
        """Compute mathematical coherence using CIMM-inspired superfluid dynamics"""
        if signal.numel() < 3:
            return torch.tensor(0.5)
        
        signal_flat = signal.flatten()
        first_grad = torch.gradient(signal_flat)[0]
        if len(first_grad) > 1:
            second_grad = torch.gradient(first_grad)[0]
            coherence = torch.exp(-torch.mean(torch.abs(second_grad)))
        else:
            coherence = torch.exp(-torch.mean(torch.abs(first_grad)))
        
        return torch.clamp(coherence, 0.0, 1.0)

class MathematicalStructureController:
    """Controller for higher-order mathematical structure adaptation with dynamic, balance-based thresholds"""
    def __init__(self, base_complexity_threshold=0.01, min_neurons=6, max_neurons=128, adaptation_window=5):
        self.base_complexity_threshold = base_complexity_threshold
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.adaptation_window = adaptation_window
        self.complexity_hist = []
        self.performance_hist = []
        self.structure_hist = []
        self.cooldown_counter = 0
        self.base_cooldown_period = 10
        
        # Dynamic threshold adaptation (CIMM-inspired)
        self.complexity_balance_history = []
        self.performance_balance_history = []
        self.structure_balance_history = []

    def _compute_dynamic_thresholds(self, complexity_metric, performance, adaptation_signal):
        """Compute dynamic, balance-based thresholds like CIMM"""
        # Update balance histories
        self.complexity_balance_history.append(complexity_metric)
        self.performance_balance_history.append(performance)
        self.structure_balance_history.append(adaptation_signal)
        
        # Keep histories manageable
        max_history = 50
        if len(self.complexity_balance_history) > max_history:
            self.complexity_balance_history.pop(0)
            self.performance_balance_history.pop(0)
            self.structure_balance_history.pop(0)
        
        if len(self.complexity_balance_history) < 5:
            # Not enough history, use base thresholds
            return {
                'complexity_threshold': self.base_complexity_threshold,
                'performance_threshold': 0.005,
                'structure_threshold': 0.005,
                'cooldown_period': self.base_cooldown_period
            }
        
        # Compute field balance metrics
        complexity_tensor = torch.tensor(self.complexity_balance_history)
        performance_tensor = torch.tensor(self.performance_balance_history)
        structure_tensor = torch.tensor(self.structure_balance_history)
        
        # Dynamic complexity threshold based on field variance
        complexity_variance = torch.var(complexity_tensor[-20:]).item()
        complexity_mean = torch.mean(complexity_tensor[-10:]).item()
        dynamic_complexity_threshold = self.base_complexity_threshold * (1 + complexity_variance) * \
                                     torch.tanh(torch.tensor(complexity_mean * 10)).item()
        
        # Dynamic performance threshold based on performance stability
        performance_variance = torch.var(performance_tensor[-20:]).item()
        performance_trend = (performance_tensor[-1] - performance_tensor[-5]).item() if len(performance_tensor) >= 5 else 0
        dynamic_performance_threshold = 0.005 * (1 + performance_variance) * (1 + abs(performance_trend))
        
        # Dynamic structure threshold based on adaptation signal variance
        structure_variance = torch.var(structure_tensor[-20:]).item()
        structure_mean = torch.mean(structure_tensor[-10:]).item()
        dynamic_structure_threshold = 0.005 * (1 + structure_variance) * \
                                    torch.sigmoid(torch.tensor(structure_mean)).item()
        
        # Dynamic cooldown period based on overall field stability
        field_stability = 1.0 / (1 + complexity_variance + performance_variance + structure_variance)
        dynamic_cooldown = int(self.base_cooldown_period * (2 - field_stability))
        
        return {
            'complexity_threshold': dynamic_complexity_threshold,
            'performance_threshold': dynamic_performance_threshold,
            'structure_threshold': dynamic_structure_threshold,
            'cooldown_period': dynamic_cooldown
        }

    def decide(self, complexity_metric, performance, adaptation_signal, num_neurons):
        """Make structure adaptation decisions based on dynamic, balance-based thresholds"""
        self.complexity_hist.append(complexity_metric)
        self.performance_hist.append(performance)
        self.structure_hist.append(adaptation_signal)
        
        # Compute dynamic thresholds
        thresholds = self._compute_dynamic_thresholds(complexity_metric, performance, adaptation_signal)
        
        # Dynamic cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return "none", 0
        
        if len(self.complexity_hist) < self.adaptation_window:
            return "none", 0
            
        complexity_arr = torch.tensor(self.complexity_hist[-self.adaptation_window:])
        performance_arr = torch.tensor(self.performance_hist[-self.adaptation_window:])
        structure_arr = torch.tensor(self.structure_hist[-self.adaptation_window:])
        
        complexity_trend = complexity_arr.mean().item()
        performance_var = performance_arr.var().item()
        structure_stability = structure_arr.std().item()
        
        action = "none"
        amount = 0
        
        # CIMM-inspired dynamic adaptation with balance-based thresholds
        if (complexity_trend > thresholds['complexity_threshold'] and 
            performance_var > thresholds['performance_threshold'] and 
            num_neurons < self.max_neurons):
            # Increase capacity for higher-order mathematical reasoning
            growth_factor = min(0.3, complexity_trend / thresholds['complexity_threshold'] * 0.15)
            amount = max(2, int(num_neurons * growth_factor))
            action = "grow"
            self.cooldown_counter = thresholds['cooldown_period']
        elif (complexity_trend < thresholds['complexity_threshold'] * 0.3 and 
              structure_stability < thresholds['structure_threshold'] and 
              num_neurons > self.min_neurons):
            # Reduce capacity when mathematical complexity is low
            prune_factor = min(0.2, thresholds['structure_threshold'] / structure_stability * 0.1)
            amount = max(1, int(num_neurons * prune_factor))
            action = "prune"
            self.cooldown_counter = thresholds['cooldown_period']
            
        return action, amount

# --- TinyCIMM-Euler: Higher-Order Mathematical Reasoning Model ---
class TinyCIMMEuler(nn.Module):
    """
    TinyCIMM-Euler: Higher-order mathematical reasoning version of TinyCIMM.
    Designed for complex mathematical pattern recognition, sequence prediction,
    and higher-order mathematical reasoning tasks like prime number prediction.
    """
    def __init__(self, input_size, hidden_size, output_size, device, **kwargs):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_size
        
        # Core parameters for higher-order mathematical reasoning
        self.W = nn.Parameter(0.05 * torch.randn(hidden_size, input_size, device=device))
        self.b = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.V = nn.Parameter(0.05 * torch.randn(output_size, hidden_size, device=device))
        self.c = nn.Parameter(torch.zeros(output_size, device=device))
        
        # CIMM-inspired controllers and loss
        self.cimm_controller = CIMMInspiredController()
        self.structure_controller = MathematicalStructureController()
        self.field_loss = CIMMInspiredLoss(lambda_qbe=0.1, lambda_entropy=0.05, lambda_coherence=0.02)
        
        # Higher-order mathematical components
        self.complexity_monitor = None
        self.complexity_factor = 0.0
        self.adaptation_steps = kwargs.get('adaptation_steps', 20)
        
        # Mathematical memory system for pattern recognition
        self.math_memory = []
        self.math_memory_size = kwargs.get('math_memory_size', 10)
        self.pattern_decay = kwargs.get('pattern_decay', 0.95)
        
        # Higher-order reasoning layers
        self.higher_order_processor = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.Tanh(),  # Better for mathematical reasoning
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        ).to(device)
        
        # Optimizer tuned for mathematical learning with CIMM-inspired parameters
        self.base_lr = 0.02  # More aggressive initial learning rate for faster mathematical learning
        self.optimizer = torch.optim.Adam(
            [self.W, self.b, self.V, self.c], 
            lr=self.base_lr,
            weight_decay=0.0005,  # Reduced weight decay for faster learning
            eps=1e-8  # Better numerical stability
        )
        
        # State tracking for mathematical reasoning
        self.last_h = None
        self.last_x = None
        self.last_prediction = None
        self.complexity_history = []
        self.pattern_history = []
        
        # Adaptation state
        self.prev_loss = None
        self.prev_complexity = None
        self.current_step = 0
        self.structure_stability = 0
        self.performance_metric = 0.0

    def set_complexity_monitor(self, monitor):
        """Set the mathematical complexity monitor"""
        self.complexity_monitor = monitor

    def forward(self, x, y_true=None):
        """Forward pass with higher-order mathematical reasoning"""
        # Standard forward pass
        h = torch.relu(x @ self.W.T + self.b)
        
        # Higher-order mathematical processing
        higher_order_signal = self.higher_order_processor(x)
        
        # Store in mathematical memory for pattern recognition
        self.math_memory.append(h.detach().cpu() * self.pattern_decay)
        if len(self.math_memory) > self.math_memory_size:
            self.math_memory.pop(0)
        
        # Output computation
        y = h @ self.V.T + self.c
        
        # Apply higher-order mathematical correction if target is available
        if y_true is not None:
            y = higher_order_transform(y, y_true, self.complexity_factor)
        
        # Update state
        self.last_h = h
        self.last_x = x
        self.last_prediction = y.detach()
        
        # Update complexity factor
        self.complexity_factor = (self.complexity_factor + 0.01) % 1.0
        
        return y

    def log_complexity_metric(self):
        """Compute mathematical complexity metric for current state"""
        if self.W.shape[0] > 1 and torch.isfinite(self.W).all():
            # Compute complexity based on weight distribution and higher-order patterns
            weight_var = torch.var(self.W).item()
            weight_entropy = -torch.sum(F.softmax(self.W.flatten(), dim=0) * 
                                      torch.log(F.softmax(self.W.flatten(), dim=0) + 1e-9)).item()
            complexity_metric = weight_var + 0.1 * weight_entropy
        else:
            complexity_metric = 0.5
        
        self.complexity_history.append(complexity_metric)
        return complexity_metric

    def compute_field_aware_performance(self, predictions, targets):
        """
        Compute field-aware performance based on pattern recognition and structural coherence
        rather than simple distance metrics
        """
        # Get CIMM-inspired error metrics for field analysis
        cimm_metrics = compute_cimm_error_metrics(targets, predictions)
        
        # Pattern Recognition Score (based on prediction accuracy, not just divergence)
        # Lower prediction error = better pattern recognition
        prediction_accuracy = 1.0 / (1.0 + torch.mean((predictions - targets) ** 2).item())
        kl_div = cimm_metrics["KL-Divergence"]
        js_div = cimm_metrics["Jensen-Shannon"]
        pattern_recognition_score = prediction_accuracy * (1.0 / (1.0 + kl_div + js_div))
        
        # Field Coherence Score (based on QWCS and Wasserstein)
        qwcs = cimm_metrics["QWCS"]
        wasserstein = cimm_metrics["Wasserstein Distance"]
        field_coherence_score = qwcs / (1.0 + wasserstein)
        
        # Structural Complexity Score (based on entropy and network state)
        entropy_value = cimm_metrics["entropy_value"]
        if hasattr(self, 'W') and self.W is not None:
            weight_complexity = torch.var(self.W).item()
            structure_stability = torch.mean(torch.abs(self.W)).item()
        else:
            weight_complexity = 0.5
            structure_stability = 0.5
            
        structural_complexity_score = entropy_value * (1 + weight_complexity) * structure_stability
        
        # Higher-Order Field Dynamics (based on prediction coherence)
        if predictions.numel() >= 3:
            pred_coherence = self.cimm_controller.compute_coherence(predictions)
            higher_order_dynamics = pred_coherence.item()
        else:
            higher_order_dynamics = 0.5
            
        # Mathematical Reasoning Consistency (based on pattern stability)
        if len(self.complexity_history) >= 3:
            complexity_stability = 1.0 / (1.0 + torch.var(torch.tensor(self.complexity_history[-5:])).item())
        else:
            complexity_stability = 0.5
            
        # Quantum Field Performance (combination of all field-aware metrics)
        quantum_field_performance = (
            0.3 * pattern_recognition_score +
            0.25 * field_coherence_score +
            0.2 * structural_complexity_score +
            0.15 * higher_order_dynamics +
            0.1 * complexity_stability
        )
        
        return {
            'pattern_recognition_score': pattern_recognition_score,
            'field_coherence_score': field_coherence_score,
            'structural_complexity_score': structural_complexity_score,
            'higher_order_dynamics': higher_order_dynamics,
            'mathematical_reasoning_consistency': complexity_stability,
            'quantum_field_performance': quantum_field_performance
        }

    def compute_mathematical_performance(self):
        """Compute mathematical reasoning performance based on current state"""
        if self.last_h is None:
            return {
                'pattern_recognition_score': 0.5,
                'field_coherence_score': 0.5,
                'structural_complexity_score': 0.5,
                'higher_order_dynamics': 0.5,
                'mathematical_reasoning_consistency': 0.5,
                'quantum_field_performance': 0.5
            }
        
        # Use current hidden state as both prediction and target for performance analysis
        h_flat = self.last_h.flatten()
        
        # Create a target based on mathematical expectations
        target = torch.tanh(h_flat)  # Mathematical transformation as expected pattern
        
        return self.compute_field_aware_performance(h_flat, target)

    def mathematical_structure_adaptation(self, complexity_metric, performance, adaptation_signal, controller):
        """Adapt network structure for higher-order mathematical reasoning using field-aware signals"""
        if self.prev_loss is None:
            self.prev_loss = adaptation_signal
        if self.prev_complexity is None:
            self.prev_complexity = complexity_metric
        
        # Mathematical structure decision making using field-aware adaptation signal
        action, amount = controller.decide(complexity_metric, performance, adaptation_signal, self.hidden_dim)
        
        min_neurons = 8
        max_neurons = 256  # Larger capacity for mathematical reasoning
        
        if action == "grow" and self.hidden_dim < max_neurons:
            # Increase capacity for complex mathematical patterns
            new_dim = min(self.hidden_dim + amount, max_neurons)
            self._grow_mathematical_network(new_dim)
            # Clear memory on growth to prevent tensor size mismatch
            self.math_memory.clear()
            print(f"Mathematical network growth: {self.hidden_dim} -> {new_dim} neurons")
            
        elif action == "prune" and self.hidden_dim > min_neurons:
            # Optimize structure while maintaining mathematical capability
            new_dim = max(self.hidden_dim - amount, min_neurons)
            self._prune_mathematical_network(new_dim)
            # Clear memory on pruning to prevent tensor size mismatch
            self.math_memory.clear()
            print(f"Mathematical network pruning: {self.hidden_dim} -> {new_dim} neurons")
        
        self.prev_loss = adaptation_signal
        self.prev_complexity = complexity_metric
        self.current_step += 1

    def _grow_mathematical_network(self, new_dim):
        """Grow network optimized for mathematical reasoning"""
        growth = new_dim - self.hidden_dim
        
        # Expand weight matrices with mathematical initialization
        new_W = torch.zeros(new_dim, self.W.shape[1], device=self.device)
        new_W[:self.hidden_dim] = self.W.data
        # Initialize new neurons with small random values for mathematical stability
        new_W[self.hidden_dim:] = 0.01 * torch.randn(growth, self.W.shape[1], device=self.device)
        
        new_b = torch.zeros(new_dim, device=self.device)
        new_b[:self.hidden_dim] = self.b.data
        
        new_V = torch.zeros(self.V.shape[0], new_dim, device=self.device)
        new_V[:, :self.hidden_dim] = self.V.data
        new_V[:, self.hidden_dim:] = 0.01 * torch.randn(self.V.shape[0], growth, device=self.device)
        
        # Update parameters
        self.W = nn.Parameter(new_W)
        self.b = nn.Parameter(new_b)
        self.V = nn.Parameter(new_V)
        self.hidden_dim = new_dim
        
        # Clear mathematical memory to avoid size mismatches
        self.math_memory = []
        
        # Update optimizer with all parameters
        self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c] + 
                                        list(self.higher_order_processor.parameters()), 
                                        lr=0.01, weight_decay=0.001)

    def _prune_mathematical_network(self, new_dim):
        """Prune network while preserving mathematical reasoning capacity"""
        if new_dim >= self.hidden_dim:
            return
        
        # Compute neuron importance for mathematical reasoning
        weight_importance = torch.abs(self.W).sum(dim=1)
        output_importance = torch.abs(self.V).sum(dim=0)
        
        # Combine importance metrics
        total_importance = weight_importance + output_importance
        
        # Keep the most mathematically important neurons
        _, keep_indices = torch.topk(total_importance, new_dim)
        keep_indices = torch.sort(keep_indices)[0]
        
        # Prune parameters
        self.W = nn.Parameter(self.W.data[keep_indices])
        self.b = nn.Parameter(self.b.data[keep_indices])
        self.V = nn.Parameter(self.V.data[:, keep_indices])
        self.hidden_dim = new_dim
        
        # Clear mathematical memory to avoid size mismatches
        self.math_memory = []
        
        # Update optimizer
        self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c] + 
                                        list(self.higher_order_processor.parameters()), 
                                        lr=0.01, weight_decay=0.001)

    def analyze_mathematical_results(self):
        """Analyze experiment results using mathematical reasoning metrics"""
        if len(self.math_memory) < 2:
            return {
                "mathematical_complexity": 0.0,
                "pattern_recognition_score": 0.0,
                "reasoning_consistency": 0.0,
                "higher_order_performance": 0.0
            }

        # Mathematical Complexity Analysis
        complexity_metric = self.log_complexity_metric()

        # Pattern Recognition Score
        if len(self.math_memory) > 1:
            pattern_correlations = []
            for i in range(len(self.math_memory)-1):
                corr = torch.corrcoef(torch.stack([self.math_memory[i].flatten(), 
                                                 self.math_memory[i+1].flatten()]))[0,1]
                if torch.isfinite(corr):
                    pattern_correlations.append(corr.item())
            pattern_score = torch.tensor(pattern_correlations).mean().item() if pattern_correlations else 0.0
        else:
            pattern_score = 0.0

        # Reasoning Consistency
        reasoning_consistency = self.compute_mathematical_performance()

        # Higher-order Performance
        if len(self.complexity_history) > 1:
            complexity_history_tensor = torch.tensor(self.complexity_history[-10:])
            complexity_stability = 1.0 / (1.0 + torch.var(complexity_history_tensor).item())
        else:
            complexity_stability = 0.5

        return {
            "mathematical_complexity": complexity_metric,
            "pattern_recognition_score": pattern_score,
            "reasoning_consistency": reasoning_consistency,
            "higher_order_performance": complexity_stability
        }

    def _compute_dynamic_adaptation_threshold(self, field_performance):
        """Compute dynamic threshold for online adaptation based on field balance"""
        if not hasattr(self, '_adaptation_history'):
            self._adaptation_history = []
        
        # Store field performance history
        self._adaptation_history.append(field_performance['pattern_recognition_score'])
        if len(self._adaptation_history) > 30:
            self._adaptation_history.pop(0)
        
        if len(self._adaptation_history) < 5:
            return 0.6  # Higher base threshold for more frequent adaptation
        
        # Compute dynamic threshold based on recent performance balance
        performance_tensor = torch.tensor(self._adaptation_history)
        performance_mean = torch.mean(performance_tensor).item()
        performance_variance = torch.var(performance_tensor).item()
        
        # Dynamic threshold balances performance level and stability - more aggressive for faster learning
        # Lower threshold when performance is consistently low
        # Higher threshold when performance is volatile
        base_threshold = 0.6  # Higher base threshold for more frequent adaptation
        variance_factor = 1 + performance_variance * 1.5  # Less sensitivity to variance
        mean_factor = 1 - (performance_mean - 0.5) * 0.6  # More aggressive adjustment based on mean performance
        
        dynamic_threshold = base_threshold * variance_factor * mean_factor
        dynamic_threshold = torch.clamp(torch.tensor(dynamic_threshold), 0.1, 0.6).item()
        
        return dynamic_threshold

    def online_adaptation_step(self, x, y_feedback=None):
        """CIMM-inspired online adaptation step with field-aware pattern recognition"""
        # Forward prediction (no training mode)
        self.eval()  # Ensure we're in evaluation mode
        with torch.no_grad():
            prediction = self.forward(x)
        
        if y_feedback is not None:
            # Compute field-aware performance metrics using CIMM metrics
            field_performance = self.compute_field_aware_performance(prediction, y_feedback)
            
            # Compute CIMM-inspired field-aware adaptation signal
            adaptation_signal, cimm_components = self.field_loss.compute_mathematical_field_loss(
                prediction, y_feedback, self.complexity_monitor, self.W
            )
            
            # Update complexity metrics based on prediction patterns
            complexity_metric = self.log_complexity_metric()
            
            # CIMM-inspired adaptive learning rate based on field performance
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = self.cimm_controller.adaptive_learning_rate(
                current_lr, complexity_metric, field_performance['quantum_field_performance'], adaptation_signal.item()
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            # Online adaptation based on actual prediction quality (like CIMM's reinforcement)
            # Use dynamic threshold based on field balance
            adaptation_threshold = self._compute_dynamic_adaptation_threshold(field_performance)
            if field_performance['pattern_recognition_score'] < adaptation_threshold:
                self.train()  # Switch to training mode for adaptation
                self.optimizer.zero_grad()
                
                # Recompute prediction with gradients enabled for backpropagation
                train_prediction = self.forward(x)
                
                # Use both simple feedback loss and QBE balance for stable learning
                feedback_loss = torch.mean((train_prediction - y_feedback) ** 2)
                qbe_balance, _ = self.field_loss.compute_qbe_balance(train_prediction, y_feedback, self.complexity_monitor, self.W)
                
                # Combine for balanced learning
                total_adaptation_loss = 0.7 * feedback_loss + 0.3 * qbe_balance
                total_adaptation_loss.backward()
                
                # Adaptive gradient clipping based on pattern quality
                max_norm = 0.1 if field_performance['pattern_recognition_score'] > 0.5 else 0.05
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
                self.optimizer.step()
                self.eval()  # Switch back to evaluation mode
            
            # CIMM-inspired entropy-aware structure adaptation based on field performance
            self.mathematical_structure_adaptation(
                complexity_metric, 
                field_performance['quantum_field_performance'], 
                adaptation_signal.item(), 
                self.structure_controller
            )
            
            # Apply field optimization periodically
            if hasattr(self, '_adaptation_counter'):
                self._adaptation_counter += 1
            else:
                self._adaptation_counter = 1
                
            if self._adaptation_counter % 100 == 0:
                self.entropy_aware_field_optimization()
            
            return {
                'prediction': prediction,
                'adaptation_signal': adaptation_signal.item(),
                'complexity_metric': complexity_metric,
                'field_performance': field_performance,
                'cimm_components': cimm_components,
                'learning_rate': new_lr
            }
        else:
            # Pure prediction without feedback
            return {
                'prediction': prediction,
                'adaptation_signal': 0.0,
                'complexity_metric': self.log_complexity_metric(),
                'field_performance': self.compute_mathematical_performance(),
                'cimm_components': {},
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }

    def entropy_aware_pruning(self):
        """CIMM-inspired entropy-aware pruning with Landauer's principle"""
        if len(self.complexity_history) < 10:
            return  # Need sufficient history
        
        # Compute dynamic pruning threshold based on complexity variance
        complexity_tensor = torch.tensor(self.complexity_history[-10:])
        complexity_variance = torch.var(complexity_tensor).item()
        
        # Landauer energy cost simulation (temperature scaling)
        temperature = 300 * (1 + complexity_variance)
        
        # Dynamic entropy threshold
        entropy_threshold = 0.05 * torch.exp(-torch.tensor(complexity_variance))
        
        # Apply soft pruning to weights below threshold
        with torch.no_grad():
            # Compute weight importance based on entropy contribution
            weight_importance = torch.abs(self.W) * torch.var(self.W, dim=1, keepdim=True)
            low_importance = weight_importance < entropy_threshold
            
            # Soft pruning (reduce rather than eliminate)
            self.W.data[low_importance] *= 0.8
            
        print(f"Applied entropy-aware pruning with threshold: {entropy_threshold:.4f}")

    def entropy_aware_field_optimization(self):
        """CIMM-inspired field optimization based on dynamic entropy thresholds"""
        if len(self.complexity_history) < 10:
            return  # Need sufficient history
        
        # Compute field entropy dynamics
        complexity_tensor = torch.tensor(self.complexity_history[-20:])
        field_entropy_variance = torch.var(complexity_tensor).item()
        field_entropy_mean = torch.mean(complexity_tensor).item()
        
        # Dynamic threshold based on field balance (like CIMM)
        base_variance_threshold = 0.1
        dynamic_variance_threshold = base_variance_threshold * (1 + field_entropy_mean) * \
                                   torch.sigmoid(torch.tensor(field_entropy_variance * 10)).item()
        
        # Field coherence optimization with dynamic thresholds
        if field_entropy_variance > dynamic_variance_threshold:
            # High variance indicates field instability - apply coherence correction
            with torch.no_grad():
                # Dynamic coherence factor based on field balance
                coherence_factor = 0.95 - min(0.1, field_entropy_variance * 0.5)
                weight_mean = torch.mean(self.W)
                self.W.data = self.W.data * coherence_factor + weight_mean * (1 - coherence_factor)
                
        # Dynamic field threshold for optimization
        base_field_threshold = 0.05
        field_threshold = base_field_threshold * torch.exp(-torch.tensor(field_entropy_variance * 2)).item()
        
        # Apply field-aware optimization with dynamic scaling
        with torch.no_grad():
            # Compute field importance based on entropy contribution
            field_importance = torch.abs(self.W) * torch.var(self.W, dim=1, keepdim=True)
            low_field_contribution = field_importance < field_threshold
            
            # Dynamic amplification/dampening factors based on field balance
            amplification_factor = 1.01 + min(0.02, field_entropy_mean * 0.1)
            dampening_factor = 0.99 - min(0.02, field_entropy_variance * 0.1)
            
            # Field optimization (strengthen important field components)
            self.W.data[~low_field_contribution] *= amplification_factor
            self.W.data[low_field_contribution] *= dampening_factor

class CIMMInspiredLoss:
    """CIMM-inspired field-aware loss functions for mathematical reasoning"""
    def __init__(self, lambda_qbe=0.1, lambda_entropy=0.05, lambda_coherence=0.02):
        self.lambda_qbe = lambda_qbe  # QBE loss scaling
        self.lambda_entropy = lambda_entropy  # Entropy penalty scaling
        self.lambda_coherence = lambda_coherence  # Coherence penalty scaling
        self.entropy_history = []
        self.qpl_target = 0.5  # Target quantum potential level
        
    def compute_qbe_balance(self, predictions, targets, complexity_monitor, model_weights=None):
        """
        CIMM-inspired Quantum Balance Equation (QBE) - field-aware balance computation with entropy-aware penalties and advanced metrics
        """
        # Ensure compatible shapes
        if predictions.shape != targets.shape:
            if predictions.numel() == targets.numel():
                predictions = predictions.view_as(targets)
            else:
                targets = targets.view_as(predictions)
        
        # Field-aware adaptation signal instead of MSE loss
        # This measures pattern mismatch, not distance-based error
        pattern_mismatch = torch.mean(torch.abs(predictions - targets))
        
        # CIMM's advanced error metrics
        cimm_metrics = compute_cimm_error_metrics(targets, predictions)
        
        # Combine CIMM metrics into a unified field-aware loss component
        kl_component = cimm_metrics["KL-Divergence"] * 0.1
        js_component = cimm_metrics["Jensen-Shannon"] * 0.15
        wd_component = cimm_metrics["Wasserstein Distance"] * 0.05
        qwcs_component = cimm_metrics["QWCS"] * 0.2
        
        # Field-aware metric loss
        field_metric_loss = torch.tensor(kl_component + js_component + wd_component + qwcs_component)
        
        # Entropy-based penalty (inspired by CIMM's QPL)
        if complexity_monitor and hasattr(complexity_monitor, 'complexity_metric'):
            current_entropy = complexity_monitor.complexity_metric
        else:
            current_entropy = 0.5
            
        # Track entropy for field dynamics
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > 20:
            self.entropy_history.pop(0)
            
        # Compute entropy field deviation
        target_entropy = self.qpl_target
        entropy_deviation = abs(current_entropy - target_entropy)
        
        # Field-aware entropy penalty (stronger for larger deviations)
        entropy_penalty = self.lambda_entropy * entropy_deviation * torch.exp(torch.tensor(entropy_deviation))
        
        # QBE balance combines pattern mismatch with field dynamics and CIMM metrics
        qbe_balance = pattern_mismatch + field_metric_loss + entropy_penalty
        
        return qbe_balance, cimm_metrics
    
    def compute_energy_information_balance(self, predictions, targets, model_weights):
        """
        CIMM-inspired energy-information balance for field-aware learning
        """
        # Compute prediction energy
        prediction_energy = torch.mean(predictions ** 2)
        
        # Compute weight complexity (information content)
        if model_weights is not None:
            weight_complexity = torch.var(model_weights) + 0.1 * torch.mean(torch.abs(model_weights))
        else:
            weight_complexity = torch.tensor(0.5)
            
        # Energy-information balance term
        energy_info_balance = torch.abs(prediction_energy - weight_complexity)
        
        return energy_info_balance
    
    def compute_superfluid_coherence_loss(self, predictions, model_weights=None):
        """
        CIMM-inspired superfluid coherence penalty for stability
        """
        if predictions.numel() < 3:
            return torch.tensor(0.0)
            
        # Compute prediction coherence (smoothness)
        pred_flat = predictions.flatten()
        first_grad = torch.gradient(pred_flat)[0]
        
        if len(first_grad) > 1:
            second_grad = torch.gradient(first_grad)[0]
            coherence_penalty = torch.mean(torch.abs(second_grad))
        else:
            coherence_penalty = torch.mean(torch.abs(first_grad))
            
        return self.lambda_coherence * coherence_penalty
    
    def compute_mathematical_field_loss(self, predictions, targets, complexity_monitor=None, model_weights=None):
        """
        Complete CIMM-inspired field-aware loss for mathematical reasoning with advanced metrics
        """
        # QBE (Quantum Balance Equation) with entropy awareness and CIMM metrics
        qbe_balance, cimm_metrics = self.compute_qbe_balance(predictions, targets, complexity_monitor, model_weights)
        
        # Energy-information balance
        energy_balance = self.compute_energy_information_balance(predictions, targets, model_weights)
        
        # Superfluid coherence for stability
        coherence_loss = self.compute_superfluid_coherence_loss(predictions, model_weights)
        
        # Einstein energy correction (inspired by CIMM's relativistic adjustments)
        if len(self.entropy_history) >= 2:
            entropy_variance = torch.var(torch.tensor(self.entropy_history[-10:]))
            einstein_correction = 1.0 / (1.0 + entropy_variance * 1e-5)
        else:
            einstein_correction = 1.0
            
        # Feynman damping for high-entropy states
        if self.entropy_history:
            current_entropy = self.entropy_history[-1]
            feynman_damping = torch.exp(-torch.tensor(current_entropy) * 5)
        else:
            feynman_damping = torch.tensor(1.0)
            
        # Combine all field-aware components
        total_loss = (qbe_balance + 
                     0.1 * energy_balance + 
                     coherence_loss) * einstein_correction * feynman_damping
        
        return total_loss, {
            'qbe_balance': qbe_balance.item(),
            'energy_balance': energy_balance.item(),
            'coherence_loss': coherence_loss.item(),
            'einstein_correction': einstein_correction,
            'feynman_damping': feynman_damping.item(),
            'cimm_kl_divergence': cimm_metrics["KL-Divergence"],
            'cimm_jensen_shannon': cimm_metrics["Jensen-Shannon"],
            'cimm_wasserstein': cimm_metrics["Wasserstein Distance"],
            'cimm_qwcs': cimm_metrics["QWCS"],
            'entropy_value': cimm_metrics["entropy_value"]
        }
        
def compute_cimm_error_metrics(y_true, y_pred):
    """
    CIMM-inspired quantum and entropy-aware error metrics for field-aware learning
    """
    # Convert to tensors and ensure proper shapes
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float64)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float64)
        
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Normalize while retaining sign (mean centering)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    
    # Optional scaling for distance metrics
    y_true = y_true / (torch.norm(y_true) + 1e-9)
    y_pred = y_pred / (torch.norm(y_pred) + 1e-9)
    
    # For KL and JS: use softmax to ensure valid probability distributions
    y_true_prob = torch.softmax(y_true, dim=0)
    y_pred_prob = torch.softmax(y_pred, dim=0)
    
    epsilon = 1e-9
    
    # KL Divergence
    kl_div = torch.sum(y_true_prob * torch.log((y_true_prob + epsilon) / (y_pred_prob + epsilon)))
    
    # Jensen-Shannon Divergence
    js_div = torch.sqrt(0.5 * (kl_div + torch.sum(y_pred_prob * torch.log((y_pred_prob + epsilon) / (y_true_prob + epsilon)))))
    js_div = torch.nan_to_num(js_div, nan=0.0)
    
    # Wasserstein Distance (Earth Mover's Distance approximation)
    wd = torch.sum(torch.abs(torch.cumsum(y_true, dim=0) - torch.cumsum(y_pred, dim=0)))
    
    # Add noise for stability in correlation computation
    noise = torch.normal(0, 1e-4, size=y_pred.shape)
    y_pred_noisy = y_pred + noise
    
    # Quantum Wave Coherence Score (QWCS)
    if torch.var(y_true) == 0 or torch.var(y_pred_noisy) == 0:
        qwcs = torch.tensor(0.5)
    else:
        correlation_matrix = torch.corrcoef(torch.stack([y_true, y_pred_noisy]))
        qwcs = 1 - torch.abs(correlation_matrix[0, 1])
    
    qwcs = torch.nan_to_num(qwcs, nan=0.5)
    
    # Entropy scaling for quantum awareness
    entropy_value = torch.sum(-y_true_prob * torch.log(y_true_prob + epsilon))
    qwcs = qwcs * (1 + 0.02 * entropy_value)
    
    # Entropy-aware scaling for KL divergence
    entropy_scaling = torch.clamp(1.0 + 0.1 * entropy_value, min=0.8, max=1.2)
    kl_div = kl_div * entropy_scaling
    
    return {
        "KL-Divergence": kl_div.item(),
        "Jensen-Shannon": js_div.item(), 
        "Wasserstein Distance": wd.item(),
        "QWCS": qwcs.item(),
        "entropy_value": entropy_value.item()
    }
    
# For backward compatibility, alias TinyCIMM to TinyCIMMEuler
TinyCIMM = TinyCIMMEuler

# End of TinyCIMM-Euler module
