import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- TinyCIMM-Planck-specific utilities and controllers ---
class EntropyMonitorLite:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.entropy = 0.0
        self.past_entropies = []

    def update(self, signal):
        prob_dist = torch.softmax(signal, dim=-1)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-9)).item()
        self.entropy = self.momentum * self.entropy + (1 - self.momentum) * entropy
        self.past_entropies.append(self.entropy)
        return self.entropy

    def get_variance(self):
        if len(self.past_entropies) < 2:
            return 0.0
        return torch.var(torch.tensor(self.past_entropies)).item()

def quantum_collapse(pred, target, entropy):
    delta = torch.abs(target - pred)
    correction = 0.05 * (target - pred) / math.sqrt(entropy + 1e-8)
    return pred + correction

def compute_coherence(signal):
    signal = torch.tensor(signal)
    first_grad = torch.gradient(signal)[0]
    curvature = torch.gradient(first_grad)[0]
    return torch.exp(-torch.mean(torch.abs(curvature)))

class FractalQBEController:
    def __init__(self, entropy_band=0.005, min_hidden=4, max_hidden=64, feedback_window=2):
        self.entropy_band = entropy_band
        self.min_hidden = min_hidden
        self.max_hidden = max_hidden
        self.feedback_window = feedback_window
        self.entropy_hist = []
        self.feedback_hist = []
        self.loss_hist = []

    def decide(self, entropy, feedback, loss, hidden_dim):
        self.entropy_hist.append(entropy)
        self.feedback_hist.append(feedback)
        self.loss_hist.append(loss)
        if len(self.entropy_hist) < self.feedback_window:
            return "none", 0
        ent_arr = torch.tensor(self.entropy_hist[-self.feedback_window:])
        feed_arr = torch.tensor(self.feedback_hist[-self.feedback_window:])
        loss_arr = torch.tensor(self.loss_hist[-self.feedback_window:])
        ent_mu, ent_std = ent_arr.mean().item(), ent_arr.std().item()
        feed_mu, feed_var = feed_arr.mean().item(), feed_arr.var().item()
        loss_mu, loss_var = loss_arr.mean().item(), loss_arr.var().item()
        lower = ent_mu - self.entropy_band
        upper = ent_mu + self.entropy_band
        grow_amt = prune_amt = 0
        action = "none"
        if feed_var > 0.001 and hidden_dim < self.max_hidden:
            grow_amt = max(1, int(hidden_dim * 0.2))
            action = "grow"
        elif ent_std > 0.005 and hidden_dim > self.min_hidden:
            prune_amt = max(1, int(hidden_dim * 0.2))
            action = "prune"
        return action, max(grow_amt, prune_amt)

# --- Modular TinyCIMM-Planck ---
class TinyCIMMPlanck(nn.Module):
    """
    TinyCIMM-Planck: The minimal, foundational version of the Cognition Index Memory Machine (CIMM).
    Implements entropy-regulated, dynamically adaptive recurrent structure for symbolic collapse benchmarking.
    """
    def __init__(self, input_size, hidden_size, output_size, device, **kwargs):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_size
        self.W = nn.Parameter(0.05 * torch.randn(hidden_size, input_size, device=device))
        self.b = nn.Parameter(torch.zeros(hidden_size, device=device))
        self.V = nn.Parameter(0.05 * torch.randn(output_size, hidden_size, device=device))
        self.c = nn.Parameter(torch.zeros(output_size, device=device))
        # Entropy/structure state
        self.entropy_monitor = None
        self.symbolic_hold = 0
        self.symbolic_hold_steps = kwargs.get('symbolic_hold_steps', 10)
        self.suppress_updates = False
        self.suppress_updates_counter = 0
        self.suppress_updates_steps = 10
        self.ema_output = None
        self.ema_alpha = 0.9
        self.prev_y = None
        self.entropy_threshold = 0.0
        self.phase_hold = 0
        # Micro-memory for local attractors
        self.micro_memory = []
        self.micro_memory_size = kwargs.get('micro_memory_size', 5)
        # Frequency estimator
        self.freq_estimator = nn.Sequential(
            nn.Linear(input_size, 8), nn.ReLU(), nn.Linear(8, 1)
        ).to(device)
        self.freq_lr = 0.01
        # Optimizer
        self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c], lr=0.03, weight_decay=0.0001)
        # State for adaptation
        self.last_h = None
        self.last_x = None
        self.last_prediction = None
        self.last_raw_prediction = None
        self.feedback_history = []
        self.feedback_entropy_window = 16
        self.feedback_buffer = []
        self.last_entropy = []
        self.symbolic_hold = 0
        self.entropy_threshold = 0.0
        self.phase_hold = 0
        self.micro_memory = []
        self.micro_memory_size = 5
        self.freq_lr = 0.01
        # For structure adaptation
        self.prev_loss = None
        self.prev_entropy = None
        self.current_step = 0
        self.loss_ma = None
        self.qbe_pressure_ma = 1.0
        self.last_prune_thresh = 0.03
        self.growth_hysteresis = 0
        self.prune_hysteresis = 0
        self.adapt_cooldown = 0
        self.min_loss_ever = None
        self.unstable_steps = 0
        self.last_growth_step = 0
        self.last_adapt_step = 0
        self.recent_loss_window = []

    def set_entropy_monitor(self, monitor):
        self.entropy_monitor = monitor

    def forward(self, x, y_true=None):
        h = torch.relu(x @ self.W.T + self.b)
        self.micro_memory.append(h.detach().cpu())
        if len(self.micro_memory) > self.micro_memory_size:
            self.micro_memory.pop(0)
        y = h @ self.V.T + self.c
        self.last_h = h
        self.last_x = x
        self.last_raw_prediction = y.detach()
        self.last_prediction = y.detach()
        # DEBUG: Print a few weights and predictions
        if self.current_step % 50 == 0:
            print(f"[TinyCIMM-Planck DEBUG] Step {getattr(self, 'current_step', 0)}")
            print(f"  W mean: {self.W.data.mean().item():.4f}, std: {self.W.data.std().item():.4f}")
            print(f"  b mean: {self.b.data.mean().item():.4f}")
            print(f"  V mean: {self.V.data.mean().item():.4f}, std: {self.V.data.std().item():.4f}")
            print(f"  c mean: {self.c.data.mean().item():.4f}")
            print(f"  Sample prediction: {y[0].detach().cpu().numpy() if hasattr(y, 'detach') else y}")
        return y

    def log_entropy(self):
        if self.W.shape[0] > 1 and torch.isfinite(self.W).all():
            ent = torch.std(self.W, dim=0).mean().item()
            if torch.isnan(torch.tensor(ent)) or torch.isinf(torch.tensor(ent)):
                ent = 1e-6
        else:
            ent = 1e-6
        return ent

    def should_hold_structure(self):
        entropy = self.entropy_monitor.entropy if self.entropy_monitor else self.log_entropy()
        threshold = self.entropy_threshold if self.entropy_threshold > 0 else 0.03
        return (entropy < threshold) or (self.symbolic_hold > 0) or (self.phase_hold > 0)

    def grow_and_prune(self, avg_feedback, entropy, loss, controller, qbe_smoothed=None, allow_prune=True, entropy_monitor=None):
        # Ensure previous loss/entropy are initialized
        if self.prev_loss is None:
            self.prev_loss = loss
        if self.prev_entropy is None:
            self.prev_entropy = entropy
        min_hidden = 8
        max_hidden = 32
        max_growth_per_step = max(1, int(self.hidden_dim * 0.25))
        growth_cooldown = 2
        prune_margin = 2
        hysteresis_steps = 4
        min_interval = 5
        cooldown_after_adapt = 15
        if not hasattr(self, 'last_growth_step'):
            self.last_growth_step = -growth_cooldown
        if not hasattr(self, 'last_prune_thresh'):
            self.last_prune_thresh = 0.03
        if not hasattr(self, 'growth_hysteresis'):
            self.growth_hysteresis = 0
        if not hasattr(self, 'prune_hysteresis'):
            self.prune_hysteresis = 0
        if not hasattr(self, 'adapt_cooldown'):
            self.adapt_cooldown = 0
        if not hasattr(self, 'min_loss_ever'):
            self.min_loss_ever = loss
        self.min_loss_ever = min(self.min_loss_ever, loss) if self.min_loss_ever is not None else loss
        stability_margin = 0.02
        in_stability_band = (loss <= self.min_loss_ever + stability_margin)
        if not hasattr(self, 'unstable_steps'):
            self.unstable_steps = 0
        if in_stability_band:
            self.unstable_steps = 0
        else:
            self.unstable_steps += 1
        required_unstable_steps = 10
        k_B = 1.380649e-23
        base_temperature = 300
        neuron_entropies = torch.std(self.W.data, dim=1).detach().cpu()
        entropy_var = float(torch.var(neuron_entropies))
        if not torch.isfinite(torch.tensor(entropy_var)) or torch.isnan(torch.tensor(entropy_var)):
            entropy_var = 0.0
        entropy_var = float(max(min(entropy_var, 10.0), 0.0))
        loss_trend = getattr(self, 'prev_loss', loss)
        entropy_trend = getattr(self, 'prev_entropy', entropy)
        loss_delta = loss - loss_trend
        entropy_delta = entropy - entropy_trend
        qbe_pressure = 1.0 + 0.5 * max(0, loss_delta) + 0.5 * max(0, entropy_delta)
        qbe_pressure = float(max(min(qbe_pressure, 1.2), 0.8))
        self.qbe_pressure_ma = qbe_pressure
        base_entropy_thresh = 0.03
        if self.last_h is not None:
            activation_var = float(torch.var(self.last_h.detach().cpu()))
        else:
            activation_var = 0.0
        if not torch.isfinite(torch.tensor(activation_var)) or torch.isnan(torch.tensor(activation_var)):
            activation_var = 0.0
        activation_var = float(max(min(activation_var, 10.0), 0.0))
        dynamic_thresh = base_entropy_thresh * (1 + 0.15 * entropy_var + 0.15 * activation_var) * self.qbe_pressure_ma
        dynamic_thresh = 0.7 * getattr(self, 'last_prune_thresh', 0.03) + 0.3 * dynamic_thresh
        if not torch.isfinite(torch.tensor(dynamic_thresh)) or torch.isnan(torch.tensor(dynamic_thresh)):
            dynamic_thresh = base_entropy_thresh
        self.last_prune_thresh = dynamic_thresh
        # Update loss_ma safely
        if self.loss_ma is None:
            self.loss_ma = loss
        else:
            self.loss_ma = 0.95 * self.loss_ma + 0.05 * loss
        loss_stable = 0.05
        loss_aggressive = 0.2
        if self.loss_ma < loss_stable:
            hysteresis_steps = 2
            min_interval = 3
            max_growth_per_step = 1
        elif self.loss_ma > loss_aggressive:
            hysteresis_steps = 2
            min_interval = 2
            max_growth_per_step = max(1, int(self.hidden_dim * 0.5))
        else:
            frac = (self.loss_ma - loss_stable) / (loss_aggressive - loss_stable)
            hysteresis_steps = int(8 - 6 * frac)
            min_interval = int(10 - 8 * frac)
            max_growth_per_step = max(1, int(self.hidden_dim * (0.1 + 0.4 * frac)))
        neuron_entropies = torch.std(self.W.data, dim=1).detach().cpu().numpy()
        entropy_mean = neuron_entropies.mean().item()
        entropy_std = neuron_entropies.std().item()
        if not torch.isfinite(torch.tensor(entropy_mean)) or torch.isnan(torch.tensor(entropy_mean)):
            entropy_mean = base_entropy_thresh
        if not torch.isfinite(torch.tensor(entropy_std)) or torch.isnan(torch.tensor(entropy_std)):
            entropy_std = 0.0
        adaptive_entropy_thresh = entropy_mean + 0.5 * entropy_std
        qbe_pressure_val = qbe_smoothed if qbe_smoothed is not None else self.qbe_pressure_ma
        if not torch.isfinite(torch.tensor(qbe_pressure_val)) or torch.isnan(torch.tensor(qbe_pressure_val)):
            qbe_pressure_val = 1.0
        dynamic_thresh = adaptive_entropy_thresh * qbe_pressure_val
        if not torch.isfinite(torch.tensor(dynamic_thresh)) or torch.isnan(torch.tensor(dynamic_thresh)):
            dynamic_thresh = base_entropy_thresh
        self.last_prune_thresh = dynamic_thresh
        # Only update loss_ma once per call
        if self.loss_ma is None:
            self.loss_ma = loss
        else:
            self.loss_ma = 0.95 * self.loss_ma + 0.05 * loss
        loss_stable = 0.05
        loss_aggressive = 0.2
        if self.loss_ma < loss_stable:
            hysteresis_steps = 2
            min_interval = 3
            max_growth_per_step = 1
        elif self.loss_ma > loss_aggressive:
            hysteresis_steps = 2
            min_interval = 2
            max_growth_per_step = max(1, int(self.hidden_dim * 0.5))
        else:
            frac = (self.loss_ma - loss_stable) / (loss_aggressive - loss_stable)
            hysteresis_steps = int(8 - 6 * frac)
            min_interval = int(10 - 8 * frac)
            max_growth_per_step = max(1, int(self.hidden_dim * (0.1 + 0.4 * frac)))
        prune_mask = (
            (torch.std(self.W.data, dim=1).detach().cpu() > dynamic_thresh) |
            (k_B * base_temperature * torch.std(self.W.data, dim=1).detach().cpu() > k_B * base_temperature * dynamic_thresh)
        )
        action, amt = controller.decide(entropy, avg_feedback, loss, self.hidden_dim)
        grew = pruned = False
        current_step = getattr(self, 'current_step', 0)
        last_adapt_step = getattr(self, 'last_adapt_step', -min_interval)
        if not hasattr(self, 'recent_loss_window'):
            self.recent_loss_window = []
        loss_window_size = 8
        self.recent_loss_window.append(loss)
        if len(self.recent_loss_window) > loss_window_size:
            self.recent_loss_window.pop(0)
        sustained_high_loss = all(l > 0.08 for l in self.recent_loss_window)
        can_adapt = (current_step - last_adapt_step) >= min_interval and sustained_high_loss
        if self.adapt_cooldown > 0 or not can_adapt or in_stability_band or self.unstable_steps < required_unstable_steps:
            self.adapt_cooldown = max(self.adapt_cooldown - 1, 0)
            action = "none"
        allow_growth = (current_step - self.last_growth_step) >= 2 or loss_delta > 0.01 or entropy < 0.1
        if action == "grow" and allow_growth and self.hidden_dim < max_hidden:
            self.growth_hysteresis += 1
            if self.growth_hysteresis >= hysteresis_steps or loss > 0.2:
                amt = min(max(amt, 2), max_growth_per_step, max_hidden - self.hidden_dim)
                for _ in range(amt):
                    scale = 0.05 * (1 if torch.rand(1).item() > 0.5 else -1) * (1.0 + torch.rand(1).item())
                    new_w = scale * torch.randn(1, self.W.shape[1], device=self.device)
                    new_b = torch.zeros(1, device=self.device)
                    new_v = scale * torch.randn(self.V.shape[0], 1, device=self.device)
                    self.W = torch.nn.Parameter(torch.cat([self.W.data, new_w], dim=0))
                    self.b = torch.nn.Parameter(torch.cat([self.b.data, new_b], dim=0))
                    self.V = torch.nn.Parameter(torch.cat([self.V.data, new_v], dim=1))
                    self.hidden_dim += 1
                    grew = True
                with torch.no_grad():
                    self.W += 0.01 * torch.randn_like(self.W)
                    self.V += 0.01 * torch.randn_like(self.V)
                self.last_growth_step = current_step
                self.growth_hysteresis = 0
                self.adapt_cooldown = 100
                self.last_adapt_step = current_step
        else:
            self.growth_hysteresis = 0
        internal_entropy = self.log_entropy()
        external_entropy = entropy_monitor.entropy if hasattr(entropy_monitor, 'entropy') else internal_entropy
        entropy_gate = (internal_entropy > dynamic_thresh) and (external_entropy > dynamic_thresh)
        can_prune = allow_prune and entropy_gate and (self.symbolic_hold == 0) and (self.phase_hold == 0)
        if can_prune and (action == "prune" or self.qbe_pressure_ma > 1.05) and self.hidden_dim > (min_hidden + prune_margin):
            self.prune_hysteresis += 1
            if self.prune_hysteresis >= hysteresis_steps:
                keep_idxs = torch.where(prune_mask)[0]
                if len(keep_idxs) < min_hidden:
                    keep_idxs = torch.arange(min_hidden)
                self.W = torch.nn.Parameter(self.W.data[keep_idxs])
                self.b = torch.nn.Parameter(self.b.data[keep_idxs])
                self.V = torch.nn.Parameter(self.V.data[:, keep_idxs])
                self.hidden_dim = self.W.shape[0]
                pruned = True
                with torch.no_grad():
                    self.W += 0.01 * torch.randn_like(self.W)
                    self.V += 0.01 * torch.randn_like(self.V)
                self.prune_hysteresis = 0
                self.adapt_cooldown = 100
                self.last_adapt_step = current_step
        else:
            self.prune_hysteresis = 0
        if grew or pruned:
            self.optimizer = torch.optim.Adam([self.W, self.b, self.V, self.c], lr=0.03, weight_decay=0.0001)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.03
        self.prev_loss = loss
        self.prev_entropy = entropy
        self.current_step = getattr(self, 'current_step', 0) + 1
        if self.last_h is not None and self.last_x is not None:
            h_shape = self.last_h.shape[1] if self.last_h.dim() > 1 else self.last_h.shape[0]
            w_shape = self.W.shape[0]
            if h_shape > w_shape:
                last_h = self.last_h[:, :w_shape]
            elif h_shape < w_shape:
                zeros = torch.zeros(self.last_h.shape[0], w_shape - h_shape, device=self.device)
                last_h = torch.cat([self.last_h, zeros], dim=1)
            else:
                last_h = self.last_h
            target = last_h.detach()
            error = target - last_h
            x_shape = self.last_x.shape[1] if self.last_x.dim() > 1 else self.last_x.shape[0]
            wx_shape = self.W.shape[1]
            if x_shape > wx_shape:
                last_x = self.last_x[:, :wx_shape]
            elif x_shape < wx_shape:
                zeros = torch.zeros(self.last_x.shape[0], wx_shape - x_shape, device=self.device)
                last_x = torch.cat([self.last_x, zeros], dim=1)
            else:
                last_x = self.last_x
            lr = max(0.005, 0.01 * (error.var().item() + 1e-5))
            if error.shape[1] == self.W.shape[0] and last_x.shape[1] == self.W.shape[1]:
                with torch.no_grad():
                    self.W.copy_(self.W + lr * error.T @ last_x)
        hebb = self.last_h.T @ self.last_h if self.last_h is not None else None
        if hebb is not None and hebb.shape == self.W.shape:
            with torch.no_grad():
                self.W.copy_(self.W + 0.001 * (hebb - self.W))

    def estimate_frequency(self, x):
        return self.freq_estimator(x)

    def analyze_results(self):
        """
        Analyze experiment results using the metrics defined in the explainability plan.
        """
        if len(self.micro_memory) < 2:
            print("Insufficient micro_memory for analysis.")
            return {
                "activation_ancestry": None,
                "entropy_alignment": None,
                "phase_alignment": None,
                "bifractal_consistency": None,
                "attractor_density": None
            }

        # Activation Ancestry Trace
        activation_ancestry = torch.mean(torch.stack([
            torch.cosine_similarity(mem, self.micro_memory[-1]) for mem in self.micro_memory[:-1]
        ]))

        # Entropy Gradient Alignment Score
        if len(self.entropy_monitor.past_entropies) > 1:
            entropy_gradients = torch.cat([
                g.unsqueeze(0) for g in torch.gradient(torch.tensor(self.entropy_monitor.past_entropies))
            ], dim=0)
            alignment_score = torch.mean(torch.abs(entropy_gradients))
        else:
            alignment_score = torch.tensor(0.0)

        # Collapse Phase Alignment
        if self.last_h is not None and self.last_x is not None:
            phase_alignment = torch.mean(torch.abs(self.last_h - self.last_x))
        else:
            phase_alignment = torch.tensor(0.0)

        # Bifractal Activation Consistency
        bifractal_consistency = torch.mean(torch.tensor([
            torch.sum(mem) for mem in self.micro_memory
        ]))

        # Semantic Attractor Density
        attractor_density = torch.mean(torch.tensor([
            torch.norm(mem) for mem in self.micro_memory
        ]))

        # Log the metrics
        print("Activation Ancestry Trace:", activation_ancestry.item())
        print("Entropy Gradient Alignment Score:", alignment_score.item())
        print("Collapse Phase Alignment:", phase_alignment.item())
        print("Bifractal Activation Consistency:", bifractal_consistency.item())
        print("Semantic Attractor Density:", attractor_density.item())

        return {
            "activation_ancestry": activation_ancestry.item(),
            "entropy_alignment": alignment_score.item(),
            "phase_alignment": phase_alignment.item(),
            "bifractal_consistency": bifractal_consistency.item(),
            "attractor_density": attractor_density.item()
        }

# For backward compatibility, alias TinyCIMM to TinyCIMMPlanck
TinyCIMM = TinyCIMMPlanck

# End of TinyCIMM-Planck module
