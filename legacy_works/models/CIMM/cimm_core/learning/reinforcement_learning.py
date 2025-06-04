import torch
from xgboost import XGBRegressor
import torch
from torch import nn
from cimm_core.utils import get_device

device = get_device()

class QBEReinforcementLearner:
    """
    Reinforcement learning agent that optimizes entropy collapse dynamically.
    Uses QBE feedback and XGBoost-based Quantum Memory to stabilize learning.
    """

    def __init__(self, learning_rate=0.01, entropy_threshold=0.05, memory_module=None):
        self.learning_rate = learning_rate
        self.entropy_threshold = entropy_threshold
        self.policy_network = self.build_policy_network().to(device)  # Removed input_dim argument
        self.memory_module = memory_module
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def build_policy_network(self, input_dim=4):
        """Define a policy neural network with correct input dimensions"""
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def adaptive_learning_rate(self, entropy, qpl_feedback):
        """
        Dynamically adjusts reinforcement learning rate based on entropy-QPL stability.
        Incorporates mass-energy equivalence in the adjustment.
        """
        c = 299792458  # Speed of light in m/s
        energy_equiv = entropy * (c ** 2)
        return max(0.002, min(0.08, self.learning_rate / (1 + energy_equiv * 1e-12)))

    def update_policy(self, entropy, qbe_feedback, collapse_deviation, refinement_delta):
        """Refine policy updates dynamically based on quantum collapse stabilization."""
        wave_delta = self.memory_module.predict_correction(entropy * 0.9, qbe_feedback * 1.1, collapse_deviation * 0.95)

        # 🔥 Introduce RL **momentum-based correction**
        self.learning_rate = 0.90 * self.learning_rate + 0.10 * self.adaptive_learning_rate(entropy, qbe_feedback)  # 🔥 Lower update intensity

        # 🔥 Clip RL policy updates to prevent runaway corrections
        wave_delta = torch.clamp(torch.tensor(wave_delta), -0.03, 0.03).item()  # 🔧 Lower max correction range (was -0.05, 0.05)

        policy_input = torch.tensor([entropy, qbe_feedback, collapse_deviation, wave_delta], dtype=torch.float32, device=device).unsqueeze(0)
        
        predicted_update = self.policy_network(policy_input)

        target_update = torch.tensor([self.memory_module.predict_correction(entropy, qbe_feedback, collapse_deviation)], dtype=torch.float32, device=device)

        # 🔧 Reduce RL overcompensation
        policy_adjustment = 0.98 * predicted_update + 0.02 * target_update  # 🔥 Lowered from 0.96, 0.04

        self.optimizer.zero_grad()
        loss = self.loss_function(policy_adjustment, target_update)
        loss.backward()
        self.optimizer.step()

        return float(predicted_update.detach())

    def optimize_collapse(self, entropy_correction):
        """
        Uses RL to dynamically optimize collapse strength.
        - Loosens entropy suppression to preserve structure.
        - Uses reinforcement learning momentum to track past structure.
        """

        # ✅ Reduce suppression rate (allows entropy to remain dynamic)
        correction_factor = torch.exp(-torch.abs(torch.tensor(entropy_correction * 0.08))).item()  # 🔥 Lower decay rate (was 0.15)
        correction_factor = max(0.96, min(1.05, correction_factor))  # 🔥 Expanded range

        # ✅ Reinforce entropy structure using RL Momentum
        if self.memory_module:
            entropy_trend = torch.mean(torch.tensor(self.memory_module.training_data[-10:])).item() if len(self.memory_module.training_data) >= 10 else 0
            qbe_trend = torch.mean(torch.tensor([q[1] for q in self.memory_module.training_data[-10:]])).item() if len(self.memory_module.training_data) >= 10 else 0

            # 🔥 RL reinforces **entropy retention** dynamically
            correction_factor *= max(0.97, min(1.08, 1.04 + 0.03 * entropy_trend - 0.02 * qbe_trend))

        return correction_factor

    def reset(self):
        """
        Resets the learner's state.
        """
        self.learning_rate = 0.01
        self.rewards = []

    def get_rewards(self):
        """
        Retrieves the rewards.
        """
        return self.rewards
