from base_agent import BaseAgent
from skopt.space import Real, Integer
import torch

class MomentumAgent(BaseAgent):
    def __init__(self, manager):
        param_space = [
            Real(1e-5, 1e-2, name='learning_rate'),
            Integer(10, 100, name='hidden_size'),
        ]

        class MomentumModel(torch.nn.Module):
            def __init__(self, input_size, hidden_size=64, output_size=1):
                super().__init__()
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(hidden_size, output_size)
                )

            def forward(self, x):
                return self.model(x)

        anchor_data = torch.randn(50, 4)  # must match input_size below
        super().__init__("MomentumAgent", MomentumModel, param_space, manager, anchor_data, hidden_size=64, role="predictor")

    def apply_feedback(self, feedback_data):
        # Example: use feedback to adjust internal entropy or trust weights
        self.agent_instance.apply_qbe_feedback(feedback_data)
    def get_entropy_state(self):
        return self.model.entropy.item()

    def set_entropy_state(self, value):
        self.model.entropy = torch.tensor(value)
