import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)  # Example layer

    def forward(self, x):
        return self.fc1(x)

    def apply_qbe_feedback(self, feedback_data):
        # Simple version: modify weights using feedback signal
        with torch.no_grad():
            adjustment = torch.tensor(feedback_data, dtype=torch.float32)
            self.fc1.bias += 0.01 * adjustment  # Adjust bias as an example