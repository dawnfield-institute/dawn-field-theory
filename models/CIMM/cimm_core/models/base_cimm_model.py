import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BaseCIMMModel(nn.Module, ABC):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseCIMMModel, self).__init__()
        self.input_size = input_size  # FIX: Store input size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ensure correct input size
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        self.device = get_device()  # Store the device
        self.to(self.device)  # Move model to device

    def common_forward(self, x):
        x = x.to(next(self.parameters()).device)  # Ensure input is on the same device as the model
        if len(x.shape) == 1:  # Fix for single-dimension input
            x = x.unsqueeze(0)

        x = torch.relu(self.linear1(x))
        return self.linear2(x)

    @abstractmethod
    def forward(self, x):
        pass