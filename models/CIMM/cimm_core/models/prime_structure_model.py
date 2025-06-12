from cimm_core.models.base_cimm_model import BaseCIMMModel
from cimm_core.utils import get_device  # Import device utility
import torch.nn as nn
import torch

device = get_device()  # Get the device


class PrimeStructureModel(BaseCIMMModel):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(PrimeStructureModel, self).__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.hidden = nn.Linear(input_size, hidden_size).to(device)
        self.activation = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_size).to(device)
        self.device = device

    def forward(self, x):
        # Convert to tensor if needed
        #import torch

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # Add batch dimension if missing
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Move all model parameters and input to the same device as self.hidden's weight
        target_device = self.hidden.weight.device
        if x.device != target_device:
            x = x.to(target_device)
        x = self.hidden(x)
        x = self.activation(x)
        return self.output_layer(x)
