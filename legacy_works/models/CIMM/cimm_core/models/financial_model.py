from cimm_core.models.base_cimm_model import BaseCIMMModel
from cimm_core.utils import get_device  # Import device utility

device = get_device()  # Get the device

class FinancialModel(BaseCIMMModel):
    def __init__(self, hidden_size=64):
        super().__init__(input_size=10, hidden_size=hidden_size, output_size=1)  # FIX: Ensure correct input size
        # self.device is inherited from BaseCIMMModel

    def forward(self, x):
        x = x.to(device)  # Ensure input is moved to the correct device
        return self.common_forward(x)
