from cimm_core.models.base_cimm_model import BaseCIMMModel
from cimm_core.utils import get_device  # Import device utility

device = get_device()  # Get the device

class CryptographyModel(BaseCIMMModel):
    def __init__(self, hidden_size):
        super(CryptographyModel, self).__init__(input_size=10, hidden_size=hidden_size, output_size=1)
        # self.device is inherited from BaseCIMMModel

    def forward(self, x):
        return self.common_forward(x)
