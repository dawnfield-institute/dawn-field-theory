import torch

def convert_to_tensor(data):
    """
    Converts input data to a PyTorch tensor.
    Handles lists, PyTorch tensors, and other iterable data.
    """
    if isinstance(data, torch.Tensor):
        return data.to(get_device())  # Ensure tensor is moved to the correct device
    elif isinstance(data, list):
        # Ensure each element is a scalar
        if all(isinstance(i, (int, float)) for i in data):
            tensor_data = torch.tensor(data, dtype=torch.float32)
        else:
            tensor_data = torch.stack([torch.tensor(i, dtype=torch.float32).flatten() for i in data])
        return tensor_data.to(get_device())  # Move to device
    else:
        raise TypeError("Unsupported data type for conversion to tensor.")
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")