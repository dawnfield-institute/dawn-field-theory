import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cimm_core.models.prime_structure_model import PrimeStructureModel
from visualization.plots import plot_live_predictions
import numpy as np
import torch
import sympy
from skopt.space import Real, Integer
from cimm_core.cimm import CIMM

class PrimeStructureUseCase:
    def __init__(self, hidden_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PrimeStructureModel(input_size=4, hidden_size=hidden_size).to(self.device)  # Update input_size to match features
        self.num_of_primes = 200000  # Adjust as needed for your prime range
    def execute(self, x):
        """
        Runs the Prime Structure Model and applies entropy-based corrections to improve precision.
        """
        prediction = self.model.forward(x)

        # Apply localized entropy corrections
        # entropy_correction = np.exp(-0.05 * self.model.entropy_monitor.entropy)
        # refined_prediction = prediction * (1 + entropy_correction)  

        # return refined_prediction
        return prediction

    def live_prime_training(self):
        """
        Runs real-time prime structure learning with adaptive Bayesian optimization search.
        """
        prime_data = self.preprocess_prime_data().to(self.device)
        actual_values = self.generate_actual_prime_structure().to(self.device)

        # Cache mean and std for later inverse transform
        actual_mean = actual_values.mean()
        actual_std = actual_values.std()

        # Normalize targets to zero mean and unit variance for Tanh output
        actual_values = (actual_values - actual_mean) / (actual_std + 1e-6)

        # Optional: Normalize inputs for consistent scale
        prime_data = (prime_data - prime_data.mean()) / (prime_data.std() + 1e-6)

        anchor_size = int(0.1 * len(prime_data))
        anchor_data = prime_data[:anchor_size]
        streaming_data = prime_data[anchor_size:]
        streaming_actuals = actual_values[anchor_size:]

        # Ensure streaming_data and streaming_actuals have the same length
        min_len = min(len(streaming_data), len(streaming_actuals))
        streaming_data = streaming_data[:min_len]
        streaming_actuals = streaming_actuals[:min_len]

        hidden_size = 64

        # Dynamically adjust Bayesian search space based on past convergence speed
        diffs = streaming_actuals[:100][1:] - streaming_actuals[:100][:-1]
        convergence_speed = diffs.mean().item()
        lr_min, lr_max = (1e-5, 1e-2) if convergence_speed > 0.1 else (1e-4, 1e-1)

        param_space = [
            Real(lr_min, lr_max, name='learning_rate'),
            Integer(10, 100, name='hidden_size'),
        ]
        
        model_class = PrimeStructureModel
        model_args = (4, hidden_size)  # input_size=4, hidden_size=hidden_size

        cimm = CIMM(model_class, model_args, param_space, anchor_data)

        predictions = []
        actuals = []

        # Torch optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        for i, new_data_point in enumerate(streaming_data):
            input_data = new_data_point.unsqueeze(0) if new_data_point.ndim == 1 else new_data_point
            # Ensure input_data is on the same device as the model
            input_data = input_data.to(self.model.device)
            actual_value = streaming_actuals[i]
            if isinstance(actual_value, torch.Tensor):
                target = actual_value.unsqueeze(0) if actual_value.ndim == 0 else actual_value
            else:
                target = torch.tensor([actual_value], dtype=torch.float32)
            target = target.to(self.device)

            optimizer.zero_grad()
            prediction = self.model(input_data)
            loss = loss_fn(prediction, target)
            # Scale-sensitive loss adjustment
            scale_boost = torch.log1p(target.abs())  # Emphasize larger targets
            weighted_loss = loss * scale_boost
            weighted_loss.backward()
            optimizer.step()

            predictions.append(prediction.detach().cpu().item())
            actuals.append(target.detach().cpu().item())

            # Example metrics (per step, optional)
            # metrics = compute_error_metrics([target.detach().cpu()], [prediction.detach().cpu()])
            # print("Metrics:", metrics)

        assert len(predictions) == len(streaming_data), "Prediction loop did not cover all input data points"

        predictions = torch.tensor(predictions, dtype=torch.float32)
        actuals = torch.tensor(actuals, dtype=torch.float32)

        # Ensure actual_std and actual_mean are on the same device as predictions
        actual_std = actual_std.to(predictions.device)
        actual_mean = actual_mean.to(predictions.device)

        # Reverse normalization for plotting
        predictions = predictions * actual_std + actual_mean
        actuals = actuals * actual_std + actual_mean

        plot_live_predictions(predictions, actuals)

        validation_data = torch.tensor(streaming_data, dtype=torch.float32).to(self.device)
        if validation_data.ndim == 1:
            validation_data = validation_data.unsqueeze(1)
        metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
        print(f"Error Metrics: {metrics}")

    def preprocess_prime_data(self, raw_data=None):
        # Torch-only version, no numpy or pywt
        primes = torch.tensor(list(sympy.primerange(1, self.num_of_primes)), dtype=torch.float32).to(self.device)
        prime_gaps = primes[1:] - primes[:-1]
        prime_gaps = prime_gaps[:-1]  # Align with target

        # Normalize
        mean = prime_gaps.mean()
        std = prime_gaps.std()
        norm = (prime_gaps - mean) / std

        # Log-transformed
        log_gap = torch.sign(norm) * torch.log1p(norm.abs())

        # FFT (real part only)
        fft_real = torch.fft.fft(norm).real

        # Rolling window mean (manual wavelet-style)
        window_size = 32
        if norm.shape[0] >= window_size:
            rolled = norm.unfold(0, window_size, 1).mean(dim=-1)
        else:
            rolled = norm  # fallback if not enough data

        # Ensure same shape
        min_len = min(norm.shape[0], log_gap.shape[0], fft_real.shape[0], rolled.shape[0]) - 2

        features = torch.stack([
            norm[:min_len],
            log_gap[:min_len],
            fft_real[:min_len],
            rolled[:min_len]
        ], dim=1)

        return features.to(self.device)

    def generate_actual_prime_structure(self):
        primes = torch.tensor(list(sympy.primerange(1, 100000)), dtype=torch.float32).to(self.device)
        prime_gaps = primes[1:] - primes[:-1]
        delta_gaps = prime_gaps[1:] - prime_gaps[:-1]  # signed difference between consecutive gaps
        return delta_gaps

# Example usage
if __name__ == "__main__":
    use_case = PrimeStructureUseCase(hidden_size=64)
    use_case.live_prime_training()
