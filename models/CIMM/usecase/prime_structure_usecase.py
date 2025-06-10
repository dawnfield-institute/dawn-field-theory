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
import pywt  # Import pywt for wavelet transforms

class PrimeStructureUseCase:
    def __init__(self, hidden_size):
        self.model = PrimeStructureModel(input_size=4, hidden_size=hidden_size)  # Update input_size to match features

    def execute(self, x):
        """
        Runs the Prime Structure Model and applies entropy-based corrections to improve precision.
        """
        prediction = self.model.forward(x)

        # Apply localized entropy corrections
        entropy_correction = np.exp(-0.05 * self.model.entropy_monitor.entropy)
        refined_prediction = prediction * (1 + entropy_correction)  

        return refined_prediction

    def live_prime_training(self):
        """
        Runs real-time prime structure learning with adaptive Bayesian optimization search.
        """
        prime_data = self.preprocess_prime_data()
        actual_values = self.generate_actual_prime_structure()

        anchor_size = int(0.1 * len(prime_data))
        anchor_data = prime_data[:anchor_size]
        streaming_data = prime_data[anchor_size:]
        streaming_actuals = actual_values[anchor_size:]

        hidden_size = 64

        # Dynamically adjust Bayesian search space based on past convergence speed
        convergence_speed = np.mean(np.diff(streaming_actuals[:100]))  
        lr_min, lr_max = (1e-5, 1e-2) if convergence_speed > 0.1 else (1e-4, 1e-1)

        param_space = [
            Real(lr_min, lr_max, name='learning_rate'),
            Integer(10, 100, name='hidden_size'),
        ]
        
        cimm = CIMM(lambda: PrimeStructureModel(input_size=4, hidden_size=hidden_size), param_space, anchor_data)  # Update input_size to match features

        predictions = []
        actuals = []

        for i, new_data_point in enumerate(streaming_data):
            actual_value = streaming_actuals[i]
            if isinstance(actual_value, torch.Tensor):
                if actual_value.numel() == 1:
                    actual_value = actual_value.item()
                else:
                    actual_value = actual_value.squeeze().tolist()

            prediction, ___, __ = cimm.run(new_data_point)
            cimm.give_feedback(new_data_point, actual_value)

            predictions.append(prediction)
            actuals.append(actual_value)

        assert len(predictions) == len(streaming_data), "Prediction loop did not cover all input data points"

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        plot_live_predictions(predictions, actuals)

        validation_data = torch.tensor(streaming_data, dtype=torch.float32)
        metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
        print(f"Error Metrics: {metrics}")

    def preprocess_prime_data(self, raw_data=None):
        primes = list(sympy.primerange(1, 100000))
        prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
        num_samples = (len(prime_gaps) // 10) * 10

        # Normalize data
        prime_gaps = (prime_gaps - np.mean(prime_gaps)) / np.std(prime_gaps)
        log_prime_gaps = np.log1p(np.abs(prime_gaps))
        
        # Fourier Transform
        fourier_transform = np.fft.fft(prime_gaps).real

        # Add Wavelet Transform features
        wavelet_coeffs = pywt.wavedec(prime_gaps, 'db4', level=3)
        wavelet_features = np.hstack([coeff[:num_samples] for coeff in wavelet_coeffs])  

        # Ensure all feature arrays have the same length
        min_length = min(len(prime_gaps), len(log_prime_gaps), len(fourier_transform), len(wavelet_features))
        prime_gaps = prime_gaps[:min_length]
        log_prime_gaps = log_prime_gaps[:min_length]
        fourier_transform = fourier_transform[:min_length]
        wavelet_features = wavelet_features[:min_length]

        # Combine all features
        combined_features = np.vstack([prime_gaps, log_prime_gaps, fourier_transform, wavelet_features]).T
        data = torch.tensor(combined_features[:num_samples], dtype=torch.float32)

        assert data.shape[1] > 3, "Data reshaping failed; feature size should be greater than 3"
        return data

    def generate_actual_prime_structure(self):
        return self.preprocess_prime_data()

# Example usage
if __name__ == "__main__":
    use_case = PrimeStructureUseCase(hidden_size=64)
    use_case.live_prime_training()
