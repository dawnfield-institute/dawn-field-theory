import torch
import numpy as np
import sympy
import matplotlib.pyplot as plt
import logging
from skopt.space import Real, Integer
from cimm import CIMM
import time

def plot_prime_structure(model, data):
    # Generate prime numbers
    primes = list(sympy.primerange(1, 100000))
    prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]

    # Normalize prime gaps
    prime_gaps = (prime_gaps - np.mean(prime_gaps)) / np.std(prime_gaps)

    # Add nonlinear transforms of prime gaps
    log_prime_gaps = np.log1p(np.abs(prime_gaps))  # Ensure no negative values for log
    fourier_transform = np.fft.fft(prime_gaps).real

    # Combine original and transformed features
    combined_features = np.vstack([prime_gaps, log_prime_gaps, fourier_transform]).T

    # Convert to tensor
    num_samples = (len(combined_features) // 10) * 10  # Ensure compatibility with model input
    data = torch.tensor(combined_features[:num_samples], dtype=torch.float32).reshape(-1, 10)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        model_output = model(data).cpu().numpy().flatten()

    # Log the model output for debugging
    logging.info(f"Model Output: {model_output}")

    # 1️⃣ Scatter Plot: Prime Numbers vs. Model Output
    plt.figure(figsize=(10, 5))
    plt.scatter(primes[:len(model_output)], model_output, alpha=0.5, label="Identified Structure")
    plt.xlabel("Prime Numbers")
    plt.ylabel("Model's Structural Output")
    plt.title("Prime Numbers vs. Identified Structure")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2️⃣ Line Plot: Prime Gaps vs. Model Output Over Iterations
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(model_output)), prime_gaps[:len(model_output)], label="Prime Gaps", linestyle='dotted')
    plt.plot(range(len(model_output)), model_output, label="Identified Structure", alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Prime Gaps vs. Model's Identified Structure")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3️⃣ Heatmap: Distribution of Prime Numbers vs. Structure
    plt.figure(figsize=(10, 5))
    plt.hist2d(primes[:len(model_output)], model_output, bins=(50, 50), cmap="plasma")
    plt.colorbar(label="Density")
    plt.xlabel("Prime Numbers")
    plt.ylabel("Model's Structural Output")
    plt.title("Heatmap of Prime Numbers vs. Identified Structure")
    plt.show()

def preprocess_prime_data(raw_data=None):
    # Function to preprocess prime number data
    primes = list(sympy.primerange(1, 100000))
    prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
    num_samples = (len(prime_gaps) // 10) * 10  # Ensure the number of samples is a multiple of 10
    prime_gaps = (prime_gaps - np.mean(prime_gaps)) / np.std(prime_gaps)  # Normalize prime gaps
    log_prime_gaps = np.log1p(np.abs(prime_gaps))  # Logarithmic scaling
    fourier_transform = np.fft.fft(prime_gaps).real  # Frequency components
    combined_features = np.vstack([prime_gaps, log_prime_gaps, fourier_transform]).T
    data = torch.tensor(combined_features[:num_samples], dtype=torch.float32).reshape(-1, 10)  # Reshape to have 10 features
    return data

def generate_actual_prime_structure():
    # Function to generate actual prime structure data
    primes = list(sympy.primerange(1, 100000))
    prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]
    num_samples = (len(prime_gaps) // 10) * 10  # Ensure the number of samples is a multiple of 10
    prime_gaps = (prime_gaps - np.mean(prime_gaps)) / np.std(prime_gaps)  # Normalize prime gaps
    log_prime_gaps = np.log1p(np.abs(prime_gaps))  # Logarithmic scaling
    fourier_transform = np.fft.fft(prime_gaps).real  # Frequency components
    combined_features = np.vstack([prime_gaps, log_prime_gaps, fourier_transform]).T
    data = torch.tensor(combined_features[:num_samples], dtype=torch.float32).reshape(-1, 10)  # Reshape to have 10 features
    return data

def plot_live_predictions(predictions, actuals):
    """Plot live predictions against actual values."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(actuals)), actuals, label="Actual Values", color='black', alpha=0.8)  # Changed color to black
    plt.plot(range(len(predictions)), predictions, label="Predicted Values", linestyle='dotted', color='red')  # Changed color to red
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Live Predictions vs. Actual Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def live_prime_training():
    """Runs CIMM on prime structure data incrementally after pretraining with an anchor dataset."""
    # Get full dataset
    prime_data = preprocess_prime_data()
    actual_values = generate_actual_prime_structure()

    # Split into anchor (20%) and streaming data (80%)
    anchor_size = int(0.2 * len(prime_data))
    anchor_data = prime_data[:anchor_size]
    streaming_data = prime_data[anchor_size:]
    streaming_actuals = actual_values[anchor_size:]

    # Initialize CIMM with pretraining
    cimm = CIMM(ExampleModel, param_space, anchor_data)

    predictions = []
    actuals = []

    # Stream live data one at a time
    for i, new_data_point in enumerate(streaming_data):
        actual_value = streaming_actuals[i]  # Keep as tensor
        if isinstance(actual_value, torch.Tensor) and actual_value.numel() == 1:
            actual_value = actual_value.item()  # Convert to scalar if it's a single number
        elif isinstance(actual_value, torch.Tensor):
            actual_value = actual_value.squeeze().tolist()  # Convert multi-element tensor to a list
        prediction = cimm.run(new_data_point, actual_value)  # Pass as single value or list
        predictions.append(prediction)
        actuals.append(actual_value)

    # Plot live predictions vs actual values
    plot_live_predictions(predictions, actuals)

    # Compute and print error metrics
    validation_data = torch.tensor(streaming_data, dtype=torch.float32)
    metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
    print(f"Error Metrics: {metrics}")

# Example usage
if __name__ == "__main__":
    class ExampleModel(torch.nn.Module):
        def __init__(self, hidden_size):
            super(ExampleModel, self).__init__()
            self.hidden_size = int(hidden_size)
            self.linear1 = torch.nn.Linear(10, 128)  # Increase from 10 to 128 neurons
            self.linear2 = torch.nn.Linear(128, 64)
            self.linear3 = torch.nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = torch.relu(self.linear2(x))
            x = self.linear3(x)
            return x

    param_space = [
        Real(1e-5, 1e-1, name='learning_rate'),
        Integer(10, 100, name='hidden_size'),
    ]

    live_prime_training()