import matplotlib.pyplot as plt
import numpy as np
import torch
import sympy
def remove_extreme_outliers(data, threshold=3.0):
    """Remove extreme outliers from the data."""
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = data[np.abs(data - mean) < threshold * std_dev]
    return filtered_data

def plot_stock_predictions(model, data, actual_prices):
    model.eval()
    with torch.no_grad():
        model_output = model(data).cpu().numpy().flatten()
    
    filtered_actual_prices = remove_extreme_outliers(actual_prices[:len(model_output)])
    filtered_model_output = remove_extreme_outliers(model_output)

    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_actual_prices, filtered_model_output, alpha=0.5, label="Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(filtered_model_output)), filtered_actual_prices, label="Actual Prices", linestyle='dotted' ,color="red")
    plt.plot(range(len(filtered_model_output)), filtered_model_output, label="Predicted Prices", alpha=0.8, color="blue")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Actual vs. Predicted Stock Prices")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_live_predictions(predictions, actuals):
    # Ensure predictions and actuals are on CPU and numpy arrays for matplotlib
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    elif isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], torch.Tensor):
        predictions = torch.stack(predictions).detach().cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.detach().cpu().numpy()
    elif isinstance(actuals, list) and len(actuals) > 0 and isinstance(actuals[0], torch.Tensor):
        actuals = torch.stack(actuals).detach().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(predictions)), predictions, label="Predicted Values", linestyle='dotted')
    plt.plot(range(len(actuals)), actuals, label="Actual Values", alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Live Predictions vs. Actual Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_prime_structure(model, data):
    primes = list(sympy.primerange(1, 100000))
    prime_gaps = [primes[i+1] - primes[i] for i in range(len(primes) - 1)]

    prime_gaps = (prime_gaps - np.mean(prime_gaps)) / np.std(prime_gaps)

    log_prime_gaps = np.log1p(np.abs(prime_gaps))
    fourier_transform = np.fft.fft(prime_gaps).real

    combined_features = np.vstack([prime_gaps, log_prime_gaps, fourier_transform]).T

    num_samples = (len(combined_features) // 10) * 10
    data = torch.tensor(combined_features[:num_samples], dtype=torch.float32).reshape(-1, 10)

    model.eval()
    with torch.no_grad():
        model_output = model(data).cpu().numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.scatter(primes[:len(model_output)], model_output, alpha=0.5, label="Identified Structure")
    plt.xlabel("Prime Numbers")
    plt.ylabel("Model's Structural Output")
    plt.title("Prime Numbers vs. Identified Structure")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(model_output)), prime_gaps[:len(model_output)], label="Prime Gaps", linestyle='dotted')
    plt.plot(range(len(model_output)), model_output, label="Identified Structure", alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Prime Gaps vs. Model's Identified Structure")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist2d(primes[:len(model_output)], model_output, bins=(50, 50), cmap="plasma")
    plt.colorbar(label="Density")
    plt.xlabel("Prime Numbers")
    plt.ylabel("Model's Structural Output")
    plt.title("Heatmap of Prime Numbers vs. Identified Structure")
    plt.show()
