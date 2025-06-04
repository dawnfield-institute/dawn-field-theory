import yfinance as yf
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from skopt.space import Real, Integer
from cimm import CIMM
import time
from scipy.signal import butter, filtfilt

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def fetch_stock_data(ticker, period="5y", interval="1d"):
    if isinstance(ticker, torch.Tensor):
        ticker = ticker[0].item()  # Extract the first element if it's a tensor with multiple elements
    if isinstance(ticker, list):
        ticker = ticker[0]  # Extract the first element if it's a list
    if isinstance(ticker, (int, float)):
        ticker = str(ticker)  # Convert to string if it's a number
    if not isinstance(ticker, str):
        raise ValueError("Ticker must be a string")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    # Extract relevant features
    df['Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log1p(df['Returns'])
    df['Volatility'] = df['Close'].rolling(window=20).std()  # Increased window
    df['Moving Average'] = df['Close'].rolling(window=50).mean()  # Longer trend capture
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal Line'] = calculate_macd(df['Close'])
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Normalize data
    for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Apply Fourier Transform
    fourier_transform = butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0)
    
    # Combine features
    num_samples = (len(df) // 10) * 10  # Ensure compatible dimensions
    combined_features = np.vstack([
        df['Returns'].values[:num_samples],
        df['Log Returns'].values[:num_samples],
        df['Volatility'].values[:num_samples],
        df['Moving Average'].values[:num_samples],
        df['RSI'].values[:num_samples],
        df['MACD'].values[:num_samples],
        df['Signal Line'].values[:num_samples],
        fourier_transform[:num_samples]
    ]).T

    return torch.tensor(combined_features, dtype=torch.float32).reshape(-1, 10)

def preprocess_stock_data(ticker):
    """Preprocess stock data for the given ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y", interval="1d")
    
    # Extract relevant features
    df['Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log1p(df['Returns'])
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Moving Average'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal Line'] = calculate_macd(df['Close'])
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Normalize data
    for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Apply Fourier Transform
    fourier_transform = butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0)
    
    # Combine features
    num_samples = (len(df) // 10) * 10  # Ensure compatible dimensions
    combined_features = np.vstack([
        df['Returns'].values[:num_samples],
        df['Log Returns'].values[:num_samples],
        df['Volatility'].values[:num_samples],
        df['Moving Average'].values[:num_samples],
        df['RSI'].values[:num_samples],
        df['MACD'].values[:num_samples],
        df['Signal Line'].values[:num_samples],
        fourier_transform[:num_samples]
    ]).T

    return torch.tensor(combined_features, dtype=torch.float32).reshape(-1, 10)

def remove_extreme_outliers(data, threshold=3.0):
    """Remove extreme outliers from the data."""
    data = np.array(data)  # Ensure data is a numpy array
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = data[np.abs(data - mean) < threshold * std_dev]
    return filtered_data

def plot_stock_predictions(model, data, actual_prices):
    model.eval()
    with torch.no_grad():
        model_output = model(data).cpu().numpy().flatten()
    
    # Remove extreme outliers
    filtered_actual_prices = remove_extreme_outliers(actual_prices[:len(model_output)])
    filtered_model_output = remove_extreme_outliers(model_output)

    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(filtered_actual_prices, filtered_model_output, alpha=0.5, label="Predicted Prices")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Stock Price Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Line plot: Trend Comparison
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(filtered_model_output)), filtered_actual_prices, label="Actual Prices", linestyle='dotted')
    plt.plot(range(len(filtered_model_output)), filtered_model_output, label="Predicted Prices", alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Actual vs. Predicted Stock Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_live_predictions(predictions, actuals):
    """Plot live predictions against actual values."""
    # Remove extreme outliers
    filtered_predictions = remove_extreme_outliers(predictions)
    filtered_actuals = remove_extreme_outliers(actuals)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(filtered_predictions)), filtered_predictions, label="Predicted Values", linestyle='dotted')
    plt.plot(range(len(filtered_actuals)), filtered_actuals, label="Actual Values", alpha=0.8)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Live Predictions vs. Actual Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def live_stock_training():
    """Runs CIMM on stock data incrementally after pretraining with an anchor dataset."""
    ticker = input("Enter stock ticker (e.g., TSLA, AMD, NIO): ").upper()
    # Get full dataset
    stock_data = preprocess_stock_data(ticker)
    actual_values = yf.Ticker(ticker).history(period="5y", interval="1d")["Close"].values

    # Split into anchor (20%) and streaming data (80%)
    anchor_size = int(0.2 * len(stock_data))
    anchor_data = stock_data[:anchor_size]
    streaming_data = stock_data[anchor_size:]
    streaming_actuals = actual_values[anchor_size:]

    # Initialize CIMM with pretraining
    cimm = CIMM(StockPredictionModel, param_space, anchor_data)

    predictions = []
    actuals = []
    smoothing_factor = 0.1

    # Stream live data one at a time
    for i, new_data_point in enumerate(streaming_data):
        actual_value = streaming_actuals[i]
        smoothed_data_point = smoothing_factor * new_data_point + (1 - smoothing_factor) * (predictions[-1] if predictions else new_data_point)
        prediction = cimm.run(smoothed_data_point, actual_value)
        predictions.append(prediction)
        actuals.append(actual_value)
        # Adjust smoothing factor based on recent error metrics
        if len(predictions) > 10:
            recent_errors = np.abs(np.array(predictions[-10:]) - np.array(actuals[-10:]))
            smoothing_factor = min(0.5, max(0.01, np.mean(recent_errors) / np.std(recent_errors)))

    # Plot live predictions vs actual values
    plot_live_predictions(predictions, actuals)

    # Compute and print error metrics
    validation_data = torch.tensor(streaming_data, dtype=torch.float32)
    metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
    print(f"Error Metrics: {metrics}")

# Example usage
if __name__ == "__main__":
    class StockPredictionModel(torch.nn.Module):
        def __init__(self, hidden_size):
            super(StockPredictionModel, self).__init__()
            self.hidden_size = int(hidden_size)
            self.linear1 = torch.nn.Linear(10, 128)
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

    live_stock_training()
