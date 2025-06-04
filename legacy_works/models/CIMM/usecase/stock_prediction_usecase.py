import yfinance as yf
import torch
import logging
from skopt.space import Real, Integer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cimm_core.cimm import CIMM
from cimm_core.models.base_cimm_model import BaseCIMMModel
from scipy.signal import butter, filtfilt
from cimm_core.entropy.entropy_monitor import EntropyMonitor
from cimm_core.models.financial_model import FinancialModel
from visualization.plots import  plot_live_predictions
from utils.logging import configure_logging
import pandas as pd
from cimm_core.utils import get_device  # Ensure device utility is imported

device = get_device()  # Ensure device is set globally

#if torch.cuda.is_available():
#    torch.set_default_tensor_type(torch.cuda.FloatTensor)

def entropy_aware_smooth(series: pd.Series, max_window: int = 50) -> pd.Series:
    ent_monitor = EntropyMonitor()
    entropy = ent_monitor.measure_entropy(series.values)
    window_size = max(5, int((1 - entropy) * max_window))
    return series.rolling(window=window_size).mean()

def entropy_weighted_collapse(outcomes, entropy_values):
    """
    Collapses outcomes based on entropy-weighted probabilities.
    """
    weights = torch.softmax(torch.tensor(entropy_values, dtype=torch.float32), dim=0)  # Ensure weights are Float
    outcomes = torch.tensor(outcomes, dtype=torch.float32)  # Ensure outcomes are Float
    return torch.dot(weights, outcomes)

class StockPredictionUseCase:
    def __init__(self, hidden_size):
        self.model = FinancialModel(hidden_size).to(device)  # Move model to device
        #configure_logging()
        print("Logging is configured.")

    def execute(self, x):
        return self.model.forward(x)

    def calculate_rsi(self, data, window=14):
        print("Calculating RSI.")
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        print("Calculating MACD.")
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        print("Applying Butterworth lowpass filter.")
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def fetch_stock_data(self, ticker, period="5y", interval="1d"):
        print(f"Fetching stock data for ticker: {ticker}")
        if isinstance(ticker, torch.Tensor):
            ticker = ticker[0].item()
        if isinstance(ticker, list):
            ticker = ticker[0]
        if isinstance(ticker, (int, float)):
            ticker = str(ticker)
        if not isinstance(ticker, str):
            raise ValueError("Ticker must be a string")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        df['Returns'] = entropy_aware_smooth(df['Close'].pct_change())
        df['Log Returns'] = entropy_aware_smooth(torch.log1p(torch.tensor(df['Returns'].values, dtype=torch.float32)))
        df['Volatility'] = entropy_aware_smooth(torch.tensor(df['Close'].rolling(window=20).std().values, dtype=torch.float32))
        df['Moving Average'] = entropy_aware_smooth(torch.tensor(df['Close'].rolling(window=50).mean().values, dtype=torch.float32))
        df['RSI'] = entropy_aware_smooth(torch.tensor(self.calculate_rsi(df['Close']).values, dtype=torch.float32))
        macd, signal_line = self.calculate_macd(df['Close'])
        df['MACD'] = entropy_aware_smooth(torch.tensor(macd.values, dtype=torch.float32))
        df['Signal Line'] = entropy_aware_smooth(torch.tensor(signal_line.values, dtype=torch.float32))

        df['Lag1_Close'] = df['Close'].shift(1)
        df['Lag2_Close'] = df['Close'].shift(2)
        df['Price Momentum'] = df['Lag1_Close'] - df['Lag2_Close']

        for col in ['Lag1_Close', 'Lag2_Close', 'Price Momentum']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        df.dropna(inplace=True)
        
        for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        entropy = EntropyMonitor().measure_entropy(df['Close'].values)
        cutoff = 0.02 + 0.1 * entropy  # Range: [0.02–0.12]

        vol_entropy = EntropyMonitor().measure_entropy(df['Volatility'].values)
        df['Volatility'] *= (1 + (1 - vol_entropy))

        fourier_transform = self.butter_lowpass_filter(df['Close'].values, cutoff=cutoff, fs=1.0, order=6).copy()
        
        num_samples = (len(df) // 10) * 10
        combined_features = torch.stack([
            torch.tensor(df['Returns'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Log Returns'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Volatility'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Moving Average'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['RSI'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['MACD'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Signal Line'].values[:num_samples], dtype=torch.float32),
            torch.tensor(fourier_transform[:num_samples], dtype=torch.float32)
        ], dim=1)

        return combined_features.to(device).reshape(-1, 10)  # Move tensor to device

    def preprocess_stock_data(self, ticker):
        print(f"Preprocessing stock data for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        df = stock.history(period="9y", interval="1d")
        
        df['Returns'] = df['Close'].pct_change()
        df['Log Returns'] = torch.log1p(torch.tensor(df['Returns'].values, dtype=torch.float32))
        df['Volatility'] = torch.tensor(df['Close'].rolling(window=20).std().values, dtype=torch.float32)
        df['Moving Average'] = torch.tensor(df['Close'].rolling(window=50).mean().values, dtype=torch.float32)
        df['RSI'] = torch.tensor(self.calculate_rsi(df['Close']).values, dtype=torch.float32)
        macd, signal_line = self.calculate_macd(df['Close'])
        df['MACD'] = torch.tensor(macd.values, dtype=torch.float32)
        df['Signal Line'] = torch.tensor(signal_line.values, dtype=torch.float32)
        
        df.dropna(inplace=True)
        
        for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        fourier_transform = self.butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0).copy()
        
        num_samples = (len(df) // 10) * 10
        combined_features = torch.stack([
            torch.tensor(df['Returns'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Log Returns'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Volatility'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Moving Average'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['RSI'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['MACD'].values[:num_samples], dtype=torch.float32),
            torch.tensor(df['Signal Line'].values[:num_samples], dtype=torch.float32),
            torch.tensor(fourier_transform[:num_samples], dtype=torch.float32)
        ], dim=1)
      
        return torch.tensor(combined_features, dtype=torch.float32).to(device).reshape(-1, 10)  # Move tensor to device

    def live_stock_training(self):
        #logging.info("Starting live stock training")
        ticker = input("Enter stock ticker (e.g., TSLA, AMD, NIO): ").upper()
        stock_data = self.preprocess_stock_data(ticker)
        actual_values = yf.Ticker(ticker).history(period="10y", interval="1d")["Close"].values

        anchor_size = int(0.1 * len(stock_data))
        anchor_data = stock_data[:anchor_size]
        streaming_data = stock_data[anchor_size:]
        streaming_actuals = actual_values[anchor_size:]

        hidden_size = 64
        model_class = StockPredictionModel
        model_args = (hidden_size,)  # Arguments for the model constructor
        cimm = CIMM(model_class, model_args, param_space, anchor_data)  # Pass model class and arguments
        # Ensure that the input tensor `x` is passed to the model during training or inference
        # Example:
        # output = cimm.model_instance(x)  # Pass the input tensor `x` here
        entropy_monitor = EntropyMonitor(initial_entropy=1.0, learning_rate=0.01)
        entropy_monitor.prev_entropy = 0.0

        optimizer = torch.optim.Adam(cimm.model_instance.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
        loss_fn = torch.nn.MSELoss()

        predictions = []
        actuals = []

        for i, new_data_point in enumerate(streaming_data):
            if i >= len(streaming_actuals):
                break

            actual_value = streaming_actuals[i]
            print(f"Processing data point {i} with actual value {actual_value}")

            # Physics-based input transformation (no smoothing)
            data_point = torch.abs(new_data_point).to(device)  # Move tensor to device
            data_point = torch.log(torch.clamp(data_point + 1, min=1e-8))
            actual_value = torch.tensor(actual_value, dtype=torch.float32).to(device)  # Move tensor to device
            actual_value = torch.log(actual_value + 1)
            actual_value = actual_value.cpu()  # ✅ Fix: move to CPU before use in metrics

            result = cimm.run(data_point)

            # Handle 1, 2, or 3-value returns
            if isinstance(result, tuple):
                if len(result) == 3:
                    prediction, probs, alternatives = result
                    probs = torch.tensor(probs, dtype=torch.float64).flatten()
                    alternatives = torch.tensor(alternatives).flatten()

                    # 🚨 Safety check: fallback if mismatch
                    if len(probs) != len(alternatives) or torch.isnan(probs).any() or probs.sum() == 0:
                        probs = torch.ones(len(alternatives)) / len(alternatives)

                    # ✅ Force-safe probability normalization
                    probs /= probs.sum()
                    probs = torch.round(probs, decimals=12)  # truncate precision beyond float64 safety
                    probs /= probs.sum()         # renormalize after rounding

                    # ✅ Confirm safe
                    assert torch.isclose(probs.sum(), torch.tensor(1.0)), f"Probabilities do not sum to 1. Current sum: {probs.sum()}"

                    # 🧠 Safe final choice
                    try:
                        selected_idx = torch.multinomial(probs, 1).item()
                    except ValueError:
                        selected_idx = torch.randint(len(alternatives), (1,)).item()  # fallback to uniform
                    prediction = alternatives[selected_idx]
                else:
                    prediction = result[0]
            else:
                prediction = result

            # Apply interference cancellation based on last 5 wave steps
            last_wave = torch.tensor(predictions[-5:]) if len(predictions) >= 5 else torch.zeros(5)
            destructive_interference = torch.std(last_wave) * 0.1
            prediction -= destructive_interference

            # Generate ensemble predictions (example placeholder)
            # Limit divergence to ±5% max to reduce volatility
            delta = 0.05 * prediction
            ensemble_features = torch.tensor([
                prediction - delta,
                prediction,
                prediction + delta
            ])
            entropy_values = torch.tensor([entropy_monitor.calculate_entropy(new_data_point)] * len(ensemble_features))

            # Apply entropy-weighted collapse
            collapsed_features = entropy_weighted_collapse(ensemble_features, entropy_values)

            # Use collapsed features for further processing
            prediction = collapsed_features

            # Replace prediction only if collapse succeeds
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                print("Invalid prediction collapse — reverting to last stable prediction")
                prediction = predictions[-1] if predictions else 0.0

            # Smooth final prediction using recent trend (moving average of past 3 predictions)
            if len(predictions) >= 3:
                prediction = 0.6 * prediction + 0.4 * torch.mean(torch.tensor(predictions[-3:]))

            # Detect anomalies
            if len(predictions) >= 20:
                delta = torch.abs(prediction - actual_value.item())
                threshold = 2.5 * torch.std(torch.tensor(predictions[-20:]))
                if delta > threshold:
                    print(f"⚠️ Anomaly detected at index {i}: Δ={float(delta):.4f}, threshold={float(threshold):.4f}")

            # Ensure prediction is scalar and safe for smoothing
            arr = torch.tensor(prediction).squeeze()
            scalar_pred = float(arr[0]) if (arr.ndimension() == 1 and len(arr) == 1) else float(torch.mean(arr))
            prediction = scalar_pred

            # --- Financial Smoothing Block ---
            if len(predictions) >= 2:
                recent_momentum = predictions[-1] - predictions[-2]
                prediction += 0.1 * recent_momentum

            if len(predictions) >= 5:
                recent_returns = torch.diff(torch.tensor(predictions[-5:]))
                avg_return = torch.mean(recent_returns)
                vol = torch.std(recent_returns) + 1e-8
                sharpe_ratio = avg_return / vol
                prediction *= (1 + 0.02 * sharpe_ratio)

            entropy = entropy_monitor.calculate_entropy(new_data_point)
            entropy_tensor = torch.tensor(entropy, dtype=torch.float32)  # Ensure entropy is a tensor
            entropy_penalty = torch.clip(entropy_tensor - 0.8, 0, 1)
            prediction *= (1 - 0.05 * entropy_penalty)

            # Append final smoothed prediction
            predictions.append(prediction)
            actuals.append(actual_value.item())

            cimm.give_feedback(data_point, actual_value)

            # Update entropy monitor
            entropy = entropy_monitor.calculate_entropy(new_data_point)
            entropy_change = torch.tensor(entropy - entropy_monitor.prev_entropy, dtype=torch.float32)  # Ensure tensor
            clipped_change = torch.clip(entropy_change, -0.01, 0.01)
            entropy_monitor.learning_rate = (0.95 * entropy_monitor.learning_rate) + (0.05 * clipped_change.item())

            if abs(entropy - entropy_monitor.prev_entropy) > 0.03:
                print("Updating model due to significant entropy change")
                cimm.controller.update_model(anchor_data, streaming_actuals[:len(anchor_data)])  # Pass targets

            entropy_monitor.prev_entropy = entropy
            new_lr = entropy_monitor.qbe_learning_rate(optimizer, entropy, entropy_monitor.qbe_baseline, base_lr=0.01)
            scheduler.step(new_lr)

        # Plot and evaluate
        plot_live_predictions(predictions, actuals, None)
        validation_data = torch.tensor(streaming_data, dtype=torch.float32)
        metrics = cimm.evaluate_model(cimm.model_instance, validation_data)
        print(f"Error Metrics: {metrics}")
        print(f"Error Metrics: {metrics}")

class StockPredictionModel(BaseCIMMModel):
    def __init__(self, hidden_size):
        super(StockPredictionModel, self).__init__(input_size=10, hidden_size=hidden_size, output_size=1)
        self.to(device)  # Ensure model is moved to device

    def forward(self, x):
        x = x.to(device)  # Ensure input tensor is moved to the correct device
        return self.common_forward(x)

param_space = [
    Real(1e-5, 1e-1, name='learning_rate'),
    Integer(10, 100, name='hidden_size'),
]

if __name__ == "__main__":
    print("Starting the stock prediction use case.")
    use_case = StockPredictionUseCase(hidden_size=64)
    use_case.live_stock_training()
    print("Stock prediction use case completed.")
