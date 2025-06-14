import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import finnhub
import torch
import logging
from skopt.space import Real, Integer

import pandas as pd
from cimm_core.cimm import CIMM
from cimm_core.models.base_cimm_model import BaseCIMMModel
from scipy.signal import butter, filtfilt
from cimm_core.entropy.entropy_monitor import EntropyMonitor
from cimm_core.models.financial_model import FinancialModel
from visualization.plots import  plot_live_predictions
from utils.logging import configure_logging
from cimm_core.utils import get_device  # Ensure device utility is imported

device = get_device()  # Ensure device is set globally

# --- Remove Finnhub API Key Setup and client ---

def entropy_aware_smooth(series, max_window: int = 50):
    # Torch-only implementation, all on GPU
    ent_monitor = EntropyMonitor()
    # Accept pd.Series, torch.Tensor, or list/array-like
    if isinstance(series, pd.Series):
        values = torch.tensor(series.values, dtype=torch.float32, device=device)
        index = series.index
    elif isinstance(series, torch.Tensor):
        values = series.to(dtype=torch.float32, device=device)
        index = None
    else:
        values = torch.tensor(series, dtype=torch.float32, device=device)
        index = None
    entropy = ent_monitor.calculate_entropy(values)
    if not torch.is_tensor(entropy):
        entropy = torch.tensor(entropy, dtype=torch.float32, device=device)
    if torch.isnan(entropy) or not torch.isfinite(entropy):
        entropy = torch.tensor(0.5, device=device)
    window_size = max(5, int((1 - entropy.item()) * max_window))
    if len(values) < window_size:
        smoothed = values.clone()
    else:
        pad_len = window_size - 1
        left_pad = values[0].repeat(pad_len)
        padded = torch.cat([left_pad, values])
        smoothed = torch.stack([padded[i:i+len(values)] for i in range(window_size)], dim=0).mean(dim=0)
    if index is not None and len(index) == len(smoothed):
        return pd.Series(smoothed.cpu().tolist(), index=index)
    else:
        return pd.Series(smoothed.cpu().tolist())

def entropy_weighted_collapse(outcomes, entropy_values):
    """
    Collapses outcomes based on entropy-weighted probabilities.
    """
    outcomes = torch.tensor(outcomes, dtype=torch.float32, device=device)
    entropy_values = torch.tensor(entropy_values, dtype=torch.float32, device=device)
    weights = torch.softmax(entropy_values, dim=0)
    return torch.dot(weights, outcomes)

class StockPredictionUseCase:
    def __init__(self, hidden_size):
        self.model = FinancialModel(hidden_size).to(device)
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

    def fetch_stock_data(self, csv_path):
        import pandas as pd

        df = pd.read_csv(csv_path)
        # Ensure 'Close' column is numeric
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        else:
            raise ValueError("No 'Close' column found in CSV.")

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

        # --- Fix: Use calculate_entropy instead of measure_entropy ---
        entropy = EntropyMonitor().calculate_entropy(torch.tensor(df['Close'].values, dtype=torch.float32, device=device))
        cutoff = 0.02 + 0.1 * entropy

        vol_entropy = EntropyMonitor().calculate_entropy(torch.tensor(df['Volatility'].values, dtype=torch.float32, device=device))
        df['Volatility'] *= (1 + (1 - vol_entropy))

        fourier_transform = self.butter_lowpass_filter(df['Close'].values, cutoff=cutoff, fs=1.0, order=6).copy()

        num_samples = (len(df) // 10) * 10
        # Stack exactly 10 features to match model input size
        combined_features = torch.stack([
            torch.tensor(df['Returns'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Log Returns'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Volatility'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Moving Average'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['RSI'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['MACD'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Signal Line'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Lag1_Close'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Lag2_Close'].values[:num_samples], dtype=torch.float32, device=device),
            torch.tensor(df['Price Momentum'].values[:num_samples], dtype=torch.float32, device=device)
        ], dim=1)
        return combined_features  # Already on device

    def preprocess_stock_data(self, csv_path="data/amd_full_history.csv"):
        features = self.fetch_stock_data(csv_path)
        # Already on device from fetch_stock_data
        return features

    def live_stock_training(self):
        csv_path = input("Enter CSV path (default: data/amd_full_history.csv): ").strip() or "data/amd_full_history.csv"
        stock_data = self.preprocess_stock_data(csv_path)
        # Already on device from preprocess_stock_data
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        except ValueError:
            df = pd.read_csv(csv_path)
            date_col = None
            for col in df.columns:
                if col.lower() == "date":
                    date_col = col
                    break
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            else:
                print("Warning: No 'Date' column found in CSV. Proceeding without date index.")
        actual_values = df['Close'].values

        # --- GPU-bound: Use only torch for all operations, no numpy ---
        actual_values = pd.to_numeric(actual_values, errors='coerce')
        actual_values = torch.tensor(actual_values, dtype=torch.float32, device=device)
        nan_mask = torch.isnan(actual_values) | torch.isinf(actual_values)
        actual_values[nan_mask] = 0.0
        # Already on device

        anchor_size = int(0.1 * len(stock_data))
        anchor_data = stock_data[:anchor_size]  # Already on device
        streaming_data = stock_data[anchor_size:]  # Already on device
        streaming_actuals = actual_values[anchor_size:]  # Already on device

        # Ensure all are on device
        anchor_data = anchor_data.to(device)
        streaming_data = streaming_data.to(device)
        streaming_actuals = streaming_actuals.to(device)

        hidden_size = 64
        model_class = StockPredictionModel
        model_args = (hidden_size,)
        cimm = CIMM(model_class, model_args, param_space, anchor_data)

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
            data_point = torch.abs(new_data_point).to(device)
            data_point = torch.log(torch.clamp(data_point + 1, min=1e-8))
            actual_value = float(actual_value)
            actual_value = torch.tensor(actual_value, dtype=torch.float32, device=device)
            actual_value = torch.log(actual_value + 1)
            actual_value = actual_value.cpu()  # Only for metrics

            result = cimm.run(data_point)

            if isinstance(result, tuple):
                if len(result) == 3:
                    prediction, probs, alternatives = result
                    probs = torch.tensor(probs, dtype=torch.float64, device=device).flatten()
                    alternatives = torch.tensor(alternatives, device=device).flatten()
                    if len(probs) != len(alternatives) or torch.isnan(probs).any() or probs.sum() == 0:
                        probs = torch.ones(len(alternatives), device=device) / len(alternatives)
                    probs /= probs.sum()
                    probs = torch.round(probs, decimals=12)
                    probs /= probs.sum()
                    assert torch.isclose(probs.sum(), torch.tensor(1.0, device=device)), f"Probabilities do not sum to 1. Current sum: {probs.sum()}"
                    try:
                        selected_idx = torch.multinomial(probs, 1).item()
                    except ValueError:
                        selected_idx = torch.randint(len(alternatives), (1,), device=device).item()
                    prediction = alternatives[selected_idx]
                else:
                    prediction = result[0]
            else:
                prediction = result

            last_wave = torch.tensor(predictions[-5:], device=device) if len(predictions) >= 5 else torch.zeros(5, device=device)
            destructive_interference = torch.std(last_wave) * 0.1
            if isinstance(prediction, torch.Tensor):
                destructive_interference = destructive_interference.to(prediction.device)
            prediction -= destructive_interference

            delta = 0.05 * prediction
            ensemble_features = torch.tensor([
                prediction - delta,
                prediction,
                prediction + delta
            ], device=device)
            entropy_values = torch.tensor([entropy_monitor.calculate_entropy(new_data_point)] * len(ensemble_features), device=device)
            collapsed_features = entropy_weighted_collapse(ensemble_features, entropy_values)
            prediction = collapsed_features

            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                print("Invalid prediction collapse — reverting to last stable prediction")
                prediction = predictions[-1].to(device) if predictions else torch.tensor(0.0, device=device)

            if len(predictions) >= 3:
                prediction = 0.6 * prediction + 0.4 * torch.mean(torch.tensor(predictions[-3:], device=device))

            if len(predictions) >= 20:
                delta = torch.abs(prediction - actual_value.item())
                threshold = 2.5 * torch.std(torch.tensor(predictions[-20:], device=device))
                if delta > threshold:
                    print(f"⚠️ Anomaly detected at index {i}: Δ={float(delta):.4f}, threshold={float(threshold):.4f}")

            arr = torch.tensor(prediction, device=device).squeeze()
            scalar_pred = float(arr[0]) if (arr.ndimension() == 1 and len(arr) == 1) else float(torch.mean(arr))
            prediction = scalar_pred

            if len(predictions) >= 2:
                recent_momentum = predictions[-1] - predictions[-2]
                prediction += 0.1 * recent_momentum

            if len(predictions) >= 5:
                recent_returns = torch.diff(torch.tensor(predictions[-5:], device=device))
                avg_return = torch.mean(recent_returns)
                vol = torch.std(recent_returns) + 1e-8
                sharpe_ratio = avg_return / vol
                prediction *= (1 + 0.02 * sharpe_ratio)

            entropy = entropy_monitor.calculate_entropy(new_data_point)
            entropy_tensor = torch.tensor(entropy, dtype=torch.float32, device=device)
            entropy_penalty = torch.clip(entropy_tensor - 0.8, 0, 1)
            prediction *= (1 - 0.05 * entropy_penalty)

            predictions.append(prediction)
            actuals.append(actual_value.item())

            cimm.give_feedback(data_point, actual_value)

            entropy = entropy_monitor.calculate_entropy(new_data_point)
            entropy_change = torch.tensor(entropy - entropy_monitor.prev_entropy, dtype=torch.float32, device=device)
            clipped_change = torch.clip(entropy_change, -0.01, 0.01)
            entropy_monitor.learning_rate = (0.95 * entropy_monitor.learning_rate) + (0.05 * clipped_change.item())

            if abs(entropy - entropy_monitor.prev_entropy) > 0.03:
                print("Updating model due to significant entropy change")
                anchor_data_device = anchor_data.to(device)
                streaming_actuals_device = streaming_actuals.to(device)
                anchor_data_device = anchor_data_device.contiguous()
                streaming_actuals_device = streaming_actuals_device.contiguous()
                if anchor_data_device.dim() == 1:
                    anchor_data_device = anchor_data_device.unsqueeze(1)
                if streaming_actuals_device.dim() == 1:
                    streaming_actuals_device = streaming_actuals_device.unsqueeze(1)
                streaming_actuals_device = streaming_actuals_device[:anchor_data_device.shape[0]]
                cimm.controller.update_model(anchor_data_device, streaming_actuals_device)

            entropy_monitor.prev_entropy = entropy
            new_lr = entropy_monitor.qbe_learning_rate(optimizer, entropy, entropy_monitor.qbe_baseline, base_lr=0.01)
            scheduler.step(new_lr)

        plot_live_predictions(predictions, actuals)
        validation_data = torch.tensor(streaming_data, dtype=torch.float32, device=device)
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
