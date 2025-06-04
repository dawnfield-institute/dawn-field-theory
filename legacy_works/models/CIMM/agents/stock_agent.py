import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import numpy as np
import yfinance as yf
from cimm_core.cimm_core_manager import CIMMCoreManager
from usecase.stock_prediction_usecase import StockPredictionUseCase
from skopt.space import Real, Integer
from agents.base_agent import BaseAgent
from cimm_core.models.financial_model import FinancialModel
import numpy as np
import torch
from skopt.space import Real, Integer
import logging
from scipy.signal import butter, filtfilt

class StockPredictionAgent(BaseAgent):
    """
    Agent for predicting stock market trends using entropy-aware intelligence.
    """

    def __init__(self, manager, hidden_size=64):
        stock_data = self.preprocess_stock_data("AMD")
        actual_values = yf.Ticker("AMD").history(period="10y", interval="1d")["Close"].values

        anchor_size = int(0.1 * len(stock_data))
        anchor_data = stock_data[:anchor_size]
        self.streaming_data = stock_data[anchor_size:]
        self.streaming_actuals = actual_values[anchor_size:]
        param_space = [
            Real(1e-5, 1e-1, name='learning_rate'),
            Integer(10, 100, name='hidden_size'),
        ]
        # FIX: Ensure input_size=4 is used
        super().__init__("StockPredictorAI", FinancialModel, param_space, manager, anchor_data, hidden_size)

    def preprocess_stock_data(self, ticker):
        print(f"Preprocessing stock data for ticker: {ticker}")
        stock = yf.Ticker(ticker)
        df = stock.history(period="9y", interval="1d")
        
        df['Returns'] = df['Close'].pct_change()
        df['Log Returns'] = np.log1p(df['Returns'])
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Moving Average'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['Signal Line'] = self.calculate_macd(df['Close'])
        
        df.dropna(inplace=True)
        
        for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        fourier_transform = self.butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0)
        
        num_samples = (len(df) // 10) * 10
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
    def calculate_rsi(self, data, window=14):
        logging.info("Calculating RSI.")
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        logging.info("Calculating MACD.")
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        logging.info("Applying Butterworth lowpass filter.")
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def predict_stock(self, ticker):
        """
        Runs stock market prediction using agentic CIMM intelligence.
        """
        predictions = []
        smoothing_factor = 0.1
        for i, new_data_point in enumerate(self.streaming_data):
            if i >= len(self.streaming_actuals):
                break
            

            if len(predictions) >= 10:
                recent_predictions = np.array(predictions[-10:])
                smoothing_factor = max(0.05, min(0.3, np.std(recent_predictions) / (np.mean(recent_predictions) + 1e-6)))

            smoothed_data_point = smoothing_factor * new_data_point + (1 - smoothing_factor) * (predictions[-1] if predictions else new_data_point)

            smoothed_data_point = torch.abs(smoothed_data_point)  # Take absolute value to remove imaginary component
            smoothed_data_point = torch.log(torch.clamp(smoothed_data_point + 1, min=1e-8))

            prediction = self.agent_instance.run(smoothed_data_point)  # FIX: Use wrapper method
            predictions.append(prediction)
            actual_value = self.streaming_actuals[i]  # Ground truth stock return

            # Compute error (reward function)
            reward_signal = -abs(prediction.item() - actual_value.item())  # Lower error = higher reward

            # Provide feedback to model
            self.agent_instance.give_feedback(smoothed_data_point, actual_value)

            # Reinforcement update
            self.agent_instance.reinforcement_update(reward_signal, entropy_level=0.02)
        self.store_quantum_state(self.streaming_data.numpy().tolist(), qbe_feedback=0.002)
        return self.retrieve_quantum_state()

    def periodic_synchronization(self):
        """
        Triggers entropy synchronization every few training steps.
        """
        return self.manager.synchronize_entropy()
    
# Example usage
if __name__ == "__main__":
    manager = CIMMCoreManager()
    stock_agent = StockPredictionAgent(manager)
    stock_predictions = stock_agent.predict_stock("AMD")
    metrics = stock_agent.agent_instance.evaluate_model(stock_agent.streaming_data)
    print(f"Error Metrics: {metrics}")
