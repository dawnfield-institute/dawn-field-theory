import yfinance as yf
import torch
import numpy as np
import logging
from scipy.signal import butter, filtfilt

def remove_extreme_outliers(data, threshold=3.0):
    """Remove extreme outliers from the data."""
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = data[np.abs(data - mean) < threshold * std_dev]
    return filtered_data

def calculate_rsi(data, window=14):
    logging.info("Calculating RSI.")
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    logging.info("Calculating MACD.")
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def butter_lowpass_filter(data, cutoff, fs, order=5):
    logging.info("Applying Butterworth lowpass filter.")
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def fetch_stock_data(ticker, period="5y", interval="1d"):
    logging.info(f"Fetching stock data for ticker: {ticker}")
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
    
    df['Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log1p(df['Returns'])
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Moving Average'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal Line'] = calculate_macd(df['Close'])
    
    df.dropna(inplace=True)
    
    for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    fourier_transform = butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0)
    
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

def preprocess_stock_data(ticker):
    logging.info(f"Preprocessing stock data for ticker: {ticker}")
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y", interval="1d")
    
    df['Returns'] = df['Close'].pct_change()
    df['Log Returns'] = np.log1p(df['Returns'])
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Moving Average'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['Signal Line'] = calculate_macd(df['Close'])
    
    df.dropna(inplace=True)
    
    for col in ['Returns', 'Log Returns', 'Volatility', 'Moving Average', 'RSI', 'MACD', 'Signal Line']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    fourier_transform = butter_lowpass_filter(df['Close'].values, cutoff=0.1, fs=1.0)
    
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
