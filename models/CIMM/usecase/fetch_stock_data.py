import yfinance as yf
import pandas as pd
from pathlib import Path

def download_amd_stock_csv(output_path="data/amd_full_history.csv"):
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("🔍 Downloading full historical AMD stock data...")
    df = yf.download(
        tickers="AMD",
        period="max",        # as much as Yahoo will provide (~30+ years)
        interval="1d",       # daily resolution
        auto_adjust=True,    # adjust for splits/dividends
        progress=True
    )

    if df.empty:
        print("❌ Failed to fetch AMD data.")
        return

    df.to_csv(output_path)
    print(f"✅ Saved AMD stock data to: {output_path}")
    print(f"📊 {len(df)} records saved.")

if __name__ == "__main__":
    download_amd_stock_csv()

