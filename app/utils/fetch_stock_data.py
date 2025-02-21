# app/utils/fetch_stock_data.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_stock_data(tickers, start_date, end_date):
    all_data = pd.DataFrame()
    for ticker in tickers:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            print(f"No data found for {ticker}. Skipping...")
            continue
        hist.reset_index(inplace=True)
        hist['Ticker'] = ticker
        all_data = pd.concat([all_data, hist], ignore_index=True)
    return all_data

def save_stock_data(df, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Stock data saved to '{file_path}'")