# app/utils/fake_data_generator.py

from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime

fake = Faker()

def generate_fake_stock_data(n, tickers, start_date, end_date):
    """
    Generate fake stock data similar to real stock data.

    Parameters:
    - n (int): Number of fake data points to generate.
    - tickers (list): List of stock tickers.
    - start_date (str): Start date in 'YYYY-MM-DD'.
    - end_date (str): End date in 'YYYY-MM-DD'.

    Returns:
    - pandas.DataFrame: Fake stock data.
    """
    date_range = pd.date_range(start=start_date, end=end_date)
    fake_data = []
    for _ in range(n):
        ticker = random.choice(tickers)
        date = random.choice(date_range)
        open_price = round(random.uniform(100, 1500), 2)
        high_price = open_price + round(random.uniform(0, 50), 2)
        low_price = open_price - round(random.uniform(0, 50), 2)
        close_price = round(random.uniform(low_price, high_price), 2)
        volume = random.randint(1_000_000, 100_000_000)
        dividends = 0.0  # Assuming no dividends for simplicity
        stock_splits = 0.0  # Assuming no splits for simplicity

        fake_data.append({
            'Date': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume,
            'Dividends': dividends,
            'Stock Splits': stock_splits,
            'Ticker': ticker
        })
    return pd.DataFrame(fake_data)