import os
import pandas as pd
import yfinance as yf


def load_prices(tickers, start_date, end_date):

    cache_file = "data/prices.csv"

    if os.path.exists(cache_file):

        print("Loading cached data...")

        prices = pd.read_csv(
            cache_file,
            index_col=0,
            parse_dates=True
        )

    else:

        print("Downloading data from Yahoo Finance...")

        prices = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )["Close"]

        prices.to_csv(cache_file)

    return prices