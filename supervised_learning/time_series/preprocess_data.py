#!/usr/bin/env python3
"""BTC Price Data Preprocessing for Forecasting"""

import pandas as pd
import numpy as np

def read_and_clean_data(csv_path):
    """
    Load the BTC dataset and clean unnecessary or missing data.
    """
    df = pd.read_csv(csv_path)
    print("Available columns:", df.columns)

    df.dropna(inplace=True)
    df = df[['Timestamp', 'Close']]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    return df

def resample_to_hourly(df):
    """
    Resample data to hourly frequency using the last available price.
    """
    df = df.set_index('Timestamp')
    hourly_df = df.resample('1H').last().dropna()
    return hourly_df

def normalize_series(series):
    """
    Normalize a pandas Series using min-max scaling.
    """
    return (series - series.min()) / (series.max() - series.min())

def generate_sequences(series, window=24):
    """
    Generate rolling window sequences for model input.
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

def preprocess_btc_data(file_path):
    """
    Full pipeline: Load, clean, resample, normalize and sequence data.
    """
    df = read_and_clean_data(file_path)
    hourly_df = resample_to_hourly(df)
    hourly_df['Close'] = normalize_series(hourly_df['Close'])

    X, y = generate_sequences(hourly_df['Close'].values)

    np.save('X.npy', X)
    np.save('y.npy', y)

    print(f"Preprocessing complete. {X.shape[0]} samples saved.")
    return hourly_df

if __name__ == "__main__":
    data_file = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    preprocess_btc_data(data_file)
