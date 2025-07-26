#!/usr/bin/env python3
"""Fills missing values appropriately"""

import pandas as pd

def fill(df):
    """Fills missing values and drops Weighted_Price"""
    df = df.drop(columns=["Weighted_Price"])
    df["Close"].fillna(method="ffill", inplace=True)
    for col in ["High", "Low", "Open"]:
        df[col].fillna(df["Close"], inplace=True)
    df["Volume_(BTC)"].fillna(0, inplace=True)
    df["Volume_(Currency)"].fillna(0, inplace=True)
    return df
