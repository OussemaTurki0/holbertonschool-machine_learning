#!/usr/bin/env python3
"""Renames Timestamp column to Datetime"""

import pandas as pd


def rename(df):
    """Renames Timestamp and converts to datetime"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df[["Datetime", "Close"]]
