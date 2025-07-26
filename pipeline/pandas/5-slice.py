#!/usr/bin/env python3
"""Slices a DataFrame every 60 rows"""

import pandas as pd


def slice(df):
    """Selects every 60th row from specific columns"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]].iloc[::60]
