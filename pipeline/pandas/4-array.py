#!/usr/bin/env python3
"""Converts DataFrame columns to numpy array"""

import pandas as pd


def array(df):
    """Selects last 10 rows of High and Close and converts to numpy"""
    return df[["High", "Close"]].tail(10).to_numpy()
