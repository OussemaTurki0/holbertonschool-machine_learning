#!/usr/bin/env python3
"""Removes rows with NaN in Close"""

import pandas as pd


def prune(df):
    """Removes NaN rows in Close column"""
    return df[df["Close"].notna()]
