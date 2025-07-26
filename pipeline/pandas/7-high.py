#!/usr/bin/env python3
"""Sorts by High column"""

import pandas as pd

def high(df):
    """Sorts DataFrame by High descending"""
    return df.sort_values(by="High", ascending=False)
