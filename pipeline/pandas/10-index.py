#!/usr/bin/env python3
"""Sets Timestamp as index"""

import pandas as pd


def index(df):
    """Sets Timestamp column as index"""
    return df.set_index("Timestamp")
