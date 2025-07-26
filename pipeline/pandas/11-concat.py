#!/usr/bin/env python3
"""Concatenates two DataFrames with keys"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concats df2 up to 1417411920 with df1"""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
