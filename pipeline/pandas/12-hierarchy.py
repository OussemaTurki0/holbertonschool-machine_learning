#!/usr/bin/env python3
"""Rearranges hierarchy of multi-indexed DataFrame"""

import pandas as pd
index = __import__('10-index').index

def hierarchy(df1, df2):
    """Rearranges MultiIndex so Timestamp is first"""
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.reorder_levels([1, 0])
    return df.sort_index()
