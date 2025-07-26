#!/usr/bin/env python3
"""Reverses and transposes a DataFrame"""

import pandas as pd

def flip_switch(df):
    """Reverses chronologically and transposes"""
    return df.sort_index(ascending=False).transpose()
