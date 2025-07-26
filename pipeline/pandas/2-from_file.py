#!/usr/bin/env python3
"""Loads data from a CSV file into a DataFrame"""

import pandas as pd


def from_file(filename, delimiter):
    """Loads data from a file"""
    return pd.read_csv(filename, delimiter=delimiter)
