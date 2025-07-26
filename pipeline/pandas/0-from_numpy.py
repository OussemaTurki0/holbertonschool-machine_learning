#!/usr/bin/env python3
"""Creates a DataFrame from a NumPy array"""

import pandas as pd

def from_numpy(array):
    """Creates a pd.DataFrame from a np.ndarray"""
    columns = [chr(i) for i in range(65, 65 + array.shape[1])]
    return pd.DataFrame(array, columns=columns)
