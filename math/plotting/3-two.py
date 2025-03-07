#!/usr/bin/env python3
"""
code to plot x -> y1 and x -> y2 as line graphs
Plots the exponential decay of two radioactive elements, C-14 and Ra-226,
as a function of time with differently styled line graphs.

The function uses NumPy to generate an array of time points and computes
the corresponding decay values using the exponential decay formula:
N(t) = N0 * exp((ln(0.5) / half_life) * t)
Usage:
This function is intended to be run within a Python environment where
matplotlib is installed and configured to display graphs. It can be
executed directly via a command line or through a driver script that imports
and calls the function.
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    The plot includes:
    - X-axis 'Time (years)'
    - Y-axis 'Fraction Remaining'
    - Title 'Exponential Decay of Radioactive Elements'
    - X-axis range from 0 to 20,000 years
    - Y-axis range from 0 to 1 (normalized decay fraction)
    - A dashed red line representing the decay of C-14
    - A solid green line representing the decay of Ra-226
    - A legend in the upper right corner indicating the element
    - each line represents
    """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc='upper right')
    plt.show()
