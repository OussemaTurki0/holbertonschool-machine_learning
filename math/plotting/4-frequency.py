#!/usr/bin/env python3
"""
This script creates and displays a histogram representing the distribution of 
student grades for "Project A". The grades are simulated using a normal 
distribution, making it easy to visualize how students performed.

The histogram divides grades into bins spanning 0 to 100, with each bin 
representing a range of 10 points. The bars of the histogram are outlined 
to improve clarity and make the distribution easier to interpret.

Key Features:
- Simulated grades follow a normal distribution.
- Grades are grouped into bins of 10.
- Each bar in the histogram has a black edge for better visibility.

Usage:
To run this script, no additional arguments are needed. Simply execute it 
directly as an executable with `./4-frequency.py` (if permissions are set) or 
by using `python3 4-frequency.py`.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Generates a histogram of student grades and customizes the plot.

    Attributes:
    - student_grades (numpy.ndarray): Array containing randomly generated grades.
    - bins (list): Defines the grade ranges for the histogram.

    Steps:
    1. Simulate grades using a normal distribution with mean=68 and std=15.
    2. Divide grades into bins from 0 to 100 (increments of 10).
    3. Create the histogram and add visual enhancements such as axis labels,
       a title, and bar edges.
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins = np.arange(0, 101, 10)
    plt.xlabel('Grades')
    plt.ylim(0, 30)
    plt.xlim(0, 100)
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins, edgecolor='black')
    plt.xticks(np.arange(0, 110, 10))
    plt.show()
