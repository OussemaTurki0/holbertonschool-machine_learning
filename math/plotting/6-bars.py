#!/usr/bin/env python3
"""
This function generates a stacked bar graph
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Requires matplotlib and numpy libraries
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ['Farrah', 'Fred', 'Felicia']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # Red,Yellow,Orange,Peach
    fruits = ['apples', 'bananas', 'oranges', 'peaches']
    bottom = np.zeros(3)
    for idx, row in enumerate(fruit):
        plt.bar(people, row, color=colors[idx], label=fruits[idx],
                bottom=bottom, width=0.5)
        bottom += row
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
