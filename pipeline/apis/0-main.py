#!/usr/bin/env python3
"""
Test file for availableShips
"""

availableShips = __import__('0-passengers').availableShips

if __name__ == "__main__":
    ships = availableShips(4)
    for ship in ships:
        print(ship)
