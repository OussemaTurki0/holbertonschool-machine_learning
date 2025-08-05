#!/usr/bin/env python3
"""
Test file for sentientPlanets
"""

sentientPlanets = __import__('1-sentience').sentientPlanets

if __name__ == "__main__":
    planets = sentientPlanets()
    for planet in planets:
        print(planet)
