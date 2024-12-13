#!/usr/bin/env python3

"""
integration functions
"""
def poly_integral(poly, C=0):
    """
    Check validity of poly and C, and handle empty polyy
    """
    if (not isinstance(poly, list) or
            not isinstance(C, (int, float)) or
            not poly):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None
    integral = [C]
    for i in range(len(poly)):
        integral.append(poly[i] / (i + 1))
    integral = [int(x) if isinstance(x, float) and x.is_integer()
                else x for x in integral]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
