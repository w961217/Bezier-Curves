# bezier_curves.py
import numpy as np
from scipy.special import comb

def bernstein_poly(i, n, t):
    """Compute Bernstein polynomial of n, i as a function of t"""
    return comb(n, i) * ( t**(n-i) ) * ( (1-t)**i )

def bezier_curve(points, n=1000):
    """Compute n points on a Bézier curve defined by control points"""
    N = len(points) - 1
    t = np.linspace(0.0, 1.0, n)
    curve = np.zeros((n, 2))
    for i in range(N+1):
        curve += np.outer(bernstein_poly(N-i, N, t), points[i])
    return curve
