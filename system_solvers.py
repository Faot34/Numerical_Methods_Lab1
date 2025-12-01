# ========== FILE: src/system_solvers.py ==========
"""Methods for solving nonlinear systems"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def newton_system(F, J, x0, tol=1e-8, max_iter=50):
    """
    Newton's method for nonlinear systems
    Returns: (solution, iterations, history)
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iter):
        F_val = F(x)
        J_val = J(x)
        
        residual = np.linalg.norm(F_val)
        if residual < tol:
            return x, k+1, history
        
        # Solve J * delta = -F
        try:
            lu, piv = lu_factor(J_val)
            delta = lu_solve((lu, piv), -F_val)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(J_val, -F_val)
        
        x = x + delta
        history.append(x.copy())
    
    return x, max_iter, history

# Test systems
def system1(x):
    """System 1: sin(x) + y^2 - 1 = 0, x^2 - cos(y) = 0"""
    return np.array([
        np.sin(x[0]) + x[1]**2 - 1,
        x[0]**2 - np.cos(x[1])
    ])

def jacobian1(x):
    """Jacobian for system 1"""
    return np.array([
        [np.cos(x[0]), 2*x[1]],
        [2*x[0], np.sin(x[1])]
    ])

def system2(x):
    """System 2: x^2 + y^2 - 4 = 0, e^x + y - 1 = 0"""
    return np.array([
        x[0]**2 + x[1]**2 - 4,
        np.exp(x[0]) + x[1] - 1
    ])

def jacobian2(x):
    """Jacobian for system 2"""
    return np.array([
        [2*x[0], 2*x[1]],
        [np.exp(x[0]), 1]
    ])