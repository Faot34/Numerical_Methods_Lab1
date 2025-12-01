# ========== FILE: src/nonlinear_solvers.py ==========
"""Methods for solving nonlinear equations"""

import numpy as np

def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    """
    Newton's method for root finding
    Returns: (root, iterations, history)
    """
    x = x0
    history = [x]
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-15:
            raise ValueError("Derivative too small")
        
        x_new = x - fx / dfx
        history.append(x_new)
        
        if abs(x_new - x) < tol or abs(f(x_new)) < tol:
            return x_new, i+1, history
        
        x = x_new
    
    return x, max_iter, history

def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    """
    Secant method for root finding
    Returns: (root, iterations, history)
    """
    x_prev = x0
    x = x1
    history = [x0, x1]
    
    for i in range(max_iter):
        fx = f(x)
        fx_prev = f(x_prev)
        
        if abs(fx - fx_prev) < 1e-15:
            raise ValueError("Function values too close")
        
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        history.append(x_new)
        
        if abs(x_new - x) < tol or abs(f(x_new)) < tol:
            return x_new, i+1, history
        
        x_prev = x
        x = x_new
    
    return x, max_iter, history

def fixed_tangent_method(f, df_fixed, x0, tol=1e-10, max_iter=100):
    """
    Fixed tangent method (modified Newton)
    Returns: (root, iterations, history)
    """
    x = x0
    history = [x]
    
    for i in range(max_iter):
        fx = f(x)
        x_new = x - fx / df_fixed
        history.append(x_new)
        
        if abs(x_new - x) < tol or abs(f(x_new)) < tol:
            return x_new, i+1, history
        
        x = x_new
    
    return x, max_iter, history