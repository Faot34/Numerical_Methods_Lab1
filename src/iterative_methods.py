# ========== FILE: src/iterative_methods.py ==========
"""Iterative methods for solving SLAE"""

import numpy as np

def jacobi_method(A, b, x0=None, tol=1e-8, max_iter=10000):
    """
    Jacobi iterative method
    Returns: (solution, iterations, residuals, errors)
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    residuals = []
    errors = []
    
    for k in range(max_iter):
        x_new = (b - R @ x) / D
        residual = np.linalg.norm(A @ x_new - b)
        error = np.linalg.norm(x_new - x)
        
        residuals.append(residual)
        errors.append(error)
        x = x_new
        
        if residual < tol:
            break
    
    return x, k+1, residuals, errors

def gauss_seidel_method(A, b, x0=None, tol=1e-8, max_iter=10000):
    """
    Gauss-Seidel iterative method
    Returns: (solution, iterations, residuals, errors)
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    residuals = []
    errors = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        residual = np.linalg.norm(A @ x - b)
        error = np.linalg.norm(x - x_old)
        
        residuals.append(residual)
        errors.append(error)
        
        if residual < tol:
            break
    
    return x, k+1, residuals, errors