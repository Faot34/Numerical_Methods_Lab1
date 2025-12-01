# ========== FILE: src/direct_methods.py ==========
"""Direct methods for solving SLAE"""

import numpy as np
import math

def gaussian_elimination(A, b, pivoting=True):
    """
    Gaussian elimination with optional partial pivoting
    Returns: (x, A_modified, b_modified)
    """
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    
    # Forward elimination
    for k in range(n-1):
        if pivoting:
            # Partial pivoting
            pivot_row = k + np.argmax(np.abs(A[k:, k]))
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                b[[k, pivot_row]] = b[[pivot_row, k]]
        
        # Elimination
        for i in range(k+1, n):
            if A[k, k] == 0:
                raise ValueError("Zero pivot encountered")
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Zero diagonal element")
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x, A, b

def lu_decomposition(A):
    """LU decomposition without pivoting"""
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j >= i:  # Upper triangular
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:   # Lower triangular (excluding diagonal)
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
    
    return L, U

def solve_lu(L, U, b):
    """Solve system using LU decomposition"""
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    
    # Back substitution: Ux = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

def cholesky_decomposition(A):
    """Cholesky decomposition for symmetric positive-definite matrices"""
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        # Diagonal element
        s = A[i, i] - np.sum(L[i, :i]**2)
        if s <= 0:
            raise ValueError("Matrix is not positive definite")
        L[i, i] = math.sqrt(s)
        
        # Off-diagonal elements
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], L[i, :i])) / L[i, i]
    
    return L

def solve_cholesky(L, b):
    """Solve system using Cholesky decomposition"""
    n = len(b)
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    
    # Back substitution: L^T x = y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(L[i+1:, i], x[i+1:])) / L[i, i]
    
    return x