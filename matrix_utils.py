# ========== FILE: src/matrix_utils.py ==========
"""Matrix generation utilities"""

import numpy as np
from scipy.linalg import hilbert

def generate_hilbert_matrix(n):
    """Generate Hilbert matrix H where H[i,j] = 1/(i+j+1)"""
    return hilbert(n)

def generate_diagonally_dominant(n, dominance_factor=2.0):
    """Generate diagonally dominant matrix that's guaranteed to converge"""
    A = np.random.randn(n, n)
    
    # Ensure the matrix is symmetric positive definite for better convergence
    A = (A + A.T) / 2  # Make symmetric
    
    for i in range(n):
        # Compute sum of absolute values of off-diagonal elements
        off_diag_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        
        # Set diagonal element to ensure strong diagonal dominance
        A[i, i] = off_diag_sum * dominance_factor + 1.0
        
        # Ensure diagonal is positive (helps convergence)
        if A[i, i] <= 0:
            A[i, i] = 1.0
    
    return A

def generate_spd_matrix(n, alpha=0.1):
    """Generate symmetric positive-definite matrix A = B^T B + alpha*I"""
    B = np.random.randn(n, n)
    A = B.T @ B + alpha * np.eye(n)
    return A

def generate_random_spd(n, alpha=0.1):
    """Alternative SPD matrix generation"""
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric
    A += (np.abs(np.min(np.linalg.eigvals(A))) + alpha) * np.eye(n)  # Make positive definite
    return A