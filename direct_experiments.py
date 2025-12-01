# ========== FILE: experiments/direct_experiments.py ==========
"""Experiments for direct methods"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from src.matrix_utils import *
from src.direct_methods import *
import time

def test_correctness():
    """Test correctness of direct methods"""
    print("=== Testing Correctness ===")
    
    for n in [10, 50]:
        print(f"\nn = {n}:")
        
        # Generate random system
        A = np.random.randn(n, n)
        A += np.eye(n) * n  # Make diagonally dominant
        x_true = np.random.randn(n)
        b = A @ x_true
        
        # Test Gaussian
        try:
            x_gauss, _, _ = gaussian_elimination(A, b)
            err_gauss = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)
            print(f"  Gaussian error: {err_gauss:.2e}")
        except Exception as e:
            print(f"  Gaussian failed: {e}")
        
        # Test LU
        try:
            L, U = lu_decomposition(A)
            x_lu = solve_lu(L, U, b)
            err_lu = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
            print(f"  LU error: {err_lu:.2e}")
        except Exception as e:
            print(f"  LU failed: {e}")
        
        # Test Cholesky (only for SPD)
        try:
            A_spd = generate_spd_matrix(n)
            x_true_spd = np.random.randn(n)
            b_spd = A_spd @ x_true_spd
            
            L_chol = cholesky_decomposition(A_spd)
            x_chol = solve_cholesky(L_chol, b_spd)
            err_chol = np.linalg.norm(x_chol - x_true_spd) / np.linalg.norm(x_true_spd)
            print(f"  Cholesky error: {err_chol:.2e}")
        except Exception as e:
            print(f"  Cholesky failed: {e}")

def hilbert_matrix_experiment():
    """Experiment with Hilbert matrices"""
    print("\n=== Hilbert Matrix Experiment ===")
    
    sizes = [5, 7, 9, 12]
    results = []
    
    for n in sizes:
        print(f"\nn = {n}:")
        
        # Generate Hilbert matrix
        H = generate_hilbert_matrix(n)
        cond = np.linalg.cond(H)
        print(f"  Condition number: {cond:.2e}")
        
        # Generate solution and RHS
        x_true = np.ones(n)
        b = H @ x_true
        
        # Test methods
        row = {'n': n, 'condition': cond}
        
        # Gaussian
        try:
            x_gauss, _, _ = gaussian_elimination(H, b)
            err_gauss = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)
            row['error_gauss'] = err_gauss
            print(f"  Gaussian error: {err_gauss:.2e}")
        except Exception as e:
            row['error_gauss'] = np.nan
            print(f"  Gaussian failed: {e}")
        
        # LU
        try:
            L, U = lu_decomposition(H)
            x_lu = solve_lu(L, U, b)
            err_lu = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
            row['error_lu'] = err_lu
            print(f"  LU error: {err_lu:.2e}")
        except Exception as e:
            row['error_lu'] = np.nan
            print(f"  LU failed: {e}")
        
        # Cholesky on H^T H
        try:
            A_spd = H.T @ H
            L_chol = cholesky_decomposition(A_spd)
            x_chol = solve_cholesky(L_chol, H.T @ b)
            err_chol = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
            row['error_chol'] = err_chol
            print(f"  Cholesky error: {err_chol:.2e}")
        except Exception as e:
            row['error_chol'] = np.nan
            print(f"  Cholesky failed: {e}")
        
        results.append(row)
    
    return results

def scalability_experiment():
    """Scalability experiment for SPD matrices"""
    print("\n=== Scalability Experiment ===")
    
    sizes = [100, 200, 300, 500]
    results = []
    
    for n in sizes:
        print(f"\nn = {n}:")
        
        # Generate SPD matrix
        A = generate_spd_matrix(n, alpha=0.1)
        x_true = np.random.randn(n)
        b = A @ x_true
        
        row = {'n': n}
        
        # Time Gaussian
        start = time.time()
        x_gauss, _, _ = gaussian_elimination(A, b)
        row['time_gauss'] = time.time() - start
        row['error_gauss'] = np.linalg.norm(x_gauss - x_true) / np.linalg.norm(x_true)
        
        # Time LU
        start = time.time()
        L, U = lu_decomposition(A)
        x_lu = solve_lu(L, U, b)
        row['time_lu'] = time.time() - start
        row['error_lu'] = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
        
        # Time Cholesky
        start = time.time()
        L_chol = cholesky_decomposition(A)
        x_chol = solve_cholesky(L_chol, b)
        row['time_chol'] = time.time() - start
        row['error_chol'] = np.linalg.norm(x_chol - x_true) / np.linalg.norm(x_true)
        
        print(f"  Times: Gaussian={row['time_gauss']:.3f}s, "
              f"LU={row['time_lu']:.3f}s, "
              f"Cholesky={row['time_chol']:.3f}s")
        
        results.append(row)
    
    return results

def plot_direct_results(hilbert_results, scalability_results):
    """Plot results for direct methods"""
    
    # Figure 1: Hilbert matrix errors
    plt.figure(figsize=(10, 6))
    
    n_values = [r['n'] for r in hilbert_results]
    errors_gauss = [r.get('error_gauss', np.nan) for r in hilbert_results]
    errors_lu = [r.get('error_lu', np.nan) for r in hilbert_results]
    errors_chol = [r.get('error_chol', np.nan) for r in hilbert_results]
    
    plt.semilogy(n_values, errors_gauss, 'mo-', label='Gaussian', linewidth=2, markersize=8)
    plt.semilogy(n_values, errors_lu, 'go-', label='LU', linewidth=2, markersize=8)
    plt.semilogy(n_values, errors_chol, 'co-', label='Cholesky', linewidth=2, markersize=8)
    
    plt.xlabel('Matrix Size n')
    plt.ylabel('Relative Error')
    plt.title('Hilbert Matrix: Error vs Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hilbert_errors.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Error vs Condition Number
    plt.figure(figsize=(10, 6))
    
    conditions = [r['condition'] for r in hilbert_results]
    
    plt.loglog(conditions, errors_gauss, 'mo', label='Gaussian', markersize=8)
    plt.loglog(conditions, errors_lu, 'go', label='LU', markersize=8)
    plt.loglog(conditions, errors_chol, 'co', label='Cholesky', markersize=8)
    
    # Add theoretical line
    x_theory = np.logspace(5, 17, 100)
    y_theory = 1e-16 * x_theory  # ε_machine * κ(A)
    plt.loglog(x_theory, y_theory, 'k--', label='Theoretical: ε·κ(A)', alpha=0.7)
    
    plt.xlabel('Condition Number κ(A)')
    plt.ylabel('Relative Error')
    plt.title('Error vs Condition Number')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('error_vs_condition.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Scalability
    plt.figure(figsize=(10, 6))
    
    n_scal = [r['n'] for r in scalability_results]
    times_gauss = [r['time_gauss'] for r in scalability_results]
    times_lu = [r['time_lu'] for r in scalability_results]
    times_chol = [r['time_chol'] for r in scalability_results]
    
    plt.loglog(n_scal, times_gauss, 'mo-', label='Gaussian', linewidth=2, markersize=8)
    plt.loglog(n_scal, times_lu, 'go-', label='LU', linewidth=2, markersize=8)
    plt.loglog(n_scal, times_chol, 'co-', label='Cholesky', linewidth=2, markersize=8)
    
    # Theoretical O(n^3) line
    n_theory = np.array(n_scal)
    time_theory = n_theory**3 * times_gauss[0] / n_theory[0]**3
    plt.loglog(n_theory, time_theory, 'k--', label='O(n³)', alpha=0.7)
    
    plt.xlabel('Matrix Size n')
    plt.ylabel('Execution Time (s)')
    plt.title('Scalability of Direct Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('scalability.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    # Run experiments
    test_correctness()
    hilbert_results = hilbert_matrix_experiment()
    scalability_results = scalability_experiment()
    
    # Save results
    import pandas as pd
    pd.DataFrame(hilbert_results).to_csv('hilbert_results.csv', index=False)
    pd.DataFrame(scalability_results).to_csv('scalability_results.csv', index=False)
    
    # Generate plots
    plot_direct_results(hilbert_results, scalability_results)