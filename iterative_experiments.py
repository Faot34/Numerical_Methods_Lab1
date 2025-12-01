"""Experiments for iterative methods"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.matrix_utils import generate_diagonally_dominant
from src.iterative_methods import jacobi_method, gauss_seidel_method

def convergence_comparison(n=100):
    """Compare convergence of Jacobi and Gauss-Seidel"""
    print("=== Convergence Comparison ===")
    
    A = generate_diagonally_dominant(n, dominance_factor=2.0)
    x_true = np.random.randn(n)
    b = A @ x_true
    x0 = np.zeros(n)
    
    print("\nRunning Jacobi method...")
    x_jacobi, iters_jacobi, res_jacobi, err_jacobi = jacobi_method(A, b, x0)
    
    print("Running Gauss-Seidel method...")
    x_gs, iters_gs, res_gs, err_gs = gauss_seidel_method(A, b, x0)
    
    print(f"\nResults:")
    print(f"  Jacobi: {iters_jacobi} iterations, error: {np.linalg.norm(x_jacobi - x_true):.2e}")
    print(f"  Gauss-Seidel: {iters_gs} iterations, error: {np.linalg.norm(x_gs - x_true):.2e}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(res_jacobi, 'b-', label='Jacobi', linewidth=2)
    plt.semilogy(res_gs, 'r-', label='Gauss-Seidel', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual ||Ax - b||₂')
    plt.title('Residual Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(err_jacobi, 'b-', label='Jacobi', linewidth=2)
    plt.semilogy(err_gs, 'r-', label='Gauss-Seidel', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Error ||x - x*||₂')
    plt.title('Error Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return iters_jacobi, iters_gs

def effect_of_size():
    """Study effect of system size on convergence"""
    print("\n=== Effect of System Size ===")
    
    sizes = [50, 100, 200, 500]
    results = []
    
    for n in sizes:
        print(f"\nn = {n}:")
        
        A = generate_diagonally_dominant(n, dominance_factor=2.0)
        x_true = np.random.randn(n)
        b = A @ x_true
        x0 = np.zeros(n)
        
        x_j, iters_j, _, _ = jacobi_method(A, b, x0)
        x_gs, iters_gs, _, _ = gauss_seidel_method(A, b, x0)
        
        speedup = iters_j / iters_gs if iters_gs > 0 else np.inf
        results.append({'n': n, 'iters_jacobi': iters_j, 'iters_gs': iters_gs, 'speedup': speedup})
        
        print(f"  Jacobi: {iters_j} iterations")
        print(f"  Gauss-Seidel: {iters_gs} iterations")
        if iters_gs > 0:
            print(f"  Speedup: {speedup:.2f}x")
    
    plt.figure(figsize=(10, 6))
    
    n_vals = [r['n'] for r in results]
    iters_j = [r['iters_jacobi'] for r in results]
    iters_gs = [r['iters_gs'] for r in results]
    
    plt.plot(n_vals, iters_j, 'bo-', label='Jacobi', linewidth=2, markersize=8)
    plt.plot(n_vals, iters_gs, 'ro-', label='Gauss-Seidel', linewidth=2, markersize=8)
    
    plt.xlabel('System Size n')
    plt.ylabel('Iterations to Convergence')
    plt.title('Effect of System Size on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('iterative_scalability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def effect_of_diagonal_dominance():
    """Study effect of diagonal dominance"""
    print("\n=== Effect of Diagonal Dominance ===")
    
    n = 100
    dominance_factors = [1.1, 1.5, 2.0, 5.0, 10.0]
    results = []
    
    for factor in dominance_factors:
        print(f"\nDominance factor = {factor}:")
        
        A = generate_diagonally_dominant(n, dominance_factor=factor)
        x_true = np.random.randn(n)
        b = A @ x_true
        x0 = np.zeros(n)
        
        x_j, iters_j, _, _ = jacobi_method(A, b, x0)
        x_gs, iters_gs, _, _ = gauss_seidel_method(A, b, x0)
        
        cond = np.linalg.cond(A)
        results.append({'factor': factor, 'iters_jacobi': iters_j, 'iters_gs': iters_gs, 'cond': cond})
        
        print(f"  Condition: {cond:.2e}")
        print(f"  Jacobi: {iters_j} iterations")
        print(f"  Gauss-Seidel: {iters_gs} iterations")
    
    plt.figure(figsize=(12, 5))
    
    factors = [r['factor'] for r in results]
    iters_j = [r['iters_jacobi'] for r in results]
    iters_gs = [r['iters_gs'] for r in results]
    
    plt.subplot(1, 2, 1)
    plt.semilogy(factors, iters_j, 'bo-', label='Jacobi', linewidth=2, markersize=8)
    plt.semilogy(factors, iters_gs, 'ro-', label='Gauss-Seidel', linewidth=2, markersize=8)
    plt.xlabel('Diagonal Dominance Factor')
    plt.ylabel('Iterations to Convergence')
    plt.title('Effect of Diagonal Dominance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    conds = [r['cond'] for r in results]
    plt.semilogy(factors, conds, 'g-', linewidth=2, markersize=8)
    plt.xlabel('Diagonal Dominance Factor')
    plt.ylabel('Condition Number')
    plt.title('Condition Number vs Dominance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagonal_dominance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def initial_approximation_study():
    """Study effect of initial approximation"""
    print("\n=== Initial Approximation Study ===")
    
    n = 100
    A = generate_diagonally_dominant(n, dominance_factor=2.0)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    initial_guesses = [
        ('Zero', np.zeros(n)),
        ('Random (norm=1)', np.random.randn(n) / np.linalg.norm(np.random.randn(n))),
        ('Near solution', x_true + 0.1 * np.random.randn(n))
    ]
    
    results = []
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, x0) in enumerate(initial_guesses):
        x_gs, iters_gs, _, err_gs = gauss_seidel_method(A, b, x0)
        
        results.append({
            'name': name,
            'initial_error': np.linalg.norm(x0 - x_true),
            'final_error': np.linalg.norm(x_gs - x_true),
            'iterations': iters_gs
        })
        
        print(f"\n{name}:")
        print(f"  Initial error: {results[-1]['initial_error']:.2e}")
        print(f"  Final error: {results[-1]['final_error']:.2e}")
        print(f"  Iterations: {iters_gs}")
        
        plt.subplot(1, 3, i+1)
        plt.semilogy(err_gs, linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Error ||x - x*||₂')
        plt.title(f'{name}\nIterations: {iters_gs}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('initial_approximation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Large system study
    print("\n=== Large System (n=500) Convergence ===")
    n_large = 500
    A_large = generate_diagonally_dominant(n_large, dominance_factor=2.0)
    x_true_large = np.random.randn(n_large)
    b_large = A_large @ x_true_large
    x0_large = np.zeros(n_large)
    
    x_j, iters_j, res_j, err_j = jacobi_method(A_large, b_large, x0_large)
    x_gs, iters_gs, res_gs, err_gs = gauss_seidel_method(A_large, b_large, x0_large)
    
    print(f"\nResults for n={n_large}:")
    print(f"  Jacobi: {iters_j} iterations, error: {np.linalg.norm(x_j - x_true_large):.2e}")
    print(f"  Gauss-Seidel: {iters_gs} iterations, error: {np.linalg.norm(x_gs - x_true_large):.2e}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    iterations_to_show = min(100, len(res_j), len(res_gs))
    plt.semilogy(res_j[:iterations_to_show], 'b-', label='Jacobi', linewidth=2)
    plt.semilogy(res_gs[:iterations_to_show], 'r-', label='Gauss-Seidel', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Residual ||Ax - b||₂')
    plt.title(f'Large System Convergence (n={n_large})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(err_j[:iterations_to_show], 'b-', label='Jacobi', linewidth=2)
    plt.semilogy(err_gs[:iterations_to_show], 'r-', label='Gauss-Seidel', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Error ||x - x*||₂')
    plt.title(f'Large System Error (n={n_large})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('large_system_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    convergence_comparison()
    size_results = effect_of_size()
    dominance_results = effect_of_diagonal_dominance()
    initial_results = initial_approximation_study()
    
    pd.DataFrame(size_results).to_csv('size_results.csv', index=False)
    pd.DataFrame(dominance_results).to_csv('dominance_results.csv', index=False)