"""Experiments for nonlinear equations"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.nonlinear_solvers import newton_method, secant_method, fixed_tangent_method

def f1(x): return x**2 - 2
def df1(x): return 2*x
def f2(x): return np.cos(x) - x
def df2(x): return -np.sin(x) - 1
def f3(x): return np.exp(-x) - x
def df3(x): return -np.exp(-x) - 1

def test_convergence():
    """Test convergence of all methods"""
    print("=== Testing Convergence ===")
    
    tests = [
        ('f1: x² - 2', f1, df1, 1.0, 1.4, df1(1.0), np.sqrt(2)),
        ('f2: cos(x) - x', f2, df2, 1.0, 1.1, df2(1.0), 0.7390851332151607),
        ('f3: exp(-x) - x', f3, df3, 0.5, 0.56, df3(0.5), 0.5671432904097838)
    ]
    
    results = []
    
    for name, f, df, x0_newton, x1_secant, df_fixed, true_root in tests:
        print(f"\n{name}:")
        print("-" * 40)
        
        root_newton, iters_newton, hist_newton = newton_method(f, df, x0_newton)
        root_secant, iters_secant, hist_secant = secant_method(f, x0_newton, x1_secant)
        root_fixed, iters_fixed, hist_fixed = fixed_tangent_method(f, df_fixed, x0_newton)
        
        print(f"Newton: {iters_newton} iterations")
        print(f"Secant: {iters_secant} iterations")
        print(f"Fixed: {iters_fixed} iterations")
        
        errors_newton = [abs(x - true_root) for x in hist_newton]
        errors_secant = [abs(x - true_root) for x in hist_secant]
        errors_fixed = [abs(x - true_root) for x in hist_fixed]
        
        results.append({
            'function': name,
            'newton': {'errors': errors_newton},
            'secant': {'errors': errors_secant},
            'fixed': {'errors': errors_fixed}
        })
    
    return results

def plot_convergence(results):
    """Plot convergence for all methods"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, result in enumerate(results):
        ax = axes[i]
        
        errors_newton = result['newton']['errors']
        errors_secant = result['secant']['errors']
        errors_fixed = result['fixed']['errors']
        
        ax.semilogy(errors_newton, 'ro-', label='Newton', linewidth=2)
        ax.semilogy(errors_secant, 'go-', label='Secant', linewidth=2)
        ax.semilogy(errors_fixed, 'bo-', label='Fixed', linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error |x - x*|')
        ax.set_title(result['function'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nonlinear_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Convergence rates for f1
    plt.figure(figsize=(10, 6))
    
    errors_newton = results[0]['newton']['errors']
    errors_secant = results[0]['secant']['errors']
    errors_fixed = results[0]['fixed']['errors']
    
    plt.semilogy(errors_newton, 'ro-', label='Newton', linewidth=2)
    plt.semilogy(errors_secant, 'go-', label='Secant', linewidth=2)
    plt.semilogy(errors_fixed, 'bo-', label='Fixed', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Error |x - x*|')
    plt.title('Convergence Rates Comparison (f1: x² - 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_rates.png', dpi=300, bbox_inches='tight')
    plt.show()

def sensitivity_analysis():
    """Analyze sensitivity to initial approximation"""
    print("\n=== Sensitivity Analysis ===")
    
    functions = [
        ('f1', f1, df1, np.sqrt(2)),
        ('f2', f2, df2, 0.7390851332151607),
        ('f3', f3, df3, 0.5671432904097838)
    ]
    
    plt.figure(figsize=(12, 4))
    
    for idx, (name, f, df, true_root) in enumerate(functions):
        iterations_needed = []
        x0_values = np.linspace(0.1, 3.0, 50)
        
        for x0 in x0_values:
            try:
                root, iters, _ = newton_method(f, df, x0, max_iter=50)
                iterations_needed.append(iters if abs(root - true_root) < 1e-6 else 50)
            except:
                iterations_needed.append(50)
        
        plt.subplot(1, 3, idx+1)
        plt.scatter(x0_values, iterations_needed, alpha=0.6, s=20)
        plt.plot(x0_values, iterations_needed, 'b-', alpha=0.5)
        plt.xlabel('Initial Guess x₀')
        plt.ylabel('Iterations Needed')
        plt.title(f'{name}')
        plt.grid(True, alpha=0.3)
        
        valid_iters = [it for it in iterations_needed if it < 50]
        if valid_iters:
            avg_iters = np.mean(valid_iters)
            print(f"{name}: Avg iterations = {avg_iters:.1f}")
    
    plt.tight_layout()
    plt.savefig('initial_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = test_convergence()
    plot_convergence(results)
    sensitivity_analysis()
    
    print("\n=== Experiments Completed ===")