"""
Compare different optimization methods head-to-head.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from scripts.simple_sgd import SimpleSGDOptimizer
from src.improved_optimizer import ImprovedECDSNAPOptimizer
from src.optimizer import ECDSNAPOptimizer
from src.snap_targets import make_snap_full_space
import time


def compare_methods():
    """Compare all optimization methods on the same problems."""
    print("METHOD COMPARISON")
    print("="*60)
    
    N_layers = 4
    N_trunc = 6
    batch_size = 16
    
    # Test problems
    problems = [
        ("Identity", np.zeros(N_trunc)),
        ("Small Linear", np.arange(N_trunc) * 0.05),
        ("Small Random", np.random.RandomState(42).randn(N_trunc) * 0.05)
    ]
    
    methods = []
    
    # Method 1: Simple SGD
    methods.append(("Simple SGD", lambda U_target: run_simple_sgd(U_target, N_layers, N_trunc)))
    
    # Method 2: Original optimizer with Adam
    methods.append(("Original Adam", lambda U_target: run_original_adam(U_target, N_layers, N_trunc, batch_size)))
    
    # Method 3: Improved with smart init
    methods.append(("Smart Init Adam", lambda U_target: run_smart_init(U_target, N_layers, N_trunc, batch_size)))
    
    # Method 4: Improved with restarts
    methods.append(("Restart Strategy", lambda U_target: run_restarts(U_target, N_layers, N_trunc, batch_size)))
    
    # Results table
    results = {}
    
    for prob_name, phases in problems:
        print(f"\n{prob_name} SNAP Gate:")
        print("-" * 40)
        
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        results[prob_name] = {}
        
        for method_name, method_func in methods:
            fidelity, time_taken = method_func(U_target_jax)
            results[prob_name][method_name] = (fidelity, time_taken)
            
            status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
            print(f"  {status} {method_name:15s}: F={fidelity:.6f} (t={time_taken:.2f}s)")
    
    return results


def run_simple_sgd(U_target, N_layers, N_trunc):
    """Run simple SGD optimizer."""
    optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
    
    start = time.time()
    params, fidelity = optimizer.optimize(
        U_target,
        max_iterations=200,
        momentum=0.9,
        verbose=False
    )
    elapsed = time.time() - start
    
    return fidelity, elapsed


def run_original_adam(U_target, N_layers, N_trunc, batch_size):
    """Run original Adam optimizer."""
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.003
    )
    
    start = time.time()
    params, fidelity, _ = optimizer.optimize(
        U_target,
        max_iterations=200,
        verbose=False
    )
    elapsed = time.time() - start
    
    return fidelity, elapsed


def run_smart_init(U_target, N_layers, N_trunc, batch_size):
    """Run improved optimizer with smart initialization."""
    optimizer = ImprovedECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.003
    )
    
    key = jax.random.PRNGKey(42)
    optimizer.params = optimizer.smart_initialize(key, "identity")
    optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
    
    start = time.time()
    params, fidelity, _ = optimizer.optimize(
        U_target,
        max_iterations=200,
        verbose=False
    )
    elapsed = time.time() - start
    
    return fidelity, elapsed


def run_restarts(U_target, N_layers, N_trunc, batch_size):
    """Run improved optimizer with restarts."""
    optimizer = ImprovedECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.003
    )
    
    start = time.time()
    params, fidelity, _ = optimizer.optimize_with_restarts(
        U_target,
        max_iterations=200,
        n_restarts=3,
        target_type="identity",
        verbose=False
    )
    elapsed = time.time() - start
    
    return fidelity, elapsed


def test_scaling():
    """Test how methods scale with problem size."""
    print("\nSCALING TEST")
    print("="*60)
    
    # Test different truncations
    truncations = [3, 4, 5, 6]
    N_layers = 3
    
    print("\nSimple SGD scaling with truncation:")
    print("-" * 40)
    
    for N_trunc in truncations:
        phases = np.zeros(N_trunc)
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        start = time.time()
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=100,
            momentum=0.9,
            verbose=False
        )
        elapsed = time.time() - start
        
        dim = 2 * N_trunc
        status = "✓" if fidelity > 0.99 else "○"
        print(f"  {status} N={N_trunc} (dim={dim:2d}): F={fidelity:.6f}, t={elapsed:.2f}s")


def test_parameter_analysis():
    """Analyze the optimized parameters."""
    print("\nPARAMETER ANALYSIS")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    # Optimize identity
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
    params, fidelity = optimizer.optimize(U_target_jax, max_iterations=200, verbose=False)
    
    print(f"Identity SNAP (F={fidelity:.6f}):")
    print(f"  Displacement amplitudes |β|:")
    for i, beta in enumerate(params['betas']):
        print(f"    Layer {i}: |β|={np.abs(beta):.4f}, arg(β)={np.angle(beta):.4f}")
    
    print(f"  Qubit rotations:")
    for i in range(len(params['phis'])):
        print(f"    Layer {i}: φ={params['phis'][i]:.4f}, θ={params['thetas'][i]:.4f}")
    
    # Compare to linear phase
    phases = np.arange(N_trunc) * 0.05
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    params, fidelity = optimizer.optimize(U_target_jax, max_iterations=200, verbose=False)
    
    print(f"\nLinear SNAP (F={fidelity:.6f}):")
    print(f"  Average |β|: {np.mean(np.abs(params['betas'])):.4f}")
    print(f"  Std |β|: {np.std(np.abs(params['betas'])):.4f}")
    print(f"  Average θ: {np.mean(np.abs(params['thetas'])):.4f}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPTIMIZATION METHOD COMPARISON")
    print("="*60)
    
    # Run comparisons
    results = compare_methods()
    
    # Scaling test
    test_scaling()
    
    # Parameter analysis
    test_parameter_analysis()
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    print("\nFidelity Comparison:")
    print("-" * 60)
    print(f"{'Method':<20} {'Identity':>10} {'Linear':>10} {'Random':>10}")
    print("-" * 60)
    
    method_names = ["Simple SGD", "Original Adam", "Smart Init Adam", "Restart Strategy"]
    for method in method_names:
        row = f"{method:<20}"
        for prob in ["Identity", "Small Linear", "Small Random"]:
            if prob in results and method in results[prob]:
                f, t = results[prob][method]
                row += f" {f:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)