"""
Test different optimization strategies on the same problems.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from scripts.simple_sgd import SimpleSGDOptimizer
from src.optimizer import ECDSNAPOptimizer
from src.snap_targets import make_snap_full_space
import time


def compare_strategies():
    """Compare different optimization strategies on the same problems."""
    print("STRATEGY COMPARISON")
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
    
    strategies = [
        ("Basic", lambda opt, U_target, target_type: run_basic(opt, U_target, target_type)),
        ("Restarts", lambda opt, U_target, target_type: run_restarts(opt, U_target, target_type)),
        ("Annealing", lambda opt, U_target, target_type: run_annealing(opt, U_target, target_type)),
        ("Two-Stage", lambda opt, U_target, target_type: run_two_stage(opt, U_target, target_type))
    ]
    
    # Results table
    results = {}
    
    for prob_name, phases in problems:
        print(f"\n{prob_name} Problem:")
        print("-" * 40)
        
        # Create target
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        # Determine target type for smart initialization
        if np.allclose(phases, 0):
            target_type = "identity"
        elif np.allclose(phases, np.arange(N_trunc) * phases[1] if len(phases) > 1 else 0):
            target_type = "linear"
        else:
            target_type = "general"
        
        results[prob_name] = {}
        
        for strat_name, strat_func in strategies:
            # Create fresh optimizer for each strategy
            optimizer = ECDSNAPOptimizer(
                N_layers=N_layers,
                N_trunc=N_trunc,
                batch_size=batch_size,
                learning_rate=0.003
            )
            
            start_time = time.time()
            fidelity = strat_func(optimizer, U_target_jax, target_type)
            elapsed = time.time() - start_time
            
            results[prob_name][strat_name] = {
                'fidelity': fidelity,
                'time': elapsed
            }
            
            status = "✓" if fidelity >= 0.99 else "○" if fidelity >= 0.9 else "✗"
            print(f"  {strat_name:15s}: F = {fidelity:.6f} | t = {elapsed:.2f}s | {status}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("-"*60)
    
    # Find best strategy for each problem
    for prob_name in results:
        best_strat = max(results[prob_name].items(), key=lambda x: x[1]['fidelity'])
        print(f"{prob_name:15s}: Best = {best_strat[0]} (F = {best_strat[1]['fidelity']:.6f})")
    
    # Average fidelity per strategy
    print("\nAverage Fidelity by Strategy:")
    for strat_name, _ in strategies:
        avg_fid = np.mean([results[p][strat_name]['fidelity'] for p in results])
        print(f"  {strat_name:15s}: {avg_fid:.6f}")


def run_basic(optimizer, U_target, target_type):
    """Run basic optimization with smart initialization."""
    key = jax.random.PRNGKey(42)
    optimizer.params = optimizer.smart_initialize(key, target_type)
    optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
    _, fidelity, _ = optimizer.optimize(U_target, max_iterations=500, verbose=False)
    return fidelity


def run_restarts(optimizer, U_target, target_type):
    """Run optimization with restarts strategy."""
    _, fidelity, _ = optimizer.optimize_with_restarts(
        U_target, max_iterations=500, n_restarts=3, 
        target_type=target_type, verbose=False
    )
    return fidelity


def run_annealing(optimizer, U_target, target_type):
    """Run optimization with annealing strategy."""
    _, fidelity, _ = optimizer.optimize_with_annealing(
        U_target, max_iterations=500, 
        target_type=target_type, verbose=False
    )
    return fidelity


def run_two_stage(optimizer, U_target, target_type):
    """Run two-stage optimization strategy."""
    _, fidelity, _ = optimizer.optimize_two_stage(
        U_target, max_iterations=500,
        target_type=target_type, verbose=False
    )
    return fidelity


def test_simple_sgd():
    """Test simple SGD for comparison."""
    print("\nSIMPLE SGD BASELINE")
    print("="*60)
    
    N_layers = 4
    N_trunc = 6
    
    problems = [
        ("Identity", np.zeros(N_trunc)),
        ("Small Linear", np.arange(N_trunc) * 0.05)
    ]
    
    for prob_name, phases in problems:
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        start_time = time.time()
        params, fidelity = optimizer.optimize(
            U_target_jax, 
            max_iterations=1000,
            momentum=0.9,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        status = "✓" if fidelity >= 0.99 else "○" if fidelity >= 0.9 else "✗"
        print(f"{prob_name:15s}: F = {fidelity:.6f} | t = {elapsed:.2f}s | {status}")


if __name__ == "__main__":
    compare_strategies()
    test_simple_sgd()
    
    print("\n" + "="*60)
    print("All strategy tests completed!")