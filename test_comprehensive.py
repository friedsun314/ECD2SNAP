"""
Comprehensive tests for the ECD-SNAP optimization system.
"""

import numpy as np
import jax
import jax.numpy as jnp
from simple_sgd import SimpleSGDOptimizer
from improved_optimizer import ImprovedECDSNAPOptimizer
from snap_targets import make_snap_full_space
from gates import build_ecd_sequence_jax_real
import time


def test_different_snap_targets():
    """Test optimization for different SNAP gate types."""
    print("Testing Different SNAP Targets")
    print("="*60)
    
    N_layers = 4
    N_trunc = 6
    
    test_cases = [
        ("Identity", np.zeros(N_trunc)),
        ("Linear", np.arange(N_trunc) * 0.1),
        ("Quadratic", np.arange(N_trunc)**2 * 0.01),
        ("Random small", np.random.randn(N_trunc) * 0.1),
        ("Periodic", np.sin(np.arange(N_trunc) * np.pi / 3) * 0.2)
    ]
    
    results = {}
    
    for name, phases in test_cases:
        print(f"\n{name} SNAP (phases std={np.std(phases):.3f}):")
        
        # Create target
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        # Try simple SGD
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        params, fidelity = optimizer.optimize(
            U_target_jax, 
            max_iterations=1000,
            momentum=0.9,
            verbose=False
        )
        
        results[name] = fidelity
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} Simple SGD: F = {fidelity:.6f}")
        
        # Show parameter scales
        print(f"     |β| avg = {jnp.mean(jnp.abs(params['betas'])):.4f}")
        print(f"     θ avg = {jnp.mean(jnp.abs(params['thetas'])):.4f}")
    
    return results


def test_layer_scaling():
    """Test how performance scales with number of layers."""
    print("\nTesting Layer Scaling")
    print("="*60)
    
    N_trunc = 4
    phases = np.zeros(N_trunc)  # Identity SNAP
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    layer_counts = [1, 2, 3, 4, 5, 6]
    results = {}
    
    for N_layers in layer_counts:
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        start_time = time.time()
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=500,
            momentum=0.9,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        results[N_layers] = (fidelity, elapsed)
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} Layers={N_layers}: F={fidelity:.6f}, Time={elapsed:.2f}s")
    
    return results


def test_truncation_scaling():
    """Test how performance scales with Fock space truncation."""
    print("\nTesting Truncation Scaling")
    print("="*60)
    
    N_layers = 3
    truncations = [3, 4, 5, 6, 8, 10]
    results = {}
    
    for N_trunc in truncations:
        phases = np.zeros(N_trunc)  # Identity SNAP
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        start_time = time.time()
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=500,
            momentum=0.9,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        results[N_trunc] = (fidelity, elapsed)
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        dim = 2 * N_trunc
        print(f"  {status} N_trunc={N_trunc:2d} (dim={dim:3d}): F={fidelity:.6f}, Time={elapsed:.2f}s")
    
    return results


def test_learning_rate_sensitivity():
    """Test sensitivity to learning rate."""
    print("\nTesting Learning Rate Sensitivity")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    results = {}
    
    for lr in learning_rates:
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=lr)
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=500,
            momentum=0.9,
            verbose=False
        )
        
        results[lr] = fidelity
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} LR={lr:5.3f}: F={fidelity:.6f}")
    
    return results


def test_robustness():
    """Test robustness with multiple random seeds."""
    print("\nTesting Robustness (10 random seeds)")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    fidelities = []
    
    for seed in range(10):
        # Create optimizer with different seed
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        # Override initialization with different seed
        key = jax.random.PRNGKey(seed * 100)
        keys = jax.random.split(key, 4)
        scale = 0.01
        
        betas = (jax.random.normal(keys[0], (N_layers + 1,)) * scale + 
                1j * jax.random.normal(keys[1], (N_layers + 1,)) * scale)
        phis = jax.random.uniform(keys[2], (N_layers + 1,)) * scale
        thetas = jax.random.uniform(keys[3], (N_layers + 1,)) * scale
        
        # Manually set initial params
        optimizer.params = {'betas': betas, 'phis': phis, 'thetas': thetas}
        
        # Run optimization
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=500,
            momentum=0.9,
            verbose=False
        )
        
        fidelities.append(fidelity)
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} Seed {seed:2d}: F={fidelity:.6f}")
    
    fidelities = np.array(fidelities)
    print(f"\n  Statistics:")
    print(f"    Mean:   {np.mean(fidelities):.6f}")
    print(f"    Std:    {np.std(fidelities):.6f}")
    print(f"    Min:    {np.min(fidelities):.6f}")
    print(f"    Max:    {np.max(fidelities):.6f}")
    print(f"    >0.99:  {np.sum(fidelities > 0.99)}/10")
    
    return fidelities


def test_gate_verification():
    """Verify that optimized gates actually implement the target."""
    print("\nVerifying Gate Implementation")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    # Test identity SNAP
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Optimize
    optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
    params, fidelity = optimizer.optimize(
        U_target_jax,
        max_iterations=500,
        momentum=0.9,
        verbose=False
    )
    
    print(f"  Optimization fidelity: {fidelity:.6f}")
    
    # Reconstruct the unitary
    U_approx = build_ecd_sequence_jax_real(
        params['betas'], params['phis'], params['thetas'], N_trunc
    )
    
    # Check unitarity
    U_dag_U = jnp.conj(U_approx.T) @ U_approx
    I = jnp.eye(2 * N_trunc)
    unitarity_error = jnp.max(jnp.abs(U_dag_U - I))
    print(f"  Unitarity error: {unitarity_error:.2e}")
    
    # Check fidelity calculation
    d = U_target_jax.shape[0]
    trace = jnp.trace(jnp.conj(U_target_jax.T) @ U_approx)
    fidelity_check = jnp.abs(trace)**2 / d**2
    print(f"  Fidelity recomputed: {fidelity_check:.6f}")
    
    # Apply to test state |0⟩
    test_state = jnp.zeros(2 * N_trunc, dtype=jnp.complex64)
    test_state = test_state.at[0].set(1.0)  # |0⟩ in cavity ⊗ |g⟩ in qubit
    
    output_target = U_target_jax @ test_state
    output_approx = U_approx @ test_state
    
    state_fidelity = jnp.abs(jnp.conj(output_target) @ output_approx)**2
    print(f"  State |0⟩ fidelity: {state_fidelity:.6f}")
    
    return fidelity > 0.99 and unitarity_error < 1e-6


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE TESTING SUITE")
    print("="*60)
    
    # Run all tests
    print("\n1. Different SNAP Targets")
    snap_results = test_different_snap_targets()
    
    print("\n2. Layer Scaling")
    layer_results = test_layer_scaling()
    
    print("\n3. Truncation Scaling")
    trunc_results = test_truncation_scaling()
    
    print("\n4. Learning Rate Sensitivity")
    lr_results = test_learning_rate_sensitivity()
    
    print("\n5. Robustness")
    robust_results = test_robustness()
    
    print("\n6. Gate Verification")
    verification = test_gate_verification()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # SNAP targets summary
    print("\nSNAP Target Performance:")
    for name, fidelity in snap_results.items():
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} {name:12s}: F={fidelity:.6f}")
    
    # Best configurations
    print("\nBest Configurations:")
    best_layers = max(layer_results.keys(), key=lambda k: layer_results[k][0])
    print(f"  Layers: {best_layers} (F={layer_results[best_layers][0]:.6f})")
    
    best_lr = max(lr_results.keys(), key=lr_results.get)
    print(f"  Learning rate: {best_lr} (F={lr_results[best_lr]:.6f})")
    
    print(f"\nRobustness: {np.mean(robust_results):.4f} ± {np.std(robust_results):.4f}")
    print(f"Gate verification: {'✓ Passed' if verification else '✗ Failed'}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)