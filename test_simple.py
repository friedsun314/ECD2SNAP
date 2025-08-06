"""
Simple test script to verify the ECD-to-SNAP optimization works.
Tests with an identity SNAP gate which should be easy to approximate.
"""

import numpy as np
import jax.numpy as jnp
from optimizer import ECDSNAPOptimizer
from snap_targets import identity_snap, linear_snap, make_snap_full_space
from gates import build_ecd_sequence
from viz import plot_convergence, visualize_gate_sequence
import matplotlib.pyplot as plt


def test_identity_snap():
    """Test optimization with identity SNAP gate."""
    print("Testing ECD-to-SNAP optimization with identity target...")
    
    # Parameters
    N_layers = 4  # Fewer layers for identity
    N_trunc = 6   # Smaller truncation for testing
    batch_size = 16
    
    # Create identity SNAP target in full space
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    print(f"Target: Identity SNAP gate")
    print(f"Layers: {N_layers}, Truncation: {N_trunc}, Batch: {batch_size}")
    
    # Initialize optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=5e-3,
        target_fidelity=0.99
    )
    
    # Run optimization
    print("\nStarting optimization...")
    best_params, best_fidelity, info = optimizer.optimize(
        U_target_jax,
        max_iterations=1000,
        verbose=True
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best fidelity: {best_fidelity:.6f}")
    print(f"  Iterations: {info['iterations']}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    plot_convergence(info['history'])
    
    visualize_gate_sequence(
        np.array(best_params['betas']),
        np.array(best_params['phis']),
        np.array(best_params['thetas'])
    )
    
    return best_fidelity > 0.99


def test_linear_snap():
    """Test optimization with linear phase SNAP gate."""
    print("\n" + "="*50)
    print("Testing ECD-to-SNAP optimization with linear phase target...")
    
    # Parameters
    N_layers = 6
    N_trunc = 8
    batch_size = 24
    slope = 0.5  # Linear phase slope
    
    # Create linear SNAP target
    phases = slope * np.arange(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    print(f"Target: Linear SNAP gate (slope={slope})")
    print(f"Phases: {phases[:5]}... (showing first 5)")
    print(f"Layers: {N_layers}, Truncation: {N_trunc}, Batch: {batch_size}")
    
    # Initialize optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=3e-3,
        target_fidelity=0.99
    )
    
    # Run optimization
    print("\nStarting optimization...")
    best_params, best_fidelity, info = optimizer.optimize(
        U_target_jax,
        max_iterations=2000,
        verbose=True
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best fidelity: {best_fidelity:.6f}")
    print(f"  Iterations: {info['iterations']}")
    
    # Check parameter values
    print("\nOptimized parameters (first layer):")
    print(f"  |β_0| = {np.abs(best_params['betas'][0]):.4f}")
    print(f"  θ_0 = {best_params['thetas'][0]:.4f} rad")
    print(f"  φ_0 = {best_params['phis'][0]:.4f} rad")
    
    return best_fidelity > 0.95


def test_gradient_flow():
    """Test that gradients flow properly through the optimization."""
    print("\n" + "="*50)
    print("Testing gradient flow...")
    
    import jax
    from optimizer import ECDSNAPOptimizer
    
    # Small test case
    N_layers = 2
    N_trunc = 4
    batch_size = 4
    
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, batch_size)
    
    # Create simple target
    phases = np.array([0, 0.5, 1.0, 1.5])
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = optimizer.initialize_parameters(key)
    
    # Compute loss and gradients
    loss_fn = lambda p: optimizer.loss_function(p, U_target_jax)
    loss_val = loss_fn(params)
    grads = jax.grad(loss_fn)(params)
    
    print(f"Initial loss: {loss_val:.4f}")
    print(f"Gradient norms:")
    print(f"  ∇β: {jnp.linalg.norm(grads['betas']):.4f}")
    print(f"  ∇θ: {jnp.linalg.norm(grads['thetas']):.4f}")
    print(f"  ∇φ: {jnp.linalg.norm(grads['phis']):.4f}")
    
    # Check that gradients are non-zero
    assert jnp.linalg.norm(grads['betas']) > 1e-6, "Beta gradients are zero!"
    assert jnp.linalg.norm(grads['thetas']) > 1e-6, "Theta gradients are zero!"
    assert jnp.linalg.norm(grads['phis']) > 1e-6, "Phi gradients are zero!"
    
    print("✓ Gradients flow correctly!")
    return True


def main():
    """Run all tests."""
    print("="*50)
    print("ECD-to-SNAP Optimization Test Suite")
    print("="*50)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_gradient_flow():
            tests_passed += 1
            print("✓ Gradient flow test PASSED")
    except Exception as e:
        print(f"✗ Gradient flow test FAILED: {e}")
    
    try:
        if test_identity_snap():
            tests_passed += 1
            print("✓ Identity SNAP test PASSED")
    except Exception as e:
        print(f"✗ Identity SNAP test FAILED: {e}")
    
    try:
        if test_linear_snap():
            tests_passed += 1
            print("✓ Linear SNAP test PASSED")
    except Exception as e:
        print(f"✗ Linear SNAP test FAILED: {e}")
    
    print("\n" + "="*50)
    print(f"Test Results: {tests_passed}/{total_tests} passed")
    print("="*50)
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed. Please review the implementation.")
    
    plt.show()  # Keep plots open


if __name__ == "__main__":
    main()