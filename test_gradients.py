"""
Test script to verify JAX gradient flow through the ECD gate operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from gates import (
    displacement_operator_jax, 
    rotation_operator_jax,
    ecd_gate_jax,
    build_ecd_sequence_jax_real
)
from optimizer import ECDSNAPOptimizer
from snap_targets import make_snap_full_space


def test_displacement_gradient():
    """Test that displacement operator is differentiable."""
    print("Testing displacement operator gradients...")
    
    N_trunc = 4
    beta = jnp.complex64(0.5 + 0.3j)
    
    def loss_fn(b):
        D = displacement_operator_jax(b, N_trunc)
        # Loss based on a specific matrix element that depends on beta
        # The (0,1) element of D should vary with beta
        return jnp.real(D[0, 1] * jnp.conj(D[0, 1]))
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(beta)
    
    print(f"  Beta: {beta}")
    print(f"  Gradient: {grad}")
    print(f"  |Gradient|: {jnp.abs(grad):.6f}")
    
    assert jnp.abs(grad) > 1e-6, "Gradient is too small!"
    print("  ✓ Displacement operator gradient flows correctly")
    return True


def test_rotation_gradient():
    """Test that rotation operator is differentiable."""
    print("\nTesting rotation operator gradients...")
    
    theta = jnp.float32(np.pi/4)
    phi = jnp.float32(np.pi/3)
    
    def loss_fn(params):
        t, p = params
        R = rotation_operator_jax(t, p)
        # Loss based on trace
        return jnp.real(jnp.trace(R))
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn([theta, phi])
    
    print(f"  Theta: {theta:.4f}, Phi: {phi:.4f}")
    print(f"  Gradient (theta): {grad[0]:.6f}")
    print(f"  Gradient (phi): {grad[1]:.6f}")
    
    assert jnp.abs(grad[0]) > 1e-6, "Theta gradient is too small!"
    print("  ✓ Rotation operator gradient flows correctly")
    return True


def test_ecd_gradient():
    """Test that ECD gate is differentiable."""
    print("\nTesting ECD gate gradients...")
    
    N_trunc = 4
    beta = jnp.complex64(0.3 + 0.2j)
    
    def loss_fn(b):
        E = ecd_gate_jax(b, N_trunc)
        # Loss based on a specific matrix element
        # ECD gate elements should vary with beta
        return jnp.real(E[0, 1] * jnp.conj(E[0, 1]) + E[1, 0] * jnp.conj(E[1, 0]))
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(beta)
    
    print(f"  Beta: {beta}")
    print(f"  Gradient: {grad}")
    print(f"  |Gradient|: {jnp.abs(grad):.6f}")
    
    assert jnp.abs(grad) > 1e-6, "Gradient is too small!"
    print("  ✓ ECD gate gradient flows correctly")
    return True


def test_full_sequence_gradient():
    """Test that full ECD sequence is differentiable."""
    print("\nTesting full ECD sequence gradients...")
    
    N_trunc = 4
    N_layers = 3
    
    # Parameters
    betas = jnp.array([0.1+0.1j, 0.2+0.1j, 0.15+0.05j, 0.05+0.02j], dtype=jnp.complex64)
    phis = jnp.array([0.5, 1.0, 1.5, 2.0], dtype=jnp.float32)
    thetas = jnp.array([0.3, 0.6, 0.9, 1.2], dtype=jnp.float32)
    
    # Create a simple target
    phases = jnp.array([0, 0.5, 1.0, 1.5])
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    def loss_fn(params):
        b, p, t = params
        U = build_ecd_sequence_jax_real(b, p, t, N_trunc)
        # Fidelity-based loss
        d = U.shape[0]
        fidelity = jnp.abs(jnp.trace(jnp.conj(U_target_jax.T) @ U))**2 / d**2
        return 1 - fidelity
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn([betas, phis, thetas])
    
    print(f"  Number of layers: {N_layers}")
    print(f"  Gradient norms:")
    print(f"    ∇β: {jnp.linalg.norm(grads[0]):.6f}")
    print(f"    ∇φ: {jnp.linalg.norm(grads[1]):.6f}")
    print(f"    ∇θ: {jnp.linalg.norm(grads[2]):.6f}")
    
    assert jnp.linalg.norm(grads[0]) > 1e-6, "Beta gradients are too small!"
    assert jnp.linalg.norm(grads[1]) > 1e-6, "Phi gradients are too small!"
    assert jnp.linalg.norm(grads[2]) > 1e-6, "Theta gradients are too small!"
    
    print("  ✓ Full sequence gradient flows correctly")
    return True


def test_optimizer_gradient_flow():
    """Test that optimizer can compute gradients."""
    print("\nTesting optimizer gradient flow...")
    
    N_layers = 2
    N_trunc = 4
    batch_size = 4
    
    # Create optimizer
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
    
    print(f"  Initial loss: {loss_val:.4f}")
    print(f"  Gradient norms:")
    print(f"    ∇β: {jnp.linalg.norm(grads['betas']):.6f}")
    print(f"    ∇φ: {jnp.linalg.norm(grads['phis']):.6f}")
    print(f"    ∇θ: {jnp.linalg.norm(grads['thetas']):.6f}")
    
    # Check that gradients are non-zero
    assert jnp.linalg.norm(grads['betas']) > 1e-6, "Beta gradients are zero!"
    assert jnp.linalg.norm(grads['phis']) > 1e-6, "Phi gradients are zero!"
    assert jnp.linalg.norm(grads['thetas']) > 1e-6, "Theta gradients are zero!"
    
    print("  ✓ Optimizer gradients flow correctly!")
    return True


def main():
    """Run all gradient tests."""
    print("="*60)
    print("JAX Gradient Flow Test Suite")
    print("="*60)
    
    tests = [
        ("Displacement operator", test_displacement_gradient),
        ("Rotation operator", test_rotation_gradient),
        ("ECD gate", test_ecd_gradient),
        ("Full sequence", test_full_sequence_gradient),
        ("Optimizer", test_optimizer_gradient_flow)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ {name} test FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 All gradient tests passed! The implementation is ready for optimization.")
    else:
        print(f"⚠️  {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)