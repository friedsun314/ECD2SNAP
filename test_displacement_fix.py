"""
Test to debug displacement operator gradient issues.
"""

import jax
import jax.numpy as jnp
from gates import displacement_operator_jax

def test_displacement_with_better_loss():
    """Test displacement operator with a loss that actually depends on beta."""
    print("Testing displacement operator with improved loss function...")
    
    N_trunc = 4
    beta = jnp.complex64(0.5 + 0.3j)
    
    # Create a target that depends on beta
    def loss_fn(b):
        D = displacement_operator_jax(b, N_trunc)
        # Use a loss that actually depends on the displacement
        # For example, the (0,1) element should depend on beta
        return jnp.real(D[0, 1] * jnp.conj(D[0, 1]))
    
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(beta)
    
    print(f"  Beta: {beta}")
    print(f"  Loss value: {loss_fn(beta):.6f}")
    print(f"  Gradient: {grad}")
    print(f"  |Gradient|: {jnp.abs(grad):.6f}")
    
    # Also test with a different loss
    def loss_fn2(b):
        D = displacement_operator_jax(b, N_trunc)
        # Measure overlap with a target state
        target = jnp.array([1.0, 0.5, 0.0, 0.0], dtype=jnp.complex64)
        target = target / jnp.linalg.norm(target)
        result = D @ target
        return jnp.real(jnp.sum(jnp.abs(result)**2))
    
    grad2 = jax.grad(loss_fn2)(beta)
    print(f"\n  Alternative loss gradient: {grad2}")
    print(f"  |Gradient|: {jnp.abs(grad2):.6f}")
    
    # Test the matrix elements directly
    D = displacement_operator_jax(beta, N_trunc)
    print(f"\n  Displacement matrix (first 2x2 block):")
    print(f"    {D[0,0]:.4f}  {D[0,1]:.4f}")
    print(f"    {D[1,0]:.4f}  {D[1,1]:.4f}")
    
    return jnp.abs(grad) > 1e-6 or jnp.abs(grad2) > 1e-6

if __name__ == "__main__":
    success = test_displacement_with_better_loss()
    if success:
        print("\n✓ Displacement operator has non-zero gradients!")
    else:
        print("\n✗ Displacement operator still has zero gradients")