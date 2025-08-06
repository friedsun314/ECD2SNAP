"""
Debug the fidelity calculation to understand why it's not working.
"""

import numpy as np
import jax.numpy as jnp
from gates import build_ecd_sequence_jax_real
from snap_targets import make_snap_full_space
from optimizer import ECDSNAPOptimizer

def test_fidelity_calculation():
    """Test if fidelity calculation is correct."""
    print("Testing fidelity calculation...")
    
    N_trunc = 4
    N_layers = 2
    
    # Create identity target in full space
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_np = U_target.full()
    U_target_jax = jnp.array(U_target_np)
    
    print(f"Target shape: {U_target_jax.shape}")
    print(f"Target is identity? {np.allclose(U_target_np, np.eye(2*N_trunc))}")
    
    # Build identity approximation with zero parameters
    betas = jnp.zeros(N_layers + 1, dtype=jnp.complex64)
    phis = jnp.zeros(N_layers + 1, dtype=jnp.float32)
    thetas = jnp.zeros(N_layers + 1, dtype=jnp.float32)
    
    U_approx = build_ecd_sequence_jax_real(betas, phis, thetas, N_trunc)
    print(f"Approx shape: {U_approx.shape}")
    print(f"Approx is identity? {np.allclose(U_approx, np.eye(2*N_trunc))}")
    
    # Manual fidelity calculation
    d = U_target_jax.shape[0]
    trace = jnp.trace(jnp.conj(U_target_jax.T) @ U_approx)
    fidelity_manual = jnp.abs(trace)**2 / d**2
    print(f"Manual fidelity: {fidelity_manual:.6f}")
    
    # Using optimizer's method
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, 1)
    fidelity_opt = optimizer.unitary_fidelity(U_target_jax, U_approx)
    print(f"Optimizer fidelity: {fidelity_opt:.6f}")
    
    # Check trace values
    print(f"Trace value: {trace}")
    print(f"|Trace|: {jnp.abs(trace)}")
    print(f"Expected (d): {d}")
    
    return fidelity_manual > 0.99


def test_batch_fidelity_issue():
    """Test if batch fidelity has issues."""
    print("\nTesting batch fidelity...")
    
    N_trunc = 4
    N_layers = 2
    batch_size = 4
    
    # Create optimizer
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, batch_size)
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Create batch of parameters (all zeros)
    params = {
        'betas': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.complex64),
        'phis': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32),
        'thetas': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32)
    }
    
    # Compute batch fidelity
    fidelities = optimizer.batch_fidelity(params, U_target_jax)
    print(f"Batch fidelities (all zeros): {fidelities}")
    print(f"All should be 1.0? {jnp.allclose(fidelities, 1.0)}")
    
    # Now with small random parameters
    key = jax.random.PRNGKey(42)
    params['betas'] = jax.random.normal(key, (batch_size, N_layers + 1)) * 0.01
    fidelities2 = optimizer.batch_fidelity(params, U_target_jax)
    print(f"Batch fidelities (small random): {fidelities2}")
    
    return jnp.allclose(fidelities, 1.0)


def test_loss_function():
    """Test the loss function behavior."""
    print("\nTesting loss function...")
    
    N_trunc = 4
    N_layers = 2
    batch_size = 4
    
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, batch_size)
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Zero parameters (perfect fidelity)
    params = {
        'betas': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.complex64),
        'phis': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32),
        'thetas': jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32)
    }
    
    fidelities = optimizer.batch_fidelity(params, U_target_jax)
    loss = optimizer.loss_function(params, U_target_jax)
    
    print(f"Fidelities: {fidelities}")
    print(f"Loss: {loss}")
    print(f"Expected loss for F=1: {jnp.log(1 - 0.999999) * batch_size:.6f}")
    
    # Check individual loss terms
    eps = 1e-10
    losses = jnp.log(jnp.maximum(1 - fidelities, eps))
    print(f"Individual losses: {losses}")
    print(f"Sum: {jnp.sum(losses)}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("Debugging Fidelity Calculation")
    print("="*60)
    
    import jax
    
    test_fidelity_calculation()
    test_batch_fidelity_issue()
    test_loss_function()
    
    print("\n" + "="*60)
    print("Debugging complete!")
    print("="*60)