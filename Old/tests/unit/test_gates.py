"""
Verify that the quantum gates are working correctly.
"""

import numpy as np
import jax.numpy as jnp
from src.gates import displacement_operator_jax, build_ecd_sequence_jax_real
from src.snap_targets import identity_snap, make_snap_full_space

def test_displacement_properties():
    """Test that displacement operator has correct properties."""
    print("Testing displacement operator properties...")
    
    N_trunc = 6
    beta = 0.5 + 0.3j
    
    D = displacement_operator_jax(beta, N_trunc)
    
    # Check unitarity: D† D = I
    D_dag_D = jnp.conj(D.T) @ D
    identity = jnp.eye(N_trunc)
    unitarity_error = jnp.max(jnp.abs(D_dag_D - identity))
    print(f"  Unitarity error: {unitarity_error:.6f}")
    
    # Check that D(0) = I
    D_zero = displacement_operator_jax(0.0, N_trunc)
    zero_error = jnp.max(jnp.abs(D_zero - identity))
    print(f"  D(0) = I error: {zero_error:.6f}")
    
    # Check coherent state generation: D(β)|0⟩
    vacuum = jnp.zeros(N_trunc, dtype=jnp.complex64)
    vacuum = vacuum.at[0].set(1.0)
    coherent = D @ vacuum
    
    print(f"  Coherent state amplitudes: {np.abs(coherent[:4])}")
    print(f"  Expected approximate pattern for small β")
    
    return unitarity_error < 1e-5 and zero_error < 1e-5


def test_simple_ecd_sequence():
    """Test if we can approximate identity with minimal gates."""
    print("\nTesting simple ECD sequence...")
    
    N_trunc = 4
    N_layers = 1
    
    # For identity, we want minimal displacements and rotations
    betas = jnp.array([0.0, 0.0], dtype=jnp.complex64)
    phis = jnp.array([0.0, 0.0], dtype=jnp.float32)
    thetas = jnp.array([0.0, 0.0], dtype=jnp.float32)
    
    U = build_ecd_sequence_jax_real(betas, phis, thetas, N_trunc)
    
    # This should be close to identity in full space
    identity_full = jnp.eye(2 * N_trunc)
    error = jnp.max(jnp.abs(U - identity_full))
    print(f"  Zero parameters -> Identity error: {error:.6f}")
    
    # Now test with target
    phases = jnp.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Compute fidelity
    d = U.shape[0]
    fidelity = jnp.abs(jnp.trace(jnp.conj(U_target_jax.T) @ U))**2 / d**2
    print(f"  Fidelity with identity SNAP: {fidelity:.6f}")
    
    return error < 1e-5


def check_optimization_target():
    """Check what the target actually looks like."""
    print("\nChecking optimization target...")
    
    N_trunc = 4
    phases = np.zeros(N_trunc)
    
    # SNAP in cavity space
    U_snap = identity_snap(N_trunc)
    print(f"  SNAP (cavity only) shape: {U_snap.shape}")
    
    # SNAP in full space
    U_snap_full = make_snap_full_space(phases, N_trunc)
    print(f"  SNAP (full space) shape: {U_snap_full.shape}")
    
    # Check what it looks like
    U_array = U_snap_full.full()
    print(f"  Is diagonal? {np.allclose(U_array, np.diag(np.diag(U_array)))}")
    print(f"  First 4x4 block:")
    print(U_array[:4, :4])
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("Verifying Quantum Gates")
    print("="*60)
    
    tests = [
        test_displacement_properties,
        test_simple_ecd_sequence,
        check_optimization_target
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
            print("  ✓ Test passed")
        else:
            print("  ✗ Test failed")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All gate operations verified!")
    else:
        print("✗ Some tests failed")