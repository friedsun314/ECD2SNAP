#!/usr/bin/env python3
"""
Basic functionality tests for the ECD-to-SNAP concept.

This test file demonstrates the testing approach and validates some 
basic quantum gate properties without depending on the exact 
ecd_snap_minimal.py implementation.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os
import json


def test_basic_imports():
    """Test that basic dependencies can be imported."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import matplotlib.pyplot as plt
    
    print(f"JAX version: {jax.__version__}")
    assert True  # If we get here, imports succeeded


def test_jax_basic_operations():
    """Test basic JAX operations work as expected."""
    # Simple array operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    expected = jnp.array([5.0, 7.0, 9.0])
    np.testing.assert_allclose(z, expected)
    
    # Matrix multiplication
    A = jnp.array([[1, 2], [3, 4]], dtype=jnp.complex64)
    B = jnp.array([[5, 6], [7, 8]], dtype=jnp.complex64)
    C = A @ B
    expected_C = jnp.array([[19, 22], [43, 50]], dtype=jnp.complex64)
    np.testing.assert_allclose(C, expected_C)


def test_quantum_gate_properties():
    """Test basic quantum gate properties using simple implementations."""
    # Test Pauli-X gate
    pauli_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    
    # Should be unitary: X† X = I
    x_dag = jnp.conj(pauli_x.T)
    identity = x_dag @ pauli_x
    expected_identity = jnp.eye(2, dtype=jnp.complex64)
    np.testing.assert_allclose(identity, expected_identity, rtol=1e-6)
    
    # Should be involutory: X² = I
    x_squared = pauli_x @ pauli_x
    np.testing.assert_allclose(x_squared, expected_identity, rtol=1e-6)


def test_displacement_operator_simple():
    """Test a simple displacement operator implementation."""
    def simple_displacement_operator(beta: complex, N_trunc: int) -> jnp.ndarray:
        """Simple displacement using series expansion (truncated)."""
        n = jnp.arange(N_trunc, dtype=jnp.float32)
        # Creation and annihilation operators
        a_dag = jnp.diag(jnp.sqrt(n[1:]), k=1).astype(jnp.complex64)
        a = jnp.diag(jnp.sqrt(n[1:]), k=-1).astype(jnp.complex64)
        
        # Generator: beta * a† - beta* * a
        generator = beta * a_dag - jnp.conj(beta) * a
        
        # Use matrix exponentiation
        return jax.scipy.linalg.expm(generator).astype(jnp.complex64)
    
    # Test with small displacement
    N_trunc = 4
    beta = 0.1 + 0.05j
    D = simple_displacement_operator(beta, N_trunc)
    
    # Check unitarity
    D_dag = jnp.conj(D.T)
    should_be_identity = D_dag @ D
    np.testing.assert_allclose(should_be_identity, jnp.eye(N_trunc, dtype=jnp.complex64), rtol=1e-6, atol=1e-6)
    
    # Check identity displacement
    D_identity = simple_displacement_operator(0.0, N_trunc)
    np.testing.assert_allclose(D_identity, jnp.eye(N_trunc), rtol=1e-6)


def test_snap_gate_simple():
    """Test a simple SNAP gate implementation."""
    def make_simple_snap(phases: jnp.ndarray, N_trunc: int) -> jnp.ndarray:
        """Create simple SNAP gate from phases."""
        if len(phases) < N_trunc:
            phases = jnp.pad(phases, (0, N_trunc - len(phases)), constant_values=0)
        else:
            phases = phases[:N_trunc]
        
        # SNAP is diagonal in Fock basis
        U_snap_cavity = jnp.diag(jnp.exp(1j * phases)).astype(jnp.complex64)
        # In full qubit-cavity space: I_qubit ⊗ U_SNAP
        U_snap_full = jnp.kron(jnp.eye(2, dtype=jnp.complex64), U_snap_cavity)
        return U_snap_full
    
    # Test with simple phases
    phases = jnp.array([0.0, jnp.pi/2, jnp.pi])
    N_trunc = 3
    U_snap = make_simple_snap(phases, N_trunc)
    
    # Check dimensions
    assert U_snap.shape == (6, 6)  # 2 * N_trunc
    
    # Check unitarity
    U_dag = jnp.conj(U_snap.T)
    should_be_identity = U_dag @ U_snap
    np.testing.assert_allclose(should_be_identity, jnp.eye(6), rtol=1e-6)
    
    # Check diagonal structure in cavity subspace
    # The |1⟩ ⊗ cavity block should have the SNAP phases on diagonal
    cavity_block_11 = U_snap[N_trunc:2*N_trunc, N_trunc:2*N_trunc]  # |1⟩ ⊗ cavity block
    expected_diag = jnp.exp(1j * phases)
    np.testing.assert_allclose(jnp.diag(cavity_block_11), expected_diag, rtol=1e-6)


def test_fidelity_calculation():
    """Test fidelity calculation between unitaries."""
    def unitary_fidelity(U1: jnp.ndarray, U2: jnp.ndarray) -> float:
        """Calculate fidelity between two unitaries: |Tr(U1† U2)|² / d²."""
        d = U1.shape[0]
        trace = jnp.trace(jnp.conj(U1.T) @ U2)
        return jnp.abs(trace)**2 / d**2
    
    # Identical matrices should have fidelity 1
    U = jnp.eye(3, dtype=jnp.complex64)
    fid = unitary_fidelity(U, U)
    assert abs(fid - 1.0) < 1e-10
    
    # Orthogonal matrices should have fidelity 0
    U1 = jnp.eye(2, dtype=jnp.complex64)
    U2 = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)  # Pauli-X
    fid_ortho = unitary_fidelity(U1, U2)
    assert abs(fid_ortho) < 1e-10


def test_phases_parsing_concept():
    """Test the concept of parsing phases from different formats."""
    def parse_phases_simple(phases_input: str) -> np.ndarray:
        """Simple phases parser."""
        # If it looks like a file path, try to read it
        if os.path.exists(phases_input):
            if phases_input.endswith('.npy'):
                return np.load(phases_input)
            else:
                with open(phases_input, 'r') as f:
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        return np.array(json.loads(content))
                    else:
                        return np.array([float(x) for x in content.replace(',', ' ').split() if x.strip()])
        else:
            # Parse as comma-separated
            return np.array([float(x.strip()) for x in phases_input.split(',') if x.strip()])
    
    # Test direct parsing
    phases_str = "0.1, 0.5, 1.0, 1.5"
    phases = parse_phases_simple(phases_str)
    expected = np.array([0.1, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(phases, expected)
    
    # Test file parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("0.2 0.6 1.1")
        temp_path = f.name
    
    try:
        phases_file = parse_phases_simple(temp_path)
        expected_file = np.array([0.2, 0.6, 1.1])
        np.testing.assert_allclose(phases_file, expected_file)
    finally:
        os.unlink(temp_path)


def test_optimization_concept():
    """Test basic optimization concept using simple loss function."""
    import optax
    
    def simple_loss(params):
        """Simple quadratic loss function."""
        return jnp.sum(params**2)
    
    # Simple optimization setup
    learning_rate = 0.1
    optimizer = optax.adam(learning_rate)
    
    # Initial parameters
    params = jnp.array([1.0, 2.0, 3.0])
    opt_state = optimizer.init(params)
    
    # Take many optimization steps for better convergence
    loss_and_grad = jax.value_and_grad(simple_loss)
    
    initial_loss = simple_loss(params)
    for _ in range(100):
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    # Should converge towards zero
    final_loss = simple_loss(params)
    # Check that we've made significant progress
    assert final_loss < 0.1 * initial_loss


def test_random_seed_reproducibility():
    """Test JAX random seed reproducibility."""
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(42)
    
    # Same seed should give same results
    rand1 = jax.random.normal(key1, (3,))
    rand2 = jax.random.normal(key2, (3,))
    
    np.testing.assert_allclose(rand1, rand2)
    
    # Different seeds should give different results
    key3 = jax.random.PRNGKey(43)
    rand3 = jax.random.normal(key3, (3,))
    
    assert not np.allclose(rand1, rand3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])