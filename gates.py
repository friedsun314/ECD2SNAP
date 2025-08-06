"""
Quantum gate operations for ECD-to-SNAP optimization.
Thin wrappers around QuTiP operators for efficient gate construction.
"""

import qutip as qt
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from typing import Union, List, Tuple
from scipy.special import factorial


def displacement_operator(beta: complex, N_trunc: int) -> qt.Qobj:
    """
    Create a displacement operator D(beta) in Fock space.
    
    Args:
        beta: Complex displacement amplitude
        N_trunc: Fock space truncation dimension
    
    Returns:
        Displacement operator as QuTiP Qobj
    """
    return qt.displace(N_trunc, beta)


def rotation_operator(theta: float, phi: float) -> qt.Qobj:
    """
    Create a qubit rotation operator R_phi(theta).
    
    Args:
        theta: Rotation angle [0, pi]
        phi: Rotation axis in equatorial plane [0, 2*pi)
    
    Returns:
        2x2 rotation matrix as QuTiP Qobj
    """
    # Rotation axis in Bloch sphere
    n = np.array([np.cos(phi), np.sin(phi), 0])
    return qt.Qobj(qt.rotation(theta, n))


def ecd_gate(beta: complex, N_trunc: int) -> qt.Qobj:
    """
    Construct an Echo Conditional Displacement (ECD) gate.
    
    The ECD gate applies conditional displacement based on qubit state:
    ECD(beta) = |0><0| ⊗ D(beta) + |1><1| ⊗ D(-beta)
    
    Args:
        beta: Complex displacement amplitude
        N_trunc: Fock space truncation dimension
    
    Returns:
        ECD gate as (2 * N_trunc) x (2 * N_trunc) QuTiP Qobj
    """
    # Qubit projectors
    proj_0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
    proj_1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
    
    # Displacement operators
    D_plus = qt.displace(N_trunc, beta)
    D_minus = qt.displace(N_trunc, -beta)
    
    # Construct ECD gate
    ecd = qt.tensor(proj_0, D_plus) + qt.tensor(proj_1, D_minus)
    return ecd


def build_ecd_sequence(betas: Union[List, np.ndarray], 
                      phis: Union[List, np.ndarray], 
                      thetas: Union[List, np.ndarray], 
                      N_trunc: int) -> qt.Qobj:
    """
    Construct the full ECD gate sequence U_ECD.
    
    U_ECD = D(beta_{N+1}/2) R_phi_{N+1}(theta_{N+1}) * 
            prod_{k=N}^{1} [ECD(beta_k) R_phi_k(theta_k)]
    
    Args:
        betas: Complex displacement amplitudes for each layer
        phis: Rotation axis angles for each layer [0, 2*pi)
        thetas: Rotation angles for each layer [0, pi]
        N_trunc: Fock space truncation dimension
    
    Returns:
        Full unitary as (2 * N_trunc) x (2 * N_trunc) QuTiP Qobj
    """
    N_layers = len(betas) - 1  # Last beta is for final displacement
    dim = 2 * N_trunc
    
    # Initialize with identity
    U = qt.identity([2, N_trunc])
    
    # Apply layers in sequence (k=1 to N)
    for k in range(N_layers):
        # Apply rotation R_phi_k(theta_k)
        R_k = qt.tensor(rotation_operator(thetas[k], phis[k]), qt.identity(N_trunc))
        U = R_k * U
        
        # Apply ECD(beta_k)
        ECD_k = ecd_gate(betas[k], N_trunc)
        U = ECD_k * U
    
    # Apply final rotation and displacement
    R_final = qt.tensor(rotation_operator(thetas[-1], phis[-1]), qt.identity(N_trunc))
    U = R_final * U
    
    # Final displacement D(beta_{N+1}/2) acts only on cavity
    D_final = qt.tensor(qt.identity(2), displacement_operator(betas[-1]/2, N_trunc))
    U = D_final * U
    
    return U


# JAX-compatible gate operations for automatic differentiation

def displacement_operator_jax(beta: complex, N_trunc: int) -> jnp.ndarray:
    """
    Create a displacement operator D(beta) in Fock space using JAX.
    
    Uses matrix exponentiation: D(beta) = exp(beta * a† - beta* * a)
    where a† and a are creation and annihilation operators.
    
    Args:
        beta: Complex displacement amplitude
        N_trunc: Fock space truncation dimension
    
    Returns:
        Displacement operator as JAX array
    """
    # Creation and annihilation operators in Fock basis
    # a|n> = sqrt(n)|n-1>, a†|n> = sqrt(n+1)|n+1>
    
    # Annihilation operator
    a = jnp.zeros((N_trunc, N_trunc), dtype=jnp.complex64)
    for n in range(1, N_trunc):
        a = a.at[n-1, n].set(jnp.sqrt(n))
    
    # Creation operator (Hermitian conjugate of a)
    a_dag = jnp.conj(a.T)
    
    # Generator of displacement: beta * a† - beta* * a
    generator = beta * a_dag - jnp.conj(beta) * a
    
    # Displacement operator via matrix exponentiation
    # Using Padé approximation for better numerical stability
    from jax.scipy.linalg import expm
    D = expm(generator)
    
    return D


def rotation_operator_jax(theta: float, phi: float) -> jnp.ndarray:
    """
    Create a qubit rotation operator R_phi(theta) using JAX.
    
    R(theta, phi) = exp(-i * theta/2 * (cos(phi) * sigma_x + sin(phi) * sigma_y))
                  = cos(theta/2) * I - i * sin(theta/2) * (cos(phi) * sigma_x + sin(phi) * sigma_y)
    
    Args:
        theta: Rotation angle [0, pi]
        phi: Rotation axis in equatorial plane [0, 2*pi)
    
    Returns:
        2x2 rotation matrix as JAX array
    """
    # Pauli matrices
    I = jnp.eye(2, dtype=jnp.complex64)
    sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    sigma_y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64)
    
    # Rotation matrix
    R = (jnp.cos(theta/2) * I - 
         1j * jnp.sin(theta/2) * (jnp.cos(phi) * sigma_x + jnp.sin(phi) * sigma_y))
    
    return R


def ecd_gate_jax(beta: complex, N_trunc: int) -> jnp.ndarray:
    """
    Construct an ECD gate using JAX operations.
    
    ECD(beta) = |0><0| ⊗ D(beta) + |1><1| ⊗ D(-beta)
    
    Args:
        beta: Complex displacement amplitude
        N_trunc: Fock space truncation dimension
    
    Returns:
        ECD gate as (2 * N_trunc) x (2 * N_trunc) JAX array
    """
    # Displacement operators
    D_plus = displacement_operator_jax(beta, N_trunc)
    D_minus = displacement_operator_jax(-beta, N_trunc)
    
    # Qubit projectors
    proj_0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
    proj_1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
    
    # Tensor products using Kronecker product
    ecd = jnp.kron(proj_0, D_plus) + jnp.kron(proj_1, D_minus)
    
    return ecd


@jit
def build_ecd_sequence_jax_real(betas: jnp.ndarray, 
                                phis: jnp.ndarray, 
                                thetas: jnp.ndarray, 
                                N_trunc: int) -> jnp.ndarray:
    """
    Build the full ECD gate sequence using only JAX operations.
    This version maintains gradient flow for automatic differentiation.
    
    U_ECD = D(beta_{N+1}/2) R_phi_{N+1}(theta_{N+1}) * 
            prod_{k=N}^{1} [ECD(beta_k) R_phi_k(theta_k)]
    
    Args:
        betas: Complex displacement amplitudes (JAX array)
        phis: Rotation axis angles (JAX array)
        thetas: Rotation angles (JAX array)
        N_trunc: Fock space truncation dimension
    
    Returns:
        Full unitary as JAX array
    """
    N_layers = len(betas) - 1  # Last beta is for final displacement
    dim = 2 * N_trunc
    
    # Initialize with identity
    U = jnp.eye(dim, dtype=jnp.complex64)
    
    # Apply layers in sequence (k=1 to N)
    for k in range(N_layers):
        # Apply rotation R_phi_k(theta_k) on qubit
        R_k = rotation_operator_jax(thetas[k], phis[k])
        R_full = jnp.kron(R_k, jnp.eye(N_trunc, dtype=jnp.complex64))
        U = R_full @ U
        
        # Apply ECD(beta_k)
        ECD_k = ecd_gate_jax(betas[k], N_trunc)
        U = ECD_k @ U
    
    # Apply final rotation
    R_final = rotation_operator_jax(thetas[-1], phis[-1])
    R_final_full = jnp.kron(R_final, jnp.eye(N_trunc, dtype=jnp.complex64))
    U = R_final_full @ U
    
    # Final displacement D(beta_{N+1}/2) acts only on cavity
    D_final = displacement_operator_jax(betas[-1]/2, N_trunc)
    D_final_full = jnp.kron(jnp.eye(2, dtype=jnp.complex64), D_final)
    U = D_final_full @ U
    
    return U


# Keep the old version for compatibility
def build_ecd_sequence_jax(betas: jnp.ndarray, 
                           phis: jnp.ndarray, 
                           thetas: jnp.ndarray, 
                           N_trunc: int) -> jnp.ndarray:
    """
    JAX-compatible version of build_ecd_sequence for gradient computation.
    Now calls the real JAX implementation.
    
    Args:
        betas: Complex displacement amplitudes (JAX array)
        phis: Rotation axis angles (JAX array)
        thetas: Rotation angles (JAX array)
        N_trunc: Fock space truncation dimension
    
    Returns:
        Full unitary as JAX array
    """
    return build_ecd_sequence_jax_real(betas, phis, thetas, N_trunc)


def extract_cavity_unitary(U_full: qt.Qobj, N_trunc: int) -> qt.Qobj:
    """
    Extract the effective cavity unitary from the full qubit-cavity unitary.
    
    Args:
        U_full: Full (2 * N_trunc) x (2 * N_trunc) unitary
        N_trunc: Fock space truncation dimension
    
    Returns:
        N_trunc x N_trunc cavity unitary
    """
    # This would require specific assumptions about initial/final qubit state
    # For now, return the block corresponding to qubit in |0>
    return U_full[:N_trunc, :N_trunc]