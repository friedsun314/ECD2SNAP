"""
SNAP (Selective Number-dependent Arbitrary Phase) gate constructors.
Provides utilities for creating target SNAP unitaries.
"""

import numpy as np
import qutip as qt
from typing import Union, List


def make_snap(phases: Union[np.ndarray, List[float]], N_trunc: int) -> qt.Qobj:
    """
    Create a SNAP gate unitary from phase angles.
    
    SNAP gate applies phase shifts that depend on the Fock state number:
    U_SNAP = Σ_n exp(i*θ_n) |n><n|
    
    Args:
        phases: Phase angles for each Fock state
        N_trunc: Fock space truncation dimension
    
    Returns:
        SNAP unitary as N_trunc x N_trunc QuTiP Qobj
    """
    # Ensure phases array has correct length
    if len(phases) < N_trunc:
        # Pad with zeros if necessary
        phases = np.pad(phases, (0, N_trunc - len(phases)), constant_values=0)
    else:
        phases = phases[:N_trunc]
    
    # Create diagonal matrix
    U_snap = np.diag(np.exp(1j * phases))
    
    return qt.Qobj(U_snap)


def make_snap_full_space(phases: Union[np.ndarray, List[float]], 
                        N_trunc: int) -> qt.Qobj:
    """
    Create a SNAP gate in the full qubit-cavity space.
    
    The SNAP gate acts only on the cavity, so in the full space:
    U_SNAP_full = I_qubit ⊗ U_SNAP_cavity
    
    Args:
        phases: Phase angles for each Fock state
        N_trunc: Fock space truncation dimension
    
    Returns:
        SNAP unitary as (2*N_trunc) x (2*N_trunc) QuTiP Qobj
    """
    U_snap_cavity = make_snap(phases, N_trunc)
    U_snap_full = qt.tensor(qt.identity(2), U_snap_cavity)
    return U_snap_full


# Pre-defined common SNAP gates

def identity_snap(N_trunc: int) -> qt.Qobj:
    """
    Identity SNAP gate (all phases = 0).
    
    Args:
        N_trunc: Fock space truncation dimension
    
    Returns:
        Identity SNAP unitary
    """
    return make_snap(np.zeros(N_trunc), N_trunc)


def phase_snap(phase: float, N_trunc: int) -> qt.Qobj:
    """
    Global phase SNAP gate (all phases = same value).
    
    Args:
        phase: Global phase to apply
        N_trunc: Fock space truncation dimension
    
    Returns:
        Global phase SNAP unitary
    """
    return make_snap(np.ones(N_trunc) * phase, N_trunc)


def linear_snap(slope: float, N_trunc: int) -> qt.Qobj:
    """
    Linear phase SNAP gate (phases = slope * n).
    
    Args:
        slope: Phase slope per Fock number
        N_trunc: Fock space truncation dimension
    
    Returns:
        Linear phase SNAP unitary
    """
    phases = slope * np.arange(N_trunc)
    return make_snap(phases, N_trunc)


def quadratic_snap(coeff: float, N_trunc: int) -> qt.Qobj:
    """
    Quadratic phase SNAP gate (phases = coeff * n²).
    
    Args:
        coeff: Quadratic coefficient
        N_trunc: Fock space truncation dimension
    
    Returns:
        Quadratic phase SNAP unitary
    """
    n = np.arange(N_trunc)
    phases = coeff * n**2
    return make_snap(phases, N_trunc)


def cubic_snap(coeff: float, N_trunc: int) -> qt.Qobj:
    """
    Cubic phase SNAP gate (phases = coeff * n³).
    
    Args:
        coeff: Cubic coefficient
        N_trunc: Fock space truncation dimension
    
    Returns:
        Cubic phase SNAP unitary
    """
    n = np.arange(N_trunc)
    phases = coeff * n**3
    return make_snap(phases, N_trunc)


def random_snap(N_trunc: int, seed: int = None) -> qt.Qobj:
    """
    Random SNAP gate with uniformly distributed phases.
    
    Args:
        N_trunc: Fock space truncation dimension
        seed: Random seed for reproducibility
    
    Returns:
        Random SNAP unitary
    """
    if seed is not None:
        np.random.seed(seed)
    phases = np.random.uniform(0, 2*np.pi, N_trunc)
    return make_snap(phases, N_trunc)


def binomial_snap(p: float, N_trunc: int) -> qt.Qobj:
    """
    SNAP gate with phases following binomial coefficients.
    
    Useful for creating superposition states.
    
    Args:
        p: Parameter for phase modulation
        N_trunc: Fock space truncation dimension
    
    Returns:
        Binomial SNAP unitary
    """
    from scipy.special import binom
    n = np.arange(N_trunc)
    phases = p * np.array([binom(N_trunc-1, k) for k in n])
    return make_snap(phases, N_trunc)


def kerr_evolution_snap(chi: float, t: float, N_trunc: int) -> qt.Qobj:
    """
    SNAP gate equivalent to Kerr evolution.
    
    Simulates the evolution under H = χ a†a(a†a - 1)/2 for time t.
    
    Args:
        chi: Kerr nonlinearity strength
        t: Evolution time
        N_trunc: Fock space truncation dimension
    
    Returns:
        Kerr evolution SNAP unitary
    """
    n = np.arange(N_trunc)
    phases = -chi * t * n * (n - 1) / 2
    return make_snap(phases, N_trunc)


def displacement_correction_snap(alpha: complex, N_trunc: int) -> qt.Qobj:
    """
    SNAP gate for correcting displacement errors.
    
    Creates phases that can help correct for unwanted displacement.
    
    Args:
        alpha: Displacement amplitude to correct
        N_trunc: Fock space truncation dimension
    
    Returns:
        Displacement correction SNAP unitary
    """
    n = np.arange(N_trunc)
    # Phase pattern that interferes with displacement
    phases = np.angle(alpha) * n - np.abs(alpha)**2 * np.sin(2*np.pi*n/N_trunc)
    return make_snap(phases, N_trunc)


def get_snap_phases(U_snap: qt.Qobj) -> np.ndarray:
    """
    Extract phase angles from a SNAP unitary.
    
    Args:
        U_snap: SNAP unitary (must be diagonal)
    
    Returns:
        Array of phase angles
    """
    diag = U_snap.diag()
    phases = np.angle(diag)
    return phases


def snap_fidelity(U1: qt.Qobj, U2: qt.Qobj) -> float:
    """
    Compute fidelity between two SNAP gates.
    
    Args:
        U1: First SNAP unitary
        U2: Second SNAP unitary
    
    Returns:
        Fidelity in [0, 1]
    """
    d = U1.shape[0]
    trace = np.trace(U1.dag() * U2)
    return np.abs(trace)**2 / d**2