"""
Shared test configuration and utilities for pytest.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure JAX for testing
jax.config.update("jax_enable_x64", True)  # Use float64 for better numerical precision in tests
jax.config.update("jax_platform_name", "cpu")  # Use CPU for consistent test results

# Common test parameters
DEFAULT_TEST_PARAMS = {
    'N_trunc': 4,
    'N_layers': 3,
    'batch_size': 8,
    'learning_rate': 0.01,
    'max_iter': 100,
    'seed': 42
}

# Test fixtures and utilities
def create_test_snap_target(N_trunc: int, target_type: str = 'identity'):
    """Create a SNAP target for testing."""
    from src.snap_targets import make_snap_full_space
    
    if target_type == 'identity':
        phases = np.zeros(N_trunc)
    elif target_type == 'linear':
        phases = np.arange(N_trunc) * 0.1
    elif target_type == 'quadratic':
        phases = np.arange(N_trunc)**2 * 0.01
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    U_target = make_snap_full_space(phases, N_trunc)
    return jnp.array(U_target.full()), phases


def assert_fidelity_improved(fidelity_before: float, fidelity_after: float, 
                            min_improvement: float = 0.01):
    """Assert that fidelity has improved by at least min_improvement."""
    improvement = fidelity_after - fidelity_before
    assert improvement >= min_improvement, \
        f"Fidelity did not improve enough: {improvement:.6f} < {min_improvement}"


def assert_valid_fidelity(fidelity: float):
    """Assert that fidelity is in valid range [0, 1]."""
    assert 0 <= fidelity <= 1, f"Invalid fidelity: {fidelity}"
    

def assert_unitary(matrix: jnp.ndarray, tolerance: float = 1e-6):
    """Assert that a matrix is unitary."""
    d = matrix.shape[0]
    identity = jnp.eye(d)
    product = matrix @ matrix.conj().T
    error = jnp.max(jnp.abs(product - identity))
    assert error < tolerance, f"Matrix is not unitary, error: {error}"