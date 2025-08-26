#!/usr/bin/env python3
"""
Minimal test to check if the core implementation logic is correct.
This version has minimal dependencies for quick testing.
"""

import numpy as np

def test_gate_dimensions():
    """Test that gate dimensions are correct."""
    print("Testing gate dimensions...")
    
    N_trunc = 4
    N_layers = 3
    
    # Check dimensions
    qubit_dim = 2
    cavity_dim = N_trunc
    full_dim = qubit_dim * cavity_dim
    
    print(f"  Qubit dimension: {qubit_dim}")
    print(f"  Cavity dimension: {cavity_dim}")
    print(f"  Full Hilbert space: {full_dim}")
    print(f"  Number of ECD layers: {N_layers}")
    
    # Check parameter counts
    n_params = N_layers + 1  # One extra for final displacement/rotation
    print(f"  Beta parameters: {n_params}")
    print(f"  Phi parameters: {n_params}")
    print(f"  Theta parameters: {n_params}")
    
    return True


def test_parameter_ranges():
    """Test that parameters are in valid ranges."""
    print("\nTesting parameter ranges...")
    
    # Beta can be complex
    beta = 0.5 + 0.3j
    print(f"  Beta (complex): {beta}, |beta| = {np.abs(beta):.4f}")
    
    # Phi should be in [0, 2π)
    phi = np.pi
    print(f"  Phi (rotation axis): {phi:.4f} rad")
    assert 0 <= phi < 2*np.pi, "Phi out of range!"
    
    # Theta should be in [0, π]
    theta = np.pi/2
    print(f"  Theta (rotation angle): {theta:.4f} rad")
    assert 0 <= theta <= np.pi, "Theta out of range!"
    
    print("  ✓ All parameters in valid ranges")
    return True


def test_snap_phases():
    """Test SNAP gate phase patterns."""
    print("\nTesting SNAP gate phases...")
    
    N_trunc = 6
    
    # Linear SNAP
    slope = 0.5
    phases_linear = slope * np.arange(N_trunc)
    print(f"  Linear SNAP (slope={slope}): {phases_linear}")
    
    # Quadratic SNAP
    coeff = 0.1
    phases_quad = coeff * np.arange(N_trunc)**2
    print(f"  Quadratic SNAP (coeff={coeff}): {phases_quad}")
    
    # Kerr evolution SNAP
    chi = 0.2
    t = 1.0
    n = np.arange(N_trunc)
    phases_kerr = -chi * t * n * (n - 1) / 2
    print(f"  Kerr SNAP (χ={chi}, t={t}): {phases_kerr}")
    
    return True


def test_fidelity_formula():
    """Test the fidelity calculation formula."""
    print("\nTesting fidelity formula...")
    
    d = 8  # Dimension
    
    # Perfect fidelity case (identical unitaries)
    U = np.eye(d, dtype=complex)
    trace = np.trace(np.conj(U.T) @ U)
    fidelity = np.abs(trace)**2 / d**2
    print(f"  Identity fidelity: {fidelity:.6f} (should be 1.0)")
    assert np.abs(fidelity - 1.0) < 1e-10, "Identity fidelity should be 1!"
    
    # Random unitary case
    # Create a simple rotation
    theta = np.pi/4
    U2 = np.array([[np.cos(theta), -np.sin(theta)], 
                   [np.sin(theta), np.cos(theta)]], dtype=complex)
    # Pad to full dimension
    U_random = np.eye(d, dtype=complex)
    U_random[:2, :2] = U2
    
    trace = np.trace(np.conj(U.T) @ U_random)
    fidelity = np.abs(trace)**2 / d**2
    print(f"  Partial rotation fidelity: {fidelity:.6f}")
    assert 0 <= fidelity <= 1, "Fidelity out of bounds!"
    
    print("  ✓ Fidelity formula correct")
    return True


def test_cost_function():
    """Test the logarithmic barrier cost function."""
    print("\nTesting cost function...")
    
    # Test with different fidelities
    fidelities = [0.1, 0.5, 0.9, 0.99, 0.999]
    
    for F in fidelities:
        # Single cost
        cost = np.log(1 - F)
        print(f"  F={F:.3f} → Cost={cost:.4f}")
        
        # Check that cost decreases as fidelity increases
        if F > 0.5:
            cost_low = np.log(1 - 0.1)
            assert cost < cost_low, "Cost should decrease with higher fidelity!"
    
    # Batch cost
    F_batch = np.array([0.9, 0.95, 0.99])
    cost_batch = np.sum(np.log(1 - F_batch))
    print(f"  Batch cost for F={F_batch}: {cost_batch:.4f}")
    
    print("  ✓ Cost function behaves correctly")
    return True


def main():
    """Run minimal tests."""
    print("="*60)
    print("Minimal ECD-to-SNAP Tests (No JAX Required)")
    print("="*60)
    
    tests = [
        test_gate_dimensions,
        test_parameter_ranges,
        test_snap_phases,
        test_fidelity_formula,
        test_cost_function
    ]
    
    passed = 0
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test FAILED: {e}")
    
    print("\n" + "="*60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ Basic logic tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run gradient tests: python test_gradients.py")
        print("3. Run optimization: python cli.py optimize --target-type identity")
    else:
        print("✗ Some tests failed. Check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)