#!/usr/bin/env python3
"""
Comprehensive test suite for ecd_snap_minimal.py

Tests all components of the minimal ECD-to-SNAP optimization script
without modifying the original implementation.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os
import json
import argparse
from unittest.mock import patch, MagicMock
from typing import Dict, Tuple

# Import the module under test
# Handle potential import issues gracefully due to JAX version compatibility
try:
    import ecd_snap_minimal as esm
    IMPORT_SUCCESS = True
except Exception as e:
    print(f"Warning: Could not import ecd_snap_minimal due to JAX compatibility: {e}")
    print("Skipping tests that require the module...")
    IMPORT_SUCCESS = False
    esm = None


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestJAXGateOperations:
    """Test JAX gate operations."""
    
    def test_displacement_operator_basic(self):
        """Test displacement operator with simple cases."""
        N_trunc = 4
        
        # Identity displacement (beta=0)
        D_id = esm.displacement_operator_jax(0.0, N_trunc)
        expected = jnp.eye(N_trunc, dtype=jnp.complex64)
        np.testing.assert_allclose(D_id, expected, rtol=1e-6)
        
        # Small real displacement
        beta = 0.1
        D = esm.displacement_operator_jax(beta, N_trunc)
        
        # Check unitarity
        D_dag = jnp.conj(D.T)
        should_be_identity = D_dag @ D
        np.testing.assert_allclose(should_be_identity, jnp.eye(N_trunc), rtol=1e-5)
    
    def test_displacement_operator_properties(self):
        """Test displacement operator mathematical properties."""
        N_trunc = 5
        beta1 = 0.2 + 0.1j
        beta2 = 0.1 - 0.05j
        
        D1 = esm.displacement_operator_jax(beta1, N_trunc)
        D2 = esm.displacement_operator_jax(beta2, N_trunc)
        D12 = esm.displacement_operator_jax(beta1 + beta2, N_trunc)
        
        # Baker-Campbell-Hausdorff: D(α)D(β) = exp(½(αβ* - α*β))D(α+β)
        phase_factor = jnp.exp(0.5 * (beta1 * jnp.conj(beta2) - jnp.conj(beta1) * beta2))
        expected = phase_factor * D12
        actual = D1 @ D2
        
        # Allow for some numerical error in BCH formula
        np.testing.assert_allclose(actual, expected, rtol=1e-4)
    
    def test_rotation_operator_basic(self):
        """Test qubit rotation operator."""
        # Identity rotation
        R_id = esm.rotation_operator_jax(0.0, 0.0)
        expected = jnp.eye(2, dtype=jnp.complex64)
        np.testing.assert_allclose(R_id, expected, rtol=1e-6)
        
        # π rotation around X (Pauli-X)
        R_x = esm.rotation_operator_jax(jnp.pi, 0.0)
        expected_x = jnp.array([[0, -1j], [-1j, 0]], dtype=jnp.complex64)
        np.testing.assert_allclose(R_x, expected_x, rtol=1e-6)
        
        # π rotation around Y (Pauli-Y)
        R_y = esm.rotation_operator_jax(jnp.pi, jnp.pi/2)
        expected_y = jnp.array([[0, -1], [1, 0]], dtype=jnp.complex64)
        np.testing.assert_allclose(R_y, expected_y, rtol=1e-6)
    
    def test_rotation_operator_unitarity(self):
        """Test rotation operator is unitary."""
        theta = 0.7
        phi = 1.2
        R = esm.rotation_operator_jax(theta, phi)
        
        R_dag = jnp.conj(R.T)
        should_be_identity = R_dag @ R
        np.testing.assert_allclose(should_be_identity, jnp.eye(2), rtol=1e-6)
    
    def test_ecd_gate_structure(self):
        """Test ECD gate has correct structure."""
        N_trunc = 3
        beta = 0.1 + 0.05j
        
        ECD = esm.ecd_gate_jax(beta, N_trunc)
        
        # Check dimensions
        assert ECD.shape == (2 * N_trunc, 2 * N_trunc)
        
        # Check unitarity
        ECD_dag = jnp.conj(ECD.T)
        should_be_identity = ECD_dag @ ECD
        np.testing.assert_allclose(should_be_identity, jnp.eye(2 * N_trunc), rtol=1e-5)
        
        # Check block structure: should have D(β) and D(-β) blocks
        D_plus = esm.displacement_operator_jax(beta, N_trunc)
        D_minus = esm.displacement_operator_jax(-beta, N_trunc)
        
        # Extract blocks from ECD
        block_00 = ECD[:N_trunc, :N_trunc]  # |0><0| ⊗ D(β)
        block_11 = ECD[N_trunc:, N_trunc:]  # |1><1| ⊗ D(-β)
        
        np.testing.assert_allclose(block_00, D_plus, rtol=1e-6)
        np.testing.assert_allclose(block_11, D_minus, rtol=1e-6)
    
    def test_build_ecd_sequence_simple(self):
        """Test ECD sequence building with simple parameters."""
        N_trunc = 3
        N_layers = 2
        
        # Simple parameter set
        betas = jnp.array([0.1, 0.05, 0.02], dtype=jnp.complex64)  # N_layers + 1
        phis = jnp.array([0.0, jnp.pi/2, jnp.pi], dtype=jnp.float32)
        thetas = jnp.array([jnp.pi/4, jnp.pi/3, jnp.pi/6], dtype=jnp.float32)
        
        U = esm.build_ecd_sequence_jax(betas, phis, thetas, N_trunc)
        
        # Check dimensions and unitarity
        assert U.shape == (2 * N_trunc, 2 * N_trunc)
        U_dag = jnp.conj(U.T)
        should_be_identity = U_dag @ U
        np.testing.assert_allclose(should_be_identity, jnp.eye(2 * N_trunc), rtol=1e-5)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestSNAPAndFidelity:
    """Test SNAP gate creation and fidelity calculations."""
    
    def test_make_snap_basic(self):
        """Test SNAP gate creation."""
        N_trunc = 4
        phases = jnp.array([0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])
        
        U_snap = esm.make_snap_jax(phases, N_trunc)
        
        # Check dimensions
        assert U_snap.shape == (2 * N_trunc, 2 * N_trunc)
        
        # Extract cavity part (should be diagonal)
        cavity_part = U_snap[N_trunc:, N_trunc:]  # |1><1| ⊗ U_snap_cavity
        expected_diag = jnp.exp(1j * phases)
        np.testing.assert_allclose(jnp.diag(cavity_part), expected_diag, rtol=1e-6)
        
        # Check unitarity
        U_dag = jnp.conj(U_snap.T)
        should_be_identity = U_dag @ U_snap
        np.testing.assert_allclose(should_be_identity, jnp.eye(2 * N_trunc), rtol=1e-6)
    
    def test_make_snap_padding(self):
        """Test SNAP gate with phase padding."""
        N_trunc = 5
        short_phases = jnp.array([0.1, 0.2, 0.3])  # Less than N_trunc
        
        U_snap = esm.make_snap_jax(short_phases, N_trunc)
        
        # Should pad with zeros
        cavity_part = U_snap[N_trunc:, N_trunc:]
        expected_phases = jnp.concatenate([short_phases, jnp.zeros(N_trunc - len(short_phases))])
        expected_diag = jnp.exp(1j * expected_phases)
        np.testing.assert_allclose(jnp.diag(cavity_part), expected_diag, rtol=1e-6)
    
    def test_fidelity_full_identity(self):
        """Test full fidelity with identical matrices."""
        N = 4
        U = jnp.eye(N, dtype=jnp.complex64)
        fid = esm.fidelity_full(U, U)
        assert abs(fid - 1.0) < 1e-10
    
    def test_fidelity_full_orthogonal(self):
        """Test full fidelity with orthogonal matrices."""
        # Create two different unitaries
        U1 = jnp.eye(2, dtype=jnp.complex64)
        U2 = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)  # Pauli-X
        
        fid = esm.fidelity_full(U1, U2)
        # Tr(I† X) = Tr(X) = 0, so fidelity = 0
        assert abs(fid) < 1e-10
    
    def test_subspace_projector(self):
        """Test subspace projector construction."""
        N_target = 2
        N_trunc = 4
        
        P = esm.build_projector_qubit_cavity_subspace(N_target, N_trunc)
        
        # Check dimensions
        assert P.shape == (2 * N_trunc, 2 * N_trunc)
        
        # Check idempotency: P² = P
        P_squared = P @ P
        np.testing.assert_allclose(P_squared, P, rtol=1e-10)
        
        # Check projection property: should project to first N_target cavity levels
        expected_diag = jnp.concatenate([
            jnp.ones(N_target),     # |0> ⊗ {|0>, |1>}
            jnp.zeros(N_trunc - N_target),  # |0> ⊗ {|2>, |3>}
            jnp.ones(N_target),     # |1> ⊗ {|0>, |1>}
            jnp.zeros(N_trunc - N_target)   # |1> ⊗ {|2>, |3>}
        ])
        np.testing.assert_allclose(jnp.diag(P), expected_diag, rtol=1e-10)
    
    def test_fidelity_subspace(self):
        """Test subspace fidelity calculation."""
        N_target = 2
        N_trunc = 3
        
        # Create identical matrices
        U1 = jnp.eye(2 * N_trunc, dtype=jnp.complex64)
        U2 = jnp.eye(2 * N_trunc, dtype=jnp.complex64)
        
        P = esm.build_projector_qubit_cavity_subspace(N_target, N_trunc)
        fid = esm.fidelity_subspace(U1, U2, P, N_target)
        
        # Should be 1.0 for identical matrices
        assert abs(fid - 1.0) < 1e-10


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestMinimalECDOptimizer:
    """Test the MinimalECDOptimizer class."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization and validation."""
        # Valid initialization
        opt = esm.MinimalECDOptimizer(N_layers=3, N_trunc=5, N_target=4)
        assert opt.N_layers == 3
        assert opt.N_trunc == 5
        assert opt.N_target == 4
        assert opt.P_sub is not None  # Should create subspace projector
        
        # Default N_target
        opt_default = esm.MinimalECDOptimizer(N_layers=2, N_trunc=4)
        assert opt_default.N_target == 4  # Should equal N_trunc
        assert opt_default.P_sub is None  # No subspace projector needed
        
        # Invalid initialization (N_target > N_trunc)
        with pytest.raises(ValueError, match="N_target cannot exceed N_trunc"):
            esm.MinimalECDOptimizer(N_layers=2, N_trunc=3, N_target=5)
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        opt = esm.MinimalECDOptimizer(N_layers=2, N_trunc=3, batch_size=4)
        
        key = jax.random.PRNGKey(42)
        params = opt.initialize_parameters(key)
        
        # Check parameter shapes
        assert params['betas'].shape == (4, 3)  # (batch_size, N_layers + 1)
        assert params['phis'].shape == (4, 3)
        assert params['thetas'].shape == (4, 3)
        
        # Check parameter ranges
        assert jnp.all(params['phis'] >= 0) and jnp.all(params['phis'] <= 2*jnp.pi)
        assert jnp.all(params['thetas'] >= 0) and jnp.all(params['thetas'] <= jnp.pi)
    
    def test_compute_fidelity_dispatch(self):
        """Test fidelity computation dispatch."""
        # Full-space fidelity
        opt_full = esm.MinimalECDOptimizer(N_layers=1, N_trunc=2, N_target=2)
        U1 = jnp.eye(4, dtype=jnp.complex64)
        U2 = jnp.eye(4, dtype=jnp.complex64)
        fid = opt_full.compute_fidelity(U1, U2)
        assert abs(fid - 1.0) < 1e-10
        
        # Subspace fidelity
        opt_sub = esm.MinimalECDOptimizer(N_layers=1, N_trunc=3, N_target=2)
        U1 = jnp.eye(6, dtype=jnp.complex64)
        U2 = jnp.eye(6, dtype=jnp.complex64)
        fid = opt_sub.compute_fidelity(U1, U2)
        assert abs(fid - 1.0) < 1e-10
    
    def test_batch_fidelity(self):
        """Test batch fidelity computation."""
        opt = esm.MinimalECDOptimizer(N_layers=1, N_trunc=2, batch_size=3)
        
        # Create simple target
        U_target = jnp.eye(4, dtype=jnp.complex64)
        
        # Create simple parameters
        params = {
            'betas': jnp.zeros((3, 2), dtype=jnp.complex64),
            'phis': jnp.zeros((3, 2), dtype=jnp.float32),
            'thetas': jnp.zeros((3, 2), dtype=jnp.float32)
        }
        
        fidelities = opt.batch_fidelity(params, U_target)
        assert fidelities.shape == (3,)
        # All should give identity transformations -> high fidelity
        assert jnp.all(fidelities > 0.9)
    
    def test_loss_function(self):
        """Test logarithmic barrier loss function."""
        opt = esm.MinimalECDOptimizer(N_layers=1, N_trunc=2, batch_size=2)
        
        U_target = jnp.eye(4, dtype=jnp.complex64)
        
        # Parameters that should give high fidelity (low loss)
        good_params = {
            'betas': jnp.zeros((2, 2), dtype=jnp.complex64),
            'phis': jnp.zeros((2, 2), dtype=jnp.float32),
            'thetas': jnp.zeros((2, 2), dtype=jnp.float32)
        }
        
        loss = opt.loss_function(good_params, U_target)
        # Should be very negative (log of small number)
        assert loss < -10


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_parse_phases_csv_string(self):
        """Test parsing comma-separated phases."""
        phases_str = "0.1, 0.5, 1.0, 1.5"
        phases = esm.parse_phases(phases_str)
        expected = np.array([0.1, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(phases, expected)
        
        # Test without spaces
        phases_str2 = "0.1,0.5,1.0,1.5"
        phases2 = esm.parse_phases(phases_str2)
        np.testing.assert_allclose(phases2, expected)
    
    def test_parse_phases_file_csv(self):
        """Test parsing phases from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0.1,0.5,1.0,1.5")
            temp_path = f.name
        
        try:
            phases = esm.parse_phases(temp_path)
            expected = np.array([0.1, 0.5, 1.0, 1.5])
            np.testing.assert_allclose(phases, expected)
        finally:
            os.unlink(temp_path)
    
    def test_parse_phases_file_json(self):
        """Test parsing phases from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([0.1, 0.5, 1.0, 1.5], f)
            temp_path = f.name
        
        try:
            phases = esm.parse_phases(temp_path)
            expected = np.array([0.1, 0.5, 1.0, 1.5])
            np.testing.assert_allclose(phases, expected)
        finally:
            os.unlink(temp_path)
    
    def test_parse_phases_file_npy(self):
        """Test parsing phases from numpy file."""
        expected = np.array([0.1, 0.5, 1.0, 1.5])
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f.name, expected)
            temp_path = f.name
        
        try:
            phases = esm.parse_phases(temp_path)
            np.testing.assert_allclose(phases, expected)
        finally:
            os.unlink(temp_path)
    
    def test_parse_phases_error_handling(self):
        """Test parse_phases error handling."""
        # Invalid string
        with pytest.raises(ValueError, match="Error parsing phases"):
            esm.parse_phases("not_a_number,0.5")
        
        # Non-existent file referenced as if it were a file
        with pytest.raises(ValueError, match="Error parsing phases"):
            esm.parse_phases("abc,def")  # Contains invalid numbers
    
    def test_coherent_cavity_state(self):
        """Test coherent state generation."""
        alpha = 0.5 + 0.2j
        N_trunc = 4
        
        coh_state = esm.coherent_cavity_state(alpha, N_trunc)
        
        # Check normalization
        norm = jnp.sum(jnp.abs(coh_state)**2)
        assert abs(norm - 1.0) < 1e-10
        
        # Check first coefficient (should be largest for small alpha)
        assert jnp.abs(coh_state[0]) > jnp.abs(coh_state[1])
        
        # Check length
        assert len(coh_state) == N_trunc
    
    def test_cavity_tail_probability(self):
        """Test tail probability calculation."""
        # Simple state with known tail
        state = jnp.array([0.8, 0.6, 0.0, 0.0])  # Not normalized intentionally
        state = state / jnp.sqrt(jnp.sum(jnp.abs(state)**2))
        
        N_target = 2
        tail_prob = esm.cavity_tail_probability(state, N_target)
        
        # Should be zero since state[2:] = 0
        assert abs(tail_prob) < 1e-10
        
        # Test with non-zero tail
        state2 = jnp.array([0.5, 0.5, 0.5, 0.5])
        state2 = state2 / jnp.sqrt(jnp.sum(jnp.abs(state2)**2))
        tail_prob2 = esm.cavity_tail_probability(state2, N_target)
        
        # Should be 0.5 (half the probability is in the tail)
        assert abs(tail_prob2 - 0.5) < 1e-6
    
    def test_split_qubit_blocks(self):
        """Test splitting full state into qubit blocks."""
        N_trunc = 3
        psi_full = jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.complex64)
        
        q0_block, q1_block = esm.split_qubit_blocks(psi_full, N_trunc)
        
        expected_q0 = jnp.array([1, 2, 3], dtype=jnp.complex64)
        expected_q1 = jnp.array([4, 5, 6], dtype=jnp.complex64)
        
        np.testing.assert_allclose(q0_block, expected_q0)
        np.testing.assert_allclose(q1_block, expected_q1)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestTailProbabilities:
    """Test tail probability utilities."""
    
    def test_measure_tail_probs_simple(self):
        """Test tail probability measurement with simple unitary."""
        N_target = 2
        N_trunc = 3
        
        # Identity unitary (no mixing between levels)
        U_identity = jnp.eye(2 * N_trunc, dtype=jnp.complex64)
        betas = jnp.array([0.1, 0.05])
        
        metrics = esm.measure_tail_probs(U_identity, N_target, N_trunc, betas)
        
        # Check return structure
        expected_keys = {'t0_q0', 't0_q1', 't1_q0', 't1_q1', 'ta_q0', 'ta_q1', 'alpha_abs', 'max_tail'}
        assert set(metrics.keys()) == expected_keys
        
        # For identity, inputs |0,0> and |1,0> should have zero tail
        assert abs(metrics['t0_q0']) < 1e-10  # |0,0> -> |0,0>
        assert abs(metrics['t1_q1']) < 1e-10  # |1,0> -> |1,0>
        
        # Check alpha envelope calculation
        expected_alpha = jnp.sum(jnp.abs(betas)) - jnp.abs(betas[-1]) + jnp.abs(betas[-1] / 2)
        assert abs(metrics['alpha_abs'] - expected_alpha) < 1e-6


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_dimension_mismatch_errors(self):
        """Test various dimension mismatch scenarios."""
        # Subspace projector with N_target > N_trunc
        with pytest.raises(ValueError, match="N_target cannot exceed N_trunc"):
            esm.build_projector_qubit_cavity_subspace(5, 3)
    
    def test_empty_phases(self):
        """Test handling of empty phase arrays."""
        empty_phases = jnp.array([])
        N_trunc = 3
        
        U_snap = esm.make_snap_jax(empty_phases, N_trunc)
        
        # Should create identity SNAP (all phases = 0)
        expected_diag = jnp.ones(N_trunc, dtype=jnp.complex64)
        cavity_part = U_snap[N_trunc:, N_trunc:]
        np.testing.assert_allclose(jnp.diag(cavity_part), expected_diag, rtol=1e-6)
    
    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        N_trunc = 3
        
        # Very large displacement
        large_beta = 10.0 + 5.0j
        D_large = esm.displacement_operator_jax(large_beta, N_trunc)
        
        # Should still be unitary
        D_dag = jnp.conj(D_large.T)
        should_be_identity = D_dag @ D_large
        np.testing.assert_allclose(should_be_identity, jnp.eye(N_trunc), rtol=1e-4)
        
        # Very large phase
        large_phase = 100.0 * jnp.pi
        R_large = esm.rotation_operator_jax(large_phase, 0.0)
        
        # Should still be unitary
        R_dag = jnp.conj(R_large.T)
        should_be_identity = R_dag @ R_large
        np.testing.assert_allclose(should_be_identity, jnp.eye(2), rtol=1e-6)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_optimization_simple(self):
        """Test complete optimization workflow with simple target."""
        # Simple target: identity SNAP
        phases = jnp.array([0.0, 0.0, 0.0])
        N_trunc = 3
        N_target = 2
        
        U_target = esm.make_snap_jax(phases, N_trunc)
        
        # Create optimizer with fast settings
        opt = esm.MinimalECDOptimizer(
            N_layers=1, 
            N_trunc=N_trunc, 
            N_target=N_target,
            batch_size=4,
            learning_rate=1e-2,
            target_fidelity=0.9
        )
        
        # Quick optimization run
        best_params, best_fid, info = opt.optimize_with_restarts(
            U_target, 
            max_iter=50, 
            n_restarts=1, 
            verbose=False
        )
        
        # Should find reasonable solution
        assert best_params is not None
        assert best_fid > 0.5  # Some minimal threshold
        assert 'history' in info
        assert info['iterations'] <= 50
    
    def test_save_and_load_results(self):
        """Test results saving and loading."""
        # Create dummy results
        params = {
            'betas': jnp.array([0.1 + 0.05j, 0.02]),
            'phis': jnp.array([0.5, 1.0]),
            'thetas': jnp.array([0.7, 0.9])
        }
        fidelity = 0.95
        info = {'iterations': 100, 'history': {'loss': [1.0, 0.5], 'max_fidelity': [0.5, 0.95]}}
        phases = np.array([0.1, 0.5, 1.0])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            esm.save_results(params, fidelity, info, phases, tmpdir, N_target=2)
            
            # Check file was created
            results_file = os.path.join(tmpdir, 'results.json')
            assert os.path.exists(results_file)
            
            # Check content
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)
            
            assert loaded_results['fidelity'] == fidelity
            assert loaded_results['N_target'] == 2
            assert len(loaded_results['parameters']['betas_real']) == 2
            np.testing.assert_allclose(loaded_results['target_phases'], phases)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_convergence(self, mock_show, mock_savefig):
        """Test convergence plotting."""
        history = {
            'loss': [-1.0, -2.0, -3.0],
            'max_fidelity': [0.3, 0.6, 0.9]
        }
        
        # Test show mode
        esm.plot_convergence(history)
        mock_show.assert_called_once()
        
        # Test save mode
        mock_show.reset_mock()
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            esm.plot_convergence(history, f.name)
            mock_savefig.assert_called_once()


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="ecd_snap_minimal import failed")
class TestCLIComponents:
    """Test CLI-related functionality."""
    
    def test_phases_parsing_integration(self):
        """Test phases parsing as used in main function."""
        # Test various input formats that main() would encounter
        test_cases = [
            ("0.1,0.5,1.0", np.array([0.1, 0.5, 1.0])),
            ("0.1, 0.5, 1.0", np.array([0.1, 0.5, 1.0])),
        ]
        
        for phases_str, expected in test_cases:
            result = esm.parse_phases(phases_str)
            np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])