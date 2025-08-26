#!/usr/bin/env python3
"""
Test suite for ecd_snap_tensorflow.py

Tests basic functionalities including:
- Gate operations (displacement, rotation, ECD)
- SNAP gate creation
- Fidelity calculations
- Optimization components
- Tail probability utilities
- Input parsing and validation
"""

import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
import json
import sys

# Add the current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from our script
from ecd_snap_tensorflow import (
    displacement_operator_tf,
    rotation_operator_tf,
    ecd_gate_tf,
    build_ecd_sequence_tf,
    make_snap_tf,
    fidelity_full_tf,
    fidelity_subspace_tf,
    build_projector_qubit_cavity_subspace_tf,
    MinimalECDOptimizer,
    parse_phases,
    _adaptive_batch_size,
    _success,
    cavity_tail_probability_tf,
    coherent_cavity_state_tf,
    DT_FLOAT,
    DT_COMPLEX
)


class TestGateOperations(unittest.TestCase):
    """Test basic quantum gate operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N_trunc = 4
        self.beta = 0.1 + 0.2j
        self.theta = np.pi / 4
        self.phi = np.pi / 3
    
    def test_displacement_operator_shape(self):
        """Test displacement operator has correct shape."""
        D = displacement_operator_tf(self.beta, self.N_trunc)
        self.assertEqual(D.shape, (self.N_trunc, self.N_trunc))
        self.assertEqual(D.dtype, DT_COMPLEX)
    
    def test_displacement_operator_unitary(self):
        """Test displacement operator is approximately unitary."""
        D = displacement_operator_tf(self.beta, self.N_trunc)
        D_dag = tf.linalg.adjoint(D)
        identity = D_dag @ D
        expected_identity = tf.eye(self.N_trunc, dtype=DT_COMPLEX)
        
        # Check if close to identity (within numerical precision)
        diff = tf.reduce_max(tf.abs(identity - expected_identity))
        self.assertLess(float(diff.numpy()), 1e-6)
    
    def test_rotation_operator_shape(self):
        """Test rotation operator has correct shape."""
        R = rotation_operator_tf(self.theta, self.phi)
        self.assertEqual(R.shape, (2, 2))
        self.assertEqual(R.dtype, DT_COMPLEX)
    
    def test_rotation_operator_unitary(self):
        """Test rotation operator is unitary."""
        R = rotation_operator_tf(self.theta, self.phi)
        R_dag = tf.linalg.adjoint(R)
        identity = R_dag @ R
        expected_identity = tf.eye(2, dtype=DT_COMPLEX)
        
        diff = tf.reduce_max(tf.abs(identity - expected_identity))
        self.assertLess(float(diff.numpy()), 1e-6)
    
    def test_ecd_gate_shape(self):
        """Test ECD gate has correct shape."""
        ECD = ecd_gate_tf(self.beta, self.N_trunc)
        expected_shape = (2 * self.N_trunc, 2 * self.N_trunc)
        self.assertEqual(ECD.shape, expected_shape)
        self.assertEqual(ECD.dtype, DT_COMPLEX)
    
    def test_ecd_gate_unitary(self):
        """Test ECD gate is approximately unitary."""
        ECD = ecd_gate_tf(self.beta, self.N_trunc)
        ECD_dag = tf.linalg.adjoint(ECD)
        identity = ECD_dag @ ECD
        expected_identity = tf.eye(2 * self.N_trunc, dtype=DT_COMPLEX)
        
        diff = tf.reduce_max(tf.abs(identity - expected_identity))
        self.assertLess(float(diff.numpy()), 1e-5)


class TestSNAPGate(unittest.TestCase):
    """Test SNAP gate creation and properties."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.phases = np.array([0.0, 0.5, 1.0, 1.5])
        self.N_trunc = 6
    
    def test_snap_gate_shape(self):
        """Test SNAP gate has correct shape."""
        U_snap = make_snap_tf(self.phases, self.N_trunc)
        expected_shape = (2 * self.N_trunc, 2 * self.N_trunc)
        self.assertEqual(U_snap.shape, expected_shape)
        self.assertEqual(U_snap.dtype, DT_COMPLEX)
    
    def test_snap_gate_unitary(self):
        """Test SNAP gate is unitary."""
        U_snap = make_snap_tf(self.phases, self.N_trunc)
        U_snap_dag = tf.linalg.adjoint(U_snap)
        identity = U_snap_dag @ U_snap
        expected_identity = tf.eye(2 * self.N_trunc, dtype=DT_COMPLEX)
        
        diff = tf.reduce_max(tf.abs(identity - expected_identity))
        self.assertLess(float(diff.numpy()), 1e-6)
    
    def test_snap_phase_padding(self):
        """Test SNAP gate handles phase padding correctly."""
        short_phases = np.array([0.0, 0.5])  # Fewer than N_trunc
        U_snap = make_snap_tf(short_phases, self.N_trunc)
        
        # Should not raise an error and should have correct shape
        expected_shape = (2 * self.N_trunc, 2 * self.N_trunc)
        self.assertEqual(U_snap.shape, expected_shape)


class TestFidelityCalculations(unittest.TestCase):
    """Test fidelity calculation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N_trunc = 4
        self.N_target = 3
        
        # Create simple test unitaries
        self.U1 = tf.eye(2 * self.N_trunc, dtype=DT_COMPLEX)
        self.U2 = tf.eye(2 * self.N_trunc, dtype=DT_COMPLEX)
        
        # Create projector for subspace fidelity
        self.P = build_projector_qubit_cavity_subspace_tf(self.N_target, self.N_trunc)
    
    def test_fidelity_identical_unitaries(self):
        """Test fidelity of identical unitaries is 1."""
        fid = fidelity_full_tf(self.U1, self.U2)
        self.assertAlmostEqual(float(fid.numpy()), 1.0, places=6)
    
    def test_subspace_fidelity_identical(self):
        """Test subspace fidelity of identical unitaries is 1."""
        fid = fidelity_subspace_tf(self.U1, self.U2, self.P, self.N_target)
        self.assertAlmostEqual(float(fid.numpy()), 1.0, places=6)
    
    def test_projector_shape(self):
        """Test projector has correct shape."""
        self.assertEqual(self.P.shape, (2 * self.N_trunc, 2 * self.N_trunc))
        self.assertEqual(self.P.dtype, DT_COMPLEX)
    
    def test_projector_properties(self):
        """Test projector is Hermitian and idempotent."""
        P_dag = tf.linalg.adjoint(self.P)
        
        # Test Hermitian property: P = P†
        diff_hermitian = tf.reduce_max(tf.abs(self.P - P_dag))
        self.assertLess(float(diff_hermitian.numpy()), 1e-6)
        
        # Test idempotent property: P² = P
        P_squared = self.P @ self.P
        diff_idempotent = tf.reduce_max(tf.abs(P_squared - self.P))
        self.assertLess(float(diff_idempotent.numpy()), 1e-6)


class TestECDSequence(unittest.TestCase):
    """Test ECD sequence building."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N_layers = 2
        self.N_trunc = 4
        
        # Create parameter arrays
        self.betas = tf.constant([0.1+0.1j, 0.2+0.2j, 0.05+0.05j], dtype=DT_COMPLEX)
        self.phis = tf.constant([0.0, np.pi/2, np.pi], dtype=DT_FLOAT)  
        self.thetas = tf.constant([np.pi/4, np.pi/3, np.pi/6], dtype=DT_FLOAT)
    
    def test_ecd_sequence_shape(self):
        """Test ECD sequence has correct shape."""
        U = build_ecd_sequence_tf(self.betas, self.phis, self.thetas, self.N_trunc)
        expected_shape = (2 * self.N_trunc, 2 * self.N_trunc)
        self.assertEqual(U.shape, expected_shape)
        self.assertEqual(U.dtype, DT_COMPLEX)
    
    def test_ecd_sequence_unitary(self):
        """Test ECD sequence is approximately unitary."""
        U = build_ecd_sequence_tf(self.betas, self.phis, self.thetas, self.N_trunc)
        U_dag = tf.linalg.adjoint(U)
        identity = U_dag @ U
        expected_identity = tf.eye(2 * self.N_trunc, dtype=DT_COMPLEX)
        
        diff = tf.reduce_max(tf.abs(identity - expected_identity))
        self.assertLess(float(diff.numpy()), 1e-4)


class TestOptimizer(unittest.TestCase):
    """Test optimizer class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N_layers = 1
        self.N_trunc = 4
        self.N_target = 3
        self.batch_size = 4
    
    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly."""
        optimizer = MinimalECDOptimizer(
            N_layers=self.N_layers,
            N_trunc=self.N_trunc, 
            N_target=self.N_target,
            batch_size=self.batch_size
        )
        
        self.assertEqual(optimizer.N_layers, self.N_layers)
        self.assertEqual(optimizer.N_trunc, self.N_trunc)
        self.assertEqual(optimizer.N_target, self.N_target)
        self.assertIsNotNone(optimizer.P_sub)  # Should create projector since N_target < N_trunc
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        optimizer = MinimalECDOptimizer(
            N_layers=self.N_layers,
            N_trunc=self.N_trunc,
            batch_size=self.batch_size
        )
        
        optimizer.initialize_parameters(seed=42)
        
        # Check parameter shapes
        self.assertEqual(optimizer.params['betas'].shape, (self.batch_size, self.N_layers + 1))
        self.assertEqual(optimizer.params['phis'].shape, (self.batch_size, self.N_layers + 1))
        self.assertEqual(optimizer.params['thetas'].shape, (self.batch_size, self.N_layers + 1))
        
        # Check parameter types
        self.assertEqual(optimizer.params['betas'].dtype, DT_COMPLEX)
        self.assertEqual(optimizer.params['phis'].dtype, DT_FLOAT)
        self.assertEqual(optimizer.params['thetas'].dtype, DT_FLOAT)


class TestTailProbabilities(unittest.TestCase):
    """Test tail probability utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.N_trunc = 6
        self.N_target = 4
        self.alpha = 0.5 + 0.3j
    
    def test_coherent_state_normalization(self):
        """Test coherent state is properly normalized."""
        psi = coherent_cavity_state_tf(self.alpha, self.N_trunc)
        norm_squared = tf.reduce_sum(tf.abs(psi) ** 2)
        self.assertAlmostEqual(float(norm_squared.numpy()), 1.0, places=6)
    
    def test_tail_probability_range(self):
        """Test tail probability is between 0 and 1."""
        psi = coherent_cavity_state_tf(self.alpha, self.N_trunc)
        tail_prob = cavity_tail_probability_tf(psi, self.N_target)
        
        tail_val = float(tail_prob.numpy())
        self.assertGreaterEqual(tail_val, 0.0)
        self.assertLessEqual(tail_val, 1.0)
    
    def test_tail_probability_zero_for_vacuum(self):
        """Test tail probability is zero for vacuum state."""
        vacuum = tf.zeros((self.N_trunc,), dtype=DT_COMPLEX)
        vacuum = vacuum.numpy()
        vacuum[0] = 1.0  # |0⟩ state
        vacuum = tf.constant(vacuum)
        
        tail_prob = cavity_tail_probability_tf(vacuum, self.N_target)
        self.assertAlmostEqual(float(tail_prob.numpy()), 0.0, places=6)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_adaptive_batch_size(self):
        """Test adaptive batch size calculation."""
        # Small N_trunc should keep full batch
        result = _adaptive_batch_size(16, 20)
        self.assertEqual(result, 16)
        
        # Large N_trunc should reduce batch
        result = _adaptive_batch_size(16, 50)
        self.assertLess(result, 16)
        self.assertGreaterEqual(result, 4)  # Should have floor of 4
    
    def test_success_function(self):
        """Test success criteria function."""
        # Without auto-guard, only check fidelity
        self.assertTrue(_success(0.999, None, 0.99, False, 1e-6))
        self.assertFalse(_success(0.98, None, 0.99, False, 1e-6))
        
        # With auto-guard, check both fidelity and tail
        metrics = {'max_tail': 1e-7}
        self.assertTrue(_success(0.999, metrics, 0.99, True, 1e-6))
        
        metrics_bad_tail = {'max_tail': 1e-5}
        self.assertFalse(_success(0.999, metrics_bad_tail, 0.99, True, 1e-6))
    
    def test_parse_phases_string(self):
        """Test parsing phases from comma-separated string."""
        phases_str = "0.0, 0.5, 1.0, 1.5"
        phases = parse_phases(phases_str)
        expected = np.array([0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(phases, expected)
    
    def test_parse_phases_file(self):
        """Test parsing phases from file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0.0, 0.5, 1.0")
            temp_filename = f.name
        
        try:
            phases = parse_phases(temp_filename)
            expected = np.array([0.0, 0.5, 1.0])
            np.testing.assert_array_almost_equal(phases, expected)
        finally:
            os.unlink(temp_filename)
    
    def test_parse_phases_json_file(self):
        """Test parsing phases from JSON file."""
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([0.0, 0.5, 1.0], f)
            temp_filename = f.name
        
        try:
            phases = parse_phases(temp_filename)
            expected = np.array([0.0, 0.5, 1.0])
            np.testing.assert_array_almost_equal(phases, expected)
        finally:
            os.unlink(temp_filename)


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling."""
    
    def test_parse_phases_empty_string(self):
        """Test parsing empty phase string returns empty array."""
        phases = parse_phases("")
        self.assertEqual(len(phases), 0)
    
    def test_parse_phases_invalid_string(self):
        """Test error handling for invalid phase string."""
        with self.assertRaises(ValueError):
            parse_phases("not,a,number")
    
    def test_parse_phases_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with self.assertRaises(ValueError):
            parse_phases("nonexistent_file.txt")
    
    def test_projector_invalid_n_target(self):
        """Test error handling for invalid N_target."""
        with self.assertRaises(ValueError):
            build_projector_qubit_cavity_subspace_tf(10, 5)  # N_target > N_trunc


if __name__ == '__main__':
    # Configure TensorFlow to suppress warnings
    tf.get_logger().setLevel('ERROR')
    
    # Run the tests
    unittest.main(verbosity=2)