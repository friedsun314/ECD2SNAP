"""
Test the improved optimization strategies.
"""

import numpy as np
import jax
import jax.numpy as jnp
from src.optimizer import ECDSNAPOptimizer
from src.snap_targets import make_snap_full_space


def test_smart_initialization():
    """Test that smart initialization gives better starting points."""
    print("Testing smart initialization...")
    
    N_layers = 4
    N_trunc = 6
    batch_size = 8
    
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, batch_size)
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Test identity initialization
    key = jax.random.PRNGKey(42)
    params_identity = optimizer.smart_initialize(key, "identity")
    fidelities_identity = optimizer.batch_fidelity(params_identity, U_target_jax)
    
    print(f"  Identity init - Max F: {jnp.max(fidelities_identity):.6f}")
    print(f"  Identity init - Mean F: {jnp.mean(fidelities_identity):.6f}")
    print(f"  First sample (should be ~1.0): {fidelities_identity[0]:.6f}")
    
    # Test regular initialization
    params_regular = optimizer.smart_initialize(key, "other")
    fidelities_regular = optimizer.batch_fidelity(params_regular, U_target_jax)
    
    print(f"  Regular init - Max F: {jnp.max(fidelities_regular):.6f}")
    print(f"  Regular init - Mean F: {jnp.mean(fidelities_regular):.6f}")
    
    return jnp.max(fidelities_identity) > 0.99


def test_restart_strategy():
    """Test optimization with restarts."""
    print("\nTesting restart strategy...")
    
    N_layers = 3
    N_trunc = 4
    batch_size = 8
    
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.01
    )
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Run with restarts
    best_params, best_fidelity, info = optimizer.optimize_with_restarts(
        U_target_jax,
        max_iterations=150,
        n_restarts=3,
        target_type="identity",
        verbose=False
    )
    
    print(f"  Final fidelity: {best_fidelity:.6f}")
    print(f"  Total iterations: {info['iterations']}")
    
    return best_fidelity


def test_annealing_strategy():
    """Test optimization with learning rate annealing."""
    print("\nTesting annealing strategy...")
    
    N_layers = 3
    N_trunc = 4
    batch_size = 8
    
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.01
    )
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Run with annealing
    best_params, best_fidelity, info = optimizer.optimize_with_annealing(
        U_target_jax,
        max_iterations=200,
        target_type="identity",
        verbose=False
    )
    
    print(f"  Final fidelity: {best_fidelity:.6f}")
    print(f"  Iterations: {info['iterations']}")
    
    return best_fidelity


def test_two_stage_strategy():
    """Test two-stage optimization."""
    print("\nTesting two-stage strategy...")
    
    N_layers = 3
    N_trunc = 4
    batch_size = 8
    
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size
    )
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Run two-stage
    best_params, best_fidelity, info = optimizer.optimize_two_stage(
        U_target_jax,
        max_iterations=200,
        target_type="identity",
        verbose=False
    )
    
    print(f"  Final fidelity: {best_fidelity:.6f}")
    print(f"  Total iterations: {len(info['history']['max_fidelity'])}")
    
    # Check progression
    history = info['history']['max_fidelity']
    mid_point = len(history) // 2
    print(f"  Stage 1 end fidelity: {history[mid_point-1]:.6f}")
    print(f"  Stage 2 end fidelity: {history[-1]:.6f}")
    
    return best_fidelity


def test_adaptive_layers_strategy():
    """Test adaptive layers strategy that increases layers as needed."""
    print("Testing adaptive layers strategy...")
    
    N_trunc = 6
    batch_size = 8
    
    # Test with a simple target that needs few layers
    print("\n1. Testing with simple target (identity)...")
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    optimizer = ECDSNAPOptimizer(
        N_layers=4,  # This will be overridden by adaptive
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.003,
        target_fidelity=0.999
    )
    
    params, fidelity, info = optimizer.optimize_adaptive_layers(
        U_target_jax,
        max_iterations=300,
        min_layers=2,
        max_layers=6,
        n_restarts=2,
        target_type='identity',
        verbose=False
    )
    
    assert fidelity > 0.99, f"Failed to reach good fidelity: {fidelity}"
    assert 'layers_used' in info, "Missing layers_used in info"
    assert 'adaptive_history' in info, "Missing adaptive history"
    
    print(f"✓ Identity target: F={fidelity:.6f} with {info['layers_used']} layers")
    print(f"  Layer attempts: {info['adaptive_history']['layer_attempts']}")
    print(f"  Fidelities: {[f'{f:.4f}' for f in info['adaptive_history']['fidelities']]}")
    
    # Test with a harder target that needs more layers
    print("\n2. Testing with harder target (linear)...")
    phases = np.arange(N_trunc) * 0.1
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    optimizer2 = ECDSNAPOptimizer(
        N_layers=4,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.003,
        target_fidelity=0.95
    )
    
    params2, fidelity2, info2 = optimizer2.optimize_adaptive_layers(
        U_target_jax,
        max_iterations=400,
        min_layers=2,
        max_layers=8,
        n_restarts=2,
        target_type='linear',
        verbose=False
    )
    
    print(f"✓ Linear target: F={fidelity2:.6f} with {info2.get('layers_used', 'N/A')} layers")
    if 'adaptive_history' in info2:
        converged = info2['adaptive_history'].get('converged_at_layer', 'Did not converge')
        print(f"  Converged at layer: {converged}")
    
    print("\n✓ Adaptive layers strategy test passed!")
    return fidelity


if __name__ == "__main__":
    print("="*60)
    print("Testing Improved Optimization Strategies")
    print("="*60)
    
    # Run tests
    init_success = test_smart_initialization()
    fidelity_restart = test_restart_strategy()
    fidelity_anneal = test_annealing_strategy()
    fidelity_two_stage = test_two_stage_strategy()
    fidelity_adaptive = test_adaptive_layers_strategy()
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Smart init works: {init_success}")
    print(f"  Restart strategy: F = {fidelity_restart:.6f}")
    print(f"  Annealing strategy: F = {fidelity_anneal:.6f}")
    print(f"  Two-stage strategy: F = {fidelity_two_stage:.6f}")
    print(f"  Adaptive layers: F = {fidelity_adaptive:.6f}")
    
    best_fidelity = max(fidelity_restart, fidelity_anneal, fidelity_two_stage, fidelity_adaptive)
    if best_fidelity > 0.9:
        print(f"\n✓ Improved strategies achieve F = {best_fidelity:.6f}")
    else:
        print(f"\n△ Best achieved: F = {best_fidelity:.6f}")
    print("="*60)