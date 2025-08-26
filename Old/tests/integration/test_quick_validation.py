"""
Quick tests for the ECD-SNAP optimization system.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from scripts.simple_sgd import SimpleSGDOptimizer
from src.snap_targets import make_snap_full_space
from src.gates import build_ecd_sequence_jax_real
import time


def test_basic_targets():
    """Test a few basic SNAP targets."""
    print("Testing Basic SNAP Targets")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    test_cases = [
        ("Identity", np.zeros(N_trunc)),
        ("Linear", np.arange(N_trunc) * 0.1),
        ("Quadratic", np.arange(N_trunc)**2 * 0.01),
    ]
    
    results = {}
    
    for name, phases in test_cases:
        print(f"\n{name} SNAP:")
        
        # Create target
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        # Try simple SGD
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        start_time = time.time()
        params, fidelity = optimizer.optimize(
            U_target_jax, 
            max_iterations=200,  # Reduced iterations
            momentum=0.9,
            verbose=False
        )
        elapsed = time.time() - start_time
        
        results[name] = fidelity
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} F = {fidelity:.6f} (time={elapsed:.2f}s)")
        print(f"     |β| = {jnp.mean(jnp.abs(params['betas'])):.4f}")
    
    return results


def test_convergence_speed():
    """Test how quickly we converge for identity."""
    print("\nTesting Convergence Speed")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Track fidelity over iterations
    class TrackingSGD(SimpleSGDOptimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fidelity_history = []
        
        def optimize(self, U_target, max_iterations=1000, momentum=0.9, verbose=False):
            # Initialize near zero
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, 4)
            scale = 0.01
            
            betas = (jax.random.normal(keys[0], (self.N_layers + 1,)) * scale + 
                    1j * jax.random.normal(keys[1], (self.N_layers + 1,)) * scale)
            phis = jax.random.uniform(keys[2], (self.N_layers + 1,)) * scale
            thetas = jax.random.uniform(keys[3], (self.N_layers + 1,)) * scale
            
            params = {'betas': betas, 'phis': phis, 'thetas': thetas}
            
            # Initialize momentum
            velocity = {
                'betas': jnp.zeros_like(betas),
                'phis': jnp.zeros_like(phis),
                'thetas': jnp.zeros_like(thetas)
            }
            
            # Gradient function
            grad_fn = jax.grad(self.loss)
            
            best_fidelity = 0.0
            best_params = params
            
            for i in range(max_iterations):
                # Compute gradient
                grads = grad_fn(params, U_target)
                
                # Update velocity (momentum)
                for key in velocity:
                    velocity[key] = momentum * velocity[key] - self.learning_rate * grads[key]
                
                # Update parameters
                for key in params:
                    params[key] = params[key] + velocity[key]
                
                # Check fidelity every 10 iterations
                if i % 10 == 0:
                    U_approx = build_ecd_sequence_jax_real(
                        params['betas'], params['phis'], params['thetas'], self.N_trunc
                    )
                    fidelity = self.unitary_fidelity(U_target, U_approx)
                    self.fidelity_history.append((i, fidelity))
                    
                    if fidelity > best_fidelity:
                        best_fidelity = fidelity
                        best_params = params.copy()
                    
                    if fidelity > 0.999:
                        print(f"  ✓ Converged at iteration {i} with F={fidelity:.6f}")
                        break
            
            return best_params, best_fidelity
    
    optimizer = TrackingSGD(N_layers, N_trunc, learning_rate=0.01)
    params, final_fidelity = optimizer.optimize(U_target_jax, max_iterations=200)
    
    print("\n  Convergence trajectory:")
    for i, f in optimizer.fidelity_history[:10]:  # Show first 10 checkpoints
        print(f"    Iter {i:3d}: F={f:.6f}")
    
    return optimizer.fidelity_history


def test_different_learning_rates():
    """Quick test of different learning rates."""
    print("\nTesting Learning Rates")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    learning_rates = [0.005, 0.01, 0.02, 0.05]
    
    for lr in learning_rates:
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=lr)
        params, fidelity = optimizer.optimize(
            U_target_jax,
            max_iterations=200,
            momentum=0.9,
            verbose=False
        )
        
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} LR={lr:5.3f}: F={fidelity:.6f}")


def test_gate_properties():
    """Verify optimized gate properties."""
    print("\nVerifying Optimized Gate Properties")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    # Optimize identity
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
    params, fidelity = optimizer.optimize(U_target_jax, max_iterations=200, verbose=False)
    
    print(f"  Optimization fidelity: {fidelity:.6f}")
    
    # Build the gate
    U_approx = build_ecd_sequence_jax_real(
        params['betas'], params['phis'], params['thetas'], N_trunc
    )
    
    # Check unitarity
    U_dag_U = jnp.conj(U_approx.T) @ U_approx
    I = jnp.eye(2 * N_trunc)
    unitarity_error = jnp.max(jnp.abs(U_dag_U - I))
    print(f"  Unitarity error: {unitarity_error:.2e}")
    
    # Check that it acts as identity on cavity space
    # Test on |n⟩ ⊗ |g⟩ states
    for n in range(min(3, N_trunc)):
        state = jnp.zeros(2 * N_trunc, dtype=jnp.complex64)
        state = state.at[n].set(1.0)  # |n⟩ ⊗ |g⟩
        
        output = U_approx @ state
        expected = U_target_jax @ state
        
        overlap = jnp.abs(jnp.conj(expected) @ output)**2
        print(f"    |{n}⟩⊗|g⟩ fidelity: {overlap:.6f}")
    
    return fidelity > 0.99


def test_robustness_quick():
    """Quick robustness test with 5 seeds."""
    print("\nQuick Robustness Test (5 seeds)")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    fidelities = []
    
    for seed in range(5):
        # Create optimizer with different initialization
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=0.01)
        
        # Different random init
        key = jax.random.PRNGKey(seed * 100)
        keys = jax.random.split(key, 4)
        scale = 0.01
        
        betas = (jax.random.normal(keys[0], (N_layers + 1,)) * scale + 
                1j * jax.random.normal(keys[1], (N_layers + 1,)) * scale)
        phis = jax.random.uniform(keys[2], (N_layers + 1,)) * scale
        thetas = jax.random.uniform(keys[3], (N_layers + 1,)) * scale
        
        optimizer.params = {'betas': betas, 'phis': phis, 'thetas': thetas}
        
        params, fidelity = optimizer.optimize(U_target_jax, max_iterations=200, verbose=False)
        fidelities.append(fidelity)
        
        status = "✓" if fidelity > 0.99 else "○"
        print(f"  {status} Seed {seed}: F={fidelity:.6f}")
    
    fidelities = np.array(fidelities)
    print(f"\n  Mean: {np.mean(fidelities):.6f} ± {np.std(fidelities):.6f}")
    print(f"  Success rate: {np.sum(fidelities > 0.99)}/5")
    
    return fidelities


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUICK TEST SUITE")
    print("="*60)
    
    # Run tests
    print("\n1. Basic Targets")
    target_results = test_basic_targets()
    
    print("\n2. Convergence Speed")
    convergence = test_convergence_speed()
    
    print("\n3. Learning Rates")
    test_different_learning_rates()
    
    print("\n4. Gate Properties")
    gate_ok = test_gate_properties()
    
    print("\n5. Robustness")
    robust_results = test_robustness_quick()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nTarget Performance:")
    for name, fidelity in target_results.items():
        status = "✓" if fidelity > 0.99 else "○" if fidelity > 0.9 else "✗"
        print(f"  {status} {name:10s}: F={fidelity:.6f}")
    
    print(f"\nGate verification: {'✓ Passed' if gate_ok else '✗ Failed'}")
    print(f"Robustness: {np.mean(robust_results):.4f} ± {np.std(robust_results):.4f}")
    print(f"Success rate: {np.sum(np.array(list(target_results.values())) > 0.99)}/{len(target_results)} targets")
    
    print("\n" + "="*60)
    print("Quick testing complete!")
    print("="*60)