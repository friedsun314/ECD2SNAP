"""
Simple SGD with momentum - sometimes simpler is better for quantum optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from gates import build_ecd_sequence_jax_real
from snap_targets import make_snap_full_space


class SimpleSGDOptimizer:
    """Simple SGD with momentum for ECD-SNAP optimization."""
    
    def __init__(self, N_layers: int, N_trunc: int, learning_rate: float = 0.1):
        self.N_layers = N_layers
        self.N_trunc = N_trunc
        self.learning_rate = learning_rate
        
    def unitary_fidelity(self, U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
        """Compute fidelity between two unitaries."""
        d = U_target.shape[0]
        return jnp.abs(jnp.trace(jnp.conj(U_target.T) @ U_approx))**2 / d**2
    
    def loss(self, params: Dict, U_target: jnp.ndarray) -> float:
        """Simple infidelity loss."""
        U_approx = build_ecd_sequence_jax_real(
            params['betas'], params['phis'], params['thetas'], self.N_trunc
        )
        fidelity = self.unitary_fidelity(U_target, U_approx)
        return 1.0 - fidelity  # Simple infidelity
    
    def optimize(
        self, 
        U_target: jnp.ndarray,
        max_iterations: int = 1000,
        momentum: float = 0.9,
        verbose: bool = False
    ) -> Tuple[Dict, float]:
        """Simple gradient descent with momentum."""
        
        # Initialize near zero for identity
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)
        
        # Very small initialization
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
            
            # Check fidelity
            if i % 50 == 0 or i == max_iterations - 1:
                U_approx = build_ecd_sequence_jax_real(
                    params['betas'], params['phis'], params['thetas'], self.N_trunc
                )
                fidelity = self.unitary_fidelity(U_target, U_approx)
                
                if fidelity > best_fidelity:
                    best_fidelity = fidelity
                    best_params = params.copy()
                
                if verbose:
                    print(f"Iter {i:4d}: F = {fidelity:.6f}, Loss = {1-fidelity:.6f}")
                
                # Early stopping
                if fidelity > 0.999:
                    print(f"✓ Target reached at iteration {i}")
                    break
        
        return best_params, best_fidelity


def test_simple_sgd():
    """Test simple SGD optimizer."""
    print("Testing Simple SGD with Momentum")
    print("="*60)
    
    N_layers = 3
    N_trunc = 4
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Test different learning rates
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    
    for lr in learning_rates:
        print(f"\nLearning rate = {lr}")
        optimizer = SimpleSGDOptimizer(N_layers, N_trunc, learning_rate=lr)
        params, fidelity = optimizer.optimize(
            U_target_jax, 
            max_iterations=500,
            momentum=0.9,
            verbose=False
        )
        print(f"  Final fidelity: {fidelity:.6f}")
        
        if fidelity > 0.99:
            print(f"  ✓ Success with lr={lr}!")
            
            # Show parameters
            print(f"  |β| = {jnp.mean(jnp.abs(params['betas'])):.4f}")
            print(f"  φ = {jnp.mean(params['phis']):.4f}")
            print(f"  θ = {jnp.mean(params['thetas']):.4f}")
            break
    
    return fidelity


if __name__ == "__main__":
    fidelity = test_simple_sgd()
    
    print("\n" + "="*60)
    if fidelity > 0.99:
        print(f"✓ Simple SGD achieves F = {fidelity:.6f}")
    else:
        print(f"△ Simple SGD achieves F = {fidelity:.6f}")
    print("="*60)