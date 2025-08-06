"""
Test optimization with better initialization strategies.
"""

import numpy as np
import jax
import jax.numpy as jnp
from optimizer import ECDSNAPOptimizer
from snap_targets import make_snap_full_space

def test_with_small_init():
    """Test optimization starting from small parameters."""
    print("Testing with small parameter initialization...")
    
    N_layers = 4
    N_trunc = 6
    batch_size = 16
    
    # Create optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.01,
        target_fidelity=0.99
    )
    
    # Custom initialization with very small parameters
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    
    # Much smaller displacements (closer to zero)
    beta_real = jax.random.normal(keys[0], (batch_size, N_layers + 1)) * 0.01
    beta_imag = jax.random.normal(keys[1], (batch_size, N_layers + 1)) * 0.01
    betas = beta_real + 1j * beta_imag
    
    # Smaller rotations
    phis = jax.random.uniform(keys[2], (batch_size, N_layers + 1)) * 0.1
    thetas = jax.random.uniform(keys[3], (batch_size, N_layers + 1)) * 0.1
    
    optimizer.params = {'betas': betas, 'phis': phis, 'thetas': thetas}
    optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Run short optimization
    best_params, best_fidelity, info = optimizer.optimize(
        U_target_jax,
        max_iterations=100,
        verbose=True
    )
    
    print(f"\nSmall init result: F = {best_fidelity:.6f}")
    return best_fidelity


def test_with_zero_init():
    """Test starting from exactly zero (should maintain high fidelity)."""
    print("\nTesting with zero initialization...")
    
    N_layers = 4
    N_trunc = 6
    batch_size = 8
    
    # Create optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=N_layers,
        N_trunc=N_trunc,
        batch_size=batch_size,
        learning_rate=0.001,  # Smaller learning rate
        target_fidelity=0.99
    )
    
    # Initialize with zeros (perfect for identity)
    betas = jnp.zeros((batch_size, N_layers + 1), dtype=jnp.complex64)
    phis = jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32)
    thetas = jnp.zeros((batch_size, N_layers + 1), dtype=jnp.float32)
    
    # Add small noise to break symmetry (except for one that stays at zero)
    key = jax.random.PRNGKey(123)
    noise = jax.random.normal(key, betas.shape) * 0.001
    betas = betas.at[1:].add(noise[1:])  # Keep first one exactly zero
    
    optimizer.params = {'betas': betas, 'phis': phis, 'thetas': thetas}
    optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
    
    # Create identity target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Check initial fidelity
    initial_fidelities = optimizer.batch_fidelity(optimizer.params, U_target_jax)
    print(f"Initial fidelities: max={jnp.max(initial_fidelities):.6f}, mean={jnp.mean(initial_fidelities):.6f}")
    
    # Run short optimization
    best_params, best_fidelity, info = optimizer.optimize(
        U_target_jax,
        max_iterations=50,
        verbose=False
    )
    
    print(f"Zero init result: F = {best_fidelity:.6f}")
    print(f"Initial max fidelity: {info['history']['max_fidelity'][0]:.6f}")
    print(f"Final max fidelity: {info['history']['max_fidelity'][-1]:.6f}")
    
    return best_fidelity


def test_gradient_direction():
    """Check if gradients point toward zero parameters for identity."""
    print("\nTesting gradient direction...")
    
    N_layers = 2
    N_trunc = 4
    
    # Create target
    phases = np.zeros(N_trunc)
    U_target = make_snap_full_space(phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Start with small random parameters
    key = jax.random.PRNGKey(999)
    beta = jax.random.normal(key, (N_layers + 1,)) * 0.1 + 1j * jax.random.normal(key, (N_layers + 1,)) * 0.1
    phi = jax.random.uniform(key, (N_layers + 1,)) * 0.1
    theta = jax.random.uniform(key, (N_layers + 1,)) * 0.1
    
    # Build optimizer just for the loss function
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, 1)
    
    # Single parameter set
    params = {'betas': beta.reshape(1, -1), 
              'phis': phi.reshape(1, -1), 
              'thetas': theta.reshape(1, -1)}
    
    # Compute loss and gradient
    def loss_fn(p):
        return optimizer.loss_function(p, U_target_jax)
    
    loss_val = loss_fn(params)
    grads = jax.grad(loss_fn)(params)
    
    print(f"Loss: {loss_val:.4f}")
    print(f"Parameter magnitudes: |β|={jnp.mean(jnp.abs(beta)):.4f}, θ={jnp.mean(theta):.4f}")
    print(f"Gradient magnitudes: |∇β|={jnp.mean(jnp.abs(grads['betas'])):.4f}, ∇θ={jnp.mean(jnp.abs(grads['thetas'])):.4f}")
    
    # Check if gradients point toward zero (negative correlation with params)
    beta_grad_dir = jnp.real(jnp.sum(jnp.conj(beta.flatten()) * grads['betas'].flatten()))
    print(f"β·∇β correlation: {beta_grad_dir:.4f} (negative is good)")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("Testing Better Initialization Strategies")
    print("="*60)
    
    # Run tests
    fidelity1 = test_with_small_init()
    fidelity2 = test_with_zero_init()
    test_gradient_direction()
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  Small init: F = {fidelity1:.6f}")
    print(f"  Zero init:  F = {fidelity2:.6f}")
    print("\nConclusion:")
    if fidelity2 > 0.99:
        print("✓ Starting near zero maintains high fidelity!")
        print("  The optimization works when initialized properly.")
    else:
        print("  The optimization landscape is challenging even with good initialization.")
    print("="*60)