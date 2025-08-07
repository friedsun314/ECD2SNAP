"""
JAX-based optimizer for ECD-to-SNAP gate decomposition.
Uses automatic differentiation and batch optimization for high performance.
Includes multiple optimization strategies for better convergence.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
import numpy as np
from typing import Tuple, Dict, Any, Optional
from tqdm import tqdm
import qutip as qt


class ECDSNAPOptimizer:
    """
    Optimizer for finding ECD gate sequences that approximate SNAP gates.
    
    Uses multi-start batch optimization with logarithmic barrier cost function
    for stable convergence near unit fidelity. Includes multiple optimization
    strategies: basic, restarts, annealing, and two-stage.
    """
    
    def __init__(self, 
                 N_layers: int = 6,
                 N_trunc: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 3e-3,
                 target_fidelity: float = 0.999):
        """
        Initialize the optimizer.
        
        Args:
            N_layers: Number of ECD-rotation layers
            N_trunc: Fock space truncation
            batch_size: Number of parallel optimizations
            learning_rate: Adam learning rate
            target_fidelity: Stop when any batch element reaches this fidelity
        """
        self.N_layers = N_layers
        self.N_trunc = N_trunc
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_fidelity = target_fidelity
        self.dim = 2 * N_trunc  # Full Hilbert space dimension
        
        # Initialize optimizer
        self.optimizer = optax.adam(learning_rate)
        
        # Initialize parameters
        self.params = None
        self.opt_state = None
    
    def initialize_parameters(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Initialize optimization parameters with smart defaults.
        
        Args:
            key: JAX random key
        
        Returns:
            Dictionary with 'betas', 'phis', 'thetas' arrays
        """
        keys = jax.random.split(key, 4)
        
        # Complex displacements: small random values
        beta_real = jax.random.normal(keys[0], (self.batch_size, self.N_layers + 1)) * 0.2
        beta_imag = jax.random.normal(keys[1], (self.batch_size, self.N_layers + 1)) * 0.2
        betas = beta_real + 1j * beta_imag
        
        # Rotation axes: uniform on [0, 2π)
        phis = jax.random.uniform(keys[2], (self.batch_size, self.N_layers + 1), 
                                  minval=0, maxval=2*np.pi)
        
        # Rotation angles: uniform on [0, π]
        thetas = jax.random.uniform(keys[3], (self.batch_size, self.N_layers + 1), 
                                    minval=0, maxval=np.pi)
        
        return {'betas': betas, 'phis': phis, 'thetas': thetas}
    
    def smart_initialize(self, key: jax.random.PRNGKey, target_type: str = "identity") -> Dict:
        """Smart initialization based on target type."""
        keys = jax.random.split(key, 4)
        
        if target_type == "identity":
            # For identity, start very close to zero
            scale = 0.01  # Much smaller than default
        else:
            # For other targets, use moderate initialization
            scale = 0.1
            
        # Initialize with small values
        beta_real = jax.random.normal(keys[0], (self.batch_size, self.N_layers + 1)) * scale
        beta_imag = jax.random.normal(keys[1], (self.batch_size, self.N_layers + 1)) * scale
        betas = beta_real + 1j * beta_imag
        
        # Small rotations
        phis = jax.random.uniform(keys[2], (self.batch_size, self.N_layers + 1)) * scale * jnp.pi
        thetas = jax.random.uniform(keys[3], (self.batch_size, self.N_layers + 1)) * scale * jnp.pi
        
        # Include one near-zero initialization for identity
        if target_type == "identity":
            betas = betas.at[0].set(jnp.zeros(self.N_layers + 1, dtype=jnp.complex64))
            phis = phis.at[0].set(jnp.zeros(self.N_layers + 1))
            thetas = thetas.at[0].set(jnp.zeros(self.N_layers + 1))
        
        return {'betas': betas, 'phis': phis, 'thetas': thetas}
    
    @staticmethod
    def build_unitary_jax(beta: jnp.ndarray, phi: jnp.ndarray, theta: jnp.ndarray, 
                         N_trunc: int) -> jnp.ndarray:
        """
        Build a single ECD sequence unitary using JAX operations.
        
        This function constructs the full gate sequence while maintaining
        gradient flow for automatic differentiation.
        """
        from .gates import build_ecd_sequence_jax_real
        return build_ecd_sequence_jax_real(beta, phi, theta, N_trunc)
    
    @staticmethod
    @jit
    def unitary_fidelity(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
        """
        Compute gate fidelity between target and approximation.
        
        F = |Tr(U_target† U_approx)|² / d²
        
        Args:
            U_target: Target unitary
            U_approx: Approximation unitary
        
        Returns:
            Fidelity in [0, 1]
        """
        d = U_target.shape[0]
        trace = jnp.trace(jnp.conj(U_target.T) @ U_approx)
        return jnp.abs(trace)**2 / d**2
    
    def batch_fidelity(self, params: Dict[str, jnp.ndarray], 
                      U_target: jnp.ndarray) -> jnp.ndarray:
        """
        Compute fidelities for entire batch.
        
        Args:
            params: Dictionary with parameter arrays
            U_target: Target SNAP unitary
        
        Returns:
            Array of fidelities for each batch element
        """
        # Build unitaries for each batch element
        build_fn = lambda b, p, t: self.build_unitary_jax(b, p, t, self.N_trunc)
        U_batch = vmap(build_fn)(params['betas'], params['phis'], params['thetas'])
        
        # Compute fidelities
        fidelity_fn = lambda U: self.unitary_fidelity(U_target, U)
        fidelities = vmap(fidelity_fn)(U_batch)
        
        return fidelities
    
    def loss_function(self, params: Dict[str, jnp.ndarray], 
                     U_target: jnp.ndarray) -> float:
        """
        Logarithmic barrier cost function for stable optimization.
        
        C = Σ_j log(1 - F_j)
        
        Args:
            params: Dictionary with parameter arrays
            U_target: Target SNAP unitary
        
        Returns:
            Scalar loss value
        """
        fidelities = self.batch_fidelity(params, U_target)
        # Add small epsilon to prevent log(0)
        eps = 1e-10
        losses = jnp.log(jnp.maximum(1 - fidelities, eps))
        return jnp.sum(losses)
    
    def optimize(self, U_target: jnp.ndarray, 
                max_iterations: int = 10000,
                verbose: bool = True) -> Tuple[Dict[str, jnp.ndarray], float, Dict[str, Any]]:
        """
        Run optimization to find ECD parameters approximating target SNAP.
        
        Args:
            U_target: Target SNAP unitary (as JAX array)
            max_iterations: Maximum optimization steps
            verbose: Whether to print progress
        
        Returns:
            Tuple of (best_params, best_fidelity, info_dict)
        """
        # Initialize if not already done
        if self.params is None:
            key = jax.random.PRNGKey(0)
            self.params = self.initialize_parameters(key)
            self.opt_state = self.optimizer.init(self.params)
        
        # Setup gradient function
        # Create a closure that captures self for JAX
        def loss_fn(params, target):
            return self.loss_function(params, target)
        loss_and_grad = jax.value_and_grad(loss_fn, argnums=0)
        
        # History tracking
        history = {
            'loss': [],
            'max_fidelity': [],
            'mean_fidelity': []
        }
        
        # Progress bar
        pbar = tqdm(range(max_iterations), disable=not verbose)
        
        best_fidelity = 0.0
        best_params = None
        best_idx = None
        
        for iteration in pbar:
            # Compute loss and gradients
            loss, grads = loss_and_grad(self.params, U_target)
            
            # Update parameters
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)
            
            # Compute fidelities for monitoring
            fidelities = self.batch_fidelity(self.params, U_target)
            max_fid = jnp.max(fidelities)
            mean_fid = jnp.mean(fidelities)
            
            # Track history
            history['loss'].append(float(loss))
            history['max_fidelity'].append(float(max_fid))
            history['mean_fidelity'].append(float(mean_fid))
            
            # Update best
            if max_fid > best_fidelity:
                best_fidelity = float(max_fid)
                best_idx = jnp.argmax(fidelities)
                best_params = {
                    'betas': self.params['betas'][best_idx],
                    'phis': self.params['phis'][best_idx],
                    'thetas': self.params['thetas'][best_idx]
                }
            
            # Update progress bar
            pbar.set_description(
                f"Loss: {loss:.4f} | Max F: {max_fid:.6f} | Mean F: {mean_fid:.6f}"
            )
            
            # Check convergence
            if max_fid >= self.target_fidelity:
                if verbose:
                    print(f"\nTarget fidelity reached! F = {max_fid:.8f}")
                break
        
        # Prepare info dictionary
        info = {
            'history': history,
            'iterations': iteration + 1,
            'final_loss': float(loss),
            'batch_fidelities': fidelities,
            'best_idx': best_idx
        }
        
        return best_params, best_fidelity, info
    
    def optimize_with_restarts(
        self,
        U_target: jnp.ndarray,
        max_iterations: int = 1000,
        n_restarts: int = 3,
        target_type: str = "identity",
        verbose: bool = False
    ) -> Tuple[Dict, float, Dict]:
        """Optimize with multiple random restarts."""
        
        best_fidelity = 0.0
        best_params = None
        best_info = None
        
        for restart in range(n_restarts):
            if verbose:
                print(f"\n--- Restart {restart + 1}/{n_restarts} ---")
            
            # Re-initialize with smart strategy
            key = jax.random.PRNGKey(42 + restart * 1000)
            self.params = self.smart_initialize(key, target_type)
            self.opt_state = self.optimizer.init(self.params)
            
            # Run optimization
            params, fidelity, info = self.optimize(
                U_target, 
                max_iterations=max_iterations // n_restarts,
                verbose=verbose
            )
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = params
                best_info = info
                
                if best_fidelity >= self.target_fidelity:
                    if verbose:
                        print(f"✓ Target fidelity reached: {best_fidelity:.6f}")
                    break
        
        return best_params, best_fidelity, best_info
    
    def optimize_with_annealing(
        self,
        U_target: jnp.ndarray,
        max_iterations: int = 1000,
        target_type: str = "identity",
        verbose: bool = False
    ) -> Tuple[Dict, float, Dict]:
        """Optimize with learning rate annealing."""
        
        # Smart initialization
        key = jax.random.PRNGKey(42)
        self.params = self.smart_initialize(key, target_type)
        
        # Use exponential decay schedule
        schedule = optax.exponential_decay(
            init_value=self.learning_rate,
            transition_steps=max_iterations // 10,
            decay_rate=0.9
        )
        
        self.optimizer = optax.adam(learning_rate=schedule)
        self.opt_state = self.optimizer.init(self.params)
        
        # Run optimization
        return self.optimize(U_target, max_iterations, verbose)
    
    def optimize_two_stage(
        self,
        U_target: jnp.ndarray,
        max_iterations: int = 1000,
        target_type: str = "identity",
        verbose: bool = False
    ) -> Tuple[Dict, float, Dict]:
        """Two-stage optimization: coarse then fine."""
        
        # Stage 1: Coarse search with higher learning rate
        key = jax.random.PRNGKey(42)
        self.params = self.smart_initialize(key, target_type)
        
        # Higher learning rate for exploration
        self.optimizer = optax.adam(learning_rate=0.01)
        self.opt_state = self.optimizer.init(self.params)
        
        if verbose:
            print("Stage 1: Coarse search...")
        
        params1, fidelity1, info1 = self.optimize(
            U_target, 
            max_iterations=max_iterations // 2,
            verbose=verbose
        )
        
        # Stage 2: Fine-tuning with lower learning rate
        if verbose:
            print(f"\nStage 2: Fine-tuning from F={fidelity1:.6f}...")
        
        # Params1 contains the best single solution, we need to expand it back to batch
        # for the second stage optimization
        self.params = {
            'betas': jnp.tile(params1['betas'][jnp.newaxis, :], (self.batch_size, 1)),
            'phis': jnp.tile(params1['phis'][jnp.newaxis, :], (self.batch_size, 1)),
            'thetas': jnp.tile(params1['thetas'][jnp.newaxis, :], (self.batch_size, 1))
        }
        self.optimizer = optax.adam(learning_rate=0.001)
        self.opt_state = self.optimizer.init(self.params)
        
        params2, fidelity2, info2 = self.optimize(
            U_target,
            max_iterations=max_iterations // 2,
            verbose=verbose
        )
        
        # Combine history
        info2['history']['max_fidelity'] = (
            info1['history']['max_fidelity'] + info2['history']['max_fidelity']
        )
        info2['history']['mean_fidelity'] = (
            info1['history']['mean_fidelity'] + info2['history']['mean_fidelity']
        )
        
        return params2, fidelity2, info2
    
    def extract_best_solution(self, params: Dict[str, jnp.ndarray], 
                             fidelities: jnp.ndarray) -> Dict[str, Any]:
        """
        Extract the best solution from batch.
        
        Args:
            params: Batch parameters
            fidelities: Fidelities for each batch element
        
        Returns:
            Dictionary with best parameters and metadata
        """
        best_idx = jnp.argmax(fidelities)
        
        return {
            'betas': params['betas'][best_idx],
            'phis': params['phis'][best_idx],
            'thetas': params['thetas'][best_idx],
            'fidelity': float(fidelities[best_idx]),
            'batch_index': int(best_idx)
        }


def optimize_ecd_to_snap(target_phases: np.ndarray,
                        N_layers: int = 6,
                        N_trunc: int = 10,
                        batch_size: int = 32,
                        **kwargs) -> Dict[str, Any]:
    """
    Convenience function to optimize ECD sequence for a SNAP gate.
    
    Args:
        target_phases: SNAP gate phases
        N_layers: Number of ECD layers
        N_trunc: Fock truncation
        batch_size: Batch size for multi-start
        **kwargs: Additional arguments for optimizer
    
    Returns:
        Dictionary with optimized parameters and metadata
    """
    # Create target SNAP unitary
    from snap_targets import make_snap
    U_target = make_snap(target_phases, N_trunc)
    U_target_jax = jnp.array(U_target.full())
    
    # Initialize and run optimizer
    optimizer = ECDSNAPOptimizer(N_layers, N_trunc, batch_size, **kwargs)
    best_params, best_fidelity, info = optimizer.optimize(U_target_jax)
    
    result = {
        'parameters': best_params,
        'fidelity': best_fidelity,
        'info': info,
        'config': {
            'N_layers': N_layers,
            'N_trunc': N_trunc,
            'batch_size': batch_size
        }
    }
    
    return result