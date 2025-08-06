"""
Improved optimizer with better initialization and optimization strategies.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, Tuple, Optional
from .optimizer import ECDSNAPOptimizer


class ImprovedECDSNAPOptimizer(ECDSNAPOptimizer):
    """Enhanced optimizer with better strategies for convergence."""
    
    def __init__(
        self,
        N_layers: int,
        N_trunc: int,
        batch_size: int = 16,
        learning_rate: float = 0.003,
        target_fidelity: float = 0.999
    ):
        super().__init__(N_layers, N_trunc, batch_size, learning_rate, target_fidelity)
        
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
        
        self.params = params1
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