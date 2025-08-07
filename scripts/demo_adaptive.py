#!/usr/bin/env python3
"""
Demonstration of the adaptive layers optimization strategy.
Shows how it automatically finds the minimum number of layers needed.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax.numpy as jnp
from src.optimizer import ECDSNAPOptimizer
from src.snap_targets import make_snap_full_space

def demo_adaptive():
    """Demonstrate adaptive layer optimization."""
    print("="*70)
    print("ADAPTIVE LAYERS OPTIMIZATION DEMO")
    print("="*70)
    print("\nThis strategy starts with few layers and adds more only if needed,")
    print("finding the minimum circuit depth required for the target fidelity.\n")
    
    N_trunc = 6
    batch_size = 16
    target_fidelity = 0.999
    
    # Test cases with increasing difficulty
    test_cases = [
        ("Identity (easiest)", np.zeros(N_trunc), "identity"),
        ("Small Linear", np.arange(N_trunc) * 0.05, "linear"),
        ("Medium Linear", np.arange(N_trunc) * 0.1, "linear"),
        ("Quadratic", np.arange(N_trunc)**2 * 0.01, "quadratic"),
    ]
    
    for name, phases, target_type in test_cases:
        print(f"\n{'='*70}")
        print(f"Target: {name}")
        print(f"Phases: {phases[:4]}..." if len(phases) > 4 else f"Phases: {phases}")
        print("-"*70)
        
        # Create target
        U_target = make_snap_full_space(phases, N_trunc)
        U_target_jax = jnp.array(U_target.full())
        
        # Create optimizer
        optimizer = ECDSNAPOptimizer(
            N_layers=4,  # Will be overridden by adaptive
            N_trunc=N_trunc,
            batch_size=batch_size,
            learning_rate=0.003,
            target_fidelity=target_fidelity
        )
        
        # Run adaptive optimization
        print("Running adaptive optimization...")
        params, fidelity, info = optimizer.optimize_adaptive_layers(
            U_target_jax,
            max_iterations=600,
            min_layers=2,
            max_layers=8,
            n_restarts=2,
            target_type=target_type,
            verbose=False
        )
        
        # Display results
        print(f"\nResults:")
        print(f"  Final fidelity: {fidelity:.6f}")
        
        if 'layers_used' in info:
            print(f"  Layers used: {info['layers_used']}")
        
        if 'adaptive_history' in info:
            history = info['adaptive_history']
            print(f"  Layers tried: {history['layer_attempts']}")
            print(f"  Fidelities: {[f'{f:.4f}' for f in history['fidelities']]}")
            
            if history['converged_at_layer']:
                print(f"  ✓ Converged at {history['converged_at_layer']} layers")
            else:
                print(f"  ✗ Did not reach target (tried up to {max(history['layer_attempts'])} layers)")
        
        # Efficiency analysis
        if 'layers_used' in info:
            efficiency = info['layers_used'] / 8 * 100  # Percentage of max layers
            print(f"  Efficiency: Using {efficiency:.0f}% of maximum layers")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("-"*70)
    print("The adaptive strategy automatically determines the minimum number")
    print("of layers needed for each target, optimizing resource usage.")
    print("This is particularly useful for quantum hardware with limited depth.")
    print("="*70)


if __name__ == "__main__":
    demo_adaptive()