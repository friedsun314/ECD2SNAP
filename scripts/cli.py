#!/usr/bin/env python3
"""
Improved CLI with better optimization strategies.
"""

import click
import numpy as np
import jax
import jax.numpy as jnp
import json
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.optimizer import ECDSNAPOptimizer
from src.snap_targets import make_snap_full_space


@click.group()
def cli():
    """Improved ECD to SNAP gate optimizer."""
    pass


@cli.command()
@click.option('--target-type', type=click.Choice(['identity', 'linear', 'quadratic']), 
              default='identity', help='Type of SNAP gate')
@click.option('--target-param', type=float, default=0.1, help='Target parameter')
@click.option('--layers', type=int, default=4, help='Number of ECD layers')
@click.option('--truncation', type=int, default=6, help='Fock space truncation')
@click.option('--batch-size', type=int, default=16, help='Batch size')
@click.option('--learning-rate', type=float, default=0.003, help='Learning rate')
@click.option('--max-iter', type=int, default=500, help='Maximum iterations')
@click.option('--strategy', type=click.Choice(['basic', 'restarts', 'annealing', 'two-stage', 'adaptive']),
              default='restarts', help='Optimization strategy')
@click.option('--n-restarts', type=int, default=3, help='Number of restarts (for restart strategy)')
@click.option('--min-layers', type=int, default=2, help='Minimum layers for adaptive strategy')
@click.option('--max-layers', type=int, default=8, help='Maximum layers for adaptive strategy')
@click.option('--output-dir', type=str, default='results', help='Output directory')
@click.option('--verbose', is_flag=True, help='Verbose output')
def optimize(target_type, target_param, layers, truncation, batch_size, 
             learning_rate, max_iter, strategy, n_restarts, min_layers, max_layers, output_dir, verbose):
    """Run improved optimization with selected strategy."""
    
    print(f"Optimizing {target_type} SNAP gate with {strategy} strategy...")
    print(f"Config: layers={layers}, truncation={truncation}, batch={batch_size}")
    
    # Create target
    if target_type == 'identity':
        phases = np.zeros(truncation)
    elif target_type == 'linear':
        phases = np.arange(truncation) * target_param
    elif target_type == 'quadratic':
        phases = np.arange(truncation)**2 * target_param
    
    U_target = make_snap_full_space(phases, truncation)
    U_target_jax = jnp.array(U_target.full())
    
    # Create optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=layers,
        N_trunc=truncation,
        batch_size=batch_size,
        learning_rate=learning_rate,
        target_fidelity=0.999
    )
    
    # Run optimization with selected strategy
    if strategy == 'basic':
        # Smart initialization only
        key = jax.random.PRNGKey(42)
        optimizer.params = optimizer.smart_initialize(key, target_type)
        optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
        best_params, best_fidelity, info = optimizer.optimize(
            U_target_jax, max_iter, verbose
        )
    elif strategy == 'restarts':
        best_params, best_fidelity, info = optimizer.optimize_with_restarts(
            U_target_jax, max_iter, n_restarts, target_type, verbose
        )
    elif strategy == 'annealing':
        best_params, best_fidelity, info = optimizer.optimize_with_annealing(
            U_target_jax, max_iter, target_type, verbose
        )
    elif strategy == 'two-stage':
        best_params, best_fidelity, info = optimizer.optimize_two_stage(
            U_target_jax, max_iter, target_type, verbose
        )
    elif strategy == 'adaptive':
        best_params, best_fidelity, info = optimizer.optimize_adaptive_layers(
            U_target_jax, max_iter, min_layers, max_layers, 
            n_restarts=2, target_type=target_type, verbose=verbose
        )
        # Update layers for output if adaptive was used
        if 'layers_used' in info:
            layers = info['layers_used']
    
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Final fidelity: {best_fidelity:.6f}")
    print(f"Iterations: {len(info['history']['max_fidelity'])}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save target info
    target_info = {
        'type': target_type,
        'parameter': target_param,
        'phases': phases.tolist(),
        'truncation': truncation
    }
    with open(os.path.join(output_dir, 'target.json'), 'w') as f:
        json.dump(target_info, f, indent=2)
    
    # Extract best parameters
    if 'best_idx' in info:
        best_idx = info['best_idx']
        betas_array = best_params['betas'][best_idx]
        phis_array = best_params['phis'][best_idx]
        thetas_array = best_params['thetas'][best_idx]
    else:
        # For single parameter set (not batch)
        betas_array = best_params['betas'].flatten() if best_params['betas'].ndim > 1 else best_params['betas']
        phis_array = best_params['phis'].flatten() if best_params['phis'].ndim > 1 else best_params['phis']
        thetas_array = best_params['thetas'].flatten() if best_params['thetas'].ndim > 1 else best_params['thetas']
    
    # Save optimization results
    results = {
        'parameters': {
            'betas_real': np.real(betas_array).tolist(),
            'betas_imag': np.imag(betas_array).tolist(),
            'phis': phis_array.tolist(),
            'thetas': thetas_array.tolist()
        },
        'fidelity': float(best_fidelity),
        'strategy': strategy,
        'iterations': len(info['history']['max_fidelity']),
        'config': {
            'layers': layers,
            'truncation': truncation,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/")
    
    # Show fidelity vs target
    if best_fidelity >= 0.999:
        print("✓ Target fidelity achieved!")
    elif best_fidelity >= 0.99:
        print("○ Good fidelity achieved")
    elif best_fidelity >= 0.9:
        print("△ Moderate fidelity achieved")
    else:
        print("✗ Low fidelity - consider more layers or different strategy")


@cli.command()
def compare_strategies():
    """Compare different optimization strategies."""
    
    print("Comparing optimization strategies for identity SNAP...")
    print("="*60)
    
    # Fixed configuration
    layers = 4
    truncation = 6
    batch_size = 16
    max_iter = 300
    
    # Create target
    phases = np.zeros(truncation)
    U_target = make_snap_full_space(phases, truncation)
    U_target_jax = jnp.array(U_target.full())
    
    strategies = ['basic', 'restarts', 'annealing', 'two-stage', 'adaptive']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        optimizer = ECDSNAPOptimizer(
            N_layers=layers,
            N_trunc=truncation,
            batch_size=batch_size,
            learning_rate=0.003
        )
        
        if strategy == 'basic':
            key = jax.random.PRNGKey(42)
            optimizer.params = optimizer.smart_initialize(key, 'identity')
            optimizer.opt_state = optimizer.optimizer.init(optimizer.params)
            _, fidelity, _ = optimizer.optimize(U_target_jax, max_iter, verbose=False)
        elif strategy == 'restarts':
            _, fidelity, _ = optimizer.optimize_with_restarts(
                U_target_jax, max_iter, n_restarts=3, target_type='identity', verbose=False
            )
        elif strategy == 'annealing':
            _, fidelity, _ = optimizer.optimize_with_annealing(
                U_target_jax, max_iter, target_type='identity', verbose=False
            )
        elif strategy == 'two-stage':
            _, fidelity, _ = optimizer.optimize_two_stage(
                U_target_jax, max_iter, target_type='identity', verbose=False
            )
        elif strategy == 'adaptive':
            _, fidelity, info = optimizer.optimize_adaptive_layers(
                U_target_jax, max_iter, min_layers=2, max_layers=6,
                n_restarts=2, target_type='identity', verbose=False
            )
            # Store extra info for adaptive
            if 'layers_used' in info:
                print(f"  (Used {info['layers_used']} layers)")
        
        results[strategy] = fidelity
        print(f"  Fidelity: {fidelity:.6f}")
    
    print("\n" + "="*60)
    print("Results Summary:")
    for strategy, fidelity in results.items():
        status = "✓" if fidelity >= 0.99 else "○" if fidelity >= 0.9 else "✗"
        print(f"  {status} {strategy:10s}: F = {fidelity:.6f}")
    
    best_strategy = max(results, key=results.get)
    print(f"\nBest strategy: {best_strategy} (F = {results[best_strategy]:.6f})")


if __name__ == '__main__':
    cli()