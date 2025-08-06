"""
Command-line interface for ECD-to-SNAP optimization.
Provides easy access to optimization and visualization functionality.
"""

import click
import numpy as np
import json
import pickle
from pathlib import Path
from optimizer import ECDSNAPOptimizer
from snap_targets import (
    make_snap, identity_snap, linear_snap, quadratic_snap, 
    cubic_snap, random_snap, kerr_evolution_snap
)
from viz import (
    plot_convergence, plot_parameter_evolution, 
    visualize_gate_sequence, analyze_gate_decomposition, 
    plot_batch_fidelities
)
from gates import build_ecd_sequence
import jax.numpy as jnp


@click.group()
def cli():
    """ECD-to-SNAP Gate Optimization Tool"""
    pass


@cli.command()
@click.option('--layers', '-l', default=6, help='Number of ECD layers')
@click.option('--batch', '-b', default=32, help='Batch size for multi-start optimization')
@click.option('--truncation', '-n', default=10, help='Fock space truncation')
@click.option('--target-type', '-t', 
              type=click.Choice(['identity', 'linear', 'quadratic', 'cubic', 
                               'random', 'kerr', 'custom']),
              default='linear', help='Type of target SNAP gate')
@click.option('--target-param', '-p', default=0.1, type=float, 
              help='Parameter for target SNAP gate')
@click.option('--target-file', '-f', type=click.Path(exists=True),
              help='File containing custom target phases (JSON or numpy)')
@click.option('--max-iter', '-i', default=10000, help='Maximum iterations')
@click.option('--learning-rate', '-r', default=0.003, type=float, 
              help='Adam learning rate')
@click.option('--target-fidelity', default=0.999, type=float,
              help='Target fidelity to stop optimization')
@click.option('--output', '-o', default='results', help='Output directory name')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def optimize(layers, batch, truncation, target_type, target_param, 
            target_file, max_iter, learning_rate, target_fidelity, 
            output, verbose):
    """
    Optimize ECD sequence to approximate a SNAP gate.
    
    Examples:
        # Optimize for a linear SNAP gate
        python cli.py optimize --target-type linear --target-param 0.5
        
        # Use custom target phases from file
        python cli.py optimize --target-type custom --target-file phases.json
        
        # High-fidelity optimization with more layers
        python cli.py optimize --layers 10 --target-fidelity 0.9999
    """
    
    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    
    # Create target SNAP gate
    if target_type == 'identity':
        U_target = identity_snap(truncation)
        phases = np.zeros(truncation)
    elif target_type == 'linear':
        U_target = linear_snap(target_param, truncation)
        phases = target_param * np.arange(truncation)
    elif target_type == 'quadratic':
        U_target = quadratic_snap(target_param, truncation)
        phases = target_param * np.arange(truncation)**2
    elif target_type == 'cubic':
        U_target = cubic_snap(target_param, truncation)
        phases = target_param * np.arange(truncation)**3
    elif target_type == 'random':
        U_target = random_snap(truncation, seed=int(target_param))
        phases = np.angle(U_target.diag())
    elif target_type == 'kerr':
        U_target = kerr_evolution_snap(target_param, 1.0, truncation)
        n = np.arange(truncation)
        phases = -target_param * n * (n - 1) / 2
    elif target_type == 'custom':
        if not target_file:
            raise click.UsageError("--target-file required for custom target")
        
        # Load phases from file
        if target_file.endswith('.json'):
            with open(target_file, 'r') as f:
                phases = np.array(json.load(f))
        elif target_file.endswith('.npy'):
            phases = np.load(target_file)
        else:
            raise click.UsageError("Target file must be .json or .npy")
        
        U_target = make_snap(phases, truncation)
    
    # Save target information
    target_info = {
        'type': target_type,
        'parameter': target_param,
        'phases': phases.tolist(),
        'truncation': truncation
    }
    with open(output_dir / 'target.json', 'w') as f:
        json.dump(target_info, f, indent=2)
    
    if verbose:
        click.echo(f"Target SNAP gate: {target_type}")
        click.echo(f"Truncation: {truncation}")
        click.echo(f"Layers: {layers}")
        click.echo(f"Batch size: {batch}")
        click.echo(f"Starting optimization...")
    
    # Initialize optimizer
    optimizer = ECDSNAPOptimizer(
        N_layers=layers,
        N_trunc=truncation,
        batch_size=batch,
        learning_rate=learning_rate,
        target_fidelity=target_fidelity
    )
    
    # Convert target to JAX array for full space
    from snap_targets import make_snap_full_space
    U_target_full = make_snap_full_space(phases, truncation)
    U_target_jax = jnp.array(U_target_full.full())
    
    # Run optimization
    best_params, best_fidelity, info = optimizer.optimize(
        U_target_jax, 
        max_iterations=max_iter,
        verbose=verbose
    )
    
    click.echo(f"\nOptimization complete!")
    click.echo(f"Best fidelity: {best_fidelity:.8f}")
    click.echo(f"Iterations: {info['iterations']}")
    
    # Save results
    results = {
        'parameters': {
            'betas': np.array(best_params['betas']).tolist(),
            'phis': np.array(best_params['phis']).tolist(),
            'thetas': np.array(best_params['thetas']).tolist()
        },
        'fidelity': best_fidelity,
        'iterations': info['iterations'],
        'config': {
            'layers': layers,
            'truncation': truncation,
            'batch_size': batch,
            'learning_rate': learning_rate
        }
    }
    
    # Save as JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full info as pickle
    with open(output_dir / 'full_results.pkl', 'wb') as f:
        pickle.dump({'params': best_params, 'info': info}, f)
    
    # Generate plots
    if verbose:
        click.echo("Generating plots...")
    
    plot_convergence(info['history'], save_path=output_dir / 'convergence.png')
    
    visualize_gate_sequence(
        np.array(best_params['betas']), 
        np.array(best_params['phis']), 
        np.array(best_params['thetas']),
        save_path=output_dir / 'gate_sequence.png'
    )
    
    plot_batch_fidelities(
        np.array(info['batch_fidelities']),
        save_path=output_dir / 'batch_fidelities.png'
    )
    
    click.echo(f"Results saved to {output_dir}/")


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--truncation', '-n', default=10, help='Fock space truncation')
def analyze(results_file, truncation):
    """
    Analyze optimization results from a previous run.
    
    Args:
        results_file: Path to results.json or full_results.pkl
    """
    # Load results
    if results_file.endswith('.json'):
        with open(results_file, 'r') as f:
            data = json.load(f)
            params = {
                'betas': np.array(data['parameters']['betas']),
                'phis': np.array(data['parameters']['phis']),
                'thetas': np.array(data['parameters']['thetas'])
            }
            fidelity = data['fidelity']
    elif results_file.endswith('.pkl'):
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
            params = data['params']
            fidelity = data.get('fidelity', -1)
    else:
        raise click.UsageError("Results file must be .json or .pkl")
    
    # Load target (should be in same directory)
    results_dir = Path(results_file).parent
    target_file = results_dir / 'target.json'
    
    if target_file.exists():
        with open(target_file, 'r') as f:
            target_info = json.load(f)
            phases = np.array(target_info['phases'])
            truncation = target_info.get('truncation', truncation)
    else:
        click.echo("Warning: target.json not found, using zero phases")
        phases = np.zeros(truncation)
    
    # Reconstruct gates
    from snap_targets import make_snap_full_space
    U_target = make_snap_full_space(phases, truncation)
    U_approx = build_ecd_sequence(
        params['betas'], params['phis'], params['thetas'], truncation
    )
    
    # Analyze decomposition
    click.echo(f"Loaded results with fidelity: {fidelity:.8f}")
    click.echo("Analyzing gate decomposition...")
    
    metrics = analyze_gate_decomposition(
        U_target, U_approx, truncation,
        save_path=results_dir / 'decomposition_analysis.png'
    )
    
    click.echo(f"Analysis complete:")
    click.echo(f"  Fidelity: {metrics['fidelity']:.8f}")
    click.echo(f"  Infidelity: {metrics['infidelity']:.2e}")
    click.echo(f"  Average Error: {metrics['avg_error']:.2e}")
    click.echo(f"  Maximum Error: {metrics['max_error']:.2e}")


@cli.command()
@click.option('--truncation', '-n', default=10, help='Fock space truncation')
@click.option('--output', '-o', help='Output file for phases')
def generate_target(truncation, output):
    """
    Generate example target SNAP phases interactively.
    """
    click.echo("Select target type:")
    click.echo("1. Linear (θ_n = a*n)")
    click.echo("2. Quadratic (θ_n = a*n²)")
    click.echo("3. Cubic (θ_n = a*n³)")
    click.echo("4. Kerr evolution")
    click.echo("5. Random")
    click.echo("6. Custom formula")
    
    choice = click.prompt("Choice", type=int)
    
    if choice in [1, 2, 3]:
        param = click.prompt("Parameter value", type=float)
        n = np.arange(truncation)
        if choice == 1:
            phases = param * n
        elif choice == 2:
            phases = param * n**2
        else:
            phases = param * n**3
    elif choice == 4:
        chi = click.prompt("Kerr strength", type=float)
        t = click.prompt("Evolution time", type=float)
        n = np.arange(truncation)
        phases = -chi * t * n * (n - 1) / 2
    elif choice == 5:
        seed = click.prompt("Random seed", type=int, default=42)
        np.random.seed(seed)
        phases = np.random.uniform(0, 2*np.pi, truncation)
    elif choice == 6:
        click.echo("Enter phases (comma-separated):")
        phases_str = click.prompt("Phases")
        phases = np.array([float(x) for x in phases_str.split(',')])
        if len(phases) < truncation:
            phases = np.pad(phases, (0, truncation - len(phases)))
        else:
            phases = phases[:truncation]
    else:
        click.echo("Invalid choice")
        return
    
    # Display phases
    click.echo("\nGenerated phases:")
    for i, phase in enumerate(phases):
        click.echo(f"  θ_{i} = {phase:.4f}")
    
    # Save if requested
    if output:
        if output.endswith('.json'):
            with open(output, 'w') as f:
                json.dump(phases.tolist(), f, indent=2)
        elif output.endswith('.npy'):
            np.save(output, phases)
        else:
            click.echo("Output file must be .json or .npy")
            return
        click.echo(f"Phases saved to {output}")


if __name__ == '__main__':
    cli()