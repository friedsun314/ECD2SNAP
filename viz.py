"""
Visualization utilities for ECD-to-SNAP optimization.
Provides plotting functions for convergence, parameters, and quantum states.
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from typing import Dict, List, Optional, Tuple
# import seaborn as sns  # Optional, not required


def plot_convergence(history: Dict[str, List[float]], 
                    save_path: Optional[str] = None) -> None:
    """
    Plot optimization convergence history.
    
    Args:
        history: Dictionary with 'loss', 'max_fidelity', 'mean_fidelity' lists
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot loss
    ax = axes[0]
    ax.plot(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log barrier)')
    ax.set_title('Optimization Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot fidelities
    ax = axes[1]
    ax.plot(history['max_fidelity'], 'g-', linewidth=2, label='Max Fidelity')
    ax.plot(history['mean_fidelity'], 'b--', linewidth=1, label='Mean Fidelity')
    ax.axhline(y=0.999, color='r', linestyle=':', label='Target (0.999)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fidelity')
    ax.set_title('Batch Fidelities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.01])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_parameter_evolution(params_history: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> None:
    """
    Plot evolution of optimization parameters over iterations.
    
    Args:
        params_history: Dictionary with parameter arrays over time
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot beta magnitudes
    if 'betas' in params_history:
        ax = axes[0]
        betas = params_history['betas']
        for i in range(betas.shape[1]):
            ax.plot(np.abs(betas[:, i]), label=f'β_{i}')
        ax.set_ylabel('|β|')
        ax.set_title('Displacement Amplitudes')
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot theta angles
    if 'thetas' in params_history:
        ax = axes[1]
        thetas = params_history['thetas']
        for i in range(thetas.shape[1]):
            ax.plot(thetas[:, i], label=f'θ_{i}')
        ax.set_ylabel('θ (rad)')
        ax.set_title('Rotation Angles')
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot phi angles
    if 'phis' in params_history:
        ax = axes[2]
        phis = params_history['phis']
        for i in range(phis.shape[1]):
            ax.plot(phis[:, i], label=f'φ_{i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('φ (rad)')
        ax.set_title('Rotation Axes')
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_gate_sequence(betas: np.ndarray, 
                           phis: np.ndarray, 
                           thetas: np.ndarray,
                           save_path: Optional[str] = None) -> None:
    """
    Visualize the ECD gate sequence as a circuit-like diagram.
    
    Args:
        betas: Displacement amplitudes
        phis: Rotation axis angles
        thetas: Rotation angles
        save_path: Optional path to save the figure
    """
    N_layers = len(betas) - 1
    
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Circuit visualization
    y_qubit = 1.0
    y_cavity = 0.0
    
    # Draw qubit and cavity lines
    ax.plot([0, N_layers + 1], [y_qubit, y_qubit], 'k-', linewidth=2)
    ax.plot([0, N_layers + 1], [y_cavity, y_cavity], 'k-', linewidth=2)
    
    # Add labels
    ax.text(-0.5, y_qubit, 'Qubit', fontsize=12, ha='right')
    ax.text(-0.5, y_cavity, 'Cavity', fontsize=12, ha='right')
    
    # Draw gates
    for k in range(N_layers):
        x = k + 0.5
        
        # Rotation gate (qubit)
        rect = plt.Rectangle((x - 0.15, y_qubit - 0.15), 0.3, 0.3, 
                            facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y_qubit, f'R\n{thetas[k]:.2f}', fontsize=8, 
                ha='center', va='center')
        
        # ECD gate (connecting)
        ax.plot([x, x], [y_cavity, y_qubit], 'b-', linewidth=2)
        circle = plt.Circle((x, y_cavity), 0.15, 
                           facecolor='lightgreen', edgecolor='black')
        ax.add_patch(circle)
        ax.text(x, y_cavity - 0.4, f'β={np.abs(betas[k]):.2f}', 
                fontsize=7, ha='center')
    
    # Final rotation and displacement
    x = N_layers + 0.5
    rect = plt.Rectangle((x - 0.15, y_qubit - 0.15), 0.3, 0.3, 
                        facecolor='lightblue', edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y_qubit, f'R\n{thetas[-1]:.2f}', fontsize=8, 
            ha='center', va='center')
    
    circle = plt.Circle((x, y_cavity), 0.15, 
                       facecolor='yellow', edgecolor='black')
    ax.add_patch(circle)
    ax.text(x, y_cavity - 0.4, f'D({np.abs(betas[-1]/2):.2f})', 
            fontsize=7, ha='center')
    
    ax.set_xlim(-1, N_layers + 2)
    ax.set_ylim(-0.8, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('ECD Gate Sequence', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_gate_decomposition(U_target: qt.Qobj, 
                              U_approx: qt.Qobj,
                              N_trunc: int,
                              save_path: Optional[str] = None) -> Dict:
    """
    Analyze the quality of gate decomposition.
    
    Args:
        U_target: Target SNAP unitary
        U_approx: Approximation from ECD sequence
        N_trunc: Fock space truncation
        save_path: Optional path to save the figure
    
    Returns:
        Dictionary with analysis metrics
    """
    # Compute error matrix
    U_error = U_target.dag() * U_approx - qt.identity(U_target.dims[0])
    
    # Compute metrics
    fidelity = np.abs(np.trace(U_target.dag() * U_approx))**2 / U_target.shape[0]**2
    avg_error = np.mean(np.abs(U_error.full()))
    max_error = np.max(np.abs(U_error.full()))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot target unitary magnitude
    ax = axes[0, 0]
    im = ax.imshow(np.abs(U_target.full()), cmap='viridis', vmin=0, vmax=1)
    ax.set_title('|U_target|')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Plot target unitary phase
    ax = axes[0, 1]
    im = ax.imshow(np.angle(U_target.full()), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.set_title('arg(U_target)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Plot approximation magnitude
    ax = axes[0, 2]
    im = ax.imshow(np.abs(U_approx.full()), cmap='viridis', vmin=0, vmax=1)
    ax.set_title('|U_approx|')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Plot approximation phase
    ax = axes[1, 0]
    im = ax.imshow(np.angle(U_approx.full()), cmap='hsv', vmin=-np.pi, vmax=np.pi)
    ax.set_title('arg(U_approx)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Plot error magnitude
    ax = axes[1, 1]
    im = ax.imshow(np.abs(U_error.full()), cmap='hot')
    ax.set_title('|U_error|')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax)
    
    # Plot metrics text
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = f"""
    Gate Decomposition Metrics:
    
    Fidelity: {fidelity:.8f}
    Infidelity: {1-fidelity:.2e}
    
    Average Error: {avg_error:.2e}
    Maximum Error: {max_error:.2e}
    
    Matrix Dimension: {U_target.shape[0]}
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=12, 
            transform=ax.transAxes, verticalalignment='center')
    
    plt.suptitle('ECD-to-SNAP Decomposition Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'fidelity': fidelity,
        'infidelity': 1 - fidelity,
        'avg_error': avg_error,
        'max_error': max_error
    }


def plot_wigner(state: qt.Qobj, 
               xvec: Optional[np.ndarray] = None,
               save_path: Optional[str] = None) -> None:
    """
    Plot Wigner function of a quantum state.
    
    Args:
        state: Quantum state (ket or density matrix)
        xvec: Grid points for Wigner function
        save_path: Optional path to save the figure
    """
    if xvec is None:
        xvec = np.linspace(-5, 5, 100)
    
    W = qt.wigner(state, xvec, xvec)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Wigner function
    vmax = np.abs(W).max()
    im = ax.contourf(xvec, xvec, W, levels=50, cmap='RdBu_r', 
                     vmin=-vmax, vmax=vmax)
    ax.contour(xvec, xvec, W, levels=[0], colors='black', linewidths=1)
    
    ax.set_xlabel('Re(α)')
    ax.set_ylabel('Im(α)')
    ax.set_title('Wigner Function')
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='W(α)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_batch_fidelities(fidelities: np.ndarray,
                         save_path: Optional[str] = None) -> None:
    """
    Plot histogram of batch fidelities.
    
    Args:
        fidelities: Array of fidelities from batch optimization
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(fidelities, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.999, color='r', linestyle='--', label='Target (0.999)')
    ax.axvline(x=np.mean(fidelities), color='g', linestyle='-', 
               label=f'Mean ({np.mean(fidelities):.4f})')
    ax.axvline(x=np.max(fidelities), color='b', linestyle='-', 
               label=f'Max ({np.max(fidelities):.6f})')
    
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Count')
    ax.set_title(f'Batch Fidelities (N={len(fidelities)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()