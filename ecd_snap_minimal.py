#!/usr/bin/env python3
"""
Minimal ECD-to-SNAP Gate Optimization Script

A self-contained script that optimizes ECD gate sequences to approximate 
SNAP gates using JAX and automatic differentiation.

Concepts:
  • target-levels (N_target): number of cavity Fock levels whose phases you actually care about.
    The score (fidelity) is computed only on this subspace.
  • truncation (N_trunc): size of the truncated Hilbert space used for simulation.
    Choose N_trunc = N_target + guard, where guard (8–20) provides headroom so boundary effects are negligible.

Usage:
    python ecd_snap_minimal.py --phases "0,0.5,1.0,1.5" --layers 4
    python ecd_snap_minimal.py --phases phases.txt --plot
"""

# Imports (JAX + Optax + utilities)
import argparse
import json
import os
from typing import Dict, Tuple, Union, List
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import lax
from jax.lib import xla_bridge
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial


# ============================================================================
# JAX Gate Operations
# ============================================================================

@partial(jit, static_argnames=("N_trunc",))
def displacement_operator_jax(beta: complex, N_trunc: int) -> jnp.ndarray:
    """Create displacement operator D(beta) using JAX matrix exponentiation (complex64)."""
    n = jnp.arange(N_trunc, dtype=jnp.float32)
    # Creation/annihilation operators (complex64 to avoid dtype promotion later)
    a_dag = jnp.diag(jnp.sqrt(n[1:]), k=1).astype(jnp.complex64)  # a†
    a = jnp.diag(jnp.sqrt(n[1:]), k=-1).astype(jnp.complex64)     # a
    beta_c64 = jnp.asarray(beta, dtype=jnp.complex64)
    # D(beta) = exp(beta * a† - beta* * a)
    operator = beta_c64 * a_dag - jnp.conj(beta_c64) * a
    return jax.scipy.linalg.expm(operator).astype(jnp.complex64)


@jit  
def rotation_operator_jax(theta: float, phi: float) -> jnp.ndarray:
    """Create qubit rotation R_phi(theta) using JAX operations."""
    cos_half = jnp.cos(theta / 2)
    sin_half = jnp.sin(theta / 2)
    exp_phi = jnp.exp(1j * phi)
    
    return jnp.array([
        [cos_half, -1j * sin_half * jnp.conj(exp_phi)],
        [-1j * sin_half * exp_phi, cos_half]
    ], dtype=jnp.complex64)


@partial(jit, static_argnames=("N_trunc",))
def ecd_gate_jax(beta: complex, N_trunc: int) -> jnp.ndarray:
    """Create ECD gate: |0><0| ⊗ D(β) + |1><1| ⊗ D(-β)."""
    D_plus = displacement_operator_jax(beta, N_trunc)
    D_minus = displacement_operator_jax(-beta, N_trunc)
    
    # Qubit projectors
    proj_0 = jnp.array([[1, 0], [0, 0]], dtype=jnp.complex64)
    proj_1 = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
    
    # ECD = |0><0| ⊗ D(β) + |1><1| ⊗ D(-β)
    term1 = jnp.kron(proj_0, D_plus)
    term2 = jnp.kron(proj_1, D_minus)
    return term1 + term2


@partial(jit, static_argnames=("N_trunc",))  # JIT the whole sequence builder; treat N_trunc as static
def build_ecd_sequence_jax(betas: jnp.ndarray, phis: jnp.ndarray, 
                           thetas: jnp.ndarray, N_trunc: int) -> jnp.ndarray:
    """Build full ECD sequence using JAX operations (scan + JIT)."""
    # betas/phis/thetas are length N_layers+1; we scan over the first N_layers
    N_layers = betas.shape[0] - 1
    dim = 2 * N_trunc
    I_cav = jnp.eye(N_trunc, dtype=jnp.complex64)
    I_qub = jnp.eye(2, dtype=jnp.complex64)
    U0 = jnp.eye(dim, dtype=jnp.complex64)

    def body(U, inputs):
        b, p, t = inputs
        # Qubit rotation for this layer
        R = rotation_operator_jax(t, p)
        R_full = jnp.kron(R, I_cav)
        U = R_full @ U
        # Conditional displacement (ECD) for this layer
        ECD = ecd_gate_jax(b, N_trunc)
        U = ECD @ U
        return U, None

    # Run scan over layers 0..N_layers-1
    U, _ = lax.scan(body, U0, (betas[:N_layers], phis[:N_layers], thetas[:N_layers]))

    # Final rotation and final displacement
    R_final = rotation_operator_jax(thetas[-1], phis[-1])
    R_final_full = jnp.kron(R_final, I_cav)
    U = R_final_full @ U

    D_final = displacement_operator_jax(betas[-1] / 2, N_trunc)
    D_final_full = jnp.kron(I_qub, D_final)
    U = D_final_full @ U

    return U


# ============================================================================
# SNAP Gate Creation
# ============================================================================

def make_snap_jax(phases: jnp.ndarray, N_trunc: int) -> jnp.ndarray:
    """Create SNAP gate unitary from phases using JAX."""
    # Ensure correct length
    if len(phases) < N_trunc:
        phases = jnp.pad(phases, (0, N_trunc - len(phases)), constant_values=0)
    else:
        phases = phases[:N_trunc]
    phases = jnp.asarray(phases, dtype=jnp.float32)
    U_snap_cavity = jnp.diag(jnp.exp(1j * phases)).astype(jnp.complex64)
    U_snap_full = jnp.kron(jnp.eye(2, dtype=jnp.complex64), U_snap_cavity)
    return U_snap_full


# ============================================================================
# Subspace Fidelity (score only the first N_target cavity levels)
# ============================================================================

def build_projector_qubit_cavity_subspace(N_target: int, N_trunc: int) -> jnp.ndarray:
    """Projector onto qubit ⊗ {|0..N_target-1>} inside a qubit⊗cavity space of size 2*N_trunc."""
    if N_target > N_trunc:
        raise ValueError("N_target cannot exceed N_trunc")
    I2 = jnp.eye(2, dtype=jnp.complex64)
    mask = jnp.concatenate([jnp.ones(N_target), jnp.zeros(N_trunc - N_target)]).astype(jnp.complex64)
    P = jnp.kron(I2, jnp.diag(mask))
    return P

@jit
def fidelity_full(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
    """Full-space unitary fidelity F = |Tr(U†V)|² / d²."""
    d = U_target.shape[0]
    trace = jnp.trace(jnp.conj(U_target.T) @ U_approx)
    return jnp.abs(trace)**2 / d**2

def fidelity_subspace(U_target: jnp.ndarray, U_approx: jnp.ndarray,
                      P: jnp.ndarray, N_target: int) -> float:
    """Subspace fidelity over qubit⊗{|0..N_target-1|}. No JIT to keep P as a captured constant."""
    d = 2 * N_target
    M = jnp.conj(U_target.T) @ U_approx
    M_sub = P @ M @ P
    tr = jnp.trace(M_sub)
    return (jnp.abs(tr)**2) / (d**2)


# ============================================================================
# Adaptive batch size heuristic
# ============================================================================
def _adaptive_batch_size(base_batch: int, N_trunc: int) -> int:
    """
    Reduce batch size as N_trunc grows to avoid excessive memory/compile times.
    Heuristic: keep full batch up to N_trunc≈24, then scale ~ 24/N_trunc with a floor of 4.
    """
    if base_batch <= 4:
        return base_batch
    scale = 24.0 / max(1, N_trunc)
    adapted = int(round(base_batch * scale))
    return max(4, min(base_batch, adapted))

# ============================================================================
# Jitted batch fidelity helper
# ============================================================================
@jit
def _batch_fidelity_jitted(params: Dict, U_target: jnp.ndarray, N_trunc: int,
                           P_sub: Union[jnp.ndarray, None], N_target: int) -> jnp.ndarray:
    build_fn = lambda b, p, t: build_ecd_sequence_jax(b, p, t, N_trunc)
    U_batch = vmap(build_fn)(params['betas'], params['phis'], params['thetas'])

    def fid(U):
        if P_sub is None:
            d = U_target.shape[0]
            trace = jnp.trace(jnp.conj(U_target.T) @ U)
            return (jnp.abs(trace) ** 2) / (d ** 2)
        else:
            d = 2 * N_target
            M = jnp.conj(U_target.T) @ U
            M_sub = P_sub @ M @ P_sub
            tr = jnp.trace(M_sub)
            return (jnp.abs(tr) ** 2) / (d ** 2)

    return vmap(fid)(U_batch)

# ============================================================================
# Optimizer
# ============================================================================

class MinimalECDOptimizer:
    """Minimal optimizer using multi-starts.

    Parameters
    ----------
    N_layers : int
        Number of ECD layers (not counting the final half displacement).
    N_trunc : int
        Truncated cavity dimension.
    N_target : int
        Number of cavity Fock levels we score on (<= N_trunc). If N_target < N_trunc,
        subspace fidelity is used automatically.
    batch_size : int
        Number of random initializations optimized in parallel.
    learning_rate : float
        Adam learning rate.
    target_fidelity : float
        Early-stop threshold (applied to the chosen fidelity definition).
    """
    def __init__(self, N_layers: int = 4, N_trunc: int = 6, N_target: int = None,
                 batch_size: int = 16, learning_rate: float = 3e-3, target_fidelity: float = 0.999):
        self.N_layers = N_layers
        self.N_trunc = N_trunc
        self.N_target = N_trunc if N_target is None else int(N_target)
        if self.N_target > self.N_trunc:
            raise ValueError("N_target cannot exceed N_trunc")

        self.batch_size = _adaptive_batch_size(batch_size, self.N_trunc)
        if self.batch_size != batch_size:
            print(f"[adaptive-batch] Reduced batch_size from {batch_size} to {self.batch_size} for N_trunc={self.N_trunc}.")
        self.learning_rate = learning_rate
        self.target_fidelity = target_fidelity

        # Precompute subspace projector if needed
        # If N_target < N_trunc, use subspace fidelity (guard-band ensures boundary is not felt)
        self.P_sub = None
        if self.N_target < self.N_trunc:
            self.P_sub = build_projector_qubit_cavity_subspace(self.N_target, self.N_trunc)

        self.optimizer = optax.adam(learning_rate)
        self.params = None
        self.opt_state = None

    def initialize_parameters(self, key: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """Initialize random parameters."""
        keys = jax.random.split(key, 4)

        # Small random displacements
        beta_real = jax.random.normal(keys[0], (self.batch_size, self.N_layers + 1)) * 0.1
        beta_imag = jax.random.normal(keys[1], (self.batch_size, self.N_layers + 1)) * 0.1
        betas = beta_real + 1j * beta_imag

        # Random rotations
        phis = jax.random.uniform(keys[2], (self.batch_size, self.N_layers + 1), 
                                 minval=0, maxval=2*jnp.pi)
        thetas = jax.random.uniform(keys[3], (self.batch_size, self.N_layers + 1), 
                                   minval=0, maxval=jnp.pi)

        return {'betas': betas, 'phis': phis, 'thetas': thetas}

    def compute_fidelity(self, U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
        """Dispatch to full-space or subspace fidelity."""
        if self.P_sub is None:
            return fidelity_full(U_target, U_approx)
        else:
            return fidelity_subspace(U_target, U_approx, self.P_sub, self.N_target)

    def batch_fidelity(self, params: Dict, U_target: jnp.ndarray) -> jnp.ndarray:
        """Compute fidelities for entire batch (jitted helper)."""
        return _batch_fidelity_jitted(params, U_target, self.N_trunc, self.P_sub, self.N_target)

    def loss_function(self, params: Dict, U_target: jnp.ndarray) -> float:
        """Logarithmic barrier loss: C = Σ log(1 - F_j)."""
        fidelities = self.batch_fidelity(params, U_target)
        eps = 1e-10
        losses = jnp.log(jnp.maximum(1 - fidelities, eps))
        return jnp.sum(losses)

    def optimize_single_run(self, U_target: jnp.ndarray, max_iter: int, 
                           verbose: bool = False) -> Tuple[Dict, float, Dict]:
        """Single optimization run."""
        loss_and_grad = jax.value_and_grad(self.loss_function, argnums=0)

        history = {'loss': [], 'max_fidelity': []}
        best_fidelity = 0.0
        best_params = None

        pbar = tqdm(range(max_iter), disable=not verbose, desc="Optimizing")

        for i in pbar:
            loss, grads = loss_and_grad(self.params, U_target)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            fidelities = self.batch_fidelity(self.params, U_target)
            max_fid = jnp.max(fidelities)

            history['loss'].append(float(loss))
            history['max_fidelity'].append(float(max_fid))

            if max_fid > best_fidelity:
                best_fidelity = float(max_fid)
                best_idx = jnp.argmax(fidelities)
                best_params = {
                    'betas': self.params['betas'][best_idx],
                    'phis': self.params['phis'][best_idx], 
                    'thetas': self.params['thetas'][best_idx]
                }

            pbar.set_postfix({'F_max': f'{max_fid:.6f}'})

            if max_fid >= self.target_fidelity:
                break

        info = {'history': history, 'iterations': i + 1}
        return best_params, best_fidelity, info

    def optimize_with_restarts(self, U_target: jnp.ndarray, max_iter: int = 1000,
                              n_restarts: int = 3, verbose: bool = True,
                              seed_params: Dict = None) -> Tuple[Dict, float, Dict]:
        """Optimize with multiple random restarts."""
        best_overall_fidelity = 0.0
        best_overall_params = None
        best_overall_info = None

        for restart in range(n_restarts):
            if verbose:
                print(f"\n--- Restart {restart + 1}/{n_restarts} ---")

            # Initialize parameters
            key = jax.random.PRNGKey(42 + restart * 1000)
            self.params = self.initialize_parameters(key)
            # If we have a seed, inject it as candidate 0 on the first restart
            if (seed_params is not None) and (restart == 0):
                self.params['betas']  = self.params['betas'].at[0].set(seed_params['betas'])
                self.params['phis']   = self.params['phis'].at[0].set(seed_params['phis'])
                self.params['thetas'] = self.params['thetas'].at[0].set(seed_params['thetas'])
            self.opt_state = self.optimizer.init(self.params)

            # Run optimization
            params, fidelity, info = self.optimize_single_run(
                U_target, max_iter // n_restarts, verbose
            )

            if fidelity > best_overall_fidelity:
                best_overall_fidelity = fidelity
                best_overall_params = params
                best_overall_info = info

                if fidelity >= self.target_fidelity:
                    if verbose:
                        print(f"✓ Target fidelity reached: {fidelity:.6f}")
                    break

        return best_overall_params, best_overall_fidelity, best_overall_info


# ============================================================================
# Tail probability utilities (to verify boundary is not "felt")
# ============================================================================

def cavity_tail_probability(cavity_state: jnp.ndarray, N_target: int) -> jnp.ndarray:
    """Probability mass above N_target in a cavity state vector."""
    p = jnp.abs(cavity_state)**2
    return jnp.sum(p[N_target:])

def split_qubit_blocks(psi_full: jnp.ndarray, N_trunc: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split a (2*N_trunc,) state into (q=0 cavity vector, q=1 cavity vector)."""
    return psi_full[:N_trunc], psi_full[N_trunc:]


def coherent_cavity_state(alpha: complex, N_trunc: int) -> jnp.ndarray:
    """Truncated coherent state |alpha> in number basis (stable via log-gamma)."""
    n = jnp.arange(N_trunc, dtype=jnp.float32)
    abs_alpha = jnp.abs(alpha)
    # log |coeff_n| = -|α|^2/2 + n*log|α| - 0.5*log(n!)
    log_mag = -0.5 * (abs_alpha ** 2) + n * jnp.log(jnp.maximum(abs_alpha, 1e-30)) - 0.5 * jax.scipy.special.gammaln(n + 1.0)
    # phase = e^{i n arg(α)}
    phase = jnp.exp(1j * n * jnp.angle(alpha))
    coeffs = jnp.exp(log_mag).astype(jnp.complex64) * phase.astype(jnp.complex64)
    norm = jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
    return coeffs / (norm + 1e-20)

# ============================================================================
# Compact tail-check helpers
# ============================================================================
def _ket(index: int, dim: int) -> jnp.ndarray:
    v = jnp.zeros((dim,), dtype=jnp.complex64)
    return v.at[index].set(1.0)

def _alpha_envelope(betas: jnp.ndarray) -> jnp.ndarray:
    """Conservative coherent amplitude envelope used for tail probing."""
    return jnp.sum(jnp.abs(betas)) - jnp.abs(betas[-1]) + jnp.abs(betas[-1] / 2)

def _tails_for_input(U: jnp.ndarray, psi_in: jnp.ndarray, N_trunc: int, N_target: int) -> Tuple[float, float]:
    out = U @ psi_in
    cav_q0, cav_q1 = split_qubit_blocks(out, N_trunc)
    t_q0 = float(cavity_tail_probability(cav_q0, N_target))
    t_q1 = float(cavity_tail_probability(cav_q1, N_target))
    return t_q0, t_q1

# ============================================================================
# Tail probability helper: measure_tail_probs
# ============================================================================
def measure_tail_probs(U_best: jnp.ndarray, N_target: int, N_trunc: int, betas: jnp.ndarray) -> Dict[str, float]:
    """Compute tail probabilities for three probe inputs and return a metrics dict."""
    dim = 2 * N_trunc

    # |q=0> ⊗ |0>
    t0_q0, t0_q1 = _tails_for_input(U_best, _ket(0, dim), N_trunc, N_target)

    # |q=1> ⊗ |0>
    t1_q0, t1_q1 = _tails_for_input(U_best, _ket(N_trunc, dim), N_trunc, N_target)

    # |q=0> ⊗ |alpha> (conservative envelope)
    alpha_env = _alpha_envelope(betas)
    cav_alpha = coherent_cavity_state(alpha_env, N_trunc)
    psi_alpha = jnp.concatenate([cav_alpha, jnp.zeros_like(cav_alpha)])
    ta_q0, ta_q1 = _tails_for_input(U_best, psi_alpha, N_trunc, N_target)

    max_tail = max(t0_q0, t0_q1, t1_q0, t1_q1, ta_q0, ta_q1)

    return {
        't0_q0': t0_q0, 't0_q1': t0_q1,
        't1_q0': t1_q0, 't1_q1': t1_q1,
        'ta_q0': ta_q0, 'ta_q1': ta_q1,
        'alpha_abs': float(jnp.abs(alpha_env)),
        'max_tail': float(max_tail),
    }



# ============================================================================
# Pretty-printer for tail metrics
# ============================================================================
def print_tail_report(metrics: Dict[str, float]):
    print("\nTail-check (probability above N_target for probe states):")
    print(f"  Input |q=0, n=0>: tail_q0={metrics['t0_q0']:.3e}, tail_q1={metrics['t0_q1']:.3e}")
    print(f"  Input |q=1, n=0>: tail_q0={metrics['t1_q0']:.3e}, tail_q1={metrics['t1_q1']:.3e}")
    print(f"  Input |q=0, |alpha|≈{metrics['alpha_abs']:.3f}>: tail_q0={metrics['ta_q0']:.3e}, tail_q1={metrics['ta_q1']:.3e}")

# ============================================================================
# Visualization
# ============================================================================

def plot_convergence(history: Dict, save_path: str = None):
    """Plot optimization convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Fidelity vs iteration
    ax1.plot(history['max_fidelity'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Fidelity')
    ax1.set_title('Convergence')
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # Loss vs iteration  
    ax2.plot(history['loss'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss Function')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    else:
        plt.show()


# ============================================================================
# Utility Functions  
# ============================================================================

def parse_phases(phases_str: str) -> np.ndarray:
    """Parse phases from string or file."""
    # Check if file exists
    if os.path.exists(phases_str):
        try:
            if phases_str.endswith('.npy'):
                return np.load(phases_str)
            else:
                with open(phases_str, 'r') as f:
                    content = f.read().strip()
                    if content.startswith('[') and content.endswith(']'):
                        # JSON format
                        return np.array(json.loads(content))
                    else:
                        # Space/comma separated
                        return np.array([float(x) for x in content.replace(',', ' ').split() if x.strip()])
        except Exception as e:
            raise ValueError(f"Error reading phases from file '{phases_str}': {e}")
    else:
        # Parse as comma-separated values
        try:
            return np.array([float(x.strip()) for x in phases_str.split(',') if x.strip()])
        except ValueError as e:
            raise ValueError(f"Error parsing phases '{phases_str}': {e}")


def save_results(params: Dict, fidelity: float, info: Dict, phases: np.ndarray, 
                output_dir: str, N_target: int = None):
    """Save optimization results."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'parameters': {
            'betas_real': np.real(params['betas']).tolist(),
            'betas_imag': np.imag(params['betas']).tolist(), 
            'phis': params['phis'].tolist(),
            'thetas': params['thetas'].tolist()
        },
        'fidelity': float(fidelity),
        'target_phases': phases.tolist(),
        'iterations': info['iterations'],
        'N_target': int(N_target) if N_target is not None else None,
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}/")


# ============================================================================
# Minimal helpers (readability without overhead)
# ============================================================================
def _success(best_fid: float,
             metrics: Dict[str, float] | None,
             target_fid: float,
             auto_guard: bool,
             tail_threshold: float) -> bool:
    """Unified acceptance criterion for a candidate solution."""
    if not auto_guard:
        return best_fid >= target_fid
    if metrics is None:
        return False
    return (metrics.get('max_tail', 1.0) <= tail_threshold) and (best_fid >= target_fid)

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Minimal ECD-to-SNAP Optimizer')
    parser.add_argument('--phases', required=True, 
                       help='Target phases: comma-separated values or file path')
    parser.add_argument('--layers', type=int, default=4, 
                       help='Max ECD layers (searches 0..L and picks minimal depth that meets target; default: 4)')
    parser.add_argument('--truncation', type=int, default=6,
                       help='Fock space simulation cutoff (use guard band, default: 6)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for multi-start (default: 16)')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Maximum iterations (default: 1000)')
    parser.add_argument('--restarts', type=int, default=3,
                       help='Number of restarts (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=3e-3,
                       help='Learning rate (default: 3e-3)')
    parser.add_argument('--plot', action='store_true',
                       help='Show convergence plot')
    parser.add_argument('--output-dir', default='results_minimal',
                       help='Output directory (default: results_minimal)')
    parser.add_argument('--target-levels', type=int, default=None,
                        help='N_target: number of cavity levels to score fidelity on (default: len(phases))')
    parser.add_argument('--tail-check', action='store_true',
                        help='After optimization, report tail probabilities above N_target for probe states')
    parser.add_argument('--auto-guard', action='store_true',
                        help='Adaptively increase N_trunc until tails fall below threshold')
    parser.add_argument('--guard-init', type=int, default=8,
                        help='Initial guard band (default: 8)')
    parser.add_argument('--guard-step', type=int, default=4,
                        help='Guard band increment when tails too large (default: 4)')
    parser.add_argument('--guard-max', type=int, default=64,
                        help='Maximum extra levels beyond N_target (default: 64)')
    parser.add_argument('--tail-threshold', type=float, default=1e-6,
                        help='Acceptable probability above N_target (default: 1e-6)')
    
    args = parser.parse_args()
    
    # Parse phases
    print(f"Parsing phases: {args.phases}")
    phases = parse_phases(args.phases)
    print(f"Target phases: {phases}")

    # Set N_target (scored levels) and print
    N_target = args.target_levels if args.target_levels is not None else len(phases)
    print(f"N_target (scored levels): {N_target}  |  N_trunc (simulated): {args.truncation}")
    if (not args.auto_guard) and (N_target > args.truncation):
        raise SystemExit("Error: --target-levels cannot exceed --truncation (use --auto-guard or increase --truncation)")
    if args.auto_guard and (args.truncation < N_target):
        print(f"Note: increasing N_trunc automatically (auto-guard) since truncation={args.truncation} < N_target={N_target}")

    # Report JAX backend/devices
    try:
        platform = xla_bridge.get_backend().platform  # 'cpu', 'gpu', or 'tpu'
    except Exception:
        platform = jax.default_backend()
    devices = jax.devices()
    first_device = devices[0] if devices else None
    device_info = str(first_device) if first_device is not None else "None"
    print(f"JAX backend: {platform.upper()} | {len(devices)} device(s). First device: {device_info}")
    
    # --------------------------------------------------------------------------
    # Inherent minimal-depth search:
    # Interpret --layers as a *maximum*; we sweep depths L = 0..L_max and
    # pick the smallest L that reaches target fidelity (and passes tail check if enabled).
    # --------------------------------------------------------------------------
    L_max = int(args.layers)
    final_params = None
    final_fidelity = 0.0
    final_info = None
    final_trunc = None
    final_layers = None
    final_metrics = None  # tail metrics for the selected solution (if measured)

    for depth in range(L_max + 1):
        print(f"\n====== Depth search: trying N_layers = {depth} (max {L_max}) ======")

        # Initialize truncation for this depth (adaptive if requested)
        if args.auto_guard:
            current_trunc = max(args.truncation, N_target + args.guard_init)
        else:
            current_trunc = args.truncation

        best_params = None
        best_fidelity = 0.0
        info = None
        metrics = None  # last tail metrics measured at this depth/trunc

        while True:
            print(f"\n=== Optimizing with N_trunc = {current_trunc} (N_target = {N_target}, N_layers = {depth}) ===")

            # Build target SNAP for this truncation
            U_target = make_snap_jax(jnp.array(phases), current_trunc)

            # Initialize optimizer with N_target for subspace fidelity
            optimizer = MinimalECDOptimizer(
                N_layers=depth,
                N_trunc=current_trunc,
                N_target=N_target,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )

            # Run optimization (seed from previous round if available)
            seed = best_params if best_params is not None else None
            print(f"\nOptimizing ECD sequence...")
            print(f"Config: layers={depth}, truncation={current_trunc}, batch_size={args.batch_size}")
            best_params, best_fidelity, info = optimizer.optimize_with_restarts(
                U_target, args.max_iter, args.restarts, verbose=True, seed_params=seed
            )

            print(f"\n--- Results at depth={depth}, N_trunc={current_trunc} ---")
            print(f"Final fidelity: {best_fidelity:.6f}")
            print(f"Iterations: {info['iterations']}")

            # If not auto-guard, we can decide success now
            if not args.auto_guard:
                # Accept if target fidelity met
                if best_fidelity >= optimizer.target_fidelity:
                    metrics = None
                    break
                else:
                    # No more truncation steps to try; break and move to next depth
                    break

            # If auto-guard, measure tails to decide if guard is sufficient
            U_best = build_ecd_sequence_jax(best_params['betas'], best_params['phis'], best_params['thetas'], current_trunc)
            U_best.block_until_ready()
            metrics = measure_tail_probs(U_best, N_target, current_trunc, best_params['betas'])
            max_tail = metrics['max_tail']
            print(f"Tail check @ N_trunc={current_trunc}: max_tail={max_tail:.3e} (threshold={args.tail_threshold:.1e})")

            # If tails below threshold and fidelity target met, we're done for this depth
            if (max_tail <= args.tail_threshold) and (best_fidelity >= optimizer.target_fidelity):
                print("✓ Guard band sufficient (tails below threshold) and target fidelity met.")
                break

            # Otherwise, increase truncation and loop again
            next_trunc = current_trunc + args.guard_step
            max_trunc = N_target + args.guard_max
            if next_trunc > max_trunc:
                print(f"Warning: Reached guard ceiling (N_trunc={next_trunc} > {max_trunc}). Stopping depth {depth}.")
                break
            current_trunc = next_trunc

        # Decide if this depth is acceptable
        # Decide if this depth is acceptable (compact check)
        if _success(best_fidelity, metrics, optimizer.target_fidelity, args.auto_guard, args.tail_threshold):
            final_params = best_params
            final_fidelity = best_fidelity
            final_info = info
            final_trunc = current_trunc
            final_layers = depth
            final_metrics = metrics
            print(f"\n✓ Selected minimal depth: N_layers = {final_layers} (fidelity {final_fidelity:.6f})")
            break  # minimal depth found
        else:
            print(f"Depth {depth} did not meet criteria; trying deeper sequence...")

    # If nothing met the criteria, fall back to the best achieved at the last tried depth
    if final_params is None:
        final_params = best_params
        final_fidelity = best_fidelity
        final_info = info
        final_trunc = current_trunc
        final_layers = depth
        final_metrics = metrics
        print(f"\n! No depth achieved target; keeping best from depth={final_layers} (fidelity {final_fidelity:.6f}).")

    # Replace the previously used variables so the rest of the script (tail check, save, plot) works unchanged
    best_params = final_params
    best_fidelity = final_fidelity
    info = final_info
    current_trunc = final_trunc

    if args.auto_guard:
        print(f"\n>>> Final choice: N_layers={final_layers}, N_trunc={current_trunc}, fidelity={best_fidelity:.6f} | target≥{optimizer.target_fidelity:.6f}, tail≤{args.tail_threshold:.1e}")
    else:
        print(f"\n>>> Final choice: N_layers={final_layers}, N_trunc={current_trunc}, fidelity={best_fidelity:.6f} | target≥{optimizer.target_fidelity:.6f}")
    
    # Optionally check tail probabilities to ensure boundary is not felt
    if args.tail_check and best_params is not None:
        U_best = build_ecd_sequence_jax(best_params['betas'], best_params['phis'], best_params['thetas'], current_trunc)
        U_best.block_until_ready()
        metrics = measure_tail_probs(U_best, N_target, current_trunc, best_params['betas'])
        print_tail_report(metrics)
    
    # Save results
    save_results(best_params, best_fidelity, info, phases, args.output_dir, N_target=N_target)
    
    # Plot
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'convergence.png')
        plot_convergence(info['history'], plot_path)


if __name__ == '__main__':
    main()