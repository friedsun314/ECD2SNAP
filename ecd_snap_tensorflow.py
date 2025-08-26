#!/usr/bin/env python3
"""
Minimal ECD-to-SNAP Gate Optimization Script

A self-contained script that optimizes ECD gate sequences to approximate 
SNAP gates using TensorFlow.

Concepts:
  • target-levels (N_target): number of cavity Fock levels whose phases you actually care about.
    The score (fidelity) is computed only on this subspace.
  • truncation (N_trunc): size of the truncated Hilbert space used for simulation.
    Choose N_trunc = N_target + guard, where guard (8–20) provides headroom so boundary effects are negligible.

Usage:
    python ecd_snap_tensorflow.py --phases "0,0.5,1.0,1.5" --layers 4
    python ecd_snap_tensorflow.py --phases phases.txt --plot
"""

# Imports (TensorFlow + utilities)
import argparse
import json
import os
from typing import Dict, Union, Optional
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# TensorFlow data types
DT_FLOAT = tf.float32
DT_COMPLEX = tf.complex64



# TensorFlow Kronecker product helper
def tf_kron(A, B):
    """Kronecker product using TensorFlow ops."""
    A = tf.convert_to_tensor(A)
    B = tf.convert_to_tensor(B)
    a0, a1 = tf.shape(A)[0], tf.shape(A)[1]
    b0, b1 = tf.shape(B)[0], tf.shape(B)[1]
    A_exp = tf.reshape(A, (a0, 1, a1, 1))
    B_exp = tf.reshape(B, (1, b0, 1, b1))
    out = A_exp * B_exp
    return tf.reshape(out, (a0 * b0, a1 * b1))

# ============================================================================
# TensorFlow Gate Operations
# ============================================================================
def displacement_operator_tf(beta: complex, N_trunc: int):
    """Create displacement operator D(beta) using TensorFlow matrix exponentiation."""
    n = tf.range(N_trunc, dtype=DT_FLOAT)
    diag_vals = tf.sqrt(n[1:])
    a_dag = tf.linalg.diag(diag_vals, k=1)
    a = tf.linalg.diag(diag_vals, k=-1)
    a_dag = tf.cast(a_dag, DT_COMPLEX)
    a = tf.cast(a, DT_COMPLEX)
    beta_c64 = tf.cast(beta, DT_COMPLEX)
    op = beta_c64 * a_dag - tf.math.conj(beta_c64) * a
    return tf.linalg.expm(op)

def rotation_operator_tf(theta: float, phi: float):
    """Create qubit rotation R_phi(theta) using TensorFlow operations."""
    theta = tf.cast(theta, DT_FLOAT)
    phi = tf.cast(phi, DT_FLOAT)
    cos_half = tf.math.cos(theta / 2.0)
    sin_half = tf.math.sin(theta / 2.0)
    ephi = tf.complex(tf.math.cos(phi), tf.math.sin(phi))
    m00 = tf.cast(cos_half, DT_COMPLEX)
    m11 = tf.cast(cos_half, DT_COMPLEX)
    m01 = -1j * tf.cast(sin_half, DT_COMPLEX) * tf.math.conj(ephi)
    m10 = -1j * tf.cast(sin_half, DT_COMPLEX) * ephi
    return tf.stack([tf.stack([m00, m01], axis=0), tf.stack([m10, m11], axis=0)], axis=0)

def ecd_gate_tf(beta: complex, N_trunc: int):
    """Create ECD gate: |0><0| ⊗ D(β) + |1><1| ⊗ D(-β)."""
    Dp = displacement_operator_tf(beta, N_trunc)
    Dm = displacement_operator_tf(-beta, N_trunc)
    P0 = tf.constant([[1, 0], [0, 0]], dtype=DT_COMPLEX)
    P1 = tf.constant([[0, 0], [0, 1]], dtype=DT_COMPLEX)
    term1 = tf_kron(P0, Dp)
    term2 = tf_kron(P1, Dm)
    return term1 + term2

def build_ecd_sequence_tf(betas, phis, thetas, N_trunc: int):
    """Build full ECD sequence using TensorFlow operations."""
    N_layers = int(betas.shape[0]) - 1
    I_cav = tf.eye(N_trunc, dtype=DT_COMPLEX)
    U = tf.eye(2 * N_trunc, dtype=DT_COMPLEX)
    for k in range(N_layers):
        R = rotation_operator_tf(thetas[k], phis[k])
        R_full = tf_kron(R, I_cav)
        U = tf.linalg.matmul(R_full, U)
        E = ecd_gate_tf(betas[k], N_trunc)
        U = tf.linalg.matmul(E, U)
    Rf = rotation_operator_tf(thetas[-1], phis[-1])
    Rf_full = tf_kron(Rf, I_cav)
    U = tf.linalg.matmul(Rf_full, U)
    Df = displacement_operator_tf(betas[-1] / 2.0, N_trunc)
    U = tf.linalg.matmul(tf_kron(tf.eye(2, dtype=DT_COMPLEX), Df), U)
    return U

def make_snap_tf(phases, N_trunc: int):
    """Create SNAP gate unitary from phases using TensorFlow."""
    phases = tf.convert_to_tensor(phases, dtype=DT_FLOAT)
    phases = tf.cond(tf.shape(phases)[0] < N_trunc,
                     lambda: tf.pad(phases, [[0, N_trunc - tf.shape(phases)[0]]]),
                     lambda: phases[:N_trunc])
    Uc = tf.linalg.diag(tf.exp(1j * tf.cast(phases, DT_COMPLEX)))
    return tf_kron(tf.eye(2, dtype=DT_COMPLEX), Uc)

# ============================================================================
# TensorFlow Fidelity and Projector Functions
# ============================================================================
def build_projector_qubit_cavity_subspace_tf(N_target: int, N_trunc: int):
    """Build projector onto qubit ⊗ {|0..N_target-1>} subspace using TensorFlow."""
    if N_target > N_trunc:
        raise ValueError("N_target cannot exceed N_trunc")
    mask = tf.concat([tf.ones((N_target,), dtype=DT_COMPLEX), tf.zeros((N_trunc - N_target,), dtype=DT_COMPLEX)], axis=0)
    P = tf.linalg.diag(mask)
    I2 = tf.eye(2, dtype=DT_COMPLEX)
    return tf_kron(I2, P)

def fidelity_full_tf(U_target: tf.Tensor, U_approx: tf.Tensor) -> tf.Tensor:
    """Full-space unitary fidelity F = |Tr(U†V)|² / d² using TensorFlow."""
    d = tf.cast(tf.shape(U_target)[0], DT_FLOAT)
    tr = tf.linalg.trace(tf.linalg.adjoint(U_target) @ U_approx)
    return tf.abs(tr) ** 2 / (d ** 2)

def fidelity_subspace_tf(U_target: tf.Tensor, U_approx: tf.Tensor, P: tf.Tensor, N_target: int) -> tf.Tensor:
    """Subspace fidelity over qubit⊗{|0..N_target-1|} using TensorFlow."""
    d = tf.cast(2 * N_target, DT_FLOAT)
    M = tf.linalg.adjoint(U_target) @ U_approx
    M_sub = P @ M @ P
    tr = tf.linalg.trace(M_sub)
    return tf.abs(tr) ** 2 / (d ** 2)



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
# TensorFlow Optimizer
# ============================================================================
class MinimalECDOptimizer:
    """Minimal ECD optimizer using TensorFlow. Supports batch multi-start optimization
    with automatic guard band adjustment for reliable simulation.
    """
    def __init__(self, N_layers: int = 4, N_trunc: int = 6, N_target: int = None,
                 batch_size: int = 16, learning_rate: float = 3e-3, target_fidelity: float = 0.999):
        self.N_layers = int(N_layers)
        self.N_trunc = int(N_trunc)
        self.N_target = int(N_trunc if N_target is None else N_target)
        if self.N_target > self.N_trunc:
            raise ValueError("N_target cannot exceed N_trunc")
        self.batch_size = _adaptive_batch_size(int(batch_size), self.N_trunc)
        if self.batch_size != batch_size:
            print(f"[adaptive-batch] Reduced batch_size from {batch_size} to {self.batch_size} for N_trunc={self.N_trunc}.")
        self.learning_rate = float(learning_rate)
        self.target_fidelity = float(target_fidelity)
        self.P_sub = None
        if self.N_target < self.N_trunc:
            self.P_sub = build_projector_qubit_cavity_subspace_tf(self.N_target, self.N_trunc)
        self.opt = tf.keras.optimizers.Adam(self.learning_rate)
        self.params = None

    def initialize_parameters(self, seed: int = 42):
        tf.random.set_seed(seed)
        beta_r = tf.random.normal((self.batch_size, self.N_layers + 1), stddev=0.1, dtype=DT_FLOAT)
        beta_i = tf.random.normal((self.batch_size, self.N_layers + 1), stddev=0.1, dtype=DT_FLOAT)
        betas = tf.complex(beta_r, beta_i)
        phis = tf.random.uniform((self.batch_size, self.N_layers + 1), minval=0.0, maxval=2*np.pi, dtype=DT_FLOAT)
        thetas = tf.random.uniform((self.batch_size, self.N_layers + 1), minval=0.0, maxval=np.pi, dtype=DT_FLOAT)
        self.params = {'betas': tf.Variable(betas), 'phis': tf.Variable(phis), 'thetas': tf.Variable(thetas)}

    def _batch_fidelity(self, U_target: tf.Tensor) -> tf.Tensor:
        def one(i):
            U = build_ecd_sequence_tf(self.params['betas'][i], self.params['phis'][i], self.params['thetas'][i], self.N_trunc)
            if self.P_sub is None:
                return fidelity_full_tf(U_target, U)
            else:
                return fidelity_subspace_tf(U_target, U, self.P_sub, self.N_target)
        idx = tf.range(self.batch_size)
        return tf.map_fn(one, idx, dtype=DT_FLOAT)

    def _loss(self, U_target: tf.Tensor) -> tf.Tensor:
        F = self._batch_fidelity(U_target)
        eps = tf.constant(1e-10, dtype=DT_FLOAT)
        return tf.reduce_sum(tf.math.log(tf.maximum(1.0 - F, eps)))

    def optimize(self, U_target: tf.Tensor, max_iter: int = 300, verbose: bool = True):
        history = {'loss': [], 'max_fidelity': []}
        best_fid = 0.0
        best = None
        for i in range(int(max_iter)):
            with tf.GradientTape() as tape:
                loss = self._loss(U_target)
            grads = tape.gradient(loss, [self.params['betas'], self.params['phis'], self.params['thetas']])
            self.opt.apply_gradients(zip(grads, [self.params['betas'], self.params['phis'], self.params['thetas']]))
            F = self._batch_fidelity(U_target)
            maxF = float(tf.reduce_max(F).numpy())
            history['loss'].append(float(loss.numpy()))
            history['max_fidelity'].append(maxF)
            if maxF > best_fid:
                best_fid = maxF
                j = int(tf.argmax(F).numpy())
                best = {
                    'betas': tf.identity(self.params['betas'][j]),
                    'phis': tf.identity(self.params['phis'][j]),
                    'thetas': tf.identity(self.params['thetas'][j]),
                }
            if verbose and (i % 50 == 0):
                print(f"Iter {i:5d} | F_max={maxF:.6f} | loss={float(loss.numpy()):.6f}")
            if maxF >= self.target_fidelity:
                break
        info = {'history': history, 'iterations': i + 1}
        return best, best_fid, info


#
# ============================================================================
# TensorFlow Tail probability utilities (Step 3)
# ============================================================================
def cavity_tail_probability_tf(cavity_state: tf.Tensor, N_target: int) -> tf.Tensor:
    p = tf.abs(cavity_state) ** 2
    return tf.reduce_sum(p[N_target:])

def split_qubit_blocks_tf(psi_full: tf.Tensor, N_trunc: int):
    return psi_full[:N_trunc], psi_full[N_trunc:]

def coherent_cavity_state_tf(alpha: complex, N_trunc: int) -> tf.Tensor:
    n = tf.range(N_trunc, dtype=DT_FLOAT)
    alpha_c = tf.cast(alpha, DT_COMPLEX)
    abs_alpha = tf.abs(alpha_c)
    abs_alpha_f = tf.cast(abs_alpha, DT_FLOAT)
    log_mag = -0.5 * (abs_alpha_f ** 2) + n * tf.math.log(tf.maximum(abs_alpha_f, 1e-30)) - 0.5 * tf.math.lgamma(n + 1.0)
    # Fix the type casting issue for the phase calculation
    n_complex = tf.cast(n, DT_COMPLEX)
    angle_alpha = tf.cast(tf.math.angle(alpha_c), DT_FLOAT)
    phase = tf.exp(1j * n_complex * tf.cast(angle_alpha, DT_COMPLEX))
    coeffs = tf.cast(tf.exp(log_mag), DT_COMPLEX) * phase
    norm = tf.sqrt(tf.reduce_sum(tf.abs(coeffs) ** 2))
    norm_complex = tf.cast(norm + 1e-20, DT_COMPLEX)
    return coeffs / norm_complex

def _ket_tf(index: int, dim: int) -> tf.Tensor:
    v = tf.zeros((dim,), dtype=DT_COMPLEX)
    return tf.tensor_scatter_nd_update(v, [[index]], [tf.constant(1.0, dtype=DT_COMPLEX)])

def _alpha_envelope_tf(betas: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.abs(betas)) - tf.abs(betas[-1]) + tf.abs(betas[-1] / 2.0)

def _tails_for_input_tf(U: tf.Tensor, psi_in: tf.Tensor, N_trunc: int, N_target: int):
    out = U @ psi_in
    cav_q0, cav_q1 = split_qubit_blocks_tf(out, N_trunc)
    t_q0 = float(cavity_tail_probability_tf(cav_q0, N_target).numpy())
    t_q1 = float(cavity_tail_probability_tf(cav_q1, N_target).numpy())
    return t_q0, t_q1

def measure_tail_probs_tf(U_best: tf.Tensor, N_target: int, N_trunc: int, betas: tf.Tensor) -> Dict[str, float]:
    dim = 2 * N_trunc
    t0_q0, t0_q1 = _tails_for_input_tf(U_best, _ket_tf(0, dim), N_trunc, N_target)
    t1_q0, t1_q1 = _tails_for_input_tf(U_best, _ket_tf(N_trunc, dim), N_trunc, N_target)
    alpha_env = _alpha_envelope_tf(betas)
    cav_alpha = coherent_cavity_state_tf(alpha_env, N_trunc)
    psi_alpha = tf.concat([cav_alpha, tf.zeros_like(cav_alpha)], axis=0)
    ta_q0, ta_q1 = _tails_for_input_tf(U_best, psi_alpha, N_trunc, N_target)
    max_tail = max(t0_q0, t0_q1, t1_q0, t1_q1, ta_q0, ta_q1)
    return {
        't0_q0': t0_q0, 't0_q1': t0_q1,
        't1_q0': t1_q0, 't1_q1': t1_q1,
        'ta_q0': ta_q0, 'ta_q1': ta_q1,
        'alpha_abs': float(tf.abs(alpha_env).numpy()),
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
             metrics: Optional[Dict[str, float]],
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
                        help='Max ECD layers; searches 0..L and picks minimal depth')
    parser.add_argument('--truncation', type=int, default=6,
                        help='Fock space simulation cutoff (use guard band, default: 6)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for multi-start (default: 16)')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum iterations (default: 1000)')
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
    if len(phases) == 0:
        raise ValueError("No phases provided. Please specify at least one phase value.")
    print(f"Target phases: {phases}")

    # Validate arguments
    if args.layers < 0:
        raise ValueError("Number of layers must be non-negative.")
    
    # Set N_target (scored levels) and print
    N_target = args.target_levels if args.target_levels is not None else len(phases)
    print(f"N_target (scored levels): {N_target}  |  N_trunc (simulated): {args.truncation}")
    if args.truncation < N_target:
        print(f"Note: increasing N_trunc automatically (auto-guard) since truncation={args.truncation} < N_target={N_target}")

    # Report TensorFlow device availability
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    print(f"TensorFlow devices -> GPU: {len(gpus)} | CPU: {len(cpus)}")

    # Depth search with guard logic
    L_max = int(args.layers)
    final_params = None
    final_fidelity = 0.0
    final_info = None
    final_trunc = None
    final_layers = None
    final_metrics = None
    
    # Initialize variables that might be referenced later
    best_params = None
    best_fidelity = 0.0
    info = None
    current_trunc = max(args.truncation, N_target + args.guard_init)
    optimizer_tf = None  # Initialize optimizer reference

    for depth in range(L_max + 1):
        print(f"\n====== Depth search: trying N_layers = {depth} (max {L_max}) ======")
        current_trunc = max(args.truncation, N_target + args.guard_init)

        best_params = None
        best_fidelity = 0.0
        info = None
        metrics = None

        while True:
            print(f"\n=== Optimizing with N_trunc = {current_trunc} (N_target = {N_target}, N_layers = {depth}) ===")
            U_target_tf = make_snap_tf(tf.constant(phases, dtype=DT_FLOAT), current_trunc)
            optimizer_tf = MinimalECDOptimizer(N_layers=depth, N_trunc=current_trunc, N_target=N_target,
                                                batch_size=args.batch_size, learning_rate=args.learning_rate)
            optimizer_tf.initialize_parameters(seed=42)
            print("Optimizing...")
            params_tf, fid_tf, info_tf = optimizer_tf.optimize(U_target_tf, max_iter=args.max_iter, verbose=True)
            print(f"Result: fidelity={fid_tf:.6f} in {info_tf['iterations']} iterations")

            best_params = params_tf
            best_fidelity = fid_tf
            info = info_tf

            # Always auto-guard: tail check using TF path (cache the computation)
            U_best_tf = build_ecd_sequence_tf(best_params['betas'], best_params['phis'], best_params['thetas'], current_trunc)
            metrics = measure_tail_probs_tf(U_best_tf, N_target, current_trunc, best_params['betas'])
            
            # Cache the computed unitary for later use
            cached_U_best = U_best_tf
            max_tail = metrics['max_tail']
            print(f"Tail check @ N_trunc={current_trunc}: max_tail={max_tail:.3e} (threshold={args.tail_threshold:.1e})")
            if (max_tail <= args.tail_threshold) and (best_fidelity >= optimizer_tf.target_fidelity):
                print("✓ Guard band sufficient (tails below threshold) and target fidelity met.")
                break

            next_trunc = current_trunc + args.guard_step
            max_trunc = N_target + args.guard_max
            if next_trunc > max_trunc:
                print(f"Warning: Reached guard ceiling (N_trunc={next_trunc} > {max_trunc}). Stopping depth {depth}.")
                break
            current_trunc = next_trunc

        if _success(best_fidelity, metrics, optimizer_tf.target_fidelity, True, args.tail_threshold):
            final_params = best_params
            final_fidelity = best_fidelity
            final_info = info
            final_trunc = current_trunc
            final_layers = depth
            final_metrics = metrics
            print(f"\n✓ Selected minimal depth: N_layers = {final_layers} (fidelity {final_fidelity:.6f})")
            break
        else:
            print(f"Depth {depth} did not meet criteria; trying deeper sequence...")

    if final_params is None:
        final_params = best_params
        final_fidelity = best_fidelity
        final_info = info
        final_trunc = current_trunc
        final_layers = depth
        final_metrics = metrics
        print(f"\n! No depth achieved target; keeping best from depth={final_layers} (fidelity {final_fidelity:.6f}).")

    # Final results
    best_params = final_params
    best_fidelity = final_fidelity
    info = final_info
    current_trunc = final_trunc

    # Use a default target fidelity if optimizer wasn't created
    target_fid_display = optimizer_tf.target_fidelity if optimizer_tf is not None else 0.999
    print(f"\n>>> Final choice: N_layers={final_layers}, N_trunc={current_trunc}, fidelity={best_fidelity:.6f} | target≥{target_fid_display:.6f}, tail≤{args.tail_threshold:.1e}")

    if args.tail_check and best_params is not None:
        # Use cached metrics if available, otherwise compute
        if final_metrics is not None:
            print_tail_report(final_metrics)
        else:
            U_best_tf = build_ecd_sequence_tf(best_params['betas'], best_params['phis'], best_params['thetas'], current_trunc)
            computed_metrics = measure_tail_probs_tf(U_best_tf, N_target, current_trunc, best_params['betas'])
            print_tail_report(computed_metrics)

    # Save results
    save_results({
        'betas': best_params['betas'].numpy(),
        'phis': best_params['phis'].numpy(),
        'thetas': best_params['thetas'].numpy(),
    }, best_fidelity, info, phases, args.output_dir, N_target=N_target)

    if args.plot:
        plot_path = os.path.join(args.output_dir, 'convergence.png')
        plot_convergence(info['history'], plot_path)


if __name__ == '__main__':
    main()