# ECD-to-SNAP Architecture Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  CLI (Click)          │  Python API        │  Jupyter Notebook  │
└───────────┬───────────┴────────┬───────────┴────────┬──────────┘
            │                    │                     │
┌───────────▼───────────────────▼─────────────────────▼──────────┐
│                      Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Optimizer  │  │   Improved   │  │   Simple     │         │
│  │   (Base)     │◄─┤   Optimizer  │  │   SGD        │         │
│  └──────┬───────┘  └──────────────┘  └──────────────┘         │
│         │                                                        │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Gates      │  │   SNAP       │  │   Viz        │         │
│  │   Module     │  │   Targets    │  │   Tools      │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
└─────────┼──────────────────┼────────────────────────────────────┘
          │                  │
┌─────────▼──────────────────▼────────────────────────────────────┐
│                     Computational Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     JAX      │  │    QuTiP     │  │    NumPy     │         │
│  │   (AD+JIT)   │  │  (Quantum)   │  │  (Numerics)  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                  │
│  ┌──────▼──────────────────▼──────────────────▼───────┐         │
│  │            XLA Compiler / GPU Backend              │         │
│  └─────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
┌─────────────┐
│ Target SNAP │
│   U_target  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         Parameter Initialization             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ β (beta)│ │ φ (phi) │ │ θ(theta)│       │
│  │ complex │ │ [0,2π)  │ │  [0,π]  │       │
│  └────┬────┘ └────┬────┘ └────┬────┘       │
│       └───────────┼────────────┘            │
│                   ▼                          │
│            [Batch Size × N_layers]          │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│          Gate Sequence Construction          │
│                                              │
│   U_ECD = D(β_N/2) R(θ_N,φ_N) ×             │
│           ∏[ECD(β_k) R(θ_k,φ_k)]            │
│                                              │
│  ┌──────────────────────────────┐           │
│  │     For each batch element:   │           │
│  │  1. Build displacement ops    │           │
│  │  2. Build rotation ops        │           │
│  │  3. Construct ECD gates       │           │
│  │  4. Matrix multiplication     │           │
│  └──────────────────────────────┘           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│           Fidelity Computation               │
│                                              │
│   F = |Tr(U_target† @ U_ECD)|² / d²         │
│                                              │
│  ┌──────────────────────────────┐           │
│  │  Parallel computation for:    │           │
│  │  - All batch elements         │           │
│  │  - Vectorized using vmap      │           │
│  └──────────────────────────────┘           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│            Cost Function                     │
│                                              │
│   Cost = Σ log(1 - F_i + ε)                 │
│                                              │
│  ┌──────────────────────────────┐           │
│  │  Logarithmic barrier for:     │           │
│  │  - Numerical stability        │           │
│  │  - Smooth gradients near F=1  │           │
│  └──────────────────────────────┘           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         Gradient Computation (JAX)           │
│                                              │
│   ∇Cost = automatic_differentiation(Cost)   │
│                                              │
│  ┌──────────────────────────────┐           │
│  │  JAX reverse-mode autodiff:   │           │
│  │  - Through all operations     │           │
│  │  - JIT compiled               │           │
│  └──────────────────────────────┘           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│         Parameter Update (Adam)              │
│                                              │
│   params_new = Adam(params, ∇Cost, lr)      │
│                                              │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
              ┌─────────┐
              │ Iterate │ ──No──→ Output best
              │  F > T? │         parameters
              └────┬────┘
                   │Yes
                   ▼
            Continue Loop
```

## Gate Construction Detail

```
ECD Gate Structure:
┌────────────────────────────────────┐
│          ECD(β) Gate               │
├────────────────────────────────────┤
│                                    │
│  |0⟩⟨0| ⊗ D(β) + |1⟩⟨1| ⊗ D(-β)  │
│                                    │
│  ┌──────┐     ┌──────┐            │
│  │Qubit │     │ Cavity│            │
│  │Space │  ⊗  │ Space │            │
│  │ (2D) │     │ (N_D) │            │
│  └──────┘     └──────┘            │
│                                    │
│  Result: 2N × 2N matrix            │
└────────────────────────────────────┘

Full Sequence:
┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐
│D(β/2)│→│R(θ,φ)│→│ECD(β)│→...→│ECD(β)│→│R(θ,φ)│
└──────┘ └──────┘ └──────┘     └──────┘ └──────┘
   ↑                                         ↑
   └──────────── N_layers times ────────────┘
```

## Optimization Strategies

```
Multi-Start Batch Optimization:
┌─────────────────────────────────────────────┐
│              Batch Size = 32                │
├─────────────────────────────────────────────┤
│ ┌──┐ ┌──┐ ┌──┐ ┌──┐         ┌──┐ ┌──┐    │
│ │P1│ │P2│ │P3│ │P4│  ...    │P31│ │P32│   │
│ └┬─┘ └┬─┘ └┬─┘ └┬─┘         └┬──┘ └┬──┘   │
│  │    │    │    │             │     │      │
│  ▼    ▼    ▼    ▼             ▼     ▼      │
│ ┌─────────────────────────────────────┐    │
│ │     Parallel Optimization (vmap)     │    │
│ └─────────────────────────────────────┘    │
│                   │                          │
│                   ▼                          │
│         Select Best Fidelity                │
└─────────────────────────────────────────────┘

Advanced Strategies:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Restarts   │  │  Annealing   │  │  Two-Stage   │
├──────────────┤  ├──────────────┤  ├──────────────┤
│              │  │              │  │              │
│  Run 1 ──►F₁ │  │ High LR ──►  │  │ Coarse ──►   │
│  Run 2 ──►F₂ │  │   ↓          │  │   ↓          │
│  Run 3 ──►F₃ │  │ Med LR  ──►  │  │ Fine   ──►   │
│      ↓       │  │   ↓          │  │              │
│  max(F₁,F₂,F₃)│  │ Low LR  ──►  │  │  Result      │
│              │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Memory Layout

```
Parameter Storage (per batch element):
┌────────────────────────────────────────┐
│ β (complex)  │ φ (real)    │ θ (real)  │
├──────────────┼─────────────┼───────────┤
│ N_layers + 1 │ N_layers + 1│N_layers +1│
│ × 2 (Re,Im)  │ × 1         │ × 1       │
└──────────────┴─────────────┴───────────┘
Total: 4 × (N_layers + 1) × batch_size floats

Matrix Storage:
┌─────────────────────────────────────────┐
│         U_target (2N × 2N)              │
│         U_ECD    (2N × 2N × batch)      │
│         Gradients (same as params)      │
└─────────────────────────────────────────┘
Memory ≈ O(N² × batch_size)
```

## Performance Optimization Flow

```
┌──────────────────────────────────────┐
│         JAX Compilation Pipeline      │
├──────────────────────────────────────┤
│                                      │
│  Python Functions                    │
│         ↓                            │
│  JAX Tracing                         │
│         ↓                            │
│  XLA IR Generation                   │
│         ↓                            │
│  ┌────────────────────┐              │
│  │  Optimizations:    │              │
│  │  - Fusion          │              │
│  │  - Vectorization   │              │
│  │  - Memory reuse    │              │
│  └────────────────────┘              │
│         ↓                            │
│  GPU/TPU Kernel                      │
│                                      │
│  Result: 10-100x speedup             │
└──────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────────────┐
│              Test Suite                      │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐                           │
│  │  Unit Tests  │                           │
│  ├──────────────┤                           │
│  │ • Gates      │                           │
│  │ • Gradients  │                           │
│  │ • Fidelity   │                           │
│  └──────┬───────┘                           │
│         │                                    │
│  ┌──────▼───────────┐                       │
│  │Integration Tests │                       │
│  ├──────────────────┤                       │
│  │ • Optimization   │                       │
│  │ • Strategies     │                       │
│  │ • Initialization │                       │
│  └──────┬───────────┘                       │
│         │                                    │
│  ┌──────▼──────┐                            │
│  │ Benchmarks  │                            │
│  ├─────────────┤                            │
│  │ • Speed     │                            │
│  │ • Scaling   │                            │
│  │ • Comparison│                            │
│  └─────────────┘                            │
│                                             │
│  Coverage: >95%                             │
└─────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────┐
│           Production Deployment             │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐     ┌──────────────┐      │
│  │   Docker     │────►│  Kubernetes  │     │
│  │   Container  │     │   Cluster    │     │
│  └──────────────┘     └──────────────┘     │
│                              │               │
│  ┌──────────────────────────▼─────────┐    │
│  │         Load Balancer              │    │
│  └────┬──────────┬─────────┬──────────┘    │
│       │          │         │                │
│  ┌────▼───┐ ┌───▼───┐ ┌──▼────┐           │
│  │ GPU-1  │ │ GPU-2 │ │ GPU-N │           │
│  │ Worker │ │ Worker│ │ Worker│           │
│  └────────┘ └───────┘ └───────┘           │
│                                             │
│  ┌─────────────────────────────────┐       │
│  │     Monitoring & Logging        │       │
│  │  • Prometheus                   │       │
│  │  • Grafana                      │       │
│  │  • ELK Stack                    │       │
│  └─────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

---

These diagrams provide a visual understanding of:
1. System architecture and layers
2. Data flow through the optimization pipeline
3. Gate construction details
4. Optimization strategies
5. Memory layout and usage
6. Performance optimization through JAX
7. Testing architecture
8. Deployment considerations