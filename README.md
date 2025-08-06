# ECD-to-SNAP Gate Optimization

A JAX-based implementation for optimizing Echo Conditional Displacement (ECD) gate sequences to approximate Selective Number-dependent Arbitrary Phase (SNAP) gates in quantum computing.

## Overview

This package implements the optimization problem:

```
{β*, φ*, θ*} = argmax F = |Tr(U_SNAP† U_ECD)|² / d²
```

where `U_ECD` is a parameterized sequence of ECD gates and qubit rotations, and `U_SNAP` is the target SNAP gate.

## Features

- **JAX-based automatic differentiation** for gradient computation
- **Multi-start batch optimization** with logarithmic barrier cost function
- **QuTiP integration** for quantum gate operations
- **Comprehensive visualization tools** for convergence and gate analysis
- **Command-line interface** for easy experimentation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ECD2SNAP.git
cd ECD2SNAP

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Optimize for a linear SNAP gate
python cli.py optimize --target-type linear --target-param 0.5 --layers 6

# Use custom target phases
python cli.py optimize --target-type custom --target-file phases.json

# Analyze results
python cli.py analyze results/results.json
```

### Python API

```python
from optimizer import ECDSNAPOptimizer
from snap_targets import linear_snap, make_snap_full_space
import jax.numpy as jnp

# Create target SNAP gate
N_trunc = 10
phases = 0.5 * np.arange(N_trunc)  # Linear phase pattern
U_target = make_snap_full_space(phases, N_trunc)
U_target_jax = jnp.array(U_target.full())

# Initialize optimizer
optimizer = ECDSNAPOptimizer(
    N_layers=6,
    N_trunc=N_trunc,
    batch_size=32,
    learning_rate=3e-3,
    target_fidelity=0.999
)

# Run optimization
best_params, best_fidelity, info = optimizer.optimize(U_target_jax)
print(f"Achieved fidelity: {best_fidelity:.6f}")
```

## Project Structure

```
ECD2SNAP/
├── gates.py           # JAX-compatible quantum gate operations
├── optimizer.py       # Main optimization engine with automatic differentiation
├── snap_targets.py    # SNAP gate constructors and utilities
├── viz.py            # Visualization tools for analysis
├── cli.py            # Command-line interface
├── test_gradients.py # Gradient flow verification tests
└── test_simple.py    # End-to-end optimization tests
```

## Key Components

### Gate Operations (`gates.py`)

- **JAX-native implementations** of displacement, rotation, and ECD gates
- **Automatic differentiation support** through all gate operations
- **Matrix exponentiation** for accurate displacement operators

### Optimizer (`optimizer.py`)

- **Batch optimization**: Run multiple random initializations in parallel
- **Logarithmic barrier cost**: `C = Σ log(1 - F_j)` for stable convergence
- **Adam optimizer** with adaptive learning rate
- **JIT compilation** for performance

### Visualization (`viz.py`)

- Convergence plots
- Parameter evolution tracking
- Gate sequence diagrams
- Wigner function visualization
- Decomposition error analysis

## Testing

```bash
# Test gradient flow
python test_gradients.py

# Test full optimization pipeline
python test_simple.py
```

## Algorithm Details

The optimization uses a parameterized ECD sequence:

```
U_ECD = D(β_{N+1}/2) R_{φ_{N+1}}(θ_{N+1}) ∏_{k=N}^{1} [ECD(β_k) R_{φ_k}(θ_k)]
```

where:
- `ECD(β)`: Echo conditional displacement gate
- `R_φ(θ)`: Qubit rotation by angle θ around axis φ
- `D(β)`: Displacement operator

The optimizer finds parameters `{β, φ, θ}` that maximize the gate fidelity with the target SNAP unitary.

## Performance

- **Identity SNAP**: Reaches F > 0.999 in ~200 iterations
- **Linear SNAP**: Reaches F > 0.99 in ~1000 iterations
- **Complex SNAP**: Typically converges in 2000-5000 iterations

Performance depends on:
- Number of layers (N_layers)
- Fock space truncation (N_trunc)
- Batch size for multi-start
- Target gate complexity

## References

Based on the ECD control paper and techniques from quantum optimal control theory.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.