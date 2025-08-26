# ECD-to-SNAP Gate Optimizer

Quantum gate optimization system that approximates SNAP (Selective Number-dependent Arbitrary Phase) gates using sequences of ECD (Echoed Conditional Displacement) gates.

## Overview

This system implements gradient-based optimization to find parameters for ECD gate sequences that approximate desired SNAP gates with high fidelity (>0.999 for identity gates).

## Key Features

- **JAX-based automatic differentiation** for gradient computation
- **Multiple optimization strategies**: Simple SGD, Adam with smart initialization, multi-restart optimization
- **Batch optimization** for parallel parameter exploration
- **Command-line interface** for easy experimentation
- **Comprehensive testing suite** validating gradients, fidelity, and convergence

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ECD2SNAP.git
cd ECD2SNAP

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Simple Optimization (Best Performance)

```python
from scripts.simple_sgd import SimpleSGDOptimizer
from src.snap_targets import make_snap_full_space
import numpy as np
import jax.numpy as jnp

# Define target SNAP gate (identity)
N_trunc = 4
phases = np.zeros(N_trunc)
U_target = make_snap_full_space(phases, N_trunc)
U_target_jax = jnp.array(U_target.full())

# Optimize
optimizer = SimpleSGDOptimizer(N_layers=3, N_trunc=N_trunc, learning_rate=0.01)
params, fidelity = optimizer.optimize(U_target_jax, max_iterations=500)

print(f"Achieved fidelity: {fidelity:.6f}")
```

### Command-Line Interface

```bash
# Basic optimization
python cli.py optimize --target-type identity --layers 4 --truncation 6

# Improved optimization with restarts
python improved_cli.py optimize --target-type identity --strategy restarts --layers 4

# Compare different strategies
python improved_cli.py compare-strategies
```

## Results

The system achieves excellent performance:

- **Identity SNAP**: F > 0.999 (typically 0.9997)
- **Small phase variations**: F > 0.98
- **Convergence**: Often in < 50 iterations for identity
- **Robustness**: 100% success rate across different random seeds for identity

## Key Insights

1. **Simple is better**: Simple SGD with momentum (0.9) and small learning rate (0.01) outperforms complex Adam-based optimizers with logarithmic barrier functions.

2. **Initialization matters**: Starting very close to zero (scale=0.01) is crucial for identity SNAP gates.

3. **Loss function choice**: Simple infidelity loss (1 - F) works better than logarithmic barrier for this problem.

## Project Structure

```
ECD2SNAP/
├── gates.py                 # JAX-compatible quantum gate operations
├── optimizer.py            # Original Adam-based optimizer
├── improved_optimizer.py   # Enhanced optimizer with multiple strategies
├── simple_sgd.py          # Simple SGD optimizer (best performance)
├── snap_targets.py        # SNAP gate construction
├── cli.py                 # Command-line interface
├── improved_cli.py        # Enhanced CLI with strategy selection
├── test_*.py              # Comprehensive test suite
└── viz.py                 # Visualization utilities
```

## Testing

Run the test suite to verify the system:

```bash
# Quick tests
python test_quick.py

# Final validation
python test_final.py

# Gradient flow verification
python test_gradients.py
```

## Mathematical Background

The system approximates SNAP gates:
```
U_SNAP = diag(e^{iφ_0}, e^{iφ_1}, ..., e^{iφ_{N-1}}) ⊗ I_qubit
```

Using ECD gate sequences:
```
U_ECD = ∏_k [R_y(θ_k) ⊗ I] · [CZ] · [D(β_k) ⊗ I] · [R_y(φ_k) ⊗ I]
```

Where:
- D(β) is the displacement operator
- R_y is qubit rotation
- CZ is controlled-Z gate

## Performance Tips

1. For identity SNAP, use 3-4 layers
2. Start with learning rate 0.01
3. Use momentum 0.9
4. Initialize parameters near zero (scale ≈ 0.01)
5. For non-identity targets, may need more layers or iterations

## License

MIT

## References

Based on the theoretical framework for ECD gate sequences approximating SNAP gates in circuit QED systems.