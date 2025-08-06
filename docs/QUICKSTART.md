# Quick Start Guide

## What Works Right Now (No Installation Needed)

You can run these immediately with just Python:

```bash
# Test basic logic
python test_minimal.py

# See algorithm demonstration
python demo_concept.py
```

## To Run Full Optimization

### Option 1: Minimal Install (Fastest)
```bash
# Install only essential packages
pip install numpy scipy matplotlib click tqdm

# Run tests that don't need JAX
python test_minimal.py
python demo_concept.py
```

### Option 2: Full Install (For Real Optimization)
```bash
# Install all dependencies (may take 5-10 minutes)
pip install -r requirements.txt

# Run actual optimization
python cli.py optimize --target-type identity --layers 4
```

### Option 3: Google Colab (Recommended)
JAX comes pre-installed in Colab. Just upload the files and run:
```python
!python cli.py optimize --target-type linear --target-param 0.5
```

## Understanding --target-param

The `--target-param` value controls the SNAP gate characteristics:

| Target Type | Parameter Meaning | Example | Result |
|------------|-------------------|---------|--------|
| linear | Phase slope | 0.5 | θ_n = 0.5 × n |
| quadratic | Quadratic coefficient | 0.1 | θ_n = 0.1 × n² |
| cubic | Cubic coefficient | 0.05 | θ_n = 0.05 × n³ |
| kerr | Kerr strength χ | 0.2 | θ_n = -0.2 × n(n-1)/2 |
| random | Random seed | 42 | Reproducible random phases |

Larger values = more complex gates = harder to optimize

## Files You Can Run Now

| File | Purpose | Dependencies |
|------|---------|--------------|
| `test_minimal.py` | Basic logic tests | None |
| `demo_concept.py` | Algorithm demonstration | numpy only |
| `test_gradients.py` | Gradient flow tests | JAX required |
| `test_simple.py` | Full optimization test | JAX + QuTiP required |
| `cli.py` | Command-line interface | All packages required |

## Common Commands

```bash
# Simple test (easiest SNAP gate)
python cli.py optimize --target-type identity --layers 4

# Medium difficulty
python cli.py optimize --target-type linear --target-param 0.5

# Harder optimization
python cli.py optimize --target-type quadratic --target-param 0.2

# Custom phases
python cli.py generate-target --truncation 8 --output my_phases.json
python cli.py optimize --target-type custom --target-file my_phases.json
```

## Troubleshooting

**"No module named 'jax'"**: 
- Run `demo_concept.py` instead (no JAX needed)
- Or use Google Colab where JAX is pre-installed

**"No module named 'qutip'"**:
- Install with: `pip install qutip`
- Or run the minimal tests that don't need it

**Installation taking too long**:
- JAXlib is ~60MB, be patient
- Or use Google Colab for instant access