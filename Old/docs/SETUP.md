# Setup Guide for ECD-to-SNAP Optimizer

## Prerequisites

- Python 3.8 or higher
- pip package manager
- ~2GB disk space for dependencies (mainly JAX/JAXlib)

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ECD2SNAP.git
cd ECD2SNAP
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Basic installation
pip install -r requirements.txt
```

**Note:** JAX installation can take 5-10 minutes due to the large size of JAXlib (~60MB).

#### Platform-Specific Notes:

**macOS (Apple Silicon):**
```bash
# If you have issues with JAX on M1/M2 Macs:
pip install --upgrade "jax[metal]"
```

**Linux with GPU:**
```bash
# For CUDA 11.x support:
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Windows:**
```bash
# JAX on Windows requires WSL2 or special builds
# Consider using Google Colab instead
```

### 4. Verify Installation

```bash
# Test basic functionality (no JAX required)
python test_minimal.py

# Test gradient flow (requires JAX)
python test_gradients.py

# Run demo (works without full JAX)
python demo_concept.py
```

## First Optimization

### Simple Test - Identity SNAP

```bash
# Optimize for identity SNAP (easiest case)
python cli.py optimize --target-type identity --layers 4 --truncation 6 --max-iter 500
```

This should converge quickly (F > 0.99 in ~200 iterations).

### Linear Phase SNAP

```bash
# More complex target
python cli.py optimize --target-type linear --target-param 0.5 --layers 6
```

### Custom SNAP Target

```bash
# Generate custom phases
python cli.py generate-target --truncation 8 --output my_phases.json

# Optimize for custom target
python cli.py optimize --target-type custom --target-file my_phases.json
```

## Troubleshooting

### Issue: JAX Installation Fails

**Solution 1:** Use pre-built wheels
```bash
# Check your Python version and platform
python --version
pip debug --verbose  # Shows compatible tags

# Install specific JAX version
pip install jax==0.4.20 jaxlib==0.4.20
```

**Solution 2:** Use Google Colab (no installation needed)
1. Upload the `.py` files to Colab
2. Run without installing (JAX is pre-installed)

### Issue: Out of Memory

**Solution:** Reduce batch size or truncation
```bash
python cli.py optimize --batch 16 --truncation 6
```

### Issue: Slow Convergence

**Solution:** Adjust learning rate and layers
```bash
python cli.py optimize --learning-rate 0.01 --layers 8
```

## Alternative: Docker Setup

If you prefer containerized environment:

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "cli.py", "optimize", "--help"]
```

Build and run:
```bash
docker build -t ecd2snap .
docker run -it ecd2snap python cli.py optimize --target-type identity
```

## Alternative: Google Colab

For quick experimentation without local setup:

```python
# In Colab notebook
!git clone https://github.com/yourusername/ECD2SNAP.git
%cd ECD2SNAP

# JAX is pre-installed in Colab
!pip install qutip optax click tqdm

# Run optimization
!python cli.py optimize --target-type linear --target-param 0.5
```

## Development Setup

For contributing to the project:

```bash
# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Check code style
flake8 .
```

## Quick Command Reference

```bash
# View help
python cli.py --help
python cli.py optimize --help

# Common optimizations
python cli.py optimize --target-type identity    # Simplest
python cli.py optimize --target-type linear      # Linear phase
python cli.py optimize --target-type quadratic   # Quadratic phase
python cli.py optimize --target-type kerr        # Kerr evolution

# Analysis
python cli.py analyze results/results.json

# Generate custom targets
python cli.py generate-target --truncation 10 --output phases.json
```

## Expected Performance

| Target Type | Typical Iterations | Final Fidelity |
|------------|-------------------|----------------|
| Identity   | 100-200          | > 0.999        |
| Linear     | 500-1000         | > 0.99         |
| Quadratic  | 1000-2000        | > 0.99         |
| Random     | 2000-5000        | > 0.98         |

## Next Steps

1. Run a simple optimization to verify setup
2. Explore different SNAP targets
3. Visualize results with generated plots
4. Adjust parameters for your specific use case

## Support

- Check `README.md` for algorithm details
- Run tests to verify installation
- Open an issue on GitHub for bugs/questions