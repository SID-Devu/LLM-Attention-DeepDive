# Contributing to LLM Attention DeepDive

Thank you for your interest in contributing!

## Development Setup

### Prerequisites
- ROCm 5.0+ / CUDA 11.0+
- HIP compiler
- Python 3.8+

```bash
git clone https://github.com/sudheerdevu/LLM-Attention-DeepDive.git
cd LLM-Attention-DeepDive

pip install -r requirements.txt
```

## Building Kernels

```bash
cd src
make all
```

## Adding Attention Variants

1. Create new `.hip` file in `src/`
2. Implement attention kernel
3. Add to benchmark comparison
4. Document optimization techniques

## Code Style

- Use descriptive variable names
- Comment memory access patterns
- Include complexity analysis

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Benchmark against existing implementations
4. Submit PR with performance data

## License

Contributions are licensed under MIT License.
