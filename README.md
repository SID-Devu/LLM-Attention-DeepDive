# LLM Attention Deep Dive (ROCm + HIP)

> Attention kernel experimentation & GPU memory behavior analysis.

## Overview

This project implements and analyzes scaled dot-product attention kernels on AMD GPUs using HIP. It demonstrates GPU memory hierarchy behavior, optimization techniques, and provides detailed performance analysis.

## What This Project Demonstrates

- **GPU Architecture Understanding**: Memory coalescing, bank conflicts, occupancy
- **Kernel Optimization**: From naive to optimized implementations
- **Profiling Skills**: Using rocprof to identify bottlenecks
- **Performance Engineering**: Quantified improvements with evidence

## Implementations

### 1. Naive Attention (`attention_naive.hip`)
- Direct implementation of scaled dot-product attention
- Global memory only
- Baseline for comparison

### 2. Shared Memory Optimized (`attention_shared.hip`)
- Uses LDS (Local Data Share) for Q, K, V tiles
- Reduced global memory traffic
- Demonstrates tiling strategy

### 3. Flash Attention Style (`attention_flash.hip`)
- Memory-efficient attention
- Online softmax computation
- Minimal memory footprint

## Key Metrics Measured

| Metric | Description |
|--------|-------------|
| Global Memory Load/Store | Data transfer to/from HBM |
| LDS Usage | Shared memory utilization |
| Occupancy | Active wavefronts vs theoretical max |
| Execution Time | Kernel duration in microseconds |

## Building

```bash
# Requires ROCm installation
export ROCM_PATH=/opt/rocm

# Build all kernels
make all

# Build specific kernel
make attention_naive
```

## Running Benchmarks

```bash
# Run all benchmarks
./run_benchmarks.sh

# Run specific benchmark
./build/attention_naive --seq-len 512 --head-dim 64 --num-heads 8

# Profile with rocprof
rocprof --stats ./build/attention_naive
```

## Expected Results

### Performance Comparison (Seq=512, HeadDim=64)

| Implementation | Time (μs) | Speedup | Memory BW (GB/s) |
|---------------|-----------|---------|------------------|
| Naive | 1250 | 1.0x | 45 |
| Shared Memory | 420 | 3.0x | 95 |
| Flash Style | 380 | 3.3x | 85 |

### Why Attention is Memory-Bound

1. **O(n²) attention matrix**: For sequence length n, requires n² memory
2. **Low arithmetic intensity**: Few FLOPs per byte loaded
3. **Large KV cache**: Dominates memory in inference

### Optimization Opportunities

1. **Tiling**: Process attention in blocks that fit in LDS
2. **Fused Softmax**: Avoid materializing full attention matrix
3. **Online Normalization**: Compute softmax in streaming fashion

## Project Structure

```
LLM-Attention-DeepDive/
├── src/
│   ├── attention_naive.hip      # Baseline implementation
│   ├── attention_shared.hip     # Shared memory optimized
│   ├── attention_flash.hip      # Flash attention style
│   ├── attention_common.h       # Shared utilities
│   └── benchmark.cpp            # Benchmark harness
├── docs/
│   ├── memory_hierarchy.md      # GPU memory explanation
│   ├── optimization_notes.md    # Optimization decisions
│   └── cuda_comparison.md       # CUDA baseline discussion
├── profiling/
│   ├── rocprof_commands.sh      # Profiling scripts
│   └── analysis.py              # Result analysis
├── results/
│   └── (profiling outputs)
├── Makefile
└── README.md
```

## GPU Memory Hierarchy (AMD RDNA/CDNA)

```
┌─────────────────────────────────────────────────┐
│                    Registers                     │
│              (Fastest, per-thread)               │
├─────────────────────────────────────────────────┤
│              LDS (Local Data Share)              │
│         64KB per Compute Unit, ~10 cycles        │
├─────────────────────────────────────────────────┤
│                 L1 Vector Cache                  │
│              16KB per CU, ~30 cycles             │
├─────────────────────────────────────────────────┤
│                    L2 Cache                      │
│               Shared, ~100 cycles                │
├─────────────────────────────────────────────────┤
│                  HBM / GDDR6                     │
│            Global Memory, ~300 cycles            │
└─────────────────────────────────────────────────┘
```

## Why This Matters

Understanding attention kernel optimization is critical for:
- LLM inference optimization
- KV cache management
- Memory-efficient inference
- Hardware selection decisions

This project provides evidence that you understand:
- GPU architecture fundamentals
- Memory hierarchy optimization
- Profiling and analysis methodology
- Performance engineering principles

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD CDNA Architecture](https://www.amd.com/en/technologies/cdna)
