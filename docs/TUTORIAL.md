# LLM Attention Kernels Tutorial

## Prerequisites

- ROCm 5.0+ installed
- HIP-enabled AMD GPU
- Python 3.8+ with numpy

## Building Kernels

### Compile All Kernels

```bash
cd kernels
hipcc -O3 attention_naive.hip -o attention_naive
hipcc -O3 attention_flash.hip -o attention_flash
```

### Verify Installation

```bash
./attention_naive --test
# Expected: All tests passed
```

## Understanding Attention

### Mathematical Foundation

Standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- Q: Query matrix [B, H, S, D]
- K: Key matrix [B, H, S, D]
- V: Value matrix [B, H, S, D]
- d_k: Head dimension

### HIP Kernel Structure

```cpp
__global__ void attention_kernel(
    const float* Q,    // Query
    const float* K,    // Key
    const float* V,    // Value
    float* Output,     // Result
    int seq_len,       // Sequence length
    int head_dim       // Head dimension
) {
    // Thread indexing
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int query_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute attention scores
    // ... (see kernels/ for full implementation)
}
```

## Benchmarking

### Run Performance Tests

```bash
python scripts/benchmark.py \
    --seq-lengths 512 1024 2048 4096 \
    --batch-size 1 \
    --heads 32 \
    --head-dim 128
```

### Expected Output

```
Sequence Length | Naive (ms) | Flash (ms) | Speedup
----------------|------------|------------|--------
512             | 2.3        | 1.1        | 2.1x
1024            | 8.5        | 2.5        | 3.4x
2048            | 33.2       | 5.2        | 6.4x
4096            | 132.1      | 10.8       | 12.2x
```

## Tuning Guide

### Adjusting Tile Sizes

In `kernels/attention_flash.hip`:

```cpp
#define BLOCK_SIZE_Q 64   // Queries per block
#define BLOCK_SIZE_KV 64  // Keys/Values per block
```

Optimal values depend on your GPU:
- **gfx1030 (RDNA2)**: 64/64
- **gfx90a (MI200)**: 128/64
