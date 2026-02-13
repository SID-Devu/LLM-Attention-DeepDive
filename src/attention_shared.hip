/**
 * Shared Memory Optimized Attention Implementation
 * 
 * Optimizations:
 * 1. Load Q, K tiles into LDS (Local Data Share / Shared Memory)
 * 2. Compute scores with data reuse from LDS
 * 3. Block-level parallel reduction for softmax
 * 4. Coalesced global memory access patterns
 * 
 * Memory Hierarchy Exploitation:
 * - L2 Cache: Sequential prefetching for K, V
 * - LDS: Q tiles kept resident, K tiles streamed
 * - Registers: Partial sums accumulated
 */

#include "attention_common.h"
#include <cfloat>

#define TILE_SIZE 32
#define TILE_DIM 32

/**
 * Tiled attention score computation with shared memory
 * 
 * Each thread block handles one query position across all keys
 * Q tile is loaded once and reused for all K tiles
 */
__global__ void attention_scores_tiled(
    const float* __restrict__ Q,      // [B, H, S, D]
    const float* __restrict__ K,      // [B, H, S, D]
    float* __restrict__ scores,        // [B, H, S, S]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory for Q and K tiles
    __shared__ float Q_tile[TILE_SIZE][TILE_DIM + 1];  // +1 to avoid bank conflicts
    __shared__ float K_tile[TILE_SIZE][TILE_DIM + 1];
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_block = blockIdx.x;
    
    int q_start = q_block * TILE_SIZE;
    int q_idx = q_start + threadIdx.y;
    
    if (b >= batch_size || h >= num_heads) return;
    
    int qk_base = ((b * num_heads + h) * seq_len) * head_dim;
    int score_base = ((b * num_heads + h) * seq_len);
    
    // For each key tile
    for (int k_block = 0; k_block < (seq_len + TILE_SIZE - 1) / TILE_SIZE; k_block++) {
        int k_start = k_block * TILE_SIZE;
        
        // Partial scores for this tile (each thread handles one Q position)
        float partial_scores[TILE_SIZE];
        for (int i = 0; i < TILE_SIZE; i++) {
            partial_scores[i] = 0.0f;
        }
        
        // Process dimension in tiles
        for (int d_block = 0; d_block < (head_dim + TILE_DIM - 1) / TILE_DIM; d_block++) {
            int d_start = d_block * TILE_DIM;
            
            // Load Q tile: each thread loads one element
            int d_idx = d_start + threadIdx.x;
            if (q_idx < seq_len && d_idx < head_dim) {
                Q_tile[threadIdx.y][threadIdx.x] = Q[qk_base + q_idx * head_dim + d_idx];
            } else {
                Q_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Load K tile: transposed for efficient dot product
            int k_idx = k_start + threadIdx.y;
            if (k_idx < seq_len && d_idx < head_dim) {
                K_tile[threadIdx.y][threadIdx.x] = K[qk_base + k_idx * head_dim + d_idx];
            } else {
                K_tile[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial dot products
            if (q_idx < seq_len) {
                for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                    int k_global = k_start + k_local;
                    if (k_global < seq_len) {
                        for (int d = 0; d < TILE_DIM && (d_start + d) < head_dim; d++) {
                            partial_scores[k_local] += Q_tile[threadIdx.y][d] * K_tile[k_local][d];
                        }
                    }
                }
            }
            
            __syncthreads();
        }
        
        // Write scores for this key tile
        if (q_idx < seq_len && threadIdx.x == 0) {
            for (int k_local = 0; k_local < TILE_SIZE; k_local++) {
                int k_global = k_start + k_local;
                if (k_global < seq_len) {
                    scores[(score_base + q_idx) * seq_len + k_global] = partial_scores[k_local] * scale;
                }
            }
        }
    }
}

/**
 * Parallel softmax with shared memory reduction
 * Each thread block handles one row (query position)
 */
__global__ void softmax_shared(
    float* __restrict__ scores,  // [B, H, S, S]
    int batch_size,
    int num_heads,
    int seq_len
) {
    __shared__ float max_vals[WARP_SIZE];
    __shared__ float sum_vals[WARP_SIZE];
    __shared__ float row_max;
    __shared__ float row_sum;
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;  // Query position
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    int offset = ((b * num_heads + h) * seq_len + i) * seq_len;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    // Step 1: Find max (parallel reduction)
    float local_max = -FLT_MAX;
    for (int j = tid; j < seq_len; j += num_threads) {
        local_max = fmaxf(local_max, scores[offset + j]);
    }
    
    // Warp reduction for max
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        local_max = fmaxf(local_max, __shfl_down(local_max, stride));
    }
    
    // Write warp results to shared memory
    if (tid % WARP_SIZE == 0) {
        max_vals[tid / WARP_SIZE] = local_max;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < WARP_SIZE) {
        local_max = (tid < (num_threads + WARP_SIZE - 1) / WARP_SIZE) ? max_vals[tid] : -FLT_MAX;
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            local_max = fmaxf(local_max, __shfl_down(local_max, stride));
        }
        if (tid == 0) row_max = local_max;
    }
    __syncthreads();
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += num_threads) {
        float val = expf(scores[offset + j] - row_max);
        scores[offset + j] = val;
        local_sum += val;
    }
    
    // Warp reduction for sum
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        local_sum += __shfl_down(local_sum, stride);
    }
    
    if (tid % WARP_SIZE == 0) {
        sum_vals[tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();
    
    if (tid < WARP_SIZE) {
        local_sum = (tid < (num_threads + WARP_SIZE - 1) / WARP_SIZE) ? sum_vals[tid] : 0.0f;
        for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            local_sum += __shfl_down(local_sum, stride);
        }
        if (tid == 0) row_sum = local_sum;
    }
    __syncthreads();
    
    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (int j = tid; j < seq_len; j += num_threads) {
        scores[offset + j] *= inv_sum;
    }
}

/**
 * Attention output computation with shared memory for attention weights
 */
__global__ void apply_attention_shared(
    const float* __restrict__ attention_weights,  // [B, H, S, S]
    const float* __restrict__ V,                  // [B, H, S, D]
    float* __restrict__ output,                   // [B, H, S, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    __shared__ float A_tile[TILE_SIZE];  // Attention weights tile
    __shared__ float V_tile[TILE_SIZE][TILE_DIM + 1];  // Value tile
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int q_idx = blockIdx.x;  // Query position
    
    if (b >= batch_size || h >= num_heads || q_idx >= seq_len) return;
    
    int attn_offset = ((b * num_heads + h) * seq_len + q_idx) * seq_len;
    int v_offset = ((b * num_heads + h) * seq_len) * head_dim;
    
    int tid = threadIdx.x;
    int d_idx = tid;  // Each thread handles one dimension
    
    float output_val = 0.0f;
    
    // Process value positions in tiles
    for (int v_block = 0; v_block < (seq_len + TILE_SIZE - 1) / TILE_SIZE; v_block++) {
        int v_start = v_block * TILE_SIZE;
        
        // Load attention weights tile
        if (tid < TILE_SIZE) {
            int v_pos = v_start + tid;
            A_tile[tid] = (v_pos < seq_len) ? attention_weights[attn_offset + v_pos] : 0.0f;
        }
        
        // Load V tile
        for (int v_local = 0; v_local < TILE_SIZE && (v_start + v_local) < seq_len; v_local++) {
            if (tid < head_dim) {
                V_tile[v_local][tid] = V[v_offset + (v_start + v_local) * head_dim + tid];
            }
        }
        
        __syncthreads();
        
        // Accumulate weighted values
        if (d_idx < head_dim) {
            for (int v_local = 0; v_local < TILE_SIZE; v_local++) {
                int v_pos = v_start + v_local;
                if (v_pos < seq_len) {
                    output_val += A_tile[v_local] * V_tile[v_local][d_idx];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    if (d_idx < head_dim) {
        output[v_offset + q_idx * head_dim + d_idx] = output_val;
    }
}

/**
 * Host function to run shared memory optimized attention
 */
void attention_shared(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    float* scores,
    const AttentionConfig& cfg,
    hipStream_t stream = 0
) {
    // Scores computation
    dim3 score_block(TILE_DIM, TILE_SIZE);
    dim3 score_grid((cfg.seq_len + TILE_SIZE - 1) / TILE_SIZE, cfg.num_heads, cfg.batch_size);
    
    attention_scores_tiled<<<score_grid, score_block, 0, stream>>>(
        Q, K, scores,
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.scale
    );
    
    // Softmax
    dim3 softmax_block(256);
    dim3 softmax_grid(cfg.seq_len, cfg.num_heads, cfg.batch_size);
    
    softmax_shared<<<softmax_grid, softmax_block, 0, stream>>>(
        scores,
        cfg.batch_size, cfg.num_heads, cfg.seq_len
    );
    
    // Apply attention
    dim3 apply_block(max(TILE_DIM, cfg.head_dim));
    dim3 apply_grid(cfg.seq_len, cfg.num_heads, cfg.batch_size);
    
    apply_attention_shared<<<apply_grid, apply_block, 0, stream>>>(
        scores, V, output,
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim
    );
}

int main(int argc, char** argv) {
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 512;
    int head_dim = 64;
    int warmup_iters = 10;
    int bench_iters = 100;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) {
            head_dim = atoi(argv[++i]);
        }
    }
    
    printf("Shared Memory Attention Benchmark\n");
    printf("Config: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    
    AttentionConfig cfg(batch_size, num_heads, seq_len, head_dim);
    
    size_t qkv_elements = batch_size * num_heads * seq_len * head_dim;
    
    float* h_Q = new float[qkv_elements];
    float* h_K = new float[qkv_elements];
    float* h_V = new float[qkv_elements];
    float* h_output = new float[qkv_elements];
    
    init_random(h_Q, qkv_elements, 42);
    init_random(h_K, qkv_elements, 43);
    init_random(h_V, qkv_elements, 44);
    
    float *d_Q, *d_K, *d_V, *d_output, *d_scores;
    HIP_CHECK(hipMalloc(&d_Q, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_K, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_V, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_output, cfg.output_size()));
    HIP_CHECK(hipMalloc(&d_scores, cfg.attention_matrix_size()));
    
    HIP_CHECK(hipMemcpy(d_Q, h_Q, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, cfg.qkv_size(), hipMemcpyHostToDevice));
    
    for (int i = 0; i < warmup_iters; i++) {
        attention_shared(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    GPUTimer timer;
    timer.record_start();
    
    for (int i = 0; i < bench_iters; i++) {
        attention_shared(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    
    timer.record_stop();
    float total_ms = timer.elapsed_ms();
    float avg_ms = total_ms / bench_iters;
    
    print_stats("Shared Memory Attention", avg_ms, cfg);
    
    HIP_CHECK(hipMemcpy(h_output, d_output, cfg.output_size(), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_scores));
    
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;
    
    return 0;
}
