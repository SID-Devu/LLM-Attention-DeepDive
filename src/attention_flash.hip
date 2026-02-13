/**
 * Flash Attention Style Implementation
 * 
 * Key Insight: Never materialize the full NxN attention matrix in HBM
 * 
 * Algorithm:
 * 1. Process Q in blocks (outer loop)
 * 2. For each Q block, stream through K,V blocks (inner loop)
 * 3. Compute and apply attention incrementally using online softmax
 * 4. Only store the final output, not intermediate attention matrices
 * 
 * Memory Complexity: O(N) instead of O(N²) for standard attention
 * 
 * This implementation focuses on demonstrating the concept rather than
 * achieving peak performance (which requires careful tuning).
 */

#include "attention_common.h"
#include <cfloat>

#define BLOCK_M 64   // Q block size
#define BLOCK_N 64   // K,V block size
#define BLOCK_D 64   // Head dimension (assumed <= 64)

/**
 * Online Softmax Statistics
 * For incrementally computing softmax without storing all scores
 * 
 * Maintains: max value (m) and normalizer (l)
 * Update rule when processing new block:
 *   m_new = max(m_old, m_block)
 *   l_new = l_old * exp(m_old - m_new) + l_block * exp(m_block - m_new)
 */

/**
 * Flash Attention Kernel (Simplified)
 * 
 * Each thread block handles a tile of output positions
 * Processes K,V in tiles, accumulating output incrementally
 */
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,       // [B, H, S, D]
    const float* __restrict__ K,       // [B, H, S, D]
    const float* __restrict__ V,       // [B, H, S, D]
    float* __restrict__ O,             // [B, H, S, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory
    extern __shared__ float smem[];
    
    float* Q_smem = smem;                                    // [BLOCK_M][head_dim]
    float* K_smem = Q_smem + BLOCK_M * head_dim;             // [BLOCK_N][head_dim]
    float* V_smem = K_smem + BLOCK_N * head_dim;             // [BLOCK_N][head_dim]
    float* S_smem = V_smem + BLOCK_N * head_dim;             // [BLOCK_M][BLOCK_N]
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block_idx = blockIdx.x;  // Which Q block
    
    if (b >= batch_size || h >= num_heads) return;
    
    int m_start = m_block_idx * BLOCK_M;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Base pointers
    int qkv_offset = ((b * num_heads + h) * seq_len) * head_dim;
    
    // Each thread handles one row in the M dimension
    int m_idx = m_start + (tid % BLOCK_M);
    bool valid_m = m_idx < seq_len;
    
    // Running statistics for online softmax (per query position)
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    
    // Accumulator for output (in registers)
    float acc[BLOCK_D];
    for (int d = 0; d < head_dim; d++) {
        acc[d] = 0.0f;
    }
    
    // Load Q block to shared memory
    for (int i = tid; i < BLOCK_M * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int global_row = m_start + row;
        
        if (global_row < seq_len) {
            Q_smem[i] = Q[qkv_offset + global_row * head_dim + col] * scale;
        } else {
            Q_smem[i] = 0.0f;
        }
    }
    __syncthreads();
    
    // Process K,V blocks
    int num_n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int n_block_idx = 0; n_block_idx < num_n_blocks; n_block_idx++) {
        int n_start = n_block_idx * BLOCK_N;
        
        // Load K block
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = n_start + row;
            
            if (global_row < seq_len) {
                K_smem[i] = K[qkv_offset + global_row * head_dim + col];
            } else {
                K_smem[i] = 0.0f;
            }
        }
        
        // Load V block
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = n_start + row;
            
            if (global_row < seq_len) {
                V_smem[i] = V[qkv_offset + global_row * head_dim + col];
            } else {
                V_smem[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute attention scores for this block: S = Q @ K^T
        // Each thread computes one row of scores
        float scores[BLOCK_N];
        float block_max = -FLT_MAX;
        
        if (valid_m) {
            int m_local = m_idx - m_start;
            
            for (int n = 0; n < BLOCK_N; n++) {
                int n_global = n_start + n;
                if (n_global < seq_len) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q_smem[m_local * head_dim + d] * K_smem[n * head_dim + d];
                    }
                    scores[n] = dot;
                    block_max = fmaxf(block_max, dot);
                } else {
                    scores[n] = -FLT_MAX;
                }
            }
        }
        
        // Online softmax update
        float new_max = fmaxf(row_max, block_max);
        float old_scale = expf(row_max - new_max);
        
        // Rescale previous accumulator
        for (int d = 0; d < head_dim; d++) {
            acc[d] *= old_scale;
        }
        row_sum *= old_scale;
        
        // Compute exp(scores - new_max) and accumulate
        float block_sum = 0.0f;
        
        if (valid_m) {
            for (int n = 0; n < BLOCK_N; n++) {
                int n_global = n_start + n;
                if (n_global < seq_len) {
                    float p = expf(scores[n] - new_max);
                    block_sum += p;
                    
                    // Accumulate weighted values
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += p * V_smem[n * head_dim + d];
                    }
                }
            }
        }
        
        row_max = new_max;
        row_sum += block_sum;
        
        __syncthreads();
    }
    
    // Write output: O = acc / row_sum
    if (valid_m) {
        float inv_sum = 1.0f / row_sum;
        for (int d = 0; d < head_dim; d++) {
            O[qkv_offset + m_idx * head_dim + d] = acc[d] * inv_sum;
        }
    }
}

/**
 * Flash Attention V2 Style - Optimized Version
 * Uses better memory access patterns and register tiling
 */
__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ L,  // Softmax normalizers (for backward pass)
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    __shared__ float K_smem[BLOCK_N * BLOCK_D];
    __shared__ float V_smem[BLOCK_N * BLOCK_D];
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_block = blockIdx.x;
    
    if (b >= batch_size || h >= num_heads) return;
    
    int m_start = m_block * BLOCK_M;
    int tid = threadIdx.x;
    
    // Each thread handles 4 query positions (register blocking)
    const int ROWS_PER_THREAD = 4;
    int m_idx[ROWS_PER_THREAD];
    bool valid[ROWS_PER_THREAD];
    float row_max[ROWS_PER_THREAD];
    float row_sum[ROWS_PER_THREAD];
    float acc[ROWS_PER_THREAD][BLOCK_D];
    
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        m_idx[r] = m_start + tid * ROWS_PER_THREAD + r;
        valid[r] = m_idx[r] < seq_len;
        row_max[r] = -FLT_MAX;
        row_sum[r] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            acc[r][d] = 0.0f;
        }
    }
    
    int qkv_base = ((b * num_heads + h) * seq_len) * head_dim;
    
    // Load Q into registers (persistent across K,V blocks)
    float Q_reg[ROWS_PER_THREAD][BLOCK_D];
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        if (valid[r]) {
            for (int d = 0; d < head_dim; d++) {
                Q_reg[r][d] = Q[qkv_base + m_idx[r] * head_dim + d] * scale;
            }
        }
    }
    
    int num_n_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int n_block = 0; n_block < num_n_blocks; n_block++) {
        int n_start = n_block * BLOCK_N;
        
        // Cooperative loading of K and V
        for (int i = tid; i < BLOCK_N * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int n_global = n_start + row;
            
            K_smem[i] = (n_global < seq_len) ? K[qkv_base + n_global * head_dim + col] : 0.0f;
            V_smem[i] = (n_global < seq_len) ? V[qkv_base + n_global * head_dim + col] : 0.0f;
        }
        __syncthreads();
        
        // Process each of our query positions
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            if (!valid[r]) continue;
            
            // Compute scores and find max
            float scores[BLOCK_N];
            float block_max = -FLT_MAX;
            
            for (int n = 0; n < BLOCK_N; n++) {
                int n_global = n_start + n;
                if (n_global < seq_len) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q_reg[r][d] * K_smem[n * head_dim + d];
                    }
                    scores[n] = dot;
                    block_max = fmaxf(block_max, dot);
                } else {
                    scores[n] = -FLT_MAX;
                }
            }
            
            // Online softmax update
            float new_max = fmaxf(row_max[r], block_max);
            float old_scale = expf(row_max[r] - new_max);
            
            // Rescale accumulator
            for (int d = 0; d < head_dim; d++) {
                acc[r][d] *= old_scale;
            }
            row_sum[r] *= old_scale;
            
            // Accumulate
            float block_sum = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                int n_global = n_start + n;
                if (n_global < seq_len) {
                    float p = expf(scores[n] - new_max);
                    block_sum += p;
                    for (int d = 0; d < head_dim; d++) {
                        acc[r][d] += p * V_smem[n * head_dim + d];
                    }
                }
            }
            
            row_max[r] = new_max;
            row_sum[r] += block_sum;
        }
        __syncthreads();
    }
    
    // Write output
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        if (valid[r]) {
            float inv_sum = 1.0f / row_sum[r];
            for (int d = 0; d < head_dim; d++) {
                O[qkv_base + m_idx[r] * head_dim + d] = acc[r][d] * inv_sum;
            }
            // Store log(row_sum) + row_max for backward pass
            if (L != nullptr) {
                L[((b * num_heads + h) * seq_len) + m_idx[r]] = row_max[r] + logf(row_sum[r]);
            }
        }
    }
}

/**
 * Host function for flash attention
 */
void flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionConfig& cfg,
    bool use_v2 = true,
    hipStream_t stream = 0
) {
    int num_m_blocks = (cfg.seq_len + BLOCK_M - 1) / BLOCK_M;
    
    if (use_v2) {
        dim3 grid(num_m_blocks, cfg.num_heads, cfg.batch_size);
        dim3 block(BLOCK_M / 4);  // Each thread handles 4 rows
        
        flash_attention_v2_kernel<<<grid, block, 0, stream>>>(
            Q, K, V, O, nullptr,
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.scale
        );
    } else {
        dim3 grid(num_m_blocks, cfg.num_heads, cfg.batch_size);
        dim3 block(BLOCK_M);
        
        size_t smem_size = (BLOCK_M + 2 * BLOCK_N) * cfg.head_dim * sizeof(float) +
                          BLOCK_M * BLOCK_N * sizeof(float);
        
        flash_attention_kernel<<<grid, block, smem_size, stream>>>(
            Q, K, V, O,
            cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.scale
        );
    }
}

int main(int argc, char** argv) {
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 2048;  // Flash attention shines at longer sequences
    int head_dim = 64;
    int warmup_iters = 10;
    int bench_iters = 100;
    bool use_v2 = true;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) {
            head_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--v1") == 0) {
            use_v2 = false;
        }
    }
    
    printf("Flash Attention %s Benchmark\n", use_v2 ? "V2" : "V1");
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
    
    float *d_Q, *d_K, *d_V, *d_O;
    HIP_CHECK(hipMalloc(&d_Q, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_K, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_V, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_O, cfg.output_size()));
    
    HIP_CHECK(hipMemcpy(d_Q, h_Q, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, cfg.qkv_size(), hipMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        flash_attention(d_Q, d_K, d_V, d_O, cfg, use_v2);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    GPUTimer timer;
    timer.record_start();
    
    for (int i = 0; i < bench_iters; i++) {
        flash_attention(d_Q, d_K, d_V, d_O, cfg, use_v2);
    }
    
    timer.record_stop();
    float total_ms = timer.elapsed_ms();
    float avg_ms = total_ms / bench_iters;
    
    print_stats(use_v2 ? "Flash Attention V2" : "Flash Attention V1", avg_ms, cfg);
    
    // Memory efficiency comparison
    printf("\nMemory Usage Comparison:\n");
    printf("  Standard Attention (stores NxN): %.2f MB\n", 
           (float)cfg.attention_matrix_size() / (1024 * 1024));
    printf("  Flash Attention (no NxN):        %.2f MB (Q+K+V+O only)\n",
           (float)(4 * cfg.qkv_size()) / (1024 * 1024));
    printf("  Memory Savings:                  %.1fx\n",
           (float)(cfg.attention_matrix_size() + 3 * cfg.qkv_size()) / (4 * cfg.qkv_size()));
    
    HIP_CHECK(hipMemcpy(h_output, d_O, cfg.output_size(), hipMemcpyDeviceToHost));
    
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_O));
    
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;
    
    return 0;
}
