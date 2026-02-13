/**
 * Naive Scaled Dot-Product Attention Implementation
 * 
 * This is the baseline implementation using only global memory.
 * It serves as a reference for correctness and performance comparison.
 * 
 * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 * 
 * Memory Access Pattern:
 * - All accesses go through global memory
 * - No data reuse optimization
 * - Each thread computes one output element
 */

#include "attention_common.h"
#include <cfloat>

/**
 * Step 1: Compute QK^T scores
 * 
 * scores[b][h][i][j] = sum_k(Q[b][h][i][k] * K[b][h][j][k]) / sqrt(d)
 */
__global__ void compute_attention_scores_naive(
    const float* __restrict__ Q,     // [B, H, S, D]
    const float* __restrict__ K,     // [B, H, S, D]
    float* __restrict__ scores,       // [B, H, S, S]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Each thread computes one score element
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Query position
    int j = threadIdx.y;  // Key position (iterated in outer loop)
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    // Base offsets for this batch/head
    int qk_offset = ((b * num_heads + h) * seq_len) * head_dim;
    int score_offset = ((b * num_heads + h) * seq_len + i) * seq_len;
    
    // Compute dot product for each key position
    for (int key_pos = 0; key_pos < seq_len; key_pos++) {
        float dot = 0.0f;
        
        // Dot product over head dimension
        for (int d = 0; d < head_dim; d++) {
            float q_val = Q[qk_offset + i * head_dim + d];
            float k_val = K[qk_offset + key_pos * head_dim + d];
            dot += q_val * k_val;
        }
        
        // Scale and store
        scores[score_offset + key_pos] = dot * scale;
    }
}

/**
 * Step 2: Softmax over the key dimension
 * 
 * attention_weights[b][h][i][j] = exp(scores[i][j]) / sum_k(exp(scores[i][k]))
 */
__global__ void softmax_naive(
    float* __restrict__ scores,  // [B, H, S, S] - in-place
    int batch_size,
    int num_heads,
    int seq_len
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Query position
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    int offset = ((b * num_heads + h) * seq_len + i) * seq_len;
    
    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) {
        max_val = fmaxf(max_val, scores[offset + j]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[offset + j] = expf(scores[offset + j] - max_val);
        sum += scores[offset + j];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int j = 0; j < seq_len; j++) {
        scores[offset + j] *= inv_sum;
    }
}

/**
 * Step 3: Apply attention weights to values
 * 
 * output[b][h][i][d] = sum_j(attention_weights[i][j] * V[b][h][j][d])
 */
__global__ void apply_attention_naive(
    const float* __restrict__ attention_weights,  // [B, H, S, S]
    const float* __restrict__ V,                  // [B, H, S, D]
    float* __restrict__ output,                   // [B, H, S, D]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Query position
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    int attn_offset = ((b * num_heads + h) * seq_len + i) * seq_len;
    int v_offset = ((b * num_heads + h) * seq_len) * head_dim;
    int out_offset = v_offset + i * head_dim;
    
    // For each output dimension
    for (int d = 0; d < head_dim; d++) {
        float weighted_sum = 0.0f;
        
        // Weighted sum over all values
        for (int j = 0; j < seq_len; j++) {
            float attn_weight = attention_weights[attn_offset + j];
            float v_val = V[v_offset + j * head_dim + d];
            weighted_sum += attn_weight * v_val;
        }
        
        output[out_offset + d] = weighted_sum;
    }
}

/**
 * Host function to run naive attention
 */
void attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    float* scores,  // Workspace
    const AttentionConfig& cfg,
    hipStream_t stream = 0
) {
    dim3 block(32);
    dim3 grid((cfg.seq_len + 31) / 32, cfg.num_heads, cfg.batch_size);
    
    // Step 1: Compute scores
    compute_attention_scores_naive<<<grid, block, 0, stream>>>(
        Q, K, scores,
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.scale
    );
    
    // Step 2: Softmax
    softmax_naive<<<grid, block, 0, stream>>>(
        scores,
        cfg.batch_size, cfg.num_heads, cfg.seq_len
    );
    
    // Step 3: Apply to values
    apply_attention_naive<<<grid, block, 0, stream>>>(
        scores, V, output,
        cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim
    );
}

// Main benchmark entry point
int main(int argc, char** argv) {
    // Default configuration
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 512;
    int head_dim = 64;
    int warmup_iters = 10;
    int bench_iters = 100;
    
    // Parse arguments
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
    
    printf("Naive Attention Benchmark\n");
    printf("Config: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    
    AttentionConfig cfg(batch_size, num_heads, seq_len, head_dim);
    
    // Allocate host memory
    size_t qkv_elements = batch_size * num_heads * seq_len * head_dim;
    size_t score_elements = batch_size * num_heads * seq_len * seq_len;
    
    float* h_Q = new float[qkv_elements];
    float* h_K = new float[qkv_elements];
    float* h_V = new float[qkv_elements];
    float* h_output = new float[qkv_elements];
    
    // Initialize
    init_random(h_Q, qkv_elements, 42);
    init_random(h_K, qkv_elements, 43);
    init_random(h_V, qkv_elements, 44);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output, *d_scores;
    HIP_CHECK(hipMalloc(&d_Q, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_K, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_V, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_output, cfg.output_size()));
    HIP_CHECK(hipMalloc(&d_scores, cfg.attention_matrix_size()));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_Q, h_Q, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, cfg.qkv_size(), hipMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        attention_naive(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    GPUTimer timer;
    timer.record_start();
    
    for (int i = 0; i < bench_iters; i++) {
        attention_naive(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    
    timer.record_stop();
    float total_ms = timer.elapsed_ms();
    float avg_ms = total_ms / bench_iters;
    
    print_stats("Naive Attention", avg_ms, cfg);
    
    // Copy back results
    HIP_CHECK(hipMemcpy(h_output, d_output, cfg.output_size(), hipMemcpyDeviceToHost));
    
    // Cleanup
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
