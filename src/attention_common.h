/**
 * Common utilities for attention kernels
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

// Configuration
#define WARP_SIZE 64  // AMD wavefront size
#define MAX_SEQ_LEN 4096
#define MAX_HEAD_DIM 128

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Timing utilities
struct GPUTimer {
    hipEvent_t start, stop;
    
    GPUTimer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    
    ~GPUTimer() {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
    
    void record_start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start, stream));
    }
    
    void record_stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop, stream));
    }
    
    float elapsed_ms() {
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// Attention configuration
struct AttentionConfig {
    int batch_size;
    int num_heads;
    int seq_len;
    int head_dim;
    float scale;  // 1/sqrt(head_dim)
    
    AttentionConfig(int b, int h, int s, int d) 
        : batch_size(b), num_heads(h), seq_len(s), head_dim(d) {
        scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    }
    
    size_t qkv_size() const {
        return batch_size * num_heads * seq_len * head_dim * sizeof(float);
    }
    
    size_t output_size() const {
        return qkv_size();
    }
    
    size_t attention_matrix_size() const {
        return batch_size * num_heads * seq_len * seq_len * sizeof(float);
    }
};

// Initialize random data
inline void init_random(float* data, size_t n, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < n; i++) {
        data[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
}

// Verify results
inline bool verify_results(const float* ref, const float* test, size_t n, float tol = 1e-3f) {
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > tol) {
            fprintf(stderr, "Mismatch at %zu: ref=%f, test=%f, diff=%f\n",
                    i, ref[i], test[i], diff);
            return false;
        }
    }
    return true;
}

// Print statistics
inline void print_stats(const char* name, float time_ms, const AttentionConfig& cfg) {
    float flops = 2.0f * cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.seq_len * cfg.head_dim;
    float bytes = cfg.qkv_size() * 3 + cfg.output_size();  // Read Q,K,V + write O
    
    float gflops = flops / (time_ms * 1e6f);
    float bandwidth_gb = bytes / (time_ms * 1e6f);
    float arithmetic_intensity = flops / bytes;
    
    printf("\n=== %s ===\n", name);
    printf("Time: %.3f ms\n", time_ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gb);
    printf("Arithmetic Intensity: %.2f FLOP/Byte\n", arithmetic_intensity);
    printf("Memory-bound: %s\n", arithmetic_intensity < 10 ? "YES" : "NO");
}
