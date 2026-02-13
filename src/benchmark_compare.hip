/**
 * Comprehensive Attention Benchmark Comparison
 * 
 * Compares Naive, Shared Memory, and Flash Attention implementations
 * with validation and detailed performance metrics
 */

#include "attention_common.h"
#include <cfloat>
#include <vector>
#include <iomanip>

// Include implementations
#include "attention_naive.hip"
#include "attention_shared.hip"
#include "attention_flash.hip"

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float throughput_tflops;
    float bandwidth_gbps;
    bool valid;
};

/**
 * CPU reference implementation for validation
 */
void attention_cpu_reference(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionConfig& cfg
) {
    int B = cfg.batch_size;
    int H = cfg.num_heads;
    int S = cfg.seq_len;
    int D = cfg.head_dim;
    float scale = cfg.scale;
    
    // Allocate attention weights
    std::vector<float> attn(S * S);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = ((b * H + h) * S) * D;
            
            // Compute QK^T
            for (int i = 0; i < S; i++) {
                float max_val = -FLT_MAX;
                for (int j = 0; j < S; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) {
                        dot += Q[offset + i * D + d] * K[offset + j * D + d];
                    }
                    attn[i * S + j] = dot * scale;
                    max_val = std::max(max_val, attn[i * S + j]);
                }
                
                // Softmax
                float sum = 0.0f;
                for (int j = 0; j < S; j++) {
                    attn[i * S + j] = expf(attn[i * S + j] - max_val);
                    sum += attn[i * S + j];
                }
                for (int j = 0; j < S; j++) {
                    attn[i * S + j] /= sum;
                }
            }
            
            // Apply to V
            for (int i = 0; i < S; i++) {
                for (int d = 0; d < D; d++) {
                    float val = 0.0f;
                    for (int j = 0; j < S; j++) {
                        val += attn[i * S + j] * V[offset + j * D + d];
                    }
                    O[offset + i * D + d] = val;
                }
            }
        }
    }
}

/**
 * Compare two outputs for validation
 */
bool validate_output(
    const float* ref,
    const float* test,
    size_t n,
    float rtol = 1e-3f,
    float atol = 1e-5f
) {
    int max_errors = 10;
    int error_count = 0;
    float max_diff = 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float threshold = atol + rtol * fabsf(ref[i]);
        max_diff = fmaxf(max_diff, diff);
        
        if (diff > threshold) {
            if (error_count < max_errors) {
                printf("  Mismatch at %zu: ref=%.6f, test=%.6f, diff=%.6f\n",
                       i, ref[i], test[i], diff);
            }
            error_count++;
        }
    }
    
    if (error_count > 0) {
        printf("  Total mismatches: %d / %zu (max diff: %.6f)\n", 
               error_count, n, max_diff);
        return false;
    }
    
    return true;
}

void print_results_table(const std::vector<BenchmarkResult>& results) {
    printf("\n%s\n", std::string(80, '=').c_str());
    printf("%-25s %12s %15s %15s %10s\n", 
           "Implementation", "Time (ms)", "TFLOPS", "BW (GB/s)", "Valid");
    printf("%s\n", std::string(80, '-').c_str());
    
    for (const auto& r : results) {
        printf("%-25s %12.3f %15.3f %15.2f %10s\n",
               r.name.c_str(),
               r.time_ms,
               r.throughput_tflops,
               r.bandwidth_gbps,
               r.valid ? "Yes" : "NO");
    }
    printf("%s\n\n", std::string(80, '=').c_str());
}

int main(int argc, char** argv) {
    int batch_size = 1;
    int num_heads = 8;
    int seq_len = 512;
    int head_dim = 64;
    int warmup_iters = 5;
    int bench_iters = 50;
    bool validate = true;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            num_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) {
            head_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-validate") == 0) {
            validate = false;
        }
    }
    
    printf("=====================================================\n");
    printf("    Attention Kernel Comparison Benchmark\n");
    printf("=====================================================\n");
    printf("Config: B=%d, H=%d, S=%d, D=%d\n", batch_size, num_heads, seq_len, head_dim);
    printf("Warmup: %d iterations, Benchmark: %d iterations\n", warmup_iters, bench_iters);
    printf("\n");
    
    AttentionConfig cfg(batch_size, num_heads, seq_len, head_dim);
    
    // Print memory requirements
    printf("Memory Requirements:\n");
    printf("  Q, K, V, O:     %.2f MB each\n", (float)cfg.qkv_size() / (1024 * 1024));
    printf("  Attention (NxN): %.2f MB\n", (float)cfg.attention_matrix_size() / (1024 * 1024));
    printf("\n");
    
    // Allocate host memory
    size_t qkv_elements = batch_size * num_heads * seq_len * head_dim;
    
    float* h_Q = new float[qkv_elements];
    float* h_K = new float[qkv_elements];
    float* h_V = new float[qkv_elements];
    float* h_ref = new float[qkv_elements];
    float* h_output = new float[qkv_elements];
    
    init_random(h_Q, qkv_elements, 42);
    init_random(h_K, qkv_elements, 43);
    init_random(h_V, qkv_elements, 44);
    
    // Compute CPU reference
    if (validate) {
        printf("Computing CPU reference...\n");
        attention_cpu_reference(h_Q, h_K, h_V, h_ref, cfg);
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output, *d_scores;
    HIP_CHECK(hipMalloc(&d_Q, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_K, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_V, cfg.qkv_size()));
    HIP_CHECK(hipMalloc(&d_output, cfg.output_size()));
    HIP_CHECK(hipMalloc(&d_scores, cfg.attention_matrix_size()));
    
    HIP_CHECK(hipMemcpy(d_Q, h_Q, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K, cfg.qkv_size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V, cfg.qkv_size(), hipMemcpyHostToDevice));
    
    std::vector<BenchmarkResult> results;
    GPUTimer timer;
    
    // Benchmark Naive Attention
    printf("Benchmarking Naive Attention...\n");
    for (int i = 0; i < warmup_iters; i++) {
        attention_naive(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    timer.record_start();
    for (int i = 0; i < bench_iters; i++) {
        attention_naive(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    timer.record_stop();
    
    float naive_ms = timer.elapsed_ms() / bench_iters;
    
    BenchmarkResult naive_result;
    naive_result.name = "Naive";
    naive_result.time_ms = naive_ms;
    naive_result.throughput_tflops = cfg.compute_tflops(naive_ms);
    naive_result.bandwidth_gbps = cfg.compute_bandwidth_gbps(naive_ms);
    
    if (validate) {
        HIP_CHECK(hipMemcpy(h_output, d_output, cfg.output_size(), hipMemcpyDeviceToHost));
        naive_result.valid = validate_output(h_ref, h_output, qkv_elements);
    } else {
        naive_result.valid = true;
    }
    results.push_back(naive_result);
    
    // Benchmark Shared Memory Attention
    printf("Benchmarking Shared Memory Attention...\n");
    for (int i = 0; i < warmup_iters; i++) {
        attention_shared(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    timer.record_start();
    for (int i = 0; i < bench_iters; i++) {
        attention_shared(d_Q, d_K, d_V, d_output, d_scores, cfg);
    }
    timer.record_stop();
    
    float shared_ms = timer.elapsed_ms() / bench_iters;
    
    BenchmarkResult shared_result;
    shared_result.name = "Shared Memory";
    shared_result.time_ms = shared_ms;
    shared_result.throughput_tflops = cfg.compute_tflops(shared_ms);
    shared_result.bandwidth_gbps = cfg.compute_bandwidth_gbps(shared_ms);
    
    if (validate) {
        HIP_CHECK(hipMemcpy(h_output, d_output, cfg.output_size(), hipMemcpyDeviceToHost));
        shared_result.valid = validate_output(h_ref, h_output, qkv_elements);
    } else {
        shared_result.valid = true;
    }
    results.push_back(shared_result);
    
    // Benchmark Flash Attention
    printf("Benchmarking Flash Attention...\n");
    for (int i = 0; i < warmup_iters; i++) {
        flash_attention(d_Q, d_K, d_V, d_output, cfg, true);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    timer.record_start();
    for (int i = 0; i < bench_iters; i++) {
        flash_attention(d_Q, d_K, d_V, d_output, cfg, true);
    }
    timer.record_stop();
    
    float flash_ms = timer.elapsed_ms() / bench_iters;
    
    BenchmarkResult flash_result;
    flash_result.name = "Flash Attention V2";
    flash_result.time_ms = flash_ms;
    flash_result.throughput_tflops = cfg.compute_tflops(flash_ms);
    flash_result.bandwidth_gbps = cfg.compute_bandwidth_gbps(flash_ms);
    
    if (validate) {
        HIP_CHECK(hipMemcpy(h_output, d_output, cfg.output_size(), hipMemcpyDeviceToHost));
        flash_result.valid = validate_output(h_ref, h_output, qkv_elements);
    } else {
        flash_result.valid = true;
    }
    results.push_back(flash_result);
    
    // Print results
    print_results_table(results);
    
    // Print speedups
    printf("Speedups (vs Naive):\n");
    printf("  Shared Memory: %.2fx\n", naive_ms / shared_ms);
    printf("  Flash Attention: %.2fx\n", naive_ms / flash_ms);
    
    // Cleanup
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_scores));
    
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_ref;
    delete[] h_output;
    
    return 0;
}
