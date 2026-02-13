#!/bin/bash
#
# Run comprehensive attention benchmarks with different configurations
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$PROJECT_DIR/bin"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"

echo "attention_type,batch_size,num_heads,seq_len,head_dim,time_ms,tflops,bandwidth_gbps" > "$RESULTS_FILE"

# Build first
echo "Building attention kernels..."
make -C "$PROJECT_DIR" all

# Configurations to test
BATCH_SIZES=(1 2 4 8)
NUM_HEADS=(8 12 16)
SEQ_LENS=(128 256 512 1024 2048 4096)
HEAD_DIMS=(64 128)

echo ""
echo "Starting Comprehensive Benchmark Suite"
echo "======================================"
echo ""

run_benchmark() {
    local impl=$1
    local binary=$2
    local batch=$3
    local heads=$4
    local seqlen=$5
    local headdim=$6
    
    echo "Testing: $impl B=$batch H=$heads S=$seqlen D=$headdim"
    
    output=$("$binary" --batch "$batch" --heads "$heads" --seq-len "$seqlen" --head-dim "$headdim" 2>&1)
    
    # Parse output
    time_ms=$(echo "$output" | grep -oP 'Latency:\s+\K[\d.]+' || echo "N/A")
    tflops=$(echo "$output" | grep -oP 'Throughput:\s+\K[\d.]+' || echo "N/A")
    bandwidth=$(echo "$output" | grep -oP 'Bandwidth:\s+\K[\d.]+' || echo "N/A")
    
    echo "$impl,$batch,$heads,$seqlen,$headdim,$time_ms,$tflops,$bandwidth" >> "$RESULTS_FILE"
}

# Quick test with default params
echo "=== Quick Validation Test ==="
for impl in naive shared flash; do
    "$BIN_DIR/attention_$impl" --seq-len 256 --heads 8 --head-dim 64
    echo ""
done

# Full benchmark matrix
echo ""
echo "=== Full Benchmark Matrix ==="
echo ""

for batch in "${BATCH_SIZES[@]}"; do
    for heads in "${NUM_HEADS[@]}"; do
        for headdim in "${HEAD_DIMS[@]}"; do
            for seqlen in "${SEQ_LENS[@]}"; do
                # Skip extremely large configs to avoid OOM
                mem_estimate=$((batch * heads * seqlen * seqlen * 4 / 1024 / 1024))
                if [ "$mem_estimate" -gt 8192 ]; then
                    echo "Skipping B=$batch H=$heads S=$seqlen (est. $mem_estimate MB)"
                    continue
                fi
                
                run_benchmark "naive" "$BIN_DIR/attention_naive" "$batch" "$heads" "$seqlen" "$headdim"
                run_benchmark "shared" "$BIN_DIR/attention_shared" "$batch" "$heads" "$seqlen" "$headdim"
                run_benchmark "flash" "$BIN_DIR/attention_flash" "$batch" "$heads" "$seqlen" "$headdim"
            done
        done
    done
done

echo ""
echo "Benchmark complete!"
echo "Results saved to: $RESULTS_FILE"

# Generate summary
echo ""
echo "=== Summary ==="
echo ""
cat "$RESULTS_FILE" | head -1
cat "$RESULTS_FILE" | tail -n +2 | sort -t',' -k3 -n | head -20
