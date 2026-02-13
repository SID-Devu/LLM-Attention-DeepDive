# LLM Attention Deep Dive - Makefile
# Build attention kernel implementations for ROCm/HIP

# ROCm installation path
ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)

# Compiler
HIPCC = $(HIP_PATH)/bin/hipcc

# Target GPU architecture
# Common architectures:
#   gfx906  - MI50/MI60
#   gfx908  - MI100
#   gfx90a  - MI210/MI250
#   gfx1030 - Navi 21 (RX 6800/6900)
#   gfx1100 - Navi 31 (RX 7900)
#   gfx1103 - Phoenix APU (Ryzen 7040)
GPU_ARCH ?= gfx1103

# Compiler flags
HIPCC_FLAGS = -O3 -std=c++17 --offload-arch=$(GPU_ARCH)
HIPCC_FLAGS += -Wall -Wextra
HIPCC_FLAGS += -ffast-math
HIPCC_FLAGS += -I$(ROCM_PATH)/include

# Debug flags
DEBUG_FLAGS = -g -O0 -DDEBUG

# Profiling flags (for rocprof)
PROFILE_FLAGS = -g -O3 --save-temps

# Linker flags
LDFLAGS = -L$(ROCM_PATH)/lib -lamdhip64

# Source and output directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Targets
NAIVE_TARGET = $(BIN_DIR)/attention_naive
SHARED_TARGET = $(BIN_DIR)/attention_shared
FLASH_TARGET = $(BIN_DIR)/attention_flash
COMPARE_TARGET = $(BIN_DIR)/benchmark_compare

# Source files
NAIVE_SRC = $(SRC_DIR)/attention_naive.hip
SHARED_SRC = $(SRC_DIR)/attention_shared.hip
FLASH_SRC = $(SRC_DIR)/attention_flash.hip
COMPARE_SRC = $(SRC_DIR)/benchmark_compare.hip

# Default target
all: dirs $(NAIVE_TARGET) $(SHARED_TARGET) $(FLASH_TARGET)

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Build naive attention
$(NAIVE_TARGET): $(NAIVE_SRC) $(SRC_DIR)/attention_common.h
	@echo "Building naive attention kernel..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $< $(LDFLAGS)

# Build shared memory attention
$(SHARED_TARGET): $(SHARED_SRC) $(SRC_DIR)/attention_common.h
	@echo "Building shared memory attention kernel..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $< $(LDFLAGS)

# Build flash attention
$(FLASH_TARGET): $(FLASH_SRC) $(SRC_DIR)/attention_common.h
	@echo "Building flash attention kernel..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $< $(LDFLAGS)

# Build comparison benchmark
$(COMPARE_TARGET): $(COMPARE_SRC) $(SRC_DIR)/attention_common.h
	@echo "Building comparison benchmark..."
	$(HIPCC) $(HIPCC_FLAGS) -o $@ $< $(LDFLAGS)

# Debug builds
debug-naive: $(NAIVE_SRC)
	$(HIPCC) $(DEBUG_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_naive_debug $< $(LDFLAGS)

debug-shared: $(SHARED_SRC)
	$(HIPCC) $(DEBUG_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_shared_debug $< $(LDFLAGS)

debug-flash: $(FLASH_SRC)
	$(HIPCC) $(DEBUG_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_flash_debug $< $(LDFLAGS)

# Profile builds (for rocprof)
profile-naive: $(NAIVE_SRC)
	$(HIPCC) $(PROFILE_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_naive_profile $< $(LDFLAGS)

profile-shared: $(SHARED_SRC)
	$(HIPCC) $(PROFILE_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_shared_profile $< $(LDFLAGS)

profile-flash: $(FLASH_SRC)
	$(HIPCC) $(PROFILE_FLAGS) --offload-arch=$(GPU_ARCH) -o $(BIN_DIR)/attention_flash_profile $< $(LDFLAGS)

# Run benchmarks
.PHONY: bench bench-naive bench-shared bench-flash

bench: all
	@echo "Running all attention benchmarks..."
	@echo ""
	@echo "=== Naive Attention ==="
	$(NAIVE_TARGET) --seq-len 512 --heads 8 --head-dim 64
	@echo ""
	@echo "=== Shared Memory Attention ==="
	$(SHARED_TARGET) --seq-len 512 --heads 8 --head-dim 64
	@echo ""
	@echo "=== Flash Attention ==="
	$(FLASH_TARGET) --seq-len 2048 --heads 8 --head-dim 64

bench-naive: $(NAIVE_TARGET)
	$(NAIVE_TARGET) --seq-len 512 --heads 8 --head-dim 64

bench-shared: $(SHARED_TARGET)
	$(SHARED_TARGET) --seq-len 512 --heads 8 --head-dim 64

bench-flash: $(FLASH_TARGET)
	$(FLASH_TARGET) --seq-len 2048 --heads 8 --head-dim 64

# Run with different sequence lengths
bench-scaling: all
	@echo "Scaling Analysis"
	@echo "================"
	@for seq in 128 256 512 1024 2048 4096; do \
		echo ""; \
		echo "Sequence Length: $$seq"; \
		$(NAIVE_TARGET) --seq-len $$seq 2>/dev/null | grep -E "Throughput|Latency:" || true; \
		$(SHARED_TARGET) --seq-len $$seq 2>/dev/null | grep -E "Throughput|Latency:" || true; \
		$(FLASH_TARGET) --seq-len $$seq 2>/dev/null | grep -E "Throughput|Latency:" || true; \
	done

# Profile with rocprof
.PHONY: profile
profile: profile-naive profile-shared profile-flash
	@echo "Running rocprof on naive attention..."
	rocprof --stats $(BIN_DIR)/attention_naive_profile --seq-len 512
	@echo ""
	@echo "Running rocprof on shared memory attention..."
	rocprof --stats $(BIN_DIR)/attention_shared_profile --seq-len 512
	@echo ""
	@echo "Running rocprof on flash attention..."
	rocprof --stats $(BIN_DIR)/attention_flash_profile --seq-len 2048

# Hardware counters profile
profile-counters: profile-naive
	@echo "Collecting hardware counters..."
	rocprof --input ../profiling/rocprof_counters.txt \
		--output results_counters.csv \
		$(BIN_DIR)/attention_naive_profile --seq-len 512

# Memory analysis
profile-memory: profile-flash
	rocprof --hsa-trace --obj-tracking \
		$(BIN_DIR)/attention_flash_profile --seq-len 2048

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f *.csv *.json *.txt *.db

# Help
help:
	@echo "LLM Attention Deep Dive - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all attention kernels"
	@echo "  bench          - Run all benchmarks"
	@echo "  bench-naive    - Run naive attention benchmark"
	@echo "  bench-shared   - Run shared memory attention benchmark"
	@echo "  bench-flash    - Run flash attention benchmark"
	@echo "  bench-scaling  - Run scaling analysis across sequence lengths"
	@echo "  profile        - Build and profile with rocprof --stats"
	@echo "  profile-counters - Profile with hardware counters"
	@echo "  clean          - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  GPU_ARCH       - Target GPU architecture (default: $(GPU_ARCH))"
	@echo "  ROCM_PATH      - ROCm installation path (default: $(ROCM_PATH))"
	@echo ""
	@echo "Examples:"
	@echo "  make GPU_ARCH=gfx90a all    # Build for MI200 series"
	@echo "  make bench-scaling          # Run scaling analysis"
	@echo "  make profile                # Profile all kernels"
