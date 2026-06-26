# ornithology — build system.
#
# Mirrors ds4's hardware-target layout. The default build is the portable
# CPU/CLI path (no GPU, no GGML) so `inspect`/`config`/`arch` and the tests
# work everywhere. GPU backends are opt-in targets.
#
#   make            # native CPU build -> ./ornith (+ tools)
#   make metal      # macOS Metal backend (primary inference target)
#   make cuda       # generic Linux CUDA
#   make cuda-spark # CUDA tuned for DGX Spark / GB10
#   make test       # build and run the test suite
#   make clean

CC      ?= cc
CSTD    ?= -std=c11
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter $(CSTD)
CFLAGS  += -pthread
# Target the host CPU (NEON/AVX) — speeds up the scalar dequant + matvec loops.
ARCH := $(shell uname -m)
ifneq (,$(filter $(ARCH),arm64 aarch64))
  CFLAGS += -mcpu=native
else
  CFLAGS += -march=native
endif
LDFLAGS ?=
LIBS    ?= -lm -pthread

SRC_DIR := src
BUILD   := build

# Core, backend-agnostic sources (compile everywhere).
CORE := \
  $(SRC_DIR)/ornith_util.c \
  $(SRC_DIR)/ornith_json.c \
  $(SRC_DIR)/ornith_config.c \
  $(SRC_DIR)/ornith_gguf.c \
  $(SRC_DIR)/ornith_gguf_write.c \
  $(SRC_DIR)/ornith_quant.c \
  $(SRC_DIR)/ornith_imatrix.c \
  $(SRC_DIR)/ornith_qdot.c \
  $(SRC_DIR)/ornith_model.c \
  $(SRC_DIR)/ornith_tensor.c \
  $(SRC_DIR)/ornith_attn.c \
  $(SRC_DIR)/ornith_moe.c \
  $(SRC_DIR)/ornith_tokenizer.c \
  $(SRC_DIR)/ornith_sample.c \
  $(SRC_DIR)/ornith_rforward.c \
  $(SRC_DIR)/ornith_forward.c \
  $(SRC_DIR)/ornith_server.c \
  $(SRC_DIR)/ornith_agent.c \
  $(SRC_DIR)/ornith_bench.c \
  $(SRC_DIR)/ornith_eval.c

CLI  := $(SRC_DIR)/ornith.c

CORE_OBJ := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(CORE))

.PHONY: all metal cuda cuda-spark cpu test clean

all: ornith

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: $(SRC_DIR)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

# Default CPU/CLI binary. The CPU backend is reference/diagnostics only.
ornith: $(CORE_OBJ) $(CLI)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -DORNITH_BACKEND_CPU $(CORE_OBJ) $(CLI) \
	  $(LDFLAGS) $(LIBS) -o $@

cpu: ornith

# macOS Metal — primary inference backend (compiles only on macOS).
#
# The Metal backend lives in src/ornith_metal.m (Objective-C + embedded MSL,
# all gated behind ORNITH_BACKEND_METAL). We compile the .m to its own object
# first (so the `-x objective-c` dialect flag does not bleed onto the C sources
# or precompiled objects in the link step), then link with the GPU frameworks.
# The core objects are backend-agnostic and reused as-is; the CLI is rebuilt
# with -DORNITH_BACKEND_METAL so any metal-guarded paths there activate.
METAL_OBJ := $(BUILD)/ornith_metal.o

$(METAL_OBJ): $(SRC_DIR)/ornith_metal.m $(SRC_DIR)/ornith_metal.h | $(BUILD)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -DORNITH_BACKEND_METAL \
	  -x objective-c -fno-objc-arc -c $(SRC_DIR)/ornith_metal.m -o $@

metal: $(CORE_OBJ) $(METAL_OBJ) $(CLI)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -DORNITH_BACKEND_METAL \
	  $(METAL_OBJ) $(CORE_OBJ) $(CLI) $(LDFLAGS) \
	  -framework Metal -framework Foundation -framework Accelerate \
	  $(LIBS) -o ornith

# Generic Linux CUDA.
cuda: $(CORE_OBJ) $(CLI)
	nvcc -O2 -I$(SRC_DIR) -DORNITH_BACKEND_CUDA \
	  $(SRC_DIR)/ornith_cuda.cu $(CORE_OBJ) $(CLI) -lm -o ornith

# DGX Spark / GB10 (sm_121a). Same sources, tuned arch flags.
cuda-spark: $(CORE_OBJ) $(CLI)
	nvcc -O2 -arch=sm_121a -I$(SRC_DIR) -DORNITH_BACKEND_CUDA \
	  $(SRC_DIR)/ornith_cuda.cu $(CORE_OBJ) $(CLI) -lm -o ornith

# Test suite (CPU only). Each test file is its own binary; all must pass.
# GOLDEN_TOP_TOKEN pins the tiny-model regression output (see test_forward.c).
GOLDEN_TOP_TOKEN ?= 9
TESTS := test_gguf test_quant test_imatrix test_qdot test_tensor test_attn \
         test_moe test_forward test_tokenizer test_server test_sample test_agent \
         test_session test_eval

test: $(CORE_OBJ)
	@set -e; for t in $(TESTS); do \
	  echo "=== building $$t ==="; \
	  $(CC) $(CFLAGS) -DGOLDEN_TOP_TOKEN=$(GOLDEN_TOP_TOKEN) -I$(SRC_DIR) \
	    tests/$$t.c $(CORE_OBJ) $(LIBS) -o $(BUILD)/$$t; \
	  echo "=== running $$t ==="; \
	  ./$(BUILD)/$$t; \
	done
	@echo "all test binaries passed"

clean:
	rm -rf $(BUILD) ornith
