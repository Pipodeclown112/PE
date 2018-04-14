# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall
SIMD_FLAGS = -ftree-vectorize -march=native -ffast-math
PAR_FLAG = -fopenmp
NVCC = nvcc
NVCC_FLAGS = -c -arch sm_52
NVCC_LFLAGS = -lcuda -lcudart
#GENERATE = -D GENERATE
GENERATE = 
TIMING = -D TIMING

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

APPS = cuda.o matmul matmul_cache matmul_cache_simd matmul_cpu matmul_gpu matmul_unroll matmul_simd transpose

.PHONY: all

all: $(APPS)

matmul: matmul.c mmio.c mm_kernel.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

matmul_cache: matmul.c mmio.c mm_kernel_cache.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

matmul_cache_simd: matmul.c mmio.c mm_kernel_cache_simd.c
	$(CXX) $(CXX_FLAGS) $(SIMD_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

matmul_cpu: matmul.c mmio.c mm_kernel_cpu.c
	$(CXX) $(CXX_FLAGS) $(SIMD_FLAGS) $(PAR_FLAG) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

cuda.o:
	$(NVCC) $(NVCC_FLAGS) mm_kernel_gpu.cu -o cuda.o

matmul_gpu: matmul.c mmio.c cuda.o
	$(CXX) $(CXX_FLAGS) $(NVCC_LFLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

matmul_unroll: matmul.c mmio.c mm_kernel_unroll.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

matmul_simd: matmul.c mmio.c mm_kernel_simd.c
	$(CXX) $(CXX_FLAGS) $(SIMD_FLAGS) $(GENERATE) $(TIMING) -o $@ $^ $(LFLAGS)

transpose: transpose.c mmio.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS) *.o
