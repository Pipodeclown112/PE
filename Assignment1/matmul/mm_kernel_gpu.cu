#include <cuda_runtime.h>
#include "mm_kernel.h"

__global__ void mat_mul(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  C[index] = A[index]*B[index];

  // for(i=0; i<m; i++) {
  //   for(l=0; l<p; l++){
  //     C[i*p+l]=0;
  //   }
  //   for(k=0; k<n; k++) {
  //     for(j=0; j<p; j++) {
  //       C[i*p+j] += A[i*n+k]*B[k*p+j];
  //     }
  //   }
  // }
}

void matrix_mult(int m, int n, int p, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int i, j, k, l;
  int size_a = (m*n),
      size_b = (n*p),
      size_c = (m*p);

  // Alloc and copy GPU memory
  cudaMalloc((void **)&GPU_A, sizeof(float) * size_a);
  cudaMalloc((void **)&GPU_B, sizeof(float) * size_b);
  cudaMalloc((void **)&GPU_C, sizeof(float) * size_c);

  cudaMemcpy(GPU_A, A, sizeof(float) * size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_B, B, sizeof(float) * size_b, cudaMemcpyHostToDevice);

  mat_mul<<<size_c, 512>>>(GPU_A, GPU_B, GPU_C);

  cudaThreadSynchronize();

  // Copy the data back to the host
  cudaMemcpy(C, GPU_C, sizeof(float) * size_c, cudaMemcpyDeviceToHost);

  cudaFree(GPU_A);
  cudaFree(GPU_B);
  cudaFree(GPU_C);
} 
