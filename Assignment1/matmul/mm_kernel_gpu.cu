#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "mm_kernel.h"

__global__ void mat_mul_kernel(int m, int n, int p, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int k;

  printf("HELLO");

  // Only let this thread compute if it is in C
  if(row < m && col < p) {
    C[row*p+col] = 0;
    for(k=0; k<n; k++){
      C[row*p+col] += A[row*n+k]*B[k*p+col];
    } 
  }
}

void matrix_mult(int m, int n, int p, float* A, float* B, float* C) {
  float* GPU_A, *GPU_B, *GPU_C;
  int size_a = (m*n),
      size_b = (n*p),
      size_c = (m*p);

  // Alloc and copy GPU memory
  cudaMalloc((void **)&GPU_A, sizeof(float) * size_a);
  cudaMalloc((void **)&GPU_B, sizeof(float) * size_b);
  cudaMalloc((void **)&GPU_C, sizeof(float) * size_c);

  cudaMemcpy(GPU_A, A, sizeof(float) * size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(GPU_B, B, sizeof(float) * size_b, cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();

  printf("%f %f\n", A[0], GPU_A[0]);
  printf("%f %f\n", B[0], GPU_B[0]);  
  dim3 threadsPerBlock(m,p);
  dim3 blocksPerGrid(1, 1);
  if (size_c > 512){
    threadsPerBlock.x = 512;
    threadsPerBlock.y = 512;
    blocksPerGrid.x = ceil(double(p)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(m)/double(threadsPerBlock.y));
  }

  mat_mul_kernel<<<blocksPerGrid,threadsPerBlock>>>(m,n,p,GPU_A,GPU_B,GPU_C);

  cudaDeviceSynchronize();
  // Copy the data back to the host
  cudaMemcpy(C, GPU_C, sizeof(float) * size_c, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("%f\n", C[0]);
  cudaFree(GPU_A);
  cudaFree(GPU_B);
  cudaFree(GPU_C);
} 
