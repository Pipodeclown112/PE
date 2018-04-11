#include "omp.h"
#include "mm_kernel.h"

#define NUM_THREADS 4


void matrix_mult(int m, int n, int p, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int i, j, k, l;

  int block_size = m/NUM_THREADS;
  #pragma omp parallel num_threads(NUM_THREADS)
  {
    int id = omp_get_thread_num();
    int stop = (id+1)*block_size;

    for(i=id*block_size; i<stop; i++) {
      for(l=0; l<p; l++){
        C[i*p+l]=0;
      }
      for(k=0; k<n; k++) {
        for(j=0; j<p; j++) {
          C[i*p+j] += A[i*n+k]*B[k*p+j];
        }
      }
    }
  }
} 
