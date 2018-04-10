#include "mm_kernel.h"


void matrix_mult(int m, int n, int p, float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  int i, j, k, l;

  for(i=0; i<m; i++) {
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
