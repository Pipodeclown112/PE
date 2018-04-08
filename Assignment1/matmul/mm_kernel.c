#include "mm_kernel.h"


void matrix_mult(int m, int n, int p, float *A, float *B, float *C) {
  int i, j, k;

  for(i=0; i<m; i++) {
    for(j=0; j<p; j++) {
      C[i*p+j]=0;
      for(k=0; k<n; k+=8) {
        C[i*p+j] += A[i*n+k]*B[k*p+j];
        C[i*p+j] += A[i*n+k+1]*B[(k+1)*p+j];
        C[i*p+j] += A[i*n+k+2]*B[(k+2)*p+j];
        C[i*p+j] += A[i*n+k+3]*B[(k+3)*p+j];
        C[i*p+j] += A[i*n+k+4]*B[(k+4)*p+j];
        C[i*p+j] += A[i*n+k+5]*B[(k+5)*p+j];
        C[i*p+j] += A[i*n+k+6]*B[(k+6)*p+j];
        C[i*p+j] += A[i*n+k+7]*B[(k+7)*p+j];
      }
    }
  }
}