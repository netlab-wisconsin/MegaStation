#include <cutlass/gemm/device/gemm.h>

#include "arti_kernel.h"

__global__ void arti_kernel(int *flag, char *data, char *result) {
  constexpr int N = 49152;
  __shared__ char shared_data[N];
  // set shared data to 0
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    shared_data[i] = 0;
  }
  __syncthreads();
  // copy data to shared memory
  while (*flag == 1) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      shared_data[i] += data[i];
    }
    __syncthreads();
    // do some computation
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
      result[i] = shared_data[i] + 2;
    }
    __syncthreads();
  }
}

void arti_launch(int *flag, char *data, char *result, int n_block,
                 cudaStream_t stream) {
  arti_kernel<<<2 * n_block, 1024, 0, stream>>>(flag, data, result);
}

void arti_launch(float *A, float *B, float *C, int n_block, int k,
                 cudaStream_t stream) {
  int m = 1280, n = 1024;
  int lda = m, ldb = n, ldc = n;
  const float alpha = 1.;
  const float beta = 0.;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                  ColumnMajor,  // Layout of A matrix
                                  float,        // Data-type of B matrix
                                  RowMajor,     // Layout of B matrix
                                  float,        // Data-type of C matrix
                                  RowMajor      // Layout of C matrix
                                  >;

  CutlassGemm gemm_operator;

  CutlassGemm::Arguments args(
      {m, n, k},  // Gemm Problem dimensions
      {A, lda},   // Tensor-ref for source matrix A
      {B, ldb},   // Tensor-ref for source matrix B
      {C, ldc},   // Tensor-ref for source matrix C
      {C, ldc},  // Tensor-ref for destination matrix D (may be different memory
                 // than source C matrix)
      {alpha, beta});

  for (int i = 0; i < n_block; i++) {
    cutlass::Status status = gemm_operator(args, nullptr, stream);
  }
}