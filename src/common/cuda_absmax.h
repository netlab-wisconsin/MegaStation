#pragma once

#include "modulation_cuda.h"

__global__ void cuda_absmax(myComplex* input, myComplex* output, size_t size, size_t batch_count, int row, int col)
{
  int tid = threadIdx.x;
  extern __shared__ float sdata[];

  for (size_t batch_idx = blockIdx.x; batch_idx < batch_count; batch_idx += gridDim.x) {
    // Find max in input[tid], input[tid + 1024], ...
    myComplex* input_batch = input + batch_idx * size;
    float local_max = 0.f;
    int idx = tid;
    while (idx < size) {
      local_max = fmaxf(local_max, input_batch[idx].abs());
      idx += blockDim.x;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }
    
    // Write max to output
    myComplex* output_batch = output + batch_idx * size;
    idx = tid;
    while (idx < size) {
      // Do transpose
      int row_idx = idx % row;
      int col_idx = idx / row;
      int transposed_idx = row_idx * col + col_idx;
      output_batch[transposed_idx] = input_batch[idx] * myComplex(1.0f / sdata[0], 0.f);
      idx += blockDim.x;
    }
  }
}