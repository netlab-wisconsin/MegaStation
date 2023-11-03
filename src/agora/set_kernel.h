#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void set_pointer(T **ptr_array, T *val, int num_ptrs, int inc, int skip) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_ptrs) {
    ptr_array[tid] = val + (tid / skip) * inc;
  }
}