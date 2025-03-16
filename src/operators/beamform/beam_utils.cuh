/**
 * @file beam_utils.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Beamforming Utils: set batch pointer kernel, scale matrix kernel.
 * @version 0.1
 * @date 2023-12-07
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "mega_complex.h"

namespace mega {
/**
 * @brief Set batched pointer array kernel.
 *
 * @tparam T type of matrix
 * @param ptr_array batched matrix pointer array
 * @param mat pointer to the first matrix
 * @param stride \p stride between two matrices
 * @param skip next matrix every \p skip matrices
 */
template <typename T>
__global__ void set_batch_ptr(T** ptr_array, T* mat, uint32_t num_ptrs,
                              uint32_t stride, uint32_t skip) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_ptrs) {
    ptr_array[tid] = mat + (tid / skip) * stride;
  }
}

__global__ void scale_absmax(const Complex* input, Complex* output,
                             uint64_t num_elements, uint64_t batch_count,
                             uint32_t row, uint32_t col) {
  uint32_t tid = threadIdx.x;
  extern __shared__ float sdata[];

  for (uint64_t batch_idx = blockIdx.x; batch_idx < batch_count;
       batch_idx += gridDim.x) {
    // Find max in input[tid], input[tid + 1024], ...
    const Complex* input_batch = input + batch_idx * num_elements;
    float local_max = 0.f;
    uint32_t idx = tid;
    while (idx < num_elements) {
      local_max = fmaxf(local_max, input_batch[idx].abs());
      idx += blockDim.x;
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (tid < s) {
        sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
    }

    // Write max to output
    Complex* output_batch = output + batch_idx * num_elements;
    idx = tid;
    while (idx < num_elements) {
      // Do transpose
      int row_idx = idx % row;
      int col_idx = idx / row;
      int transposed_idx = row_idx * col + col_idx;
      output_batch[transposed_idx] =
          input_batch[idx] * Complex(1.0f / sdata[0], 0.f);
      idx += blockDim.x;
    }
  }
}

}  // namespace mega