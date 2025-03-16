/**
 * @file scrambler_kernel.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Scrambling and descrambling functions
 * @version 0.1
 * @date 2023-11-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "scrambler.h"

namespace mega {

__constant__ uint8_t
    scramble_buffer[Scrambler::kScramblerLength];  //!< Scrambler buffer

/**
 * @brief Scramble/Descramble the input data
 * Different batches are contiguous in memory
 *
 * @param out_data Scrambled/Descrambled output
 * @param in_data Input data to be scrambled/descrambled
 * @param num_bytes Number of bytes to be scrambled/descrambled
 * @param batch_count Number of batches
 */
__global__ void scrambler_kernel(uint8_t *out_data, uint8_t *in_data,
                                 uint64_t num_bytes, uint32_t batch_count) {
  for (uint32_t batch_idx = blockIdx.z; batch_idx < batch_count;
       batch_idx += gridDim.z) {
    uint32_t idx_i = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t idx_j = (idx_i * 8) % Scrambler::kScramblerLength;
    uint8_t scram_byte = 0;
    // coallesce 8 bits scramble_buffer[j:j+8) into a byte
    for (uint8_t k = 0; k < 8; k++) {
      scram_byte = (scram_byte << 1) | scramble_buffer[idx_j];
      idx_j = (idx_j + 1) % Scrambler::kScramblerLength;
    }

    uint8_t *out_ptr = out_data + batch_idx * num_bytes;
    uint8_t *in_ptr = in_data + batch_idx * num_bytes;
    if ((uint64_t)idx_i < num_bytes) {
      out_ptr[idx_i] = scram_byte ^ in_ptr[idx_i];
    }
  }
}

/**
 * @brief Initialize the scrambler buffer
 *
 * @param scramble_cpu Scrambler buffer to be copied to GPU
 * @param stream CUDA stream
 */
__host__ void init_scrambler_cuda_buffer(uint8_t *scramble_cpu,
                                         cudaStream_t stream = nullptr) {
  cudaMemcpyToSymbolAsync(scramble_buffer, scramble_cpu,
                          Scrambler::kScramblerLength, 0,
                          cudaMemcpyHostToDevice, stream);
}

}  // namespace mega