#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

static const uint8_t kScramblerlength_cuda = 127;
__constant__ uint8_t scram_buffer_cuda[kScramblerlength_cuda];

__global__ void scrambler_cuda(uint8_t *out_data, uint8_t *in_data,
                      size_t num_bytes, size_t batch_count) {
  for (int batch_idx = blockIdx.z; batch_idx < batch_count; batch_idx += gridDim.z) {
    int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_i >= num_bytes) {
      continue;
    }

    int idx_j = (idx_i * 8) % kScramblerlength_cuda;
    uint8_t scram_byte = 0;
    // coallesce 8 bits scram_buffer[j:j+8] into a byte
    for (size_t k = 0; k < 8; k++) {
      //scram_byte |= ((unsigned char*)input_buffer)[i * 8 + j] << (7 - j);
      scram_byte = (scram_byte << 1) | scram_buffer_cuda[idx_j];
      idx_j = (idx_j + 1) % kScramblerlength_cuda;
    }

    uint8_t *out_ptr = out_data + batch_idx * num_bytes;
    uint8_t *in_ptr = in_data + batch_idx * num_bytes;
    out_ptr[idx_i] = scram_byte ^ in_ptr[idx_i];
  }
}

__host__ void init_scrambler_cuda_buffer(uint8_t *scrambler, cudaStream_t stream) {
  cudaMemcpyToSymbol(scram_buffer_cuda, scrambler, kScramblerlength_cuda);
}