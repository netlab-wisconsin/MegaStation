/**
 * @file scrambler.cu
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief launch kernel for scrambler
 * @version 0.1
 * @date 2023-11-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstdint>

#include "mmlt.cuh"
#include "scrambler.h"
#include "scrambler_kernel.cuh"

using namespace mega;

bool Scrambler::initialized = false;

void Scrambler::init(uint8_t num_device, cudaStream_t stream) {
  if (!initialized) {
    uint8_t scrambler_init = kScramblerInitState;
    uint8_t res_xor = 0;
    for (uint8_t i = 0; i < kScramblerLength; i++) {
      res_xor = (scrambler_init ^ (scrambler_init >> 3)) & 0x01;
      scrambler_buffer_cpu[i] = res_xor;
      scrambler_init >>= 1;
      scrambler_init |= res_xor << 6;
    }
    for (uint8_t i = 0; i < num_device; i++) {
      cudaSetDevice(i);
      init_scrambler_cuda_buffer(scrambler_buffer_cpu, stream);
    }
    initialized = true;
  }
}

void Scrambler::scrambler(uint8_t *out_data, uint8_t *in_data,
                          uint64_t num_bytes, uint64_t batch_count,
                          cudaStream_t stream) {
  Scrambler::init(1, stream);  // For compatibility with the old API

  dim3 block = get_block_shape();
  dim3 grid = get_grid_shape({num_bytes, 1, batch_count}, block);

  scrambler_kernel<<<grid, block, 0, stream>>>(out_data, in_data, num_bytes,
                                               batch_count);
}

void Scrambler::scrambler(const Matrix &out_data, const Matrix &in_data,
                          cudaStream_t stream) {
  const uint64_t num_bytes = out_data.szBytes(1);
  const uint64_t batch_count = out_data.nDim() == 1 ? 1 : out_data.dim(1);
  const uint64_t in_batch_count = in_data.nDim() == 1 ? 1 : in_data.dim(1);
  if (out_data.nDim() != in_data.nDim() || batch_count != in_batch_count ||
      num_bytes != in_data.szBytes(1)) {
    throw std::runtime_error("Input and output mismatch");
  }

  scrambler(out_data.ptr<uint8_t>(), in_data.ptr<uint8_t>(), num_bytes,
            batch_count, stream);
}