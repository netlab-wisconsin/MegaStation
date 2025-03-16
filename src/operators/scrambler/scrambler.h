/**
 * @file scrambler.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Scrambling and descrambling Launch functions
 * s(x) = x7 + x4 + 1
 * @version 0.1
 * @date 2023-11-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "matrix/matrix.h"

namespace mega {

/**
 * @brief Scrambler wrapper for CUDA Scrambler kernel
 *
 * @details
 * Scrambler is a 7-bit LFSR with polynomial x7 + x4 + 1
 * Check whether the scrambler is initialized, if not, initialize it with
 * init_scrambler_launch
 */
class Scrambler {
 public:
  static constexpr uint8_t kScramblerLength = 127;  // !< Scrambler length
 private:
  static bool
      initialized;  // !< Whether the scrambler gpu buffer is initialized
  static constexpr uint8_t kScramblerInitState =
      0x5D;  // !< Scrambler init state

  static inline uint8_t
      scrambler_buffer_cpu[kScramblerLength];  // !< Scrambler buffer in cpu

  /**
   * @brief Scramble the input data
   *
   * @param out_data Output data
   * @param in_data Input data
   * @param num_bytes Number of bytes to scramble
   * @param batch_count Number of batches
   * @param stream CUDA stream for async copy
   */
  static void scrambler(uint8_t *out_data, uint8_t *in_data, uint64_t num_bytes,
                        uint64_t batch_count, cudaStream_t stream = nullptr);

 public:
  /**
   * @brief Initialize the scrambler buffer
   *
   * @param num_device Number of devices
   * @param stream CUDA stream for async copy
   */
  static void init(uint8_t num_device = 1, cudaStream_t stream = nullptr);
  /**
   * @brief Scramble the input data
   *
   * @param out_data Output data
   * @param in_data Input data
   * @param num_bytes Number of bytes to scramble
   * @param batch_count Number of batches
   * @param stream CUDA stream for async copy
   */
  static void scrambler(const Matrix &out_data, const Matrix &in_data,
                        cudaStream_t stream = nullptr);
};

}  // namespace mega