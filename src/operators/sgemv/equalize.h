/**
 * @file equalize.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Static method for launching equalization kernels
 * @version 0.1
 * @date 2023-12-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

class Equalize {
 private:
  /**
   * @brief Equalize the received signal using the CSI matrix and codeblock
   *
   * @param problem_size_row number of rows in the CSI matrix
   * @param problem_size_col number of columns in the CSI matrix
   * @param batch_count number of subcarriers
   * @param bCSI pointer to CSI matrix (batched, column-major)
   * @param bBsVec pointer to received BS signal (batched, column-major)
   * @param bUeVec pointer to equalized UE signal (batched, row-major)
   * @param ue_stride stride of UE signal between batches
   * @param csi_skip move to next batch of batched_csi every csi_skip
   * @param mod_order modulation order (M = 2^mod_order for M-QAM)
   * @param cb_size size of codeblock (under ldpc context)
   * @param cb_stride stride of codeblock (under ldpc context, with punctured
   * zeros)
   * @param zeros number of punctured zeros in codeblock (under ldpc context)
   * @param valid_bits number of valid bits of one user (under ldpc context)
   * @param stream cuda stream
   */
  static void equalize(const uint64_t problem_size_row,
                       const uint64_t problem_size_col,
                       const uint64_t batch_count, const Complex* bCSI,
                       const Complex* bBsVec, half* bUeVec,
                       const uint64_t ue_stride, uint64_t csi_skip,
                       uint8_t mod_order, uint64_t cb_size, uint64_t cb_stride,
                       uint64_t zeros, uint64_t valid_bits,
                       cudaStream_t stream = nullptr);

 public:
  /**
   * @brief Equalize the received signal using the CSI matrix and codeblock
   *
   * @param csi CSI matrix (ue * bs, column-major)
   * @param bsVec received BS signal (bs * ofdm, column-major)
   * @param ueVec equalized UE signal ((cb_size+zeros) * (blocks*ue), row-major,
   * aligned)
   * @param csi_skip subcarriers group size
   * @param mod_order modulation order (M = 2^mod_order for M-QAM)
   * @param cb_size size of codeblock (under ldpc context)
   * @param zeros number of punctured zeros in codeblock (under ldpc context)
   * @param cb_blocks number of code blocks of one user (cb_size * blocks)
   * @param stream cuda stream
   */
  static void equalize(const Matrix& csi, const Matrix& bsVec,
                       const Matrix& ueVec, uint64_t csi_skip,
                       uint8_t mod_order, uint64_t cb_size, uint64_t zeros,
                       uint16_t cb_blocks, cudaStream_t stream = nullptr);
};

}  // namespace mega