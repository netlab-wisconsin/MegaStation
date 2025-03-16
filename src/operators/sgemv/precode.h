/**
 * @file precode.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Static method for launching the precode kernel
 * @version 0.1
 * @date 2023-12-03
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

class Precode {
 private:
  /**
   * @brief Launch the precode kernel
   *
   * @param problem_size_row Number of rows in the precode matrix
   * @param problem_size_col Number of columns in the precode matrix
   * @param batch_count Number of subcarriers
   * @param bPrecode pointer to Precode matrix (batched, column-major)
   * @param bUeVec pointer to modulated UE vector (batched, column-major)
   * @param bBsVec pointer to output BS signal vector (batched, row-major)
   * @param bs_stride Stride of the BS vector for next batch
   * @param precode_skip move to next batch of batched_precode every
   * precode_skip
   * @param bs_start Starting valid BS subcarrier index
   * @param stream CUDA stream
   */
  static void precode(uint64_t problem_size_row, uint64_t problem_size_col,
                      uint64_t batch_count, const Complex* bPrecoded,
                      const Complex* bUeVec, Complex* bBsVec,
                      uint64_t bs_stride, uint64_t precode_skip,
                      uint64_t bs_start, cudaStream_t stream = nullptr);

 public:
  /**
   * @brief Launch the precode kernel
   *
   * @param precoded Precode matrix (bs * ue, column-major)
   * @param ueVec modulated UE vector (ue * ofdm, column-major)
   * @param bsVec output BS signal vector (bs * ofdmCa, row-major)
   * @param precode_skip move to next batch of batched_precode every
   * precode_skip
   * @param bs_start Starting valid BS subcarrier index
   * @param stream CUDA stream
   */
  static void precode(const Matrix& precoded, const Matrix& ueVec,
                      const Matrix& bsVec, uint64_t precode_skip,
                      uint64_t bs_start, cudaStream_t stream = nullptr);
};

}  // namespace mega