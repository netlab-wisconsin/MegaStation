/**
 * @file precode.cu
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Launch the precode kernel
 * @version 0.1
 * @date 2023-12-03
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "mmlt.cuh"
#include "precode.h"
#include "precode_kernel.cuh"

using namespace mega;

void Precode::precode(uint64_t problem_size_row, uint64_t problem_size_col,
                      uint64_t batch_count, const Complex* bPrecoded,
                      const Complex* bUeVec, Complex* bBsVec,
                      uint64_t bs_stride, uint64_t precode_skip,
                      uint64_t bs_start, cudaStream_t stream) {
  coord<3> problem_size = {problem_size_row, problem_size_col, batch_count};
  dim3 block = dim3(kThreadCount * 2, 1, 1);
  dim3 grid =
      dim3((problem_size_row + 127) / 128, 1, batch_count / precode_skip);
  PrecodeKernel::Params params(problem_size, bPrecoded, bUeVec, bBsVec,
                               bs_stride, precode_skip, bs_start);
  if (problem_size_row <= 8) {
    grid.y = (precode_skip + 7) / 8;
    precode_kernel<1, 1><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 16) {
    grid.y = (precode_skip + 7) / 8;
    block.x *= 2;
    precode_kernel<2, 1><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 32) {
    grid.y = (precode_skip + 15) / 16;
    block.x *= 2;
    precode_kernel<2, 2><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 64) {
    if (precode_skip <= 16) {
      grid.x = 2;
      grid.y = 1;
      block.x *= 2;
      precode_kernel<2, 2><<<grid, block, 0, stream>>>(params);
    } else {
      grid.y = (precode_skip + 31) / 32;
      block.x *= 2;
      precode_kernel<2, 4><<<grid, block, 0, stream>>>(params);
    }
  } else {
    grid.y = (precode_skip + 31) / 32;
    block.x *= 4;
    precode_kernel<4, 4><<<grid, block, 0, stream>>>(params);
  }
}

void Precode::precode(const Matrix& precoded, const Matrix& ueVec,
                      const Matrix& bsVec, uint64_t precode_skip,
                      uint64_t bs_start, cudaStream_t stream) {
  uint64_t problem_size_row = precoded.dim(0);
  uint64_t problem_size_col = precoded.dim(1);
  uint64_t batch_count = ueVec.dim(0);
  uint64_t bs_stride = bsVec.stride(1);  // ofdmca
  precode(problem_size_row, problem_size_col, batch_count,
          precoded.ptr<Complex>(), ueVec.ptr<Complex>(), bsVec.ptr<Complex>(),
          bs_stride, precode_skip, bs_start, stream);
}