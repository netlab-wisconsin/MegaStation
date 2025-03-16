/**
 * @file equalize.cu
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Launches the equalize kernel
 * @version 0.1
 * @date 2023-12-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstdio>

#include "equalize.h"
#include "equalize_kernel.cuh"
#include "mmlt.cuh"

using namespace mega;

void Equalize::equalize(uint64_t problem_size_row, uint64_t problem_size_col,
                        uint64_t batch_count, const Complex* bCSI,
                        const Complex* bBsVec, half* bUeVec, uint64_t ue_stride,
                        uint64_t csi_skip, uint8_t mod_order, uint64_t cb_size,
                        uint64_t cb_stride, uint64_t zeros, uint64_t valid_bits,
                        cudaStream_t stream) {
  coord<3> problem_size = {problem_size_row, problem_size_col, batch_count};
  dim3 block = dim3(kThreadCount * 2, 1, 1);
  dim3 grid = dim3((problem_size_row + 127) / 128, 1, batch_count / csi_skip);
  EqualizeKernel::Params params(problem_size, bCSI, bBsVec, bUeVec, ue_stride,
                                csi_skip, mod_order, cb_size, cb_stride, zeros,
                                valid_bits);
  if (problem_size_row <= 8) {
    grid.y = (csi_skip + 7) / 8;
    equalize_kernel<1, 1><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 16) {
    grid.y = (csi_skip + 15) / 16;
    equalize_kernel<1, 2><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 32) {
    grid.y = (csi_skip + 15) / 16;
    block.x *= 2;
    equalize_kernel<2, 2><<<grid, block, 0, stream>>>(params);
  } else if (problem_size_row <= 64) {
    grid.y = (csi_skip + 31) / 32;
    block.x *= 2;
    equalize_kernel<2, 4><<<grid, block, 0, stream>>>(params);
  } else {
    grid.y = (csi_skip + 31) / 32;
    block.x *= 4;
    equalize_kernel<4, 4><<<grid, block, 0, stream>>>(params);
  }
}

void Equalize::equalize(const Matrix& csi, const Matrix& bsVec,
                        const Matrix& ueVec, uint64_t csi_skip,
                        uint8_t mod_order, uint64_t cb_size, uint64_t zeros,
                        uint16_t cb_blocks, cudaStream_t stream) {
  uint64_t problem_size_row = csi.dim(0);
  uint64_t problem_size_col = csi.dim(1);
  uint64_t batch_count = bsVec.dim(0);

  uint64_t cb_stride = ueVec.stride(1);
  uint64_t ue_stride = cb_stride * cb_blocks;
  uint64_t valid_bits = cb_size * cb_blocks;
  equalize(problem_size_row, problem_size_col, batch_count, csi.ptr<Complex>(),
           bsVec.ptr<Complex>(), ueVec.ptr<half>(), ue_stride, csi_skip,
           mod_order, cb_size, cb_stride, zeros, valid_bits, stream);
}