#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "modulation_cuda.h"


#ifndef MMLT_DEVICE
	#define MMLT_DEVICE __forceinline__ __device__
#endif
#ifndef MMLT_UNROLL
	#define MMLT_UNROLL _Pragma("unroll")
#endif
#ifndef MMLT_LOOP
	#define MMLT_LOOP _Pragma("unroll 1")
#endif

#ifndef COORD
#define COORD
struct coord
{
  unsigned int row, col;
};
#endif

class BatchedGemvR {
private:
  
public:
  static const int kThreadCount = 32;

  struct Params {
    coord problem_size;
    coord a_shape;
    unsigned int batch_count;
    const myComplex *AMat; // Row Major
    const myComplex *BVec;
    myComplex *CMat; // Row Major
    unsigned long a_stride;
    unsigned long b_stride;
    unsigned long c_stride;
    unsigned long c_start;
    unsigned long a_skip;

    Params(
      coord problem_size,
      coord a_shape,
      unsigned int batch_count,
      const myComplex *AMat,
      const myComplex *BVec,
      myComplex *CMat,
      unsigned long a_stride,
      unsigned long b_stride,
      unsigned long c_stride,
      unsigned long c_start = 0,
      unsigned long a_skip = 1
    ):
      problem_size(problem_size),
      a_shape(a_shape),
      batch_count(batch_count),
      AMat(AMat),
      BVec(BVec),
      CMat(CMat),
      a_stride(a_stride),
      b_stride(b_stride),
      c_stride(c_stride),
      c_start(c_start),
      a_skip(a_skip)
    {}

    Params(
      coord problem_size,
      unsigned int batch_count,
      const myComplex *AMat,
      const myComplex *BVec,
      myComplex *CMat,
      unsigned long c_stride,
      unsigned long c_start = 0,
      unsigned long a_skip = 1
    ):
      problem_size(problem_size),
      a_shape(problem_size),
      batch_count(batch_count),
      AMat(AMat),
      BVec(BVec),
      CMat(CMat),
      a_stride(problem_size.row * problem_size.col),
      b_stride(problem_size.col),
      c_stride(c_stride),
      c_start(c_start),
      a_skip(a_skip)
    {}

    static inline dim3 get_block_shape() {
      return dim3(kThreadCount, 1, 1);
    }

    static inline dim3 get_grid_shape(const Params &params, const dim3 &block) {
      return dim3((params.problem_size.row + block.x - 1) / block.x, 1, params.batch_count % 65536);
    }
  };

  MMLT_DEVICE
  BatchedGemvR() {}

  MMLT_DEVICE
  void operator()(const Params &params) {
    for (int batch_idx = blockIdx.z;
        batch_idx < params.batch_count;
        batch_idx += gridDim.z) {
      int i = blockIdx.x * kThreadCount + threadIdx.x;

      const myComplex *ptr_A = params.AMat + i;
      const myComplex *ptr_B = params.BVec;

      // every a_skip batches, we point to the next batch
      ptr_A += (batch_idx / params.a_skip) * params.a_stride;
      ptr_B += batch_idx * params.b_stride;

      myComplex accum = myComplex();

      MMLT_LOOP
      for (int j = 0; j < params.problem_size.col; ++j) {
        myComplex a = myComplex();
        if (i < params.problem_size.row) {
          a = *ptr_A;
        }
        ptr_A += params.a_shape.row;

        myComplex b = *ptr_B;
        ptr_B += 1;

        accum += a * b;
      }

      int out_batch_idx = (batch_idx + params.c_start + (params.c_stride / 2))
        % params.c_stride;
      myComplex *ptr_C = params.CMat + i * params.c_stride + out_batch_idx;
      if (i < params.problem_size.row) {
        ptr_C[0] = accum;
      }
    }
  }
};

__global__
void batched_rgemv_kernel(const BatchedGemvR::Params params) {
  BatchedGemvR()(params);
}

void batched_rgemv(
  const BatchedGemvR::Params &params,
  cudaStream_t stream = nullptr
) {
  dim3 block = BatchedGemvR::Params::get_block_shape();
  dim3 grid = BatchedGemvR::Params::get_grid_shape(params, block);

  batched_rgemv_kernel<<<grid, block, 0, stream>>>(params);
}