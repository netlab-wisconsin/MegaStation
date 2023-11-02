#pragma once

#include <cuda.h>
#include <stdio.h>

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

struct coord
{
  unsigned int row, col;
};

class BatchedGemv {
private:
  
public:
  static const int kThreadCount = 32;

  struct Params {
    coord problem_size;
    coord a_shape;
    unsigned int batch_count;
    unsigned int mod_func;
    const myComplex *AMat; // Column Major
    const myComplex *BVec;
    signed char *CMat; // Row Major
    unsigned long a_stride;
    unsigned long b_stride;
    unsigned long c_stride;
    unsigned long a_skip;

    Params(
      coord problem_size,
      coord a_shape,
      unsigned int batch_count,
      unsigned int mod_func,
      const myComplex *AMat,
      const myComplex *BVec,
      signed char *CMat,
      unsigned long a_stride,
      unsigned long b_stride,
      unsigned long c_stride,
      unsigned long a_skip = 1
    ):
      problem_size(problem_size),
      a_shape(a_shape),
      batch_count(batch_count),
      mod_func(mod_func),
      AMat(AMat),
      BVec(BVec),
      CMat(CMat),
      a_stride(a_stride),
      b_stride(b_stride),
      c_stride(c_stride),
      a_skip(a_skip)
    {}

    Params(
      coord problem_size,
      unsigned int batch_count,
      unsigned int mod_func,
      const myComplex *AMat,
      const myComplex *BVec,
      signed char *CMat,
      unsigned long c_stride,
      unsigned long a_skip = 1
    ):
      problem_size(problem_size),
      a_shape(problem_size),
      batch_count(batch_count),
      mod_func(mod_func),
      AMat(AMat),
      BVec(BVec),
      CMat(CMat),
      a_stride(problem_size.row),
      b_stride(problem_size.col),
      c_stride(c_stride),
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
  BatchedGemv() {}

  MMLT_DEVICE
  void operator()(const Params &params) {
    static const demodPtr kDemodPtrs[] = {nullptr, demodQPSK, nullptr, demod16QAM, nullptr, demod64QAM, nullptr, demod256QAM};
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

      signed char *ptr_C = params.CMat +
        i * params.c_stride + batch_idx * params.mod_func;
      if (i < params.problem_size.row) {
        kDemodPtrs[params.mod_func-1](accum, ptr_C);
      }
    }
  }
};

__global__
void batched_gemv_kernel(const BatchedGemv::Params params) {
  BatchedGemv()(params);
}

void batched_gemv(
  const BatchedGemv::Params &params,
  cudaStream_t stream = nullptr
) {
  dim3 block = BatchedGemv::Params::get_block_shape();
  dim3 grid = BatchedGemv::Params::get_grid_shape(params, block);

  batched_gemv_kernel<<<grid, block, 0, stream>>>(params);
}