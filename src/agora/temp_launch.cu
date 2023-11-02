#include "batched_gemv.h"

void temp_launch(
  unsigned int problem_size_row,
  unsigned int problem_size_col,
  unsigned int batch_count,
  unsigned int mod_func,
  const void *AMat,
  const void *BVec,
  signed char *CMat,
  unsigned long c_stride,
  unsigned long a_skip = 1,
  cudaStream_t stream = nullptr) {
  BatchedGemv::Params params(
    {problem_size_row, problem_size_col},
    batch_count,
    mod_func,
    (myComplex *)AMat,
    (myComplex *)BVec,
    CMat,
    c_stride,
    a_skip
  );
  batched_gemv(params, stream);
}