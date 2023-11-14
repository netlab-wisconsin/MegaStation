#include "batched_gemv.h"
#include "set_kernel.h"
#include "scrambler_cuda.h"

void demul_launch(
  unsigned int problem_size_row,
  unsigned int problem_size_col,
  unsigned int batch_count,
  unsigned int mod_func,
  const void *AMat,
  const void *BVec,
  short *CMat,
  unsigned long c_stride,
  unsigned int cb_size,
  unsigned int cb_stride,
  unsigned int zeros,
  unsigned int valid_bit_count,
  unsigned long a_skip = 1,
  cudaStream_t stream = nullptr) {
  BatchedGemv::Params params(
    {problem_size_row, problem_size_col},
    batch_count,
    mod_func,
    (myComplex *)AMat,
    (myComplex *)BVec,
    (half *)CMat,
    c_stride,
    cb_size,
    cb_stride,
    zeros,
    valid_bit_count,
    a_skip
  );
  batched_gemv(params, stream);
}

template <typename T>
void set_ptr_launch(
  T **ptr_array,
  T *val,
  int num_ptrs,
  int inc,
  int skip,
  cudaStream_t stream = nullptr) {
  dim3 block = dim3(32, 1, 1);
  dim3 grid = dim3((num_ptrs + block.x - 1) / block.x, 1, 1);

  set_pointer<T><<<grid, block, 0, stream>>>(ptr_array, val, num_ptrs, inc, skip);
}

template void set_ptr_launch<float2>(
  float2 **ptr_array,
  float2 *val,
  int num_ptrs,
  int inc,
  int skip,
  cudaStream_t stream);

void scrambler_launch(
  uint8_t *out_data,
  uint8_t *in_data,
  size_t num_bytes,
  size_t num_symbols,
  cudaStream_t stream = nullptr
) {
  dim3 block = dim3(32, 1, 1);
  dim3 grid = dim3((num_bytes + block.x - 1) / block.x, 1, num_symbols % 65536);

  scrambler_cuda<<<grid, block, 0, stream>>>(out_data, in_data, num_bytes, num_symbols);
}

void init_scrambler_launch(
    uint8_t *scrambler_buffer,
    cudaStream_t stream = nullptr
)
{
  init_scrambler_cuda_buffer(scrambler_buffer, stream);
}