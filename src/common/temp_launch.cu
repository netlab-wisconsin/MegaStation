#include "batched_gemv.h"
#include "set_kernel.h"
#include "scrambler_cuda.h"
#include "modulation_cuda.h"
#include "batched_rgemv.h"
#include "cuda_absmax.h"

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

void modulation_launch(
  const uint8_t *input,
  void *output,
  const void *pilot_table, int pilot_spacing,
  uint8_t mod,
  size_t in_bytes,
  size_t out_bytes,
  size_t batch_count,
  cudaStream_t stream = nullptr
) {
  dim3 block = dim3(32, 1, 1);
  dim3 grid = dim3((out_bytes + block.x - 1) / block.x, 1, batch_count % 65536);
  // printf("[OUT]mod: %d, in_bytes: %d, out_bytes: %d\n", mod, in_bytes, out_bytes);

  modulateKernel<<<grid, block, 0, stream>>>(input, (myComplex *)output, (myComplex *)pilot_table, pilot_spacing, mod, in_bytes, out_bytes, batch_count);
}

void init_modulation_launch(
  void *modulation,
  size_t sz,
  cudaStream_t stream = nullptr
)
{
  init_modulation_table((myComplex *)modulation, sz, stream);
}

void precode_launch(
  unsigned int problem_size_row,
  unsigned int problem_size_col,
  unsigned int batch_count,
  const void *AMat,
  const void *BVec,
  void *CMat,
  unsigned long c_stride,
  unsigned long c_start,
  unsigned long a_skip = 1,
  cudaStream_t stream = nullptr
) {
  BatchedGemvR::Params params(
    {problem_size_row, problem_size_col},
    batch_count,
    (myComplex *)AMat,
    (myComplex *)BVec,
    (myComplex *)CMat,
    c_stride,
    c_start,
    a_skip
  );
  batched_rgemv(params, stream);
}

void absmax_launch(
  const void *input,
  void *output,
  size_t num_elements,
  size_t batch_count,
  int row, int col,
  cudaStream_t stream = nullptr
) {
  size_t num_threads = num_elements < 1024 ? ((num_elements / 32 + 1) * 32) : 1024;
  dim3 block = dim3(num_threads, 1, 1);
  dim3 grid = dim3(batch_count % 65536, 1, 1);

  size_t smem_size = sizeof(float) * num_threads;

  cuda_absmax<<<grid, block, smem_size, stream>>>((myComplex *)input, (myComplex *)output, num_elements, batch_count, row, col);
}