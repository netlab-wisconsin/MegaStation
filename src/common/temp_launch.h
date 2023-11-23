#pragma once

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
    cudaStream_t stream = nullptr);

template <typename T>
void set_ptr_launch(
    T **ptr_array,
    T *val,
    int num_ptrs,
    int inc,
    int skip,
    cudaStream_t stream = nullptr);

void scrambler_launch(
    uint8_t *out_data,
    uint8_t *in_data,
    size_t num_bytes,
    size_t num_symbols,
    cudaStream_t stream = nullptr);

void init_scrambler_launch(
    uint8_t *scrambler_buffer,
    cudaStream_t stream = nullptr);

void modulation_launch(
  const uint8_t *input,
  void *output,
  const void *pilot_table, int pilot_spacing,
  uint8_t mod,
  size_t in_bytes,
  size_t out_bytes,
  size_t batch_count,
  cudaStream_t stream = nullptr
);

void init_modulation_launch(
  void *modulation,
  size_t sz,
  cudaStream_t stream = nullptr
);

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
);

void absmax_launch(
  const void *input,
  void *output,
  size_t num_elements,
  size_t batch_count,
  int row, int col,
  cudaStream_t stream = nullptr
);