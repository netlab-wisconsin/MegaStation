#pragma once

void demul_launch(
    unsigned int problem_size_row,
    unsigned int problem_size_col,
    unsigned int batch_count,
    unsigned int mod_func,
    const void *AMat,
    const void *BVec,
    signed char *CMat,
    unsigned long c_stride,
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