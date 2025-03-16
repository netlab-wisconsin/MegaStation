#pragma once

#include <cuda_runtime.h>

void arti_launch(int *flag, char *data, char *result, int n_block,
                 cudaStream_t stream);

void arti_launch(float *A, float *B, float *C, int n_block, int k,
                 cudaStream_t stream);
