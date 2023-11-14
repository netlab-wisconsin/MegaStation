/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(UTIL_HPP_INCLUDED_)
#define UTIL_HPP_INCLUDED_

#include "cuphy.h"

#define CUDA_CHECK(result)                        \
    if((cudaError_t)result != cudaSuccess)        \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%s\n", \
                __FILE__,                         \
                __LINE__,                         \
                cudaGetErrorString(result));      \
    }

void gpu_ms_delay(uint32_t delay_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
void gpu_us_delay(uint32_t delay_us, int gpuId = 0, cudaStream_t cuStrm = 0, bool singleThrdBlk = false);
void gpu_ms_sleep(uint32_t sleep_ms, int gpuId = 0, cudaStream_t cuStrm = 0);
void gpu_ns_delay_until(uint64_t* start_time_d, uint64_t time_offset_ns, cudaStream_t cuStrm);
void gpu_empty_kernel(cudaStream_t cuStrm = 0);
void get_sm_ids(int gpuId, uint32_t* pSmIds, uint32_t smIdsCnt, cudaStream_t cuStrm = 0, uint32_t delay_us = 1000);
void get_gpu_time(uint64_t *ptimer_d, cudaStream_t cuStrm);

#endif // !defined(UTIL_HPP_INCLUDED_)
