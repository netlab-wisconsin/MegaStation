/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <cstdint>
#include "util.hpp"


__global__ void sleep_kernel(uint32_t sleep_ms)
{
#if __CUDA_ARCH__ >= 700 // __nanosleep supportd in sm_70 or higher
    constexpr uint32_t NS_PER_MS = 1000000UL;
    for(uint32_t i = 0; i < sleep_ms; ++i) __nanosleep(NS_PER_MS);
#else
    if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) &&
       (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    {
        printf("Sleep not supported\n");
    }
#endif // __CUDA_ARCH__
}

void gpu_ms_sleep(uint32_t sleep_ms, int gpuId, cudaStream_t cuStrm)
{
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));

    const uint32_t N_THRD_BLKS_PER_SM       = deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock;
    const uint32_t N_THRD_BLKS              = deviceProp.multiProcessorCount*N_THRD_BLKS_PER_SM;
    const uint32_t N_MAX_THRDS_PER_THRD_BLK = deviceProp.maxThreadsPerBlock;
    sleep_kernel<<<dim3(N_THRD_BLKS), dim3(N_MAX_THRDS_PER_THRD_BLK), 0, cuStrm>>>(sleep_ms);
}

__device__ __forceinline__ unsigned long long __globaltimer()
{
    unsigned long long globaltimer;
    // 64-bit global nanosecond timer
    asm volatile("mov.u64 %0, %globaltimer;"
                 : "=l"(globaltimer));
    return globaltimer;
}

__global__ void delay_kernel_us(uint32_t delay_us)
{
    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    // 64-bit timer has a long range so skipping wrap around check
    while(__globaltimer() < end_time)
    {
    };
}

__global__ void delay_ns_wait_until(uint64_t* start_time_d, uint64_t time_offset_ns)
{
    uint64_t time_ns = *start_time_d + time_offset_ns;
//    if (threadIdx.x == 0) {
//        printf(">> start %lu, now %lu, until %lu\n", *start_time_d, __globaltimer(), time_ns);
//    }
    while(__globaltimer() < time_ns)
    {
    };
}

__global__ void get_ptimer_kernel(volatile uint64_t* ptimer)
{
    if (threadIdx.x == 0)
    {
        *ptimer = __globaltimer();
    }
}

void get_gpu_time(uint64_t *ptimer_d, cudaStream_t cuStrm)
{
    get_ptimer_kernel<<<1, 32, 0, cuStrm>>>(ptimer_d);
}

void gpu_ns_delay_until(uint64_t* start_time_d, uint64_t time_offset_ns, cudaStream_t cuStrm)
{
    delay_ns_wait_until<<<1, 32, 0, cuStrm>>>(start_time_d, time_offset_ns);
}

void gpu_us_delay(uint32_t delay_us, int gpuId, cudaStream_t cuStrm, bool singleThrdBlk)
{
#if 0
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));
    const uint32_t N_MAX_THRDS_PER_THRD_BLK = singleThrdBlk ? 32 : deviceProp.maxThreadsPerBlock;
    const uint32_t N_THRD_BLKS_PER_SM       = deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock;
    const uint32_t N_SM                     = deviceProp.multiProcessorCount;
    const uint32_t N_THRD_BLKS              = singleThrdBlk ? 1 : (N_SM*N_THRD_BLKS_PER_SM);
#else
    int maxThreadsPerMultiProcessor = 0;
    int maxThreadsPerBlock          = 0;
    int multiProcessorCount         = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, gpuId));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuId));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, gpuId));

    const uint32_t N_MAX_THRDS_PER_THRD_BLK = singleThrdBlk ? 32 :  maxThreadsPerBlock;
    const uint32_t N_THRD_BLKS_PER_SM       = maxThreadsPerMultiProcessor/maxThreadsPerBlock;
    const uint32_t N_SM                     = multiProcessorCount;
    const uint32_t N_THRD_BLKS              = singleThrdBlk ? 1 : (N_SM*N_THRD_BLKS_PER_SM);
#endif

    delay_kernel_us<<<dim3(N_THRD_BLKS), dim3(N_MAX_THRDS_PER_THRD_BLK), 0, cuStrm>>>(delay_us);
}

void gpu_ms_delay(uint32_t delay_ms, int gpuId, cudaStream_t cuStrm)
{
    gpu_us_delay(delay_ms*1000, gpuId, cuStrm);
}

__global__ void empty_kernel()
{
}

void gpu_empty_kernel(cudaStream_t cuStrm)
{
    empty_kernel<<<dim3(1), dim3(32), 0, cuStrm>>>();
}

__global__ void get_sm_id_kernel(uint32_t delay_us, uint32_t* pSmIds)
{

    // Delay to ensure SMs are in use long before reading SM Id (for multiple sub-contexts)

    // 64-bit global nanosecond timer
    constexpr uint64_t NS_PER_US = 1000UL;

    uint64_t start_time = __globaltimer();
    uint64_t end_time   = start_time + (delay_us * NS_PER_US);

    // 64-bit timer has a long range so skipping wrap around check
    while(__globaltimer() < end_time)
    {
    };

    if(0 == threadIdx.x)
    {
        uint32_t smId;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
        pSmIds[blockIdx.x] = smId;
    }
}

void get_sm_ids(int gpuId, uint32_t* pSmIds, uint32_t smIdsCnt, cudaStream_t cuStrm, uint32_t delay_us)
{
#if 0
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, gpuId));
    const uint32_t N_THRD_BLKS_PER_SM       = deviceProp.maxThreadsPerMultiProcessor/deviceProp.maxThreadsPerBlock;
    //const uint32_t N_THRD_BLKS              = deviceProp.multiProcessorCount;
    const uint32_t N_THRD_BLKS              = min(smIdsCnt, deviceProp.multiProcessorCount);
    const uint32_t N_MAX_THRDS_PER_THRD_BLK = deviceProp.maxThreadsPerBlock;
#else
    int maxThreadsPerMultiProcessor = 0;
    int maxThreadsPerBlock = 0;
    int multiProcessorCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, gpuId));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, gpuId));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, gpuId));
    const uint32_t N_THRD_BLKS_PER_SM       = maxThreadsPerMultiProcessor/maxThreadsPerBlock;
    const uint32_t N_THRD_BLKS              = min(smIdsCnt, multiProcessorCount);
    const uint32_t N_MAX_THRDS_PER_THRD_BLK = maxThreadsPerBlock;
#endif
    // printf("get_sm_ids N_THRD_BLKS %d N_MAX_THRDS_PER_THRD_BLK %d N_THRD_BLKS_PER_SM %d\n", N_THRD_BLKS, N_MAX_THRDS_PER_THRD_BLK, N_THRD_BLKS_PER_SM);
    get_sm_id_kernel<<<dim3(N_THRD_BLKS*N_THRD_BLKS_PER_SM), dim3(N_MAX_THRDS_PER_THRD_BLK), 0, cuStrm>>>(delay_us, pSmIds);
}
