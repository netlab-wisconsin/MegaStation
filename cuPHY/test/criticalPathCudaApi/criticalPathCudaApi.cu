/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//#define ENABLE_MI_MEMTRACE
#define ENABLE_DRIVER_KERNEL_LAUNCH

#include <gtest/gtest.h>
#include <cstdlib>
#include <array>
#include "cuphy.hpp"
#include "util.hpp"

#ifdef ENABLE_MI_MEMTRACE
#include "../../mimalloc/include/mimalloc-memtrace.h" // Set correct path to mimalloc library
#endif

static constexpr size_t NUM_NODES = 5;
using cpuCuPhyBufR32_t = cuphy::buffer<float, cuphy::pinned_alloc>;
using gpuCuPhyBufR32_t = cuphy::buffer<float, cuphy::device_alloc>;
using uqPtrCpuCuPhyBufR32_t = std::unique_ptr<cpuCuPhyBufR32_t>;
using uqPtrGpuCuPhyBufR32_t = std::unique_ptr<gpuCuPhyBufR32_t>;

using uqPtrsCpuCuPhyBufR32_t = std::unique_ptr<cpuCuPhyBufR32_t[]>;
using uqPtrsGpuCuPhyBufR32_t = std::unique_ptr<gpuCuPhyBufR32_t[]>;

template<size_t N>
using cudaStreamArr_t = std::array<cudaStream_t, N>;

template<size_t N>
using cudaEventArr_t = std::array<cudaEvent_t, N>;


////////////////////////////////////////////////////////////////////////
// Input/Output buffers
typedef struct _cpuInBufs
{
    uqPtrCpuCuPhyBufR32_t  uqPtrBuf1;
    uqPtrCpuCuPhyBufR32_t  uqPtrBuf2;
} cpuInBufs_t;

typedef struct _cpuOutBufs
{
    uqPtrCpuCuPhyBufR32_t  uqPtrAddBuf;
    uqPtrCpuCuPhyBufR32_t  uqPtrSubBuf;
    uqPtrCpuCuPhyBufR32_t  uqPtrMulBuf;
    uqPtrCpuCuPhyBufR32_t  uqPtrDivBuf;
    uqPtrCpuCuPhyBufR32_t  uqPtrSqrBuf;
} cpuOutBufs_t;

typedef struct _gpuInBufs
{
    uqPtrGpuCuPhyBufR32_t  uqPtrBuf1;
    uqPtrGpuCuPhyBufR32_t  uqPtrBuf2;
} gpuInBufs_t;

typedef struct _gpuOutBufs
{
    uqPtrGpuCuPhyBufR32_t  uqPtrAddBuf;
    uqPtrGpuCuPhyBufR32_t  uqPtrSubBuf;
    uqPtrGpuCuPhyBufR32_t  uqPtrMulBuf;
    uqPtrGpuCuPhyBufR32_t  uqPtrDivBuf;
    uqPtrGpuCuPhyBufR32_t  uqPtrSqrBuf;
} gpuOutBufs_t;

typedef struct _graphNodes
{
    CUgraphNode emptyRootNode;
    CUgraphNode addKernelNode;
    CUgraphNode subKernelNode;
    CUgraphNode mulKernelNode;
    CUgraphNode divKernelNode;
    CUgraphNode sqrKernelNode;
} graphNodes_t;

// malloc, calloc, realloc, new; free, delete, delete[]

// cudaMemsetAsync
// cudaEventRecord
// cudaMemcpyAsync
// cudaGetFuncBySymbol
// cuGraphExecKernelNodeSetParams
// cuGraphLaunch
// cuGraphNodeSetEnabled
// cudaLaunchKernel
// cuGraphUpload
// cudaStreamWaitEvent
// cudaEventQuery
// cudaEventWait

////////////////////////////////////////////////////////////////////////
// Kernels
__global__ void vecAddKernel(size_t len, float const* pOp1, float const* pOp2, float* pOut)
{
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(idx < len) pOut[idx] = pOp1[idx] + pOp2[idx];
}

__global__ void vecSubKernel(size_t len, float const* pOp1, float const* pOp2, float* pOut)
{
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(idx < len) pOut[idx] = pOp1[idx] - pOp2[idx];
}

__global__ void vecMulKernel(size_t len, float const* pOp1, float const* pOp2, float* pOut)
{
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(idx < len) pOut[idx] = pOp1[idx] * pOp2[idx];
}

__global__ void vecDivKernel(size_t len, float const* pOp1, float const* pOp2, float* pOut)
{
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(idx < len) pOut[idx] = pOp1[idx] / pOp2[idx];
}

__global__ void vecSqrKernel(size_t len, float const* pOp1, float* pOut)
{
    size_t idx = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(idx < len) pOut[idx] = pOp1[idx] * pOp1[idx];
}

__global__ void emptyKernel(void)
{
}

////////////////////////////////////////////////////////////////////////
// CPU functions

// Helper to calculate expected results
void calcExpectedResults(size_t             len,
                         cpuInBufs_t const& cpuInBufs,
                         cpuOutBufs_t&      cpuExpectedOutBufs)
{
    // Calculate expected results
    float const* pIn1Buf = cpuInBufs.uqPtrBuf1->addr();
    float const* pIn2Buf = cpuInBufs.uqPtrBuf2->addr();
    float*       pOutAddBuf = cpuExpectedOutBufs.uqPtrAddBuf->addr();
    float*       pOutSubBuf = cpuExpectedOutBufs.uqPtrSubBuf->addr();
    float*       pOutMulBuf = cpuExpectedOutBufs.uqPtrMulBuf->addr();
    float*       pOutDivBuf = cpuExpectedOutBufs.uqPtrDivBuf->addr();
    float*       pOutSqrBuf = cpuExpectedOutBufs.uqPtrSqrBuf->addr();

    for(uint32_t i = 0; i < len; ++i)
    {
        pOutAddBuf[i] = pIn1Buf[i] + pIn2Buf[i];
        pOutSubBuf[i] = pIn1Buf[i] - pIn2Buf[i];
        pOutMulBuf[i] = pIn1Buf[i] * pIn2Buf[i];
        pOutDivBuf[i] = pIn1Buf[i] / pIn2Buf[i];
        pOutSqrBuf[i] = pIn1Buf[i] * pIn1Buf[i];        
    }
}


// Check results
void checkResults(size_t              len,
                  cpuOutBufs_t const& cpuExpectedOutBufs,
                  cpuOutBufs_t const& cpuResOutBufs)
{
    float const* pResAddBuf = cpuResOutBufs.uqPtrAddBuf->addr();
    float const* pResSubBuf = cpuResOutBufs.uqPtrSubBuf->addr();
    float const* pResMulBuf = cpuResOutBufs.uqPtrMulBuf->addr();
    float const* pResDivBuf = cpuResOutBufs.uqPtrDivBuf->addr();
    float const* pResSqrBuf = cpuResOutBufs.uqPtrSqrBuf->addr();

    float const* pExpectedAddBuf = cpuExpectedOutBufs.uqPtrAddBuf->addr();
    float const* pExpectedSubBuf = cpuExpectedOutBufs.uqPtrSubBuf->addr();
    float const* pExpectedMulBuf = cpuExpectedOutBufs.uqPtrMulBuf->addr();
    float const* pExpectedDivBuf = cpuExpectedOutBufs.uqPtrDivBuf->addr();
    float const* pExpectedSqrBuf = cpuExpectedOutBufs.uqPtrSqrBuf->addr();

    for(uint32_t i = 0; i < len; ++i)
    {
        EXPECT_FLOAT_EQ(pExpectedAddBuf[i], pResAddBuf[i]);
        EXPECT_FLOAT_EQ(pExpectedSubBuf[i], pResSubBuf[i]);
        EXPECT_FLOAT_EQ(pExpectedMulBuf[i], pResMulBuf[i]);
        EXPECT_FLOAT_EQ(pExpectedDivBuf[i], pResDivBuf[i]);
        EXPECT_FLOAT_EQ(pExpectedSqrBuf[i], pResSqrBuf[i]);
    }
}


// Factory to instantiate CPU buffers used in tests
void createCpuBufs(size_t        nElems, 
                   cpuInBufs_t&  cpuInBufs,
                   cpuOutBufs_t& cpuExpectedOutBufs,
                   cpuOutBufs_t& cpuOutBufs)
{
#if 1
    cpuInBufs.uqPtrBuf1 = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuInBufs.uqPtrBuf2 = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    
    cpuExpectedOutBufs.uqPtrAddBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuExpectedOutBufs.uqPtrSubBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuExpectedOutBufs.uqPtrMulBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuExpectedOutBufs.uqPtrDivBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuExpectedOutBufs.uqPtrSqrBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);

    cpuOutBufs.uqPtrAddBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuOutBufs.uqPtrSubBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuOutBufs.uqPtrMulBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuOutBufs.uqPtrDivBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
    cpuOutBufs.uqPtrSqrBuf = std::make_unique<cpuCuPhyBufR32_t>(nElems);
#else
    std::vector<std::reference_wrapper<uqPtrCpuCuPhyBufR32_t>> uqPtrs {cpuInBufs.uqPtrBuf1, cpuInBufs.uqPtrBuf2,
     cpuExpectedOutBufs.uqPtrAddBuf, cpuExpectedOutBufs.uqPtrSubBuf, cpuExpectedOutBufs.uqPtrMulBuf, cpuExpectedOutBufs.uqPtrDivBuf, cpuExpectedOutBufs.uqPtrSqrBuf,
     cpuOutBufs.uqPtrAddBuf, cpuOutBufs.uqPtrSubBuf, cpuOutBufs.uqPtrMulBuf, cpuOutBufs.uqPtrDivBuf, cpuOutBufs.uqPtrSqrBuf};

    std::generate(uqPtrs.begin(), uqPtrs.end(), []() { return std::make_unique<cpuCuPhyBufR32_t>(); });
#endif    
}

// Factory to instantiate GPU buffers used in tests
void createGpuBufs(size_t        nElems,
                   gpuInBufs_t&  gpuInBufs,
                   gpuOutBufs_t& gpuOutBufs)
{
    gpuInBufs.uqPtrBuf1 = std::make_unique<gpuCuPhyBufR32_t>(nElems);
    gpuInBufs.uqPtrBuf2 = std::make_unique<gpuCuPhyBufR32_t>(nElems);

    gpuOutBufs.uqPtrAddBuf = std::make_unique<gpuCuPhyBufR32_t>(nElems);
    gpuOutBufs.uqPtrSubBuf = std::make_unique<gpuCuPhyBufR32_t>(nElems);
    gpuOutBufs.uqPtrMulBuf = std::make_unique<gpuCuPhyBufR32_t>(nElems);
    gpuOutBufs.uqPtrDivBuf = std::make_unique<gpuCuPhyBufR32_t>(nElems);
    gpuOutBufs.uqPtrSqrBuf = std::make_unique<gpuCuPhyBufR32_t>(nElems);   
}

void testInputGen(cpuInBufs_t& cpuInBufs)
{
    // Lambda to generate random numbers in the range [0, 1.0]
    auto randf = [](float &val){val = static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);};
    std::for_each(cpuInBufs.uqPtrBuf1->addr(), cpuInBufs.uqPtrBuf1->addr()+cpuInBufs.uqPtrBuf1->size(), randf);
    std::for_each(cpuInBufs.uqPtrBuf2->addr(), cpuInBufs.uqPtrBuf2->addr()+cpuInBufs.uqPtrBuf2->size(), randf);    
}

// Helper to set kernel node parameters
void setKernelNodePrms(void** kernelArgs, void* kernelFunc, dim3 const& blockDim, dim3 const& gridDim, CUDA_KERNEL_NODE_PARAMS& kernelNodePrmsDriver)
{    
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, kernelFunc));
    kernelNodePrmsDriver.blockDimX      = blockDim.x;
    kernelNodePrmsDriver.blockDimY      = blockDim.y;
    kernelNodePrmsDriver.blockDimZ      = blockDim.z;
    kernelNodePrmsDriver.gridDimX       = gridDim.x;
    kernelNodePrmsDriver.gridDimY       = gridDim.y;
    kernelNodePrmsDriver.gridDimZ       = gridDim.z;
    kernelNodePrmsDriver.kernelParams   = kernelArgs;
    kernelNodePrmsDriver.extra          = nullptr;
    kernelNodePrmsDriver.sharedMemBytes = 0;    
}


////////////////////////////////////////////////////////////////////////
// Helper to set Graph node parameters
void setGraphNode(cudaGraphExec_t graphExec, CUgraphNode node, void** kernelArgs, void* kernelFunc, dim3 const& blockDim, dim3 const& gridDim)
{
    CUDA_KERNEL_NODE_PARAMS kernelNodePrmsDriver;
    setKernelNodePrms(kernelArgs, kernelFunc, blockDim, gridDim, kernelNodePrmsDriver);
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, node, 1));

#ifdef ENABLE_MI_MEMTRACE        
    // auto saveCfg = mi_memtrace_get_config();
    // mi_memtrace_set_config(0);
#endif // ENABLE_MI_MEMTRACE    
    
    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(graphExec, node, &kernelNodePrmsDriver));

#ifdef ENABLE_MI_MEMTRACE        
   // mi_memtrace_set_config(saveCfg);
#endif // ENABLE_MI_MEMTRACE  
}

////////////////////////////////////////////////////////////////////////
// Test stream based work submission

// Helper to exercise critical path CUDA APIs used in stream based work submission
void criticalPathStreams(cudaStream_t                 cuStrm,
                         cudaStreamArr_t<NUM_NODES>&  cuStrms,
                         cudaEvent_t&                 cuEvent,
                         dim3                  const& blockDim,                         
                         dim3                  const& gridDim,
                         cpuInBufs_t const&           cpuInBufs,
                         cpuOutBufs_t&                cpuOutBufs,
                         gpuInBufs_t&                 gpuInBufs,
                         gpuOutBufs_t&                gpuOutBufs)
{
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrAddBuf->addr(), 0xFF, gpuOutBufs.uqPtrAddBuf->size()*sizeof(float), cuStrms[0]));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrSubBuf->addr(), 0xFF, gpuOutBufs.uqPtrSubBuf->size()*sizeof(float), cuStrms[1]));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrMulBuf->addr(), 0xFF, gpuOutBufs.uqPtrMulBuf->size()*sizeof(float), cuStrms[2]));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrDivBuf->addr(), 0xFF, gpuOutBufs.uqPtrDivBuf->size()*sizeof(float), cuStrms[3]));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrSqrBuf->addr(), 0xFF, gpuOutBufs.uqPtrSqrBuf->size()*sizeof(float), cuStrms[4]));

    CUDA_CHECK(cudaMemcpyAsync(gpuInBufs.uqPtrBuf1->addr(), cpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf1->size()*sizeof(float), cudaMemcpyHostToDevice, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(gpuInBufs.uqPtrBuf2->addr(), cpuInBufs.uqPtrBuf2->addr(), gpuInBufs.uqPtrBuf2->size()*sizeof(float), cudaMemcpyHostToDevice, cuStrm));
    CUDA_CHECK(cudaEventRecord(cuEvent, cuStrm));

    // Wait for input buffers to be copied before fanning out kernels on respective streams
    for(uint32_t i = 0; i < NUM_NODES; ++i)
    {
        CUDA_CHECK(cudaStreamWaitEvent(cuStrms[i], cuEvent, 0));
    }

    // CUDA_CHECK(cudaEventSynchronize(cuEvent));
#ifdef ENABLE_DRIVER_KERNEL_LAUNCH
    CUDA_KERNEL_NODE_PARAMS kernelNodePrmsDriver;

    auto len  = gpuInBufs.uqPtrBuf1->size();    
    auto pIn1 = gpuInBufs.uqPtrBuf1->addr();
    auto pIn2 = gpuInBufs.uqPtrBuf2->addr();
    auto pOutAdd = gpuOutBufs.uqPtrAddBuf->addr();
    std::array<void*, 4> kernelArgs = {(void*)(&len), (void*)(&pIn1), (void*)(&pIn2), (void*)(&pOutAdd)};

    setKernelNodePrms(kernelArgs.data(), reinterpret_cast<void*>(vecAddKernel), blockDim, gridDim, kernelNodePrmsDriver);    
    launch_kernel(kernelNodePrmsDriver, cuStrms[0]);

    auto pOutSub = gpuOutBufs.uqPtrSubBuf->addr();
    kernelArgs[3] = (void*)(&pOutSub);
    setKernelNodePrms(kernelArgs.data(), reinterpret_cast<void*>(vecSubKernel), blockDim, gridDim, kernelNodePrmsDriver);    
    launch_kernel(kernelNodePrmsDriver, cuStrms[1]);

    auto pOutMul = gpuOutBufs.uqPtrMulBuf->addr();
    kernelArgs[3] = (void*)(&pOutMul);
    setKernelNodePrms(kernelArgs.data(), reinterpret_cast<void*>(vecMulKernel), blockDim, gridDim, kernelNodePrmsDriver);    
    launch_kernel(kernelNodePrmsDriver, cuStrms[2]);

    auto pOutDiv = gpuOutBufs.uqPtrDivBuf->addr();
    kernelArgs[3] = (void*)(&pOutDiv);
    setKernelNodePrms(kernelArgs.data(), reinterpret_cast<void*>(vecDivKernel), blockDim, gridDim, kernelNodePrmsDriver);    
    launch_kernel(kernelNodePrmsDriver, cuStrms[3]);

    auto pOutSqr = gpuOutBufs.uqPtrSqrBuf->addr();
    kernelArgs[2] = (void*)(&pOutSqr);
    setKernelNodePrms(kernelArgs.data(), reinterpret_cast<void*>(vecSqrKernel), blockDim, gridDim, kernelNodePrmsDriver);    
    launch_kernel(kernelNodePrmsDriver, cuStrms[4]);
#else
    vecAddKernel<<<gridDim, blockDim, 0, cuStrms[0]>>>(gpuInBufs.uqPtrBuf1->size(), gpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf2->addr(), gpuOutBufs.uqPtrAddBuf->addr());
    vecSubKernel<<<gridDim, blockDim, 0, cuStrms[1]>>>(gpuInBufs.uqPtrBuf1->size(), gpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf2->addr(), gpuOutBufs.uqPtrSubBuf->addr());
    vecMulKernel<<<gridDim, blockDim, 0, cuStrms[2]>>>(gpuInBufs.uqPtrBuf1->size(), gpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf2->addr(), gpuOutBufs.uqPtrMulBuf->addr());
    vecDivKernel<<<gridDim, blockDim, 0, cuStrms[3]>>>(gpuInBufs.uqPtrBuf1->size(), gpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf2->addr(), gpuOutBufs.uqPtrDivBuf->addr());
    vecSqrKernel<<<gridDim, blockDim, 0, cuStrms[4]>>>(gpuInBufs.uqPtrBuf1->size(), gpuInBufs.uqPtrBuf1->addr(), gpuOutBufs.uqPtrSqrBuf->addr());
#endif

    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrAddBuf->addr(), gpuOutBufs.uqPtrAddBuf->addr(), cpuOutBufs.uqPtrAddBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrms[0]));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrSubBuf->addr(), gpuOutBufs.uqPtrSubBuf->addr(), cpuOutBufs.uqPtrSubBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrms[1]));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrMulBuf->addr(), gpuOutBufs.uqPtrMulBuf->addr(), cpuOutBufs.uqPtrMulBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrms[2]));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrDivBuf->addr(), gpuOutBufs.uqPtrDivBuf->addr(), cpuOutBufs.uqPtrDivBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrms[3]));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrSqrBuf->addr(), gpuOutBufs.uqPtrSqrBuf->addr(), cpuOutBufs.uqPtrSqrBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrms[4]));

    for(uint32_t i = 0; i < NUM_NODES; ++i)
    {
        CUDA_CHECK(cudaEventRecord(cuEvent, cuStrms[i]));
    }
    // Place a wait event on the main CUDA stream to wait for kernels and memcopies of results to complete
    CUDA_CHECK(cudaStreamWaitEvent(cuStrm, cuEvent, 0));
}


// Test stream based work submission

// Note: CUDA APIs tested in function criticalPathStreams:
// cudaMemsetAsync
// cudaEventRecord
// cudaMemcpyAsync
// cudaStreamWaitEvent
// cudaLaunchKernel
// cudaEventQuery
TEST(cudaApi, cudaStreams)
{
    static constexpr size_t NUM_ELEMS = 1024;

    // Test setup    
    cpuInBufs_t cpuInBufs{};
    cpuOutBufs_t cpuExpectedOutBufs{}, cpuOutBufs{};
    createCpuBufs(NUM_ELEMS, cpuInBufs, cpuExpectedOutBufs, cpuOutBufs);

    gpuInBufs_t gpuInBufs{};
    gpuOutBufs_t gpuOutBufs{};

    createGpuBufs(NUM_ELEMS, gpuInBufs, gpuOutBufs);
    
    static constexpr size_t N_CUDA_EVENTS = 1;
    cudaEventArr_t<N_CUDA_EVENTS> cuEventArr;
    cudaError_t cuEventStatusArr[N_CUDA_EVENTS];
    for(auto& cuEventArrElem : cuEventArr)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&cuEventArrElem, cudaEventDisableTiming));
    }
    
    cudaStream_t cuStrm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cuStrm, cudaStreamDefault));
    cudaStreamArr_t<NUM_NODES> cuStrmArr;        
    for(auto& cuStrmArrElem : cuStrmArr)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&cuStrmArrElem, cudaStreamDefault));
    }

    testInputGen(cpuInBufs);

    cudaDeviceSynchronize();

    static constexpr size_t THRD_BLK_SIZE = 128;
    dim3 gridDim = dim3((NUM_ELEMS + THRD_BLK_SIZE - 1)/ THRD_BLK_SIZE);
    dim3 blockDim = dim3(THRD_BLK_SIZE);

#ifdef ENABLE_MI_MEMTRACE        
    mi_memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif // ENABLE_MI_MEMTRACE

    // Start "Critical path"
    criticalPathStreams(cuStrm, cuStrmArr, cuEventArr[0], blockDim, gridDim, 
                        cpuInBufs, cpuOutBufs, gpuInBufs, gpuOutBufs);

    uint32_t i = 0;
    for(auto& cuEventArrElem : cuEventArr)
    {
        cuEventStatusArr[i++] = cudaEventQuery(cuEventArrElem);
    }
    // End "Critical path"

#ifdef ENABLE_MI_MEMTRACE    
    mi_memtrace_set_config(0);
#endif // ENABLE_MI_MEMTRACE    

    calcExpectedResults(cpuInBufs.uqPtrBuf1->size(), cpuInBufs, cpuExpectedOutBufs);

    // Wait for GPU to complete
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));
    for(auto& cuStrmArrElem : cuStrmArr)
    {
        CUDA_CHECK(cudaStreamSynchronize(cuStrmArrElem));
    }

    i = 0;
    for(auto& cuEventArrElem : cuEventArr)
    {
        printf("Event[%d] Status after work submission: %d\n", i, cuEventStatusArr[i]);
        i++;
    }

    // Check results
    checkResults(cpuOutBufs.uqPtrAddBuf->size(), cpuExpectedOutBufs, cpuOutBufs);

    // Cleanup
    for(auto& cuEventArrElem : cuEventArr)
    {
        CUDA_CHECK(cudaEventDestroy(cuEventArrElem));
    }
    CUDA_CHECK(cudaStreamDestroy(cuStrm));    
    for(auto& cuStrmArrElem : cuStrmArr)
    {
        CUDA_CHECK(cudaStreamDestroy(cuStrmArrElem));
    }
}

////////////////////////////////////////////////////////////////////////
// Test graph based work submission path

// Helper function to exercise critical path CUDA APIs graph based work submission
void criticalPathGraphs(cudaStream_t             cuStrm,
                         dim3             const& blockDim,                         
                         dim3             const& gridDim, 
                         cpuInBufs_t      const& cpuInBufs,
                         cudaGraphExec_t&        graphExec,
                         graphNodes_t&           graphNodes,
                         cpuOutBufs_t&           cpuOutBufs,
                         gpuInBufs_t&            gpuInBufs,
                         gpuOutBufs_t&           gpuOutBufs)
{
    printf("Entering criticalPathGraphs\n");
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrAddBuf->addr(), 0xFF, gpuOutBufs.uqPtrAddBuf->size()*sizeof(float), cuStrm));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrSubBuf->addr(), 0xFF, gpuOutBufs.uqPtrSubBuf->size()*sizeof(float), cuStrm));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrMulBuf->addr(), 0xFF, gpuOutBufs.uqPtrMulBuf->size()*sizeof(float), cuStrm));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrDivBuf->addr(), 0xFF, gpuOutBufs.uqPtrDivBuf->size()*sizeof(float), cuStrm));
    CUDA_CHECK(cudaMemsetAsync(gpuOutBufs.uqPtrSqrBuf->addr(), 0xFF, gpuOutBufs.uqPtrSqrBuf->size()*sizeof(float), cuStrm));

    CUDA_CHECK(cudaMemcpyAsync(gpuInBufs.uqPtrBuf1->addr(), cpuInBufs.uqPtrBuf1->addr(), gpuInBufs.uqPtrBuf1->size()*sizeof(float), cudaMemcpyHostToDevice, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(gpuInBufs.uqPtrBuf2->addr(), cpuInBufs.uqPtrBuf2->addr(), gpuInBufs.uqPtrBuf2->size()*sizeof(float), cudaMemcpyHostToDevice, cuStrm));
    // CUDA_CHECK(cudaEventSynchronize(cuEventArr[0]));

    // Disable all nodes
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.emptyRootNode, 1));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.addKernelNode, 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.subKernelNode, 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.mulKernelNode, 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.divKernelNode, 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, graphNodes.sqrKernelNode, 0));

    CUDA_CHECK_EXCEPTION(cudaGraphUpload(graphExec, cuStrm));
    CUDA_CHECK_EXCEPTION(cudaGraphLaunch(graphExec, cuStrm));

    // Initialize and enable nodes

    setGraphNode(graphExec, graphNodes.emptyRootNode, nullptr, reinterpret_cast<void*>(emptyKernel), blockDim, gridDim);

    auto len  = gpuInBufs.uqPtrBuf1->size();    
    auto pIn1 = gpuInBufs.uqPtrBuf1->addr();
    auto pIn2 = gpuInBufs.uqPtrBuf2->addr();
    auto pOutAdd = gpuOutBufs.uqPtrAddBuf->addr();
    std::array<void*, 4> kernelArgs = {(void*)(&len), (void*)(&pIn1), (void*)(&pIn2), (void*)(&pOutAdd)};

    setGraphNode(graphExec, graphNodes.addKernelNode, kernelArgs.data(), reinterpret_cast<void*>(vecAddKernel), blockDim, gridDim);

    auto pOutSub = gpuOutBufs.uqPtrSubBuf->addr();
    kernelArgs[3] = (void*)(&pOutSub);
    setGraphNode(graphExec, graphNodes.subKernelNode, kernelArgs.data(), reinterpret_cast<void*>(vecSubKernel), blockDim, gridDim);

    auto pOutMul = gpuOutBufs.uqPtrMulBuf->addr();
    kernelArgs[3] = (void*)(&pOutMul);
    setGraphNode(graphExec, graphNodes.mulKernelNode, kernelArgs.data(), reinterpret_cast<void*>(vecMulKernel), blockDim, gridDim);

    auto pOutDiv = gpuOutBufs.uqPtrDivBuf->addr();
    kernelArgs[3] = (void*)(&pOutDiv);
    setGraphNode(graphExec, graphNodes.divKernelNode, kernelArgs.data(), reinterpret_cast<void*>(vecDivKernel), blockDim, gridDim);

    auto pOutSqr = gpuOutBufs.uqPtrSqrBuf->addr();
    kernelArgs[2] = (void*)(&pOutSqr);
    setGraphNode(graphExec, graphNodes.sqrKernelNode, kernelArgs.data(), reinterpret_cast<void*>(vecSqrKernel), blockDim, gridDim);

    CUDA_CHECK_EXCEPTION(cudaGraphUpload(graphExec, cuStrm));
    CUDA_CHECK_EXCEPTION(cudaGraphLaunch(graphExec, cuStrm));

    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrAddBuf->addr(), gpuOutBufs.uqPtrAddBuf->addr(), cpuOutBufs.uqPtrAddBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrSubBuf->addr(), gpuOutBufs.uqPtrSubBuf->addr(), cpuOutBufs.uqPtrSubBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrMulBuf->addr(), gpuOutBufs.uqPtrMulBuf->addr(), cpuOutBufs.uqPtrMulBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrDivBuf->addr(), gpuOutBufs.uqPtrDivBuf->addr(), cpuOutBufs.uqPtrDivBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrm));
    CUDA_CHECK(cudaMemcpyAsync(cpuOutBufs.uqPtrSqrBuf->addr(), gpuOutBufs.uqPtrSqrBuf->addr(), cpuOutBufs.uqPtrSqrBuf->size()*sizeof(float), cudaMemcpyDeviceToHost, cuStrm));
    printf("Exiting criticalPathGraphs\n");
}

// Helper function to create a Graph executable
void createGraphExec(graphNodes_t&           graphNodes,
                     cudaGraph_t&            graph,
                     cudaGraphExec_t&        graphExec,
                     gpuInBufs_t&            gpuInBufs,
                     gpuOutBufs_t&           gpuOutBufs)                     
{
    printf("Entering createGraphExec\n");
    std::vector<CUgraphNode> currNodeDeps, nextNodeDeps;

    // Initialize empty node and use it as a root
    CUDA_KERNEL_NODE_PARAMS kernelNodePrmsDriver;
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(emptyKernel)));
    kernelNodePrmsDriver.blockDimX      = 32;
    kernelNodePrmsDriver.blockDimY      = 1;
    kernelNodePrmsDriver.blockDimZ      = 1;
    kernelNodePrmsDriver.gridDimX       = 1;
    kernelNodePrmsDriver.gridDimY       = 1;
    kernelNodePrmsDriver.gridDimZ       = 1;
    kernelNodePrmsDriver.kernelParams   = nullptr;
    kernelNodePrmsDriver.extra          = nullptr;
    kernelNodePrmsDriver.sharedMemBytes = 0;
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.emptyRootNode, graph, currNodeDeps.data(), currNodeDeps.size(), &kernelNodePrmsDriver));
    currNodeDeps.emplace_back(graphNodes.emptyRootNode);

    auto len  = gpuInBufs.uqPtrBuf1->size();    
    auto pIn1 = gpuInBufs.uqPtrBuf1->addr();
    auto pIn2 = gpuInBufs.uqPtrBuf2->addr();
    auto pOutAdd = gpuOutBufs.uqPtrAddBuf->addr();
    std::array<void*, 4> kernelArgs = {(void*)(&len), (void*)(&pIn1), (void*)(&pIn2), (void*)(&pOutAdd)};

    // fan-out Add, Sub kernel nodes from root empty node
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(vecAddKernel)));
    kernelNodePrmsDriver.kernelParams = kernelArgs.data();
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.addKernelNode, graph, currNodeDeps.data(), currNodeDeps.size(), &(kernelNodePrmsDriver)));
    nextNodeDeps.emplace_back(graphNodes.addKernelNode);

    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(vecSubKernel)));
    auto pOutSub = gpuOutBufs.uqPtrSubBuf->addr();
    kernelArgs[3] = (void*)(&pOutSub);
    kernelNodePrmsDriver.kernelParams = kernelArgs.data();
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.subKernelNode, graph, currNodeDeps.data(), currNodeDeps.size(), &(kernelNodePrmsDriver)));
    nextNodeDeps.emplace_back(graphNodes.subKernelNode);

    currNodeDeps.clear();
    currNodeDeps = nextNodeDeps;
    nextNodeDeps.clear();

    // fan-out Mul, Div kernel nodes after Add, Sub kernel nodes complete
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(vecMulKernel)));
    auto pOutMul = gpuOutBufs.uqPtrMulBuf->addr();
    kernelArgs[3] = (void*)(&pOutMul);
    kernelNodePrmsDriver.kernelParams = kernelArgs.data();    
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.mulKernelNode, graph, currNodeDeps.data(), currNodeDeps.size(), &(kernelNodePrmsDriver)));
    nextNodeDeps.emplace_back(graphNodes.mulKernelNode);
    
    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(vecDivKernel)));
    auto pOutDiv = gpuOutBufs.uqPtrDivBuf->addr();
    kernelArgs[3] = (void*)(&pOutDiv);
    kernelNodePrmsDriver.kernelParams = kernelArgs.data();       
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.divKernelNode, graph, currNodeDeps.data(), currNodeDeps.size(), &(kernelNodePrmsDriver)));
    nextNodeDeps.emplace_back(graphNodes.divKernelNode);

    currNodeDeps.clear();
    currNodeDeps = nextNodeDeps;
    nextNodeDeps.clear();

    CUDA_CHECK_EXCEPTION(cudaGetFuncBySymbol(&kernelNodePrmsDriver.func, reinterpret_cast<void*>(vecSqrKernel)));
    auto pOutSqr = gpuOutBufs.uqPtrSqrBuf->addr();
    kernelArgs[2] = (void*)(&pOutSqr);
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&graphNodes.sqrKernelNode, graph, currNodeDeps.data(), currNodeDeps.size(), &(kernelNodePrmsDriver)));    

    CUDA_CHECK_EXCEPTION(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("Exiting createGraphExec\n");
}

// Test graph based work submission

// CUDA APIs tested in function criticalPathGraphs:
// cudaGetFuncBySymbol
// cuGraphExecKernelNodeSetParams
// cuGraphLaunch
// cuGraphNodeSetEnabled
// cuGraphUpload
TEST(cudaApi, cudaGraphs)
{
    static constexpr size_t NUM_ELEMS = 1024;

    // Test setup
    cpuInBufs_t cpuInBufs{};
    cpuOutBufs_t cpuExpectedOutBufs{}, cpuOutBufs{};
    createCpuBufs(NUM_ELEMS, cpuInBufs, cpuExpectedOutBufs, cpuOutBufs);

    gpuInBufs_t gpuInBufs{};
    gpuOutBufs_t gpuOutBufs{};

    createGpuBufs(NUM_ELEMS, gpuInBufs, gpuOutBufs);

    cudaStream_t cuStrm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cuStrm, cudaStreamDefault));
  
    graphNodes_t graphNodes;

    cudaGraph_t graph;    
    CUDA_CHECK_EXCEPTION(cudaGraphCreate(&graph, 0));

    cudaGraphExec_t graphExec;
    createGraphExec(graphNodes, graph, graphExec, gpuInBufs, gpuOutBufs);

    testInputGen(cpuInBufs);

    cudaDeviceSynchronize();

    static constexpr size_t THRD_BLK_SIZE = 128;
    dim3 gridDim = dim3((NUM_ELEMS + THRD_BLK_SIZE - 1)/ THRD_BLK_SIZE);
    dim3 blockDim = dim3(THRD_BLK_SIZE);

#ifdef ENABLE_MI_MEMTRACE        
    mi_memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif // ENABLE_MI_MEMTRACE

    // Start "Critical path"
    criticalPathGraphs(cuStrm, blockDim, gridDim, cpuInBufs, graphExec, graphNodes, cpuOutBufs, gpuInBufs, gpuOutBufs);
    // End "Critical path"

#ifdef ENABLE_MI_MEMTRACE    
    mi_memtrace_set_config(0);
#endif // ENABLE_MI_MEMTRACE    

    calcExpectedResults(cpuInBufs.uqPtrBuf1->size(), cpuInBufs, cpuExpectedOutBufs);

    // Wait for GPU to complete
    CUDA_CHECK(cudaStreamSynchronize(cuStrm));

    // Check results
    checkResults(cpuOutBufs.uqPtrAddBuf->size(), cpuExpectedOutBufs, cpuOutBufs);

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(cuStrm));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
