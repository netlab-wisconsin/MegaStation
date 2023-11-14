/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <cuda_runtime.h>
#include <stdio.h>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "descrambling.cuh"
#include "descrambling.hpp"
#include "crc.hpp"
using namespace cuphy_i;
using namespace crc;

namespace descrambling
{


// KERNEL for descrambling

__global__ void descrambleKernel(float*          llrs,
                                 uint32_t        size,
                                 const uint32_t* tbBoundaryArray,
                                 const uint32_t* cinitArray)
{
    extern __shared__ uint32_t sharedSeq[];

    int      tid      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t myTBBase = tbBoundaryArray[blockIdx.y];
    uint32_t myTBEnd  = tbBoundaryArray[blockIdx.y + 1];
    uint32_t myCinit  = cinitArray[blockIdx.y];

    uint32_t seq;

    uint32_t blockEnd = myTBEnd + ((blockDim.x - myTBEnd % blockDim.x) % blockDim.x);

    // apply descrambling (sign change)
    for(int t = tid + myTBBase; t < blockEnd; t += blockDim.x * gridDim.x)
    {
        __syncthreads();
        if(threadIdx.x < blockDim.x / WARP_SIZE)
            sharedSeq[threadIdx.x] = gold32(myCinit, t + threadIdx.x * WARP_SIZE - myTBBase);

        __syncthreads();

        if(t < myTBEnd)
        { // end of transport block guard
            seq         = sharedSeq[threadIdx.x / WARP_SIZE];
            uint32_t s  = (seq >> (threadIdx.x & (WARP_SIZE - 1))) & 1; // modulo warp size
            uint32_t sn = (s + 1) & 0x1;
            // change sign based on scrambling sequence bit
            llrs[t] = -llrs[t] * s + llrs[t] * sn;
        }
    }
}
// cuphyDescramble class

class cuphyDescramble //
{
public:
    cuphyDescramble() :
        d_llrs_(nullptr),
        d_tbBoundaryArray_(nullptr),
        d_cinitArray_(nullptr),
        nTBs_(0),
        maxNCodeBlocks_(0),
        totalSize_(0){};

    cuphyStatus_t loadParams(const uint32_t* tbBoundaryArray,
                             const uint32_t* cinitArray,
                             uint32_t        nTBs,
                             uint32_t        maxNCodeBlocks);
    cuphyStatus_t loadInput(float* llrs);

    cuphyStatus_t launch(float*       llrs   = nullptr,
                         bool         timeIt = false,
                         uint32_t     NRUNS  = 10000,
                         cudaStream_t strm   = 0);

    cuphyStatus_t storeOutput(float* llrs);

    void cleanup();

private:
    unique_device_ptr<float>    d_llrs_;
    unique_device_ptr<uint32_t> d_tbBoundaryArray_;
    unique_device_ptr<uint32_t> d_cinitArray_;
    uint32_t                    maxNCodeBlocks_;
    uint32_t                    nTBs_;
    uint32_t                    totalSize_;
};

cuphyStatus_t cuphyDescramble::loadInput(float* llrs)
{
    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;

    d_llrs_ = make_unique_device<float>(totalSize_);
    if(d_llrs_ != nullptr)
    {
        CUDA_CHECK(cudaMemcpy(d_llrs_.get(),
                              llrs,
                              sizeof(float) * totalSize_,
                              cudaMemcpyHostToDevice));
    }
    else
    {
        status = CUPHY_STATUS_ALLOC_FAILED;
    }

    return status;
}

cuphyStatus_t cuphyDescramble::loadParams(const uint32_t* tbBoundaryArray,
                                          const uint32_t* cinitArray,
                                          uint32_t        nTBs,
                                          uint32_t        maxNCodeBlocks)
{
    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
    nTBs_                = nTBs;
    maxNCodeBlocks_      = maxNCodeBlocks;
    totalSize_           = tbBoundaryArray[nTBs];

    d_tbBoundaryArray_ = make_unique_device<uint32_t>(nTBs + 1);
    d_cinitArray_      = make_unique_device<uint32_t>(nTBs);

    CUDA_CHECK(cudaMemcpy(d_tbBoundaryArray_.get(),
                          tbBoundaryArray,
                          sizeof(uint32_t) * (nTBs + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cinitArray_.get(),
                          cinitArray,
                          sizeof(uint32_t) * nTBs,
                          cudaMemcpyHostToDevice));

    return status;
}

cuphyStatus_t cuphyDescramble::launch(float*       d_llrs_i,
                                      bool         timeIt,
                                      uint32_t     NRUNS,
                                      cudaStream_t strm)
{
    const int blockSize = GLOBAL_BLOCK_SIZE;
    int       gridSizeX = maxNCodeBlocks_;
    int       gridSizeY = nTBs_;
    dim3      gridSize(gridSizeX, gridSizeY);

    float* d_input = (d_llrs_i == nullptr) ? d_llrs_.get() : d_llrs_i;

    descrambleKernel<<<gridSize, blockSize, (blockSize / WARP_SIZE) * sizeof(uint32_t), strm>>>(d_input,
                                                                                                totalSize_,
                                                                                                d_tbBoundaryArray_.get(),
                                                                                                d_cinitArray_.get());

    if(timeIt)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time1 = 0.0;
        cudaEventRecord(start);

        for(int i = 0; i < NRUNS; i++)
        {
            descrambleKernel<<<gridSize, blockSize, (blockSize / WARP_SIZE) * sizeof(uint32_t), strm>>>(d_input,
                                                                                                        totalSize_,
                                                                                                        d_tbBoundaryArray_.get(),
                                                                                                        d_cinitArray_.get());
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time1, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        time1 /= NRUNS;

        printf(
            "Descramble Kernel"
            "\n %.2f us",
            time1 * 1000);
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyDescramble::storeOutput(float* llrs)
{
    CUDA_CHECK(cudaMemcpy(llrs,
                          d_llrs_.get(),
                          sizeof(float) * totalSize_,
                          cudaMemcpyDeviceToHost));

    return CUPHY_STATUS_SUCCESS;
}

void cuphyDescramble::cleanup()
{
    d_tbBoundaryArray_ = nullptr;

    d_cinitArray_ = nullptr;

    d_llrs_ = nullptr;
}

} // namespace descrambling

void cuphyDescrambleInit(void** descrambleEnv)
{
    descrambling::cuphyDescramble* descramblePtr = new descrambling::cuphyDescramble();
    *descrambleEnv                               = descramblePtr;
}

void cuphyDescrambleCleanUp(void** descrambleEnv)
{
    delete static_cast<descrambling::cuphyDescramble*>((*descrambleEnv));
}

cuphyStatus_t cuphyDescrambleLoadParams(void**          descrambleEnv,
                                        uint32_t        nTBs,
                                        uint32_t        maxNCodeBlocks,
                                        const uint32_t* tbBoundaryArray,
                                        const uint32_t* cinitArray)
{
    descrambling::cuphyDescramble* descramblePtr = static_cast<descrambling::cuphyDescramble*>(*descrambleEnv);
    cuphyStatus_t                  status        = descramblePtr->loadParams(tbBoundaryArray,
                                                     cinitArray,
                                                     nTBs,
                                                     maxNCodeBlocks);
    return status;
}

cuphyStatus_t cuphyDescrambleLoadInput(void** descrambleEnv, float* llrs)

{
    descrambling::cuphyDescramble* descramblePtr = static_cast<descrambling::cuphyDescramble*>(*descrambleEnv);
    cuphyStatus_t                  status        = descramblePtr->loadInput(llrs);
    return status;
}

cuphyStatus_t cuphyDescramble(void** descrambleEnv, float* d_llrs, bool timeIt, uint32_t NRUNS, cudaStream_t strm)
{
    descrambling::cuphyDescramble* descramblePtr = static_cast<descrambling::cuphyDescramble*>(*descrambleEnv);

    cuphyStatus_t status = descramblePtr->launch(d_llrs, timeIt, NRUNS, strm);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyDescrambleStoreOutput(void** descrambleEnv, float* llrs)
{
    descrambling::cuphyDescramble* descramblePtr = static_cast<descrambling::cuphyDescramble*>(*descrambleEnv);
    cuphyStatus_t                  status        = descramblePtr->storeOutput(llrs);

    return status;
}

cuphyStatus_t cuphyDescrambleAllParams(float*          llrs,
                                       const uint32_t* tbBoundaryArray,
                                       const uint32_t* cinitArray,
                                       uint32_t        nTBs,
                                       uint32_t        maxNCodeBlocks,
                                       int             timeIt,
                                       uint32_t        NRUNS,
                                       cudaStream_t    stream)
{
    descrambling::cuphyDescramble descramble;

    cuphyStatus_t status = descramble.loadParams(tbBoundaryArray, cinitArray, nTBs, maxNCodeBlocks);

    if(status != CUPHY_STATUS_SUCCESS) return status;

    descramble.loadInput(llrs);

    if(status != CUPHY_STATUS_SUCCESS) return status;

    status = descramble.launch(nullptr, timeIt, NRUNS, stream);

    if(status != CUPHY_STATUS_SUCCESS) return status;

    status = descramble.storeOutput(llrs);

    return CUPHY_STATUS_SUCCESS;
}
