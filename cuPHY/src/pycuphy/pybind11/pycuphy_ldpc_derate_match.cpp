/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>
#include <bitset>
#include <map>

#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;

using namespace std::complex_literals;
namespace py = pybind11;

#ifndef AERIAL_PYTHON_LDPC_DERATE_MATCH_CPP
#define AERIAL_PYTHON_LDPC_DERATE_MATCH_CPP

typedef struct {
    void *              inputDevicePtr;     // Input Device (GPU) Memory Pointer
    void *              outputHostPtr;      // Output Host (CPU) Memory Pointer

    bool                scrambling;         // Whether to descramble
    uint16_t            numIterations;      // Number of iterations

} cuphyLdpcDeRateMatchPrms_t;


namespace pycuphy {

class LdpcDerateMatch {
    public:
        LdpcDerateMatch(const uint64_t inputDevicePtr,
                        const uint64_t outputHostPtr,
                        const bool scrambling);

        py::array_t<float> derateMatch(const py::array& inputData,
                                       const uint32_t tbSize,
                                       const float codeRate,
                                       const uint32_t rateMatchLen,
                                       const uint8_t qamMod,
                                       const uint32_t rv,
                                       const uint32_t ndi,
                                       const uint32_t cinit,
                                       uint64_t cuStream);

        void destroy();

        void setProfilingIterations(const uint32_t numIterations);

    private:
        cuphyLdpcDeRateMatchPrms_t ldpcDeRateMatchPrms;
        cuphyPuschRxRateMatchHndl_t puschRmHndl;

        uint32_t getNumCodeBlocks(const uint32_t tbSize, const uint32_t baseGraph);

        // Only one TB/UE supported here. Also only one layer.
        const uint32_t nUes = 1;
        const uint32_t numLayers = 1;
        const uint16_t nUeGrps = 1;
        const uint32_t nMaxTbs = 1;

};


uint32_t LdpcDerateMatch::getNumCodeBlocks(const uint32_t tbSize, const uint32_t baseGraph) {
    uint32_t K_cb = (baseGraph == 1) ? 8448 : 3840;
    uint32_t tbSizeWithCrc = tbSize + compute_TB_CRC(tbSize);
    uint32_t C = 1;
    if (tbSizeWithCrc > K_cb) { // The TB will be segmented into multiple CBs.
        uint32_t L = 24;
        C = ceil((tbSizeWithCrc * 1.0f) / (K_cb - L));
    }
    return C;
}


LdpcDerateMatch::LdpcDerateMatch(const uint64_t inputDevicePtr,
                                 const uint64_t outputHostPtr,
                                 bool scrambling) {
    // Floating point config:
    // 0: FP32 in, FP32 out
    // 1: FP16 in, FP32 out
    // 2: FP32 in, FP16 out
    // 3: FP16 in, FP16 out
    // other values: invalid
    int fpConfig = 0;

    ldpcDeRateMatchPrms.inputDevicePtr = (void *)inputDevicePtr;
    ldpcDeRateMatchPrms.outputHostPtr = (void *)outputHostPtr;

    ldpcDeRateMatchPrms.scrambling = scrambling;

    ldpcDeRateMatchPrms.numIterations = 0;

    // Create the PUSCH rate match object.
    cuphyCreatePuschRxRateMatch(&puschRmHndl, fpConfig, (int)scrambling);
}


void LdpcDerateMatch::setProfilingIterations(const uint32_t numIterations) {
    ldpcDeRateMatchPrms.numIterations = numIterations;
}


py::array_t<float> LdpcDerateMatch::derateMatch(const py::array& inputData,
                                                const uint32_t tbSize,
                                                const float codeRate,
                                                const uint32_t rateMatchLen,
                                                const uint8_t qamMod,
                                                const uint32_t rv,
                                                const uint32_t ndi,
                                                const uint32_t cinit,
                                                uint64_t cuStream) {
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> tbPrmsCpu(nUes);

    tbPrmsCpu[0].codeRate = codeRate;           // Code rate.
    tbPrmsCpu[0].Qm = qamMod;                   // Modulation order per TB: [2, 4, 6, 8].
    tbPrmsCpu[0].ndi = ndi;                     // Indicates if this is new data or a retransmission, 0=retransmission, 1=new data
    tbPrmsCpu[0].rv = rv;                       // Redundancy version per TB, one of {0, 1, 2, 3}.
    // Base graph per TB; options are 1 or 2.
    tbPrmsCpu[0].bg = get_base_graph(codeRate, tbSize);
    tbPrmsCpu[0].Nl = numLayers;                // Number of transmission layers per TB.
    // Number of code blocks (CBs) per TB.
    tbPrmsCpu[0].num_CBs = getNumCodeBlocks(tbSize, tbPrmsCpu[0].bg);
    uint32_t Kprime = get_K_prime(tbSize, tbPrmsCpu[0].bg, tbPrmsCpu[0].num_CBs);
    // Lifting size.
    tbPrmsCpu[0].Zc = get_lifting_size(tbSize, tbPrmsCpu[0].bg, Kprime);
    // Number of bits in a code block.
    tbPrmsCpu[0].N = (tbPrmsCpu[0].bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * tbPrmsCpu[0].Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * tbPrmsCpu[0].Zc;
    tbPrmsCpu[0].Ncb = tbPrmsCpu[0].N;          // Same as N for now
    // Ncb w/ padding for LDPC decoder alignment requirements.
    tbPrmsCpu[0].Ncb_padded = (tbPrmsCpu[0].N + 2 * tbPrmsCpu[0].Zc + 7) / 8;
    tbPrmsCpu[0].Ncb_padded *= 8;
    // Number of rate-matched bits available for TB transmission.
    tbPrmsCpu[0].G = rateMatchLen;
    // Non-punctured systematic bits.
    tbPrmsCpu[0].K = (tbPrmsCpu[0].bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * tbPrmsCpu[0].Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * tbPrmsCpu[0].Zc;
    // Filler bits.
    tbPrmsCpu[0].F = tbPrmsCpu[0].K - Kprime;
    tbPrmsCpu[0].cinit = cinit;                 // Used to generate scrambling sequence; seed2 arg. of gold32
    // Number of data bytes in transport block (no CRCs).
    tbPrmsCpu[0].nDataBytes = tbSize / 8;
    // Number of zero padded encoded bits per codeblock (input to LDPC decoder).
    tbPrmsCpu[0].nZpBitsPerCb = 0;
    tbPrmsCpu[0].firstCodeBlockIndex = 0;       // For symbol-by-symbol processing
    tbPrmsCpu[0].encodedSize = tbPrmsCpu[0].G;
    tbPrmsCpu[0].layer_map_array[0] = 0;        // First Nl elements of array specify the layer(s) this TB maps to.
    tbPrmsCpu[0].userGroupIndex = 0;            // User group/cell index.
    tbPrmsCpu[0].nBBULayers = 1;                // Number of BBU layers for current user group/cell.
    tbPrmsCpu[0].mScUciSum = 0;                 // Total number of REs available for UCI transmission.
    tbPrmsCpu[0].isDataPresent = 1;             // Bit0 = 1 in pduBitmap, if data is present.
    tbPrmsCpu[0].uciOnPuschFlag = 0;            // Indicates if UCI on PUSCH.
    tbPrmsCpu[0].csi2Flag = 0;                  // Indicates if CSI2 present.

    // Copy to GPU.
    cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu(nUes);
    cudaMemcpyAsync(tbPrmsGpu.addr(),
                    tbPrmsCpu.addr(),
                    sizeof(PerTbParams) * nUes,
                    cudaMemcpyHostToDevice,
                    (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    const PerTbParams* pTbPrmsCpu = tbPrmsCpu.addr();
    const PerTbParams* pTbPrmsGpu = tbPrmsGpu.addr();

    // Convert the input Numpy array to a device tensor.
    cuphy::tensor_device inputDataTensor = deviceFromNumpy<float>(inputData,
                                                                  ldpcDeRateMatchPrms.inputDevicePtr,
                                                                  CUPHY_R_32F,
                                                                  CUPHY_R_32F,
                                                                  tensor_flags::align_tight,
                                                                  (cudaStream_t)cuStream);

    std::vector<cuphyTensorPrm_t> tPrmEqOutLLRsVec(nUeGrps);
    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx) {
        tPrmEqOutLLRsVec[ueGrpIdx].desc = inputDataTensor.desc().handle();
        tPrmEqOutLLRsVec[ueGrpIdx].pAddr = inputDataTensor.addr();
    }
    cuphyTensorPrm_t* pTPrmRmIn = tPrmEqOutLLRsVec.data();

    uint16_t nSchUes = 0;
    std::vector<uint16_t> schUserIdxsVec(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        if(pTbPrmsCpu[ueIdx].isDataPresent) {
            schUserIdxsVec[nSchUes] = ueIdx;
            nSchUes++;
        }
    }

    // Output buffers.
    uint32_t NUM_BYTES_PER_LLR = 4;
    uint32_t maxBytesRateMatch = NUM_BYTES_PER_LLR * nMaxTbs * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_RM_LLRS_PER_CB;
    cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(maxBytesRateMatch);

    void** ppRmOut;
    cudaError_t status = cudaHostAlloc(&ppRmOut, sizeof(uint8_t*) * nMaxTbs, cudaHostAllocPortable | cudaHostAllocMapped);
    if (status != cudaSuccess) {
        throw std::runtime_error("LdpcDerateMatch::derateMatch: Failure with cudaHostAlloc!");
    }

    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        size_t nBytesDeRm = NUM_BYTES_PER_LLR * tbPrmsCpu[ueIdx].Ncb_padded * tbPrmsCpu[ueIdx].num_CBs;
        ppRmOut[ueIdx] = linearAlloc.alloc(nBytesDeRm);
    }

    // PUSCH rate match descriptors.
    // Descriptors hold kernel parameters in GPU.
    size_t dynDescrSizeBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRateMatchGetDescrInfo(&dynDescrSizeBytes,
                                                                             &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcDerateMatch::derateMatch: Failed to get descriptor info!");
    }

    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

    // Setup PUSCH rate match object.
    // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
    cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;
    // Setup function populates dynamic descriptor and launch config
    bool enableCpuToGpuDescrAsyncCpy = false;

    cuphyStatus_t puschRmSetupStatus = cuphySetupPuschRxRateMatch(puschRmHndl,
                                                                  nSchUes,
                                                                  schUserIdxsVec.data(),
                                                                  pTbPrmsCpu,
                                                                  pTbPrmsGpu,
                                                                  pTPrmRmIn,
                                                                  pTPrmRmIn,
                                                                  ppRmOut,
                                                                  dynDescrBufCpu.addr(),
                                                                  dynDescrBufGpu.addr(),
                                                                  enableCpuToGpuDescrAsyncCpy,
                                                                  &puschRmLaunchCfg,
                                                                  (cudaStream_t)cuStream);
    if(puschRmSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcDerateMatch::derateMatch: Setup failed!");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
    }

    // Run PUSCH rate match.
    // Launch kernel using the CUDA driver API.
    CUresult r;
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = puschRmLaunchCfg.kernelNodeParamsDriver;
    if(!ldpcDeRateMatchPrms.numIterations) {
        r = cuLaunchKernel(kernelNodeParamsDriver.func,
                           kernelNodeParamsDriver.gridDimX,
                           kernelNodeParamsDriver.gridDimY,
                           kernelNodeParamsDriver.gridDimZ,
                           kernelNodeParamsDriver.blockDimX,
                           kernelNodeParamsDriver.blockDimY,
                           kernelNodeParamsDriver.blockDimZ,
                           kernelNodeParamsDriver.sharedMemBytes,
                           static_cast<CUstream>((cudaStream_t)cuStream),
                           kernelNodeParamsDriver.kernelParams,
                           kernelNodeParamsDriver.extra);
        if(r != CUDA_SUCCESS) {
            throw std::runtime_error("LdpcDerateMatch::derateMatch: Kernel launch failed!");
        }
    }
    else {

        cudaEvent_t start, stop;
        float time = 0.0f;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < ldpcDeRateMatchPrms.numIterations; iter++) {

            r = cuLaunchKernel(kernelNodeParamsDriver.func,
                               kernelNodeParamsDriver.gridDimX,
                               kernelNodeParamsDriver.gridDimY,
                               kernelNodeParamsDriver.gridDimZ,
                               kernelNodeParamsDriver.blockDimX,
                               kernelNodeParamsDriver.blockDimY,
                               kernelNodeParamsDriver.blockDimZ,
                               kernelNodeParamsDriver.sharedMemBytes,
                               static_cast<CUstream>((cudaStream_t)cuStream),
                               kernelNodeParamsDriver.kernelParams,
                               kernelNodeParamsDriver.extra);
            if(r != CUDA_SUCCESS) {
                throw std::runtime_error("LdpcDerateMatch::derateMatch: Kernel launch failed!");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double int_tput = ((double)rateMatchLen * (double)ldpcDeRateMatchPrms.numIterations) / ((double)1000 * (double)time);
        std::cout << "Total time from C++ is " << time * 1000 << " us." << std::endl;
        std::cout << "Internal throughput is " << int_tput / 1000 << " Gbps." << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Convert to host memory and return the Numpy array.
    uint32_t dim0 = tbPrmsCpu[0].Ncb_padded;
    uint32_t dim1 = tbPrmsCpu[0].num_CBs;
    cuphy::tensor_device dOutputTensor = cuphy::tensor_device(ppRmOut[0], CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(ldpcDeRateMatchPrms.outputHostPtr, CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        ppRmOut[ueIdx] = nullptr;
    }
    cudaError_t cudaFreeStatus = cudaFreeHost(ppRmOut);
    if (cudaFreeStatus != cudaSuccess) {
        throw std::runtime_error("LdpcDerateMatch::destroy: Failure with cudaFreeHost!");
    }

    // Create the Numpy array for output.
    return py::array_t<float>(
        {tbPrmsCpu[0].N + 2 * tbPrmsCpu[0].Zc, dim1},  // Shape
        {sizeof(float), sizeof(float) * tbPrmsCpu[0].Ncb_padded},  // Strides (in bytes) for each index
        (float*)ldpcDeRateMatchPrms.outputHostPtr
    );
}


void LdpcDerateMatch::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxRateMatch(puschRmHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcDerateMatch::destroy: Failed to destroy the rate matching object!");
    }
}


}

#endif
