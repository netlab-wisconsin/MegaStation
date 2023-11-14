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
#include "utils.cuh"
#include "pycuphy_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace cuphy;

using namespace std::complex_literals;
namespace py = pybind11;

#ifndef AERIAL_PYTHON_LDPC_RATE_MATCH_CPP
#define AERIAL_PYTHON_LDPC_RATE_MATCH_CPP

typedef struct {
    void *                inputDevicePtr;     // Input device (GPU) memory pointer
    void *                outputDevicePtr;    // Output device (GPU) memory pointer
    void *                inputHostPtr;       // Input host (CPU) memory pointer
    void *                tempOutputHostPtr;  // Temporary output host (CPU) memory pointer
    void *                outputHostPtr;      // Output host (CPU) memory pointer

    PdschPerTbParams      tbParams;           // Transport block parameters

    bool                  scrambling;
    bool                  layerMapping;

    uint32_t              numIterations;      // Number of iterations

} cuphyLdpcRateMatchPrms_t;


namespace pycuphy {

class LdpcRateMatch {
    public:

        LdpcRateMatch(const uint64_t inputDevicePtr,
                      const uint64_t outputDevicePtr,
                      const uint64_t inputHostPtr,
                      const uint64_t tempOutputHostPtr,
                      const uint64_t outputHostPtr,
                      const bool scrambling);

        py::array_t<float> rateMatch(const py::array& inputBits,
                                     const uint32_t tbSize,
                                     const float codeRate,
                                     const uint32_t rateMatchLen,
                                     const uint8_t qamMod,
                                     const uint8_t rv,
                                     const uint32_t cinit,
                                     uint64_t cuStream);

        void setProfilingIterations(const uint32_t numIterations);

    private:
        cuphyLdpcRateMatchPrms_t ldpcRateMatchPrms;

        void setTransportBlockParams(const uint32_t tbSize,
                                     const float codeRate,
                                     const uint32_t rateMatchLen,
                                     const uint8_t qamMod,
                                     const uint32_t numCodeBlocks,
                                     const uint32_t numCodedBits,
                                     const uint8_t rv,
                                     const uint8_t numLayers,
                                     const uint32_t cinit);
};


void LdpcRateMatch::setTransportBlockParams(const uint32_t tbSize,
                                            const float codeRate,
                                            const uint32_t rateMatchLen,
                                            const uint8_t qamMod,
                                            const uint32_t numCodeBlocks,
                                            const uint32_t numCodedBits,
                                            const uint8_t rv,
                                            const uint8_t numLayers,
                                            const uint32_t cinit) {

    ldpcRateMatchPrms.tbParams.tbStartAddr = 0;
    ldpcRateMatchPrms.tbParams.tbStartOffset = 0;
    ldpcRateMatchPrms.tbParams.cumulativeTbSizePadding = 0;

    ldpcRateMatchPrms.tbParams.tbSize = tbSize / 8;
    ldpcRateMatchPrms.tbParams.bg = get_base_graph(codeRate, tbSize);
    uint32_t tmp_numCodeBlocks = numCodeBlocks; // Assuming numCodeBlocks is non zero and so won't be modified by Kprime
    uint16_t Kprime = get_K_prime(tbSize, ldpcRateMatchPrms.tbParams.bg, tmp_numCodeBlocks);
    ldpcRateMatchPrms.tbParams.Zc = get_lifting_size(tbSize, ldpcRateMatchPrms.tbParams.bg, Kprime);
    ldpcRateMatchPrms.tbParams.K = (ldpcRateMatchPrms.tbParams.bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * ldpcRateMatchPrms.tbParams.Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * ldpcRateMatchPrms.tbParams.Zc;
    ldpcRateMatchPrms.tbParams.F = ldpcRateMatchPrms.tbParams.K - Kprime;
    ldpcRateMatchPrms.tbParams.N = (ldpcRateMatchPrms.tbParams.bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * ldpcRateMatchPrms.tbParams.Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * ldpcRateMatchPrms.tbParams.Zc;
    if(ldpcRateMatchPrms.tbParams.N != numCodedBits) {  // Just a check as these should match.
        throw std::runtime_error("Invalid number of coded bits!");
    }

    ldpcRateMatchPrms.tbParams.Ncb = ldpcRateMatchPrms.tbParams.N;
    ldpcRateMatchPrms.tbParams.G = rateMatchLen;
    ldpcRateMatchPrms.tbParams.Qm = qamMod;
    ldpcRateMatchPrms.tbParams.num_CBs = numCodeBlocks;
    ldpcRateMatchPrms.tbParams.rv = rv;
    ldpcRateMatchPrms.tbParams.Nl = numLayers;
    ldpcRateMatchPrms.tbParams.max_REs = rateMatchLen / (qamMod * numLayers);
    ldpcRateMatchPrms.tbParams.cinit = cinit;
    ldpcRateMatchPrms.tbParams.firstCodeBlockIndex = 0;
}


void LdpcRateMatch::setProfilingIterations(const uint32_t numIterations) {
    ldpcRateMatchPrms.numIterations = numIterations;
}


LdpcRateMatch::LdpcRateMatch(const uint64_t inputDevicePtr,
                             const uint64_t outputDevicePtr,
                             const uint64_t inputHostPtr,
                             const uint64_t tempOutputHostPtr,
                             const uint64_t outputHostPtr,
                             bool scrambling) {
    ldpcRateMatchPrms.inputDevicePtr = (void *)inputDevicePtr;
    ldpcRateMatchPrms.outputDevicePtr = (void *)outputDevicePtr;
    ldpcRateMatchPrms.inputHostPtr = (void *)inputHostPtr;
    ldpcRateMatchPrms.tempOutputHostPtr = (void *)tempOutputHostPtr;
    ldpcRateMatchPrms.outputHostPtr = (void *)outputHostPtr;

    ldpcRateMatchPrms.scrambling = scrambling;
    ldpcRateMatchPrms.layerMapping = false;

    ldpcRateMatchPrms.numIterations = 0;
}


py::array_t<float> LdpcRateMatch::rateMatch(const py::array& inputBits,
                                            const uint32_t tbSize,
                                            const float codeRate,
                                            const uint32_t rateMatchLen,
                                            const uint8_t qamMod,
                                            const uint8_t rv,
                                            const uint32_t cinit,
                                            uint64_t cuStream) {

    if (!ldpcRateMatchPrms.inputDevicePtr || !ldpcRateMatchPrms.outputDevicePtr || !ldpcRateMatchPrms.outputHostPtr) {
        throw std::runtime_error("LdpcRateMatch::rateMatch: Memory not allocated!");
    }

    // TODO: These are fixed and hard-coded for now.
    uint8_t numLayers = 1;
    int numTbs = 1;

    // Access the input data address
    py::array_t<float, py::array::c_style | py::array::forcecast> array = inputBits;
    py::buffer_info buf = array.request();
    uint32_t N = buf.shape[0];
    uint32_t C = buf.shape[1];

    // Allocate input device buffer.

    // Convert the input float array data to 32 bit array data.
    fromNumpyBitArray<uint32_t>((float*)buf.ptr,
                                (uint32_t*)ldpcRateMatchPrms.inputHostPtr,
                                N,
                                C);

    uint32_t roundedN = round_up_to_next(N, (uint32_t)32);
    cuphy::tensor_device dInputTensor(ldpcRateMatchPrms.inputDevicePtr, CUPHY_BIT, roundedN, C, numTbs);
    CUDA_CHECK(cudaMemcpyAsync(dInputTensor.addr(),
                               ldpcRateMatchPrms.inputHostPtr,
                               dInputTensor.desc().get_size_in_bytes(),
                               cudaMemcpyHostToDevice,
                               (cudaStream_t)cuStream));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Allocate output device buffer.
    uint32_t rateMatchLen32 = div_round_up(rateMatchLen, (uint32_t)32);
    cuphy::tensor_device dOutputTensor(ldpcRateMatchPrms.outputDevicePtr, CUPHY_R_32U, rateMatchLen32, numTbs);

    // Set transport block parameters.
    setTransportBlockParams(
        tbSize,
        codeRate,
        rateMatchLen,
        qamMod,
        C,
        N,
        rv,
        numLayers,
        cinit
    );

    // Allocate workspace and copy config params
    size_t allocatedWorkspaceSize = cuphyDlRateMatchingWorkspaceSize(numTbs);
    unique_device_ptr<uint32_t> dWorkspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(allocatedWorkspaceSize, sizeof(uint32_t)));
    unique_pinned_ptr<uint32_t> hWorkspace = make_unique_pinned<uint32_t>((2 + 2) * numTbs);

    // Copy TB parameters from host to device.
    cuphy::unique_device_ptr<PdschPerTbParams> dTbPrmsArray = make_unique_device<PdschPerTbParams>(numTbs);
    CUDA_CHECK(cudaMemcpyAsync(dTbPrmsArray.get(), &ldpcRateMatchPrms.tbParams, sizeof(PdschPerTbParams) * numTbs, cudaMemcpyHostToDevice, (cudaStream_t)cuStream));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    cuphy::unique_pinned_ptr<PdschDmrsParams> hDmrsParams = make_unique_pinned<PdschDmrsParams>(numTbs);
    cuphy::unique_device_ptr<PdschDmrsParams> dDmrsParams = make_unique_device<PdschDmrsParams>(numTbs);
    CUDA_CHECK(cudaMemcpyAsync(dDmrsParams.get(), hDmrsParams.get(), numTbs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice, (cudaStream_t)cuStream));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Allocate launch config struct.
    std::unique_ptr<cuphyDlRateMatchingLaunchConfig> rmHandle = std::make_unique<cuphyDlRateMatchingLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t descAsyncCopy = 1; // Copy descriptor to the GPU during setup.

    size_t descSize = 0, allocSize = 0;
    cuphyStatus_t status = cuphyDlRateMatchingGetDescrInfo(&descSize, &allocSize);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("cuphyDlRateMatchingGetDescrInfo error!");
    }
    cuphy::unique_device_ptr<uint8_t> dRmDesc = make_unique_device<uint8_t>(descSize);
    cuphy::unique_pinned_ptr<uint8_t> hRmDesc = make_unique_pinned<uint8_t>(descSize);


    cuphyPdschStatusOut_t dl_out_status; // populated during cuphySetupDlRateMatching, but contents not used here

    // Setup DL Rate Matching object
    status = cuphySetupDlRateMatching(rmHandle.get(),
                                      &dl_out_status,
                                      (const uint32_t*)dInputTensor.addr(),
                                      (uint32_t*)dOutputTensor.addr(),
                                      nullptr,
                                      nullptr, // d_modulation_output
                                      nullptr, // d_xtf_re_map
                                      273,
                                      numTbs,
                                      numLayers,
                                      ldpcRateMatchPrms.scrambling,
                                      ldpcRateMatchPrms.layerMapping,
                                      false,  // enable_modulation
                                      0,  // precoding
                                      false,  // restructure_kernel
                                      false,  // batching
                                      hWorkspace.get(),
                                      dWorkspace.get(),  // Explicit H2D copy as part of setup
                                      &ldpcRateMatchPrms.tbParams,
                                      dTbPrmsArray.get(),
                                      dDmrsParams.get(),
                                      nullptr,  // d_ue_grp_params
                                      hRmDesc.get(),
                                      dRmDesc.get(),
                                      descAsyncCopy,
                                      (cudaStream_t)cuStream);

    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupDlRateMatching!");
    }

    // Run the kernel.
    CUresult r;
    if(!ldpcRateMatchPrms.numIterations) {
        r = launch_kernel(rmHandle.get()->m_kernelNodeParams[0], (cudaStream_t)cuStream);
        if(r != CUDA_SUCCESS) {
            throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
        }
    }
    else {

        cudaEvent_t start, stop;
        float time = 0.0f;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < ldpcRateMatchPrms.numIterations; iter++) {
            r = launch_kernel(rmHandle.get()->m_kernelNodeParams[0], (cudaStream_t)cuStream);
            if(r != CUDA_SUCCESS) {
                throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double int_tput = ((double)rateMatchLen * (double)numTbs  / (double)1000) * ((double)ldpcRateMatchPrms.numIterations / (double)time);
        std::cout << "Total time from C++ is " << time * 1000 << " us." << std::endl;
        std::cout << "Internal throughput is " << int_tput / 1000 << " Gbps." << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Convert to host memory.
    cuphy::tensor_pinned hOutputTensor = tensor_pinned(ldpcRateMatchPrms.tempOutputHostPtr,
                                                       CUPHY_R_32U,
                                                       rateMatchLen32,
                                                       numTbs,
                                                       cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);

    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Unpack the 32-bit integers to floats for Numpy.
    toNumpyBitArray<uint32_t>((uint32_t*)ldpcRateMatchPrms.tempOutputHostPtr,
                              (float*)ldpcRateMatchPrms.outputHostPtr,
                              rateMatchLen,
                              numTbs);
    // Create the Numpy array for output.
    return hostToNumpy<float>((float*)ldpcRateMatchPrms.outputHostPtr, rateMatchLen, (uint32_t)numTbs);

}

}

#endif
