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

#ifndef AERIAL_PYTHON_LDPC_ENCODER_CPP
#define AERIAL_PYTHON_LDPC_ENCODER_CPP

typedef struct {
    void *              inputDevicePtr;     // Input Device (GPU) Memory Pointer
    void *              outputDevicePtr;    // Output Device (GPU) Memory Pointer
    void *              outputHostPtr;      // Output Host (CPU) Memory Pointer
    void *              tempInputHostPtr;   // Temporary Input Host Pointer

    uint8_t             BG;                 // Base Graph Type (1 or 2)
    uint16_t            maxParityNodes;     // Maximum Parity Nodes
    int                 K;                  // Padded Information Bits length
    int                 C;                  // Number of code blocks
    int                 nCwNodes;           // Total Codeword nodes
    int                 Kb;                 // Information Nodes
    int                 Z;                  // Lifting Size
    int                 N;                  // Codeword length
    int                 N_padding;          // Padded codeword length bits
    int                 K_padding;          // Padded information bits
    int                 rv;                 // Redundancy version

    uint8_t             puncture;           // Puncturing flag
    uint16_t            numIterations;      // Number of profiling iterations

} cuphyLdpcEncoderPrms_t;


namespace pycuphy {

class LdpcEncoder {

public:
    LdpcEncoder(
        const uint64_t inputDevicePtr,
        const uint64_t tempInputHostPtr,
        const uint64_t outputDevicePtr,
        const uint64_t outputHostPtr
    );

    py::array_t<float> encode(
        const py::array& inputData,
        const uint32_t tbSize,
        const float codeRate,
        const int rv,
        uint64_t cuStream
    );

    void setProfilingIterations(const uint16_t numIterations);
    void setPuncturing(const uint8_t puncture);

private:
    // Encoder parameters.
    cuphyLdpcEncoderPrms_t ldpcEncoderPrms;

    void setEncoderParameters(
        const uint32_t tbSize,
        const float codeRate,
        const int rv
    );
};


LdpcEncoder::LdpcEncoder(const uint64_t inputDevicePtr,
                         const uint64_t tempInputHostPtr,
                         const uint64_t outputDevicePtr,
                         const uint64_t outputHostPtr) {

    ldpcEncoderPrms.inputDevicePtr = (void *)inputDevicePtr;
    ldpcEncoderPrms.tempInputHostPtr = (void *)tempInputHostPtr;
    ldpcEncoderPrms.outputDevicePtr = (void *)outputDevicePtr;
    ldpcEncoderPrms.outputHostPtr = (void *)outputHostPtr;
    ldpcEncoderPrms.numIterations = 0;
    ldpcEncoderPrms.puncture = 1;
}


void LdpcEncoder::setProfilingIterations(const uint16_t numIterations) {
    ldpcEncoderPrms.numIterations = numIterations;
}


void LdpcEncoder::setPuncturing(const uint8_t puncture) {
    ldpcEncoderPrms.puncture = puncture;
}


py::array_t<float> LdpcEncoder::encode(const py::array& inputData,
                                       const uint32_t tbSize,
                                       const float codeRate,
                                       const int rv,
                                       uint64_t cuStream) {

    if (!ldpcEncoderPrms.tempInputHostPtr || !ldpcEncoderPrms.inputDevicePtr ||
        !ldpcEncoderPrms.outputDevicePtr || !ldpcEncoderPrms.outputHostPtr) {
        throw std::runtime_error("LdpcEncoder::encode: Memory not allocated!");
    }

    // Access the input data address.
    py::array_t<float, py::array::c_style | py::array::forcecast> inputArray = inputData;
    py::buffer_info buf = inputArray.request();
    ldpcEncoderPrms.C = buf.shape[1];
    ldpcEncoderPrms.K = buf.shape[0];

    // Set the encoder parameters into the struct.
    setEncoderParameters(tbSize, codeRate, rv);

    // Convert the float array data to 32 bit array data. Store under tempInputHostPtr.
    fromNumpyBitArray<uint32_t>((float*)buf.ptr,
                                (uint32_t *)ldpcEncoderPrms.tempInputHostPtr,
                                ldpcEncoderPrms.K,
                                ldpcEncoderPrms.C);

    // Tensor to hold input uncoded data in device memory.
    cuphy::tensor_device dInputTensor(ldpcEncoderPrms.inputDevicePtr, CUPHY_BIT, ldpcEncoderPrms.K + ldpcEncoderPrms.K_padding, ldpcEncoderPrms.C);
    CUDA_CHECK(cudaMemcpyAsync(dInputTensor.addr(), ldpcEncoderPrms.tempInputHostPtr, dInputTensor.desc().get_size_in_bytes(), cudaMemcpyHostToDevice, (cudaStream_t)cuStream));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Output tensor in device memory.
    cuphy::tensor_device dOutputTensor(ldpcEncoderPrms.outputDevicePtr, CUPHY_BIT, ldpcEncoderPrms.N + ldpcEncoderPrms.N_padding, ldpcEncoderPrms.C);

    // Allocate launch config struct.
    std::unique_ptr<cuphyLDPCEncodeLaunchConfig> ldpcHandle = std::make_unique<cuphyLDPCEncodeLaunchConfig>();

    // Allocate descriptors and setup LDPC encoder.
    uint8_t descAsyncCopy = 1; // Copy descriptor to the GPU during setup
    size_t  workspaceSize = 0;
    int maxUes = PDSCH_MAX_UES_PER_CELL_GROUP;

    size_t descSize = 0, allocSize = 0;
    cuphyStatus_t status = cuphyLDPCEncodeGetDescrInfo(&descSize, &allocSize, maxUes, &workspaceSize);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcEncoder::encode: cuphyLDPCEncodeGetDescrInfo error!");
    }

    cuphy::unique_device_ptr<uint8_t> dLdpcDesc = make_unique_device<uint8_t>(descSize);
    cuphy::unique_pinned_ptr<uint8_t> hLdpcDesc = make_unique_pinned<uint8_t>(descSize);

    cuphy::unique_device_ptr<uint8_t> dWorkspace = make_unique_device<uint8_t>(workspaceSize);
    cuphy::unique_pinned_ptr<uint8_t> hWorkspace = make_unique_pinned<uint8_t>(workspaceSize);

    // Setup the LDPC Encoder
    status = cuphySetupLDPCEncode(ldpcHandle.get(),
                                  dInputTensor.desc().handle(),
                                  dInputTensor.addr(),
                                  dOutputTensor.desc().handle(),
                                  dOutputTensor.addr(),
                                  ldpcEncoderPrms.BG,
                                  ldpcEncoderPrms.Z,
                                  ldpcEncoderPrms.puncture,
                                  ldpcEncoderPrms.maxParityNodes,
                                  ldpcEncoderPrms.rv,
                                  0,
                                  1,
                                  nullptr,
                                  nullptr,
                                  hWorkspace.get(),
                                  dWorkspace.get(),
                                  hLdpcDesc.get(),
                                  dLdpcDesc.get(),
                                  descAsyncCopy,
                                  (cudaStream_t)cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcEncoder::encode: Invalid argument(s) for cuphySetupLDPCEncode!");
    }

    if(!ldpcEncoderPrms.numIterations) {
        CUresult r;
        r = launch_kernel(ldpcHandle.get()->m_kernelNodeParams, (cudaStream_t)cuStream);
        if(r != CUDA_SUCCESS) {
            throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
        }
    }
    else {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int N_ITER = ldpcEncoderPrms.numIterations;

        float time = 0.0f;
        cudaEventRecord(start);

        // Launch the Kernel
        CUresult r;
        for(unsigned int i = 0; i < N_ITER; i++)
        {
            r = launch_kernel(ldpcHandle.get()->m_kernelNodeParams, (cudaStream_t)cuStream);
            if(r != CUDA_SUCCESS) {
                throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double tput = (ldpcEncoderPrms.K * ldpcEncoderPrms.C / 1000000) * (N_ITER / time);
        std::cout << "Total time from C++ is " << time * 1000 << " us" << std::endl;
        std::cout << "Internal throughput is " << tput << " Gbps" << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    ldpcHandle.reset();
    dLdpcDesc.reset();
    hLdpcDesc.reset();
    dWorkspace.reset();
    hWorkspace.reset();

    // Convert output from tensor device to host numpy
    uint32_t dim0 = dOutputTensor.dimensions()[0];
    uint32_t dim1 = dOutputTensor.dimensions()[1];

    cuphy::tensor_pinned hOutputTensor = tensor_pinned(ldpcEncoderPrms.outputHostPtr, CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Create the Numpy array for output.
    return hostToNumpy<float>((float*)ldpcEncoderPrms.outputHostPtr, (uint32_t)ldpcEncoderPrms.N, dim1);
}


void LdpcEncoder::setEncoderParameters(const uint32_t tbSize, const float codeRate, const int rv) {

    uint8_t BG = get_base_graph(codeRate, tbSize);
    ldpcEncoderPrms.BG = BG;
    ldpcEncoderPrms.rv = rv;

    // Parameter selection based on base graph.
    if(ldpcEncoderPrms.BG == 1) {
        ldpcEncoderPrms.Kb = CUPHY_LDPC_BG1_INFO_NODES;
        ldpcEncoderPrms.Z = ldpcEncoderPrms.K / ldpcEncoderPrms.Kb;
        ldpcEncoderPrms.nCwNodes = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES;
    }
    else {
        ldpcEncoderPrms.Kb = CUPHY_LDPC_MAX_BG2_INFO_NODES;
        ldpcEncoderPrms.Z = ldpcEncoderPrms.K / 10;
        ldpcEncoderPrms.nCwNodes = CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
    }

    // N and K to be in multiples of 32.
    ldpcEncoderPrms.N = ldpcEncoderPrms.Z * (ldpcEncoderPrms.nCwNodes + (ldpcEncoderPrms.puncture ? 0 : 2));
    ldpcEncoderPrms.N_padding = (ldpcEncoderPrms.N % 32 != 0) * (32 - (ldpcEncoderPrms.N % 32));
    ldpcEncoderPrms.K_padding = (ldpcEncoderPrms.K % 32 != 0) * (32 - (ldpcEncoderPrms.K % 32));

    if(ldpcEncoderPrms.BG == 1) {
        ldpcEncoderPrms.maxParityNodes = CUPHY_LDPC_MAX_BG1_PARITY_NODES;
    }
    else {
        ldpcEncoderPrms.maxParityNodes = CUPHY_LDPC_MAX_BG2_PARITY_NODES;
    }

}


} // namespace pycuphy

#endif // AERIAL_PYTHON_LDPC_ENCODER_CPP