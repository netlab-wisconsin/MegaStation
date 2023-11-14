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

#ifndef AERIAL_PYTHON_LDPC_DECODER_CPP
#define AERIAL_PYTHON_LDPC_DECODER_CPP


typedef struct {

    void *                  inputDevicePtr;        // Input device (GPU) memory pointer
    void *                  outputDevicePtr;       // Output device (GPU) memory pointer
    void *                  outputHostPtr;         // Output host (CPU) memory pointer

    uint8_t                 BG;                    // Base graph (1 or 2).
    uint32_t                N;                     // Total number of nodes (information + parity)
    uint32_t                K;                     // Number of information bits
    uint32_t                maxParityNodes;        // Maximum parity nodes
    int                     C;                     // Number of code blocks
    int                     nCwNodes;              // Total codeword nodes
    int                     Kb;                    // Information nodes
    int                     Z;                     // Lifting size

    uint8_t                 halfPrecisionFlag;     // Flag to indicate half precision. Single by default.
    uint16_t                numIterations;         // Number of iteration to decode. 10 by default.
    uint16_t                numProfilingIterations;// Number of profiling iterations to run.
    uint8_t                 algoIndex;             // Algorithm selection index. 0 by default.
    float                   normalizationFactor;   // Normalization factor for min-sum. 0.1 by default
    uint8_t                 interfaceFlag;         // Interface selection flag // 0 Tensor (default), 1 Transport Block
    uint32_t                decodeFlags;           // For enabling throughput mode - prefer high throughput over low latency

} cuphyLdpcDecoderPrms_t;


namespace pycuphy {

class LdpcDecoder {

    public:

        LdpcDecoder(
            const uint64_t inputDevicePtr,
            const uint64_t outputDevicePtr,
            const uint64_t outputHostPtr
        );

        py::array_t<float> decode(
            const py::array& inputLlr,
            const uint32_t tbSize,
            const float codeRate,
            uint64_t cuStream
        );

        void setNumIterations(const uint16_t numIterations);
        void setProfilingIterations(const uint16_t numProfilingIterations);
        void setThroughputMode(const uint8_t throughputMode);
        void setHalfPrecision(const uint8_t halfPrecisionFlag);

    private:
        cuphyLdpcDecoderPrms_t ldpcDecoderPrms;
        cuphy::context ctx;

        void setDecoderParameters(const uint32_t tbSize, const float codeRate, const uint32_t N);
        uint32_t getNumCodeBlocks(const uint32_t tbSize, const uint32_t baseGraph);

};




LdpcDecoder::LdpcDecoder(const uint64_t inputDevicePtr, const uint64_t outputDevicePtr, const uint64_t outputHostPtr) {

    ldpcDecoderPrms.inputDevicePtr = (void *)inputDevicePtr;
    ldpcDecoderPrms.outputDevicePtr = (void *)outputDevicePtr;
    ldpcDecoderPrms.outputHostPtr = (void *)outputHostPtr;

    ldpcDecoderPrms.halfPrecisionFlag = 0;
    ldpcDecoderPrms.numIterations = 10;
    ldpcDecoderPrms.algoIndex = 0;
    ldpcDecoderPrms.normalizationFactor = 0.0;
    ldpcDecoderPrms.numProfilingIterations = 0;
    ldpcDecoderPrms.interfaceFlag = 0;
    ldpcDecoderPrms.decodeFlags = 0;
}


void LdpcDecoder::setProfilingIterations(const uint16_t numProfilingIterations) {
    ldpcDecoderPrms.numProfilingIterations = numProfilingIterations;
}


void LdpcDecoder::setNumIterations(const uint16_t numIterations) {
    ldpcDecoderPrms.numIterations = numIterations;
}

void LdpcDecoder::setThroughputMode(const uint8_t throughputMode) {
    ldpcDecoderPrms.decodeFlags = throughputMode ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
}


void LdpcDecoder::setHalfPrecision(const uint8_t halfPrecisionFlag) {
    ldpcDecoderPrms.halfPrecisionFlag = halfPrecisionFlag;
}


uint32_t LdpcDecoder::getNumCodeBlocks(const uint32_t tbSize, const uint32_t baseGraph) {
    uint32_t K_cb = (baseGraph == 1) ? 8448 : 3840;
    uint32_t tbSizeWithCrc = tbSize + compute_TB_CRC(tbSize);
    uint32_t C = 1;
    if (tbSizeWithCrc > K_cb) { // The TB will be segmented into multiple CBs.
        uint32_t L = 24;
        C = ceil((tbSizeWithCrc * 1.0f) / (K_cb - L));
    }
    return C;
}


void LdpcDecoder::setDecoderParameters(const uint32_t tbSize,
                                       const float codeRate,
                                       const uint32_t N) {
    uint8_t BG = get_base_graph(codeRate, tbSize);
    uint32_t C = getNumCodeBlocks(tbSize, BG);
    uint16_t Kprime = get_K_prime(tbSize, BG, C);
    int Z = get_lifting_size(tbSize, BG, Kprime);

    ldpcDecoderPrms.BG = BG;
    ldpcDecoderPrms.N = N;
    ldpcDecoderPrms.C = C;
    ldpcDecoderPrms.Z = Z;

    // Parameter selection based on base graph.
    if(ldpcDecoderPrms.BG == 1) {
        ldpcDecoderPrms.Kb = CUPHY_LDPC_BG1_INFO_NODES;
        ldpcDecoderPrms.nCwNodes = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES;
        ldpcDecoderPrms.maxParityNodes = CUPHY_LDPC_MAX_BG1_PARITY_NODES;
        ldpcDecoderPrms.K = ldpcDecoderPrms.Z * 22;
    }
    else {
        ldpcDecoderPrms.Kb = CUPHY_LDPC_MAX_BG2_INFO_NODES;
        ldpcDecoderPrms.nCwNodes = CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
        ldpcDecoderPrms.maxParityNodes = CUPHY_LDPC_MAX_BG2_PARITY_NODES;
        ldpcDecoderPrms.K = ldpcDecoderPrms.Z * 10;
    }
}


py::array_t<float> LdpcDecoder::decode(const py::array& inputLlr,
                                       const uint32_t tbSize,
                                       const float codeRate,
                                       uint64_t cuStream) {

    if (!ldpcDecoderPrms.inputDevicePtr || !ldpcDecoderPrms.outputDevicePtr || !ldpcDecoderPrms.outputHostPtr) {
        throw std::runtime_error("LdpcDecoder::decode: Memory not allocated!");
    }

    // Single precision by default.
    cuphyDataType_t convertToType = CUPHY_R_32F;
    if(ldpcDecoderPrms.halfPrecisionFlag == 1) {
        convertToType = CUPHY_R_16F;
    }

    // Convert input numpy array to tensor.
    cuphy::tensor_device inputLlrTensor = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
        inputLlr,
        ldpcDecoderPrms.inputDevicePtr,
        CUPHY_R_32F,
        convertToType,
        cuphy::tensor_flags::align_tight,
        (cudaStream_t)cuStream);

    uint32_t N = inputLlrTensor.dimensions()[0];
    uint32_t C = inputLlrTensor.dimensions()[1];
    setDecoderParameters(tbSize, codeRate, N);

    // Allocate an output buffer for decoded bits.
    cuphy::tensor_device dOutputTensor(ldpcDecoderPrms.outputDevicePtr,
                                       CUPHY_BIT,
                                       ldpcDecoderPrms.K,
                                       C,
                                       cuphy::tensor_flags::align_tight);

    // Create an LDPC decoder instance.
    cuphy::LDPC_decoder decoder(ctx);

    // Initialize the LDPC decode configuration.
    int numParityNodes = static_cast<int>(std::ceil((ldpcDecoderPrms.N - ldpcDecoderPrms.K) / static_cast<float>(ldpcDecoderPrms.Z)));
    cuphy::LDPC_decode_config decoderConfig(convertToType,                       // LLR type (fp16 or fp32)
                                            numParityNodes,                      // num parity nodes
                                            ldpcDecoderPrms.Z,                   // lifting size
                                            ldpcDecoderPrms.numIterations,       // max num iterations
                                            ldpcDecoderPrms.Kb,                  // info nodes
                                            ldpcDecoderPrms.normalizationFactor, // normalization value
                                            ldpcDecoderPrms.decodeFlags,         // flags
                                            ldpcDecoderPrms.BG,                  // base graph
                                            ldpcDecoderPrms.algoIndex,           // algorithm index
                                            nullptr);                            // workspace address

    // If no normalization value was provided, query the library for
    // an appropriate value.
    if(ldpcDecoderPrms.normalizationFactor <= 0.0f) {
        decoder.set_normalization(decoderConfig);
    }

    // Initialize an LDPC decode descriptor structure.
    // Used only when the transport block interface is selected.
    LDPC_decode_desc decoderDesc(decoderConfig);
    decoderDesc.add_tensor_as_tb(inputLlrTensor.desc(),
                                 inputLlrTensor.addr(),
                                 dOutputTensor.desc(),
                                 dOutputTensor.addr());

    // Initialize an LDPC decode tensor params structure. (This is
    // only used when the tensor-based decoder interface is selected.)
    LDPC_decode_tensor_params decoderTensor(decoderConfig,                   // LDPC configuration
                                            dOutputTensor.desc().handle(),   // output descriptor
                                            dOutputTensor.addr(),            // output address
                                            inputLlrTensor.desc().handle(),  // LLR descriptor
                                            inputLlrTensor.addr());          // LLR address

    if(!ldpcDecoderPrms.numProfilingIterations) {
        if(ldpcDecoderPrms.interfaceFlag) {
            decoder.decode(decoderDesc, (cudaStream_t)cuStream);
        }
        else {
            decoder.decode(decoderTensor, (cudaStream_t)cuStream);
        }
    }
    else {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time = 0.0f;
        cudaEventRecord(start);

        // Run the Decoder
        for(unsigned int i = 0; i < ldpcDecoderPrms.numProfilingIterations; i++)
        {
            if(ldpcDecoderPrms.interfaceFlag) {
                decoder.decode(decoderDesc, (cudaStream_t)cuStream);
            }
            else {
                decoder.decode(decoderTensor, (cudaStream_t)cuStream);
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double int_tput = (ldpcDecoderPrms.K * C / 1000000.0f) * (ldpcDecoderPrms.numProfilingIterations / time);
        std::cout << "Total time from C++ is " << time * 1000 << " us" << std::endl;
        std::cout << "Internal throughput is " << int_tput << " Gbps for " << ldpcDecoderPrms.numProfilingIterations << " runs" << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    uint32_t dim0 = dOutputTensor.dimensions()[0];
    uint32_t dim1 = dOutputTensor.dimensions()[1];

    cuphy::tensor_pinned hOutputTensor = tensor_pinned(ldpcDecoderPrms.outputHostPtr, CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);

    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Create the Numpy array for output.
    return hostToNumpy<float>((float*)ldpcDecoderPrms.outputHostPtr, dim0, dim1);
}

} // pycuphy

#endif // AERIAL_PYTHON_LDPC_DECODER_CPP