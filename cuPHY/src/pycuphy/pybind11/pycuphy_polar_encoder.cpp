/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

#ifndef AERIAL_PYTHON_POLAR_ENCODER_CPP
#define AERIAL_PYTHON_POLAR_ENCODER_CPP

typedef struct
{
    void *              inputDevicePtr;       // Pointer to device memory of input data
    void *              outputEncDevicePtr;   // Pointer to device memory of encoded data
    void *              outputRmDevicePtr;    // Pointer to device memory of rate matched data
    void *              outputHostPtr;        // Pointer to host memory of output data
    void *              tempOutputHostPtr;    // Pointer to host memory of temporary output data

    uint8_t             encMode;              // Encoder mode - 0 downlink (default), 1 uplink
    uint16_t            numIterations;        // Number of profiling iterations

} cuphyPolarEncoderPrms_t;


namespace pycuphy {

class PolarEncoder {

public:
    PolarEncoder(
        const uint64_t inputDevicePtr,
        const uint64_t outputEncDevicePtr,
        const uint64_t outputRmDevicePtr,
        const uint64_t outputHostPtr,
        const uint64_t tempOutputHostPtr,
        const uint8_t encMode
    );
    py::array_t<float> encode(
        const py::array& inputData,
        const uint16_t K,
        uint32_t N,
        const uint16_t E,
        uint64_t cuStream
    );

    void setProfilingIterations(const uint16_t numIterations);

private:
    cuphyPolarEncoderPrms_t polarEncoderPrms;

};


PolarEncoder::PolarEncoder(const uint64_t inputDevicePtr,
                           const uint64_t outputEncDevicePtr,
                           const uint64_t outputRmDevicePtr,
                           const uint64_t outputHostPtr,
                           const uint64_t tempOutputHostPtr,
                           const uint8_t encMode) {

    polarEncoderPrms.inputDevicePtr             = (void *)inputDevicePtr;
    polarEncoderPrms.outputEncDevicePtr         = (void *)outputEncDevicePtr;
    polarEncoderPrms.outputRmDevicePtr          = (void *)outputRmDevicePtr;
    polarEncoderPrms.outputHostPtr              = (void *)outputHostPtr;
    polarEncoderPrms.tempOutputHostPtr          = (void *)tempOutputHostPtr;
    polarEncoderPrms.encMode                    = encMode;
    polarEncoderPrms.numIterations              = 0;
}


void PolarEncoder::setProfilingIterations(const uint16_t numIterations) {
    polarEncoderPrms.numIterations = numIterations;
}


py::array_t<float> PolarEncoder::encode(const py::array& inputData,
                                        const uint16_t K,
                                        uint32_t N,
                                        const uint16_t E,
                                        uint64_t cuStream) {

    if (!polarEncoderPrms.inputDevicePtr || !polarEncoderPrms.outputEncDevicePtr ||
        !polarEncoderPrms.outputRmDevicePtr || !polarEncoderPrms.outputHostPtr) {
        throw std::runtime_error("PolarEncoder::encode: Memory not allocated!");
    }

    // Allocate output tensors.
    // For coded bits provide the worst case storage.
    cuphy::tensor_device tGpuCodedBits(polarEncoderPrms.outputEncDevicePtr,
                                       CUPHY_R_8U,
                                       div_round_up(CUPHY_POLAR_ENC_MAX_CODED_BITS, 8),
                                       cuphy::tensor_flags::align_tight);

    cuphy::tensor_device tGpuTxBits(polarEncoderPrms.outputRmDevicePtr,
                                    CUPHY_R_8U,
                                    round_up_to_next(CUPHY_POLAR_ENC_MAX_TX_BITS, 32) / 8, // roundup to nearest 32b boundary (multiple of words)
                                    cuphy::tensor_flags::align_tight);

    // Convert input numpy array to tensor.
    cuphy::tensor_device inputDataTensor = deviceFromNumpy<uint8_t>(inputData,
                                                                    polarEncoderPrms.inputDevicePtr,
                                                                    CUPHY_R_8U,
                                                                    CUPHY_R_8U,
                                                                    cuphy::tensor_flags::align_coalesce,
                                                                    cuphy::tensor_flags::align_coalesce,
                                                                    (cudaStream_t)cuStream);

    cuphyStatus_t polarEncStat;
    if(!polarEncoderPrms.numIterations){
        polarEncStat = cuphyPolarEncRateMatch(K,
                                              E,
                                              static_cast<uint8_t const*>(inputDataTensor.addr()),
                                              &N,
                                              static_cast<uint8_t*>(tGpuCodedBits.addr()),
                                              static_cast<uint8_t*>(tGpuTxBits.addr()),
                                              polarEncoderPrms.encMode,
                                              (cudaStream_t)cuStream);
        if(polarEncStat != CUPHY_STATUS_SUCCESS) {
            throw cuphy::cuphy_exception(polarEncStat);
        }
    }
    else {
        cudaEvent_t start, stop;
        float time = 0.0f;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Encoder and rate match function.
        for(uint32_t i = 0; i < polarEncoderPrms.numIterations; ++i)
        {
            polarEncStat = cuphyPolarEncRateMatch(K,
                                                  E,
                                                  static_cast<uint8_t const*>(inputDataTensor.addr()),
                                                  &N,
                                                  static_cast<uint8_t*>(tGpuCodedBits.addr()),
                                                  static_cast<uint8_t*>(tGpuTxBits.addr()),
                                                  polarEncoderPrms.encMode,
                                                  (cudaStream_t)cuStream);
            if(polarEncStat != CUPHY_STATUS_SUCCESS) {
                throw cuphy::cuphy_exception(polarEncStat);
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        double int_tput = ((double)K * (double)polarEncoderPrms.numIterations) / (1000.0f * (double)time);
        std::cout << "Total time from C++ is " << time * 1000 << " us." << std::endl;
        std::cout << "Internal throughput is " << int_tput << " Mbps." << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Device to host memory transfer.
    uint32_t dim0 = tGpuTxBits.dimensions()[0];
    uint32_t dim1 = tGpuTxBits.dimensions()[1];
    cuphy::tensor_device dOutputTensor = tensor_device(tGpuTxBits.addr(), CUPHY_R_8U, dim0, dim1, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hOutputTensor = tensor_pinned((uint8_t*)polarEncoderPrms.tempOutputHostPtr, CUPHY_R_8U, dim0, dim1, cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, (cudaStream_t)cuStream);
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));

    // Unpack the 8-bit integers to floats for Numpy.
    toNumpyBitArray<uint8_t>((uint8_t*)polarEncoderPrms.tempOutputHostPtr,
                             (float*)polarEncoderPrms.outputHostPtr,
                             E,
                             1);

    // Return the Numpy array.
    return hostToNumpy<float>((float*)polarEncoderPrms.outputHostPtr, E, 1);
}

}

#endif
