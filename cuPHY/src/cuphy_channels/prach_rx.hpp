/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(PRACH_RX_HPP_INCLUDED_)
#define PRACH_RX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "prach_receiver/prach_receiver.hpp"
#include "util.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>

struct cuphyPrachRx
{};

class PrachRx : public cuphyPrachRx {

public:
    /**
     * @brief: Construct PrachRx class.
     */
     PrachRx(cuphyPrachStatPrms_t const* pStatPrms, cuphyStatus_t* status);

    /**
     * @brief: PrachRx cleanup.
     */
    ~PrachRx();

    /**
     * @brief: PrachRx setup
     * @param[in] dyn_params: input parameters to PRACH.
     */
    cuphyStatus_t expandParameters(cuphyPrachDynPrms_t* pDynPrms);

    /**
     * @brief: Run PRACH
     */
    cuphyStatus_t Run();

    const void* getMemoryTracker();

    /**
     * @brief Print Static API Parameters
     * 
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStatApiPrms(cuphyPrachStatPrms_t const* pStatPrms);
    /**
     * @brief Print Dynamic API Parameters
     * 
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPrachDynPrms_t* pDynPrm);

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // number of cells passed during create pipeline
    uint16_t nMaxCells;

    // total number of occasion across all cells passed during create pipeline
    uint16_t nTotCellOcca;

    // max number of occasion that will be processed in a single pipeline processing
    uint16_t nMaxOccasions;

    // number of occasion to be processed, passed during setup pipeline
    uint16_t nOccaProc;

    // processing mode - graph (1) or stream(0)
    uint64_t procModeBmsk = 0;

    uint16_t maxAntenna = 0;
    uint max_l_oran_ant = 0;
    uint max_ant_u = 0;
    uint max_nfft = 0;
    int max_zoneSizeExt = 0;
    uint cudaDeviceArch = 0;

    std::vector<PrachInternalStaticParamPerOcca> staticParam;

    // Vector of size nMaxOccasions containing state of each occasion
    // value at each index is 0 or 1
    // 1 - occasion is active, 0 - not active
    std::vector<char> activeOccasions;
    // same as above - active occasions in previous step
    std::vector<char> prevActiveOccasions;

    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc> d_staticParam;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc> d_dynParam;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::pinned_alloc> h_dynParam;

    cuphyTensorPrm_t numDetectedPrmb;
    cuphyTensorPrm_t prmbIndexEstimates;
    cuphyTensorPrm_t prmbDelayEstimates;
    cuphyTensorPrm_t prmbPowerEstimates;
    cuphyTensorPrm_t antRssi;
    cuphyTensorPrm_t rssi;
    cuphyTensorPrm_t interference;

    // dynamic parameters used in previous step
    // Need to maintain to see if CUDA graph requires updating
    uint32_t* prev_numDetectedPrmb = nullptr;
    uint32_t* prev_prmbIndexEstimates = nullptr;
    float* prev_prmbDelayEstimates = nullptr;
    float* prev_prmbPowerEstimates = nullptr;
    float* prev_antRssi = nullptr;
    float* prev_rssi = nullptr;
    float* prev_interference = nullptr;
    uint16_t nPrevOccaProc;

    cudaStream_t cuStream;

    cudaGraph_t graph;
    cudaGraphExec_t graphInstance;
    std::vector<cudaGraphNode_t> nodes;

};

#endif // !defined(PRACH_RX_HPP_INCLUDED_)
