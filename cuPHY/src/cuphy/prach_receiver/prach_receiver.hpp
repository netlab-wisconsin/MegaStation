/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(PRACH_RECEIVER_HPP_INCLUDED_)
#define PRACH_RECEIVER_HPP_INCLUDED_

#define USE_CUFFTDX 1

// enum used as index into graph node container
enum GraphNodeType
{
    ComputePDPNode = 0,
    SearchPDPNode,
    MemsetRSSI,
    ComputeRSSI,
    MemcpyRSSI,
    ComputeCorrelationNode,
    FFTNode                     // FFT node. This is last because FFTNode + i is used to create FFT node for ith occassion
};

template<typename Tscalar>
struct prach_pdp_t 
{
    Tscalar power;   // averaged power
    Tscalar max;     // peak power 
    int loc;        // peak location
};

template<typename Tscalar>
struct prach_det_t 
{
    Tscalar power[CUPHY_PRACH_RX_NUM_PREAMBLE];     // array for detected peak power 
    int prmbIdx[CUPHY_PRACH_RX_NUM_PREAMBLE];      // array for detected preamble index
    int loc[CUPHY_PRACH_RX_NUM_PREAMBLE];          // array for detected peak location
    int Ndet;                       // number of detected preambles
};

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * @brief Intermediate structure generated from cuphyPrachCellStatPrms_t needed for for PRACH receiver processing.
 */
struct PrachParams
{
    uint32_t N_CS;       /*!< Cyclic shift step */
    uint32_t uCount;     /*!< number of u for preamble search */
    uint32_t L_RA;       /*!< length of preamble sequence */
    uint32_t N_rep;      /*!< number of preamble repetition */
    uint32_t delta_f_RA; /*!< subcarrier spacing (Hz) for PRACH */
    uint32_t N_ant;      /*!< number of antennas */
    uint32_t mu;         /*!< numerology */
    uint32_t Nfft;       /*!< FFT size */
    uint32_t N_nc;       /*!< number of non-coherent combining for repetitve preamble */
    uint32_t kBar;       /*!< number of guard subcarriers */
};

/**
 * @brief Per occasion Intermediate structure generated from cuphyPrachOccaStatPrms_t needed for for PRACH receiver processing.
 */
struct PrachInternalStaticParamPerOcca
{
    PrachParams prach_params;
    cuphy::buffer<float, cuphy::device_alloc> prach_workspace_buffer;
    cuphy::buffer<__half2, cuphy::device_alloc> d_y_u_ref;
#ifndef USE_CUFFTDX
    cufftHandle fft_plan;
#endif
};

/**
 * @brief Per occasion Intermediate structure on device generated from cuphyPrachOccaStatPrms_t needed for for PRACH receiver processing.
 */
struct PrachDeviceInternalStaticParamPerOcca
{
    PrachParams prach_params;
    float* prach_workspace_buffer;
    __half2* d_y_u_ref;
};

/**
 * @brief Per cell Intermediate structure on device generated from cuphyPrachDynPrms_t needed for for PRACH receiver processing.
 */
struct PrachInternalDynParamPerOcca
{
    __half2* dataRx;
    uint16_t occaPrmStatIdx; /*!< Index to occasion-static parameter information */
    uint16_t occaPrmDynIdx;  /*!< Index to occasion-dynamic parameter information */
    float thr0;
};

/** @brief: Create initial CUDA graph with some default values as parameters which can be changed later
 *  @return cuphy status
 */
cuphyStatus_t cuphyPrachCreateGraph(cudaGraph_t* graph, cudaGraphExec_t* graphInstance, std::vector<cudaGraphNode_t>& nodes,  cudaStream_t strm,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalStaticParamPerOcca* h_staticParam,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t nMaxOccasions,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    uint cudaDeviceArch);

/** @brief: Update CUDA graph in case any of dynamic parameters changed
 *  @return cuphy status
 */
cuphyStatus_t cuphyPrachUpdateGraph(cudaGraphExec_t graphInstance, std::vector<cudaGraphNode_t>& nodes,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalDynParamPerOcca* h_dynParam,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint32_t*& prev_num_detectedPrmb_addr,
                                    uint32_t*& prev_prmbIndex_estimates_addr,
                                    float*& prev_prmbDelay_estimates_addr,
                                    float*& prev_prmbPower_estimates_addr,
                                    float*& prev_ant_rssi_addr,
                                    float*& prev_rssi_addr,
                                    float*& prev_interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t& nPrevOccaProc,
                                    uint16_t nOccaProc,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    std::vector<char>& prevActiveOccasions);

/** @brief: Launch CUDA graph with all PRACH receiver kernels that do processing at receive end of PRACH.
 *  @return cuphy status
 */
cuphyStatus_t cuphyPrachLaunchGraph(cudaGraphExec_t graphInstance, cudaStream_t strm);

/** @brief: Launch PRACH receiver kernels that do processing at receive end of PRACH.
 *  @return cuphy status
 */
cuphyStatus_t cuphyPrachReceiver(const PrachInternalDynParamPerOcca* d_dynParam,
                                const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                const PrachInternalDynParamPerOcca* h_dynParam,
                                const PrachInternalStaticParamPerOcca* h_staticParam,
                                uint32_t* num_detectedPrmb_addr,
                                uint32_t* prmbIndex_estimates_addr,
                                float* prmbDelay_estimates_addr,
                                float* prmbPower_estimates_addr,
                                float* ant_rssi_addr,
                                float* rssi_addr,
                                float* interference_addr,
                                uint16_t nOccaProc,
                                uint16_t maxAntenna,
                                uint max_l_oran_ant,
                                uint max_ant_u,
                                uint max_nfft,
                                int max_zoneSizeExt,
                                uint cudaDeviceArch,
                                cudaStream_t strm);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */
#endif // PRACH_RECEIVER_HPP_INCLUDED_
