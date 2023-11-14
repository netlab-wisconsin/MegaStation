/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CH_EST_HPP_INCLUDED_)
#define CH_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <vector>

// Implementation of the channel estimator interface exposed as an opaque data type to abstract out implementation
// details (channel estimator C++ class). The channel estimator is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPuschRxChEst
{};

namespace ch_est
{

enum class dmrsCfg_t : uint32_t
{
    DMRS_CFG0 = CUPHY_DMRS_CFG0, // 1 layer : DMRS grid 0   ; fOCC = [+1, +1]          ; 1 DMRS symbol
    DMRS_CFG1 = CUPHY_DMRS_CFG1, // 2 layers: DMRS grids 0,1; fOCC = [+1, +1]          ; 1 DMRS symbol
    DMRS_CFG2 = CUPHY_DMRS_CFG2  // 4 layers: DMRS grids 0,1; fOCC = [+1, +1], [+1, -1]; 1 DMRS symbol
};

// Tensor parameters needed to access Channel estimator input/output tensors
template <size_t NDim>
struct puschRxChEstTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxChEstTensorPrm_t = puschRxChEstTensorPrm<NDim>;

// Channel estimator static descriptor
struct puschRxChEstStatDescr
{
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefs;       // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefs4;      // 25 x 24 x 3
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefsSmall;  // 37 x 18 x 3

    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmShiftSeq;         // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_UNSHIFT_SEQ>       tPrmUnShiftSeq;       // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)

    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmShiftSeq4;        // Small shift sequence. Dim: 24 x 1
    puschRxChEstTensorPrm_t<CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ>         tPrmUnShiftSeq4;      // Small un-shift sequence. Dim: 49 x 1

    const uint32_t *                                                       pSymbolRxStatus;      // Pointer to GPU array indicating if symbol received for all cells scheduled for each time slot
};
typedef struct puschRxChEstStatDescr puschRxChEstStatDescr_t;

// Channel estimator dynamic descriptor
typedef struct _puschRxChEstDynDescr
{
    uint8_t                   chEstTimeInst; // Time domain instance of channel estimation
    uint8_t                   dmrsSymPos[N_MAX_DMRS_SYMS];
    cuphyPuschRxUeGrpPrms_t*  pDrvdUeGrpPrms;
    uint32_t                  hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // Mapping of Heterogenous config to UE group
    uint8_t*                  pPreEarlyHarqWaitKernelStatus_d;       
    uint8_t*                  pPostEarlyHarqWaitKernelStatus_d;   
    uint16_t                  waitTimeOutPreEarlyHarqUs;       // timeout threshold for wait kernel prior to starting early HARQ processing
    uint16_t                  waitTimeOutPostEarlyHarqUs;      // timeout threshold for wait kernel after finishing early HARQ processing
    uint64_t                  mPuschStartTimeNs;      // start time as reference point to measure timeout
}puschRxChEstDynDescr_t;



// Channel estimator kernel arguments (supplied via descriptors)
typedef struct _puschRxChEstKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    // puschRxChEstStatDescr_t const* pStatDescr;
    puschRxChEstStatDescr_t* pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CH_EST_N_HOM_CFG)
    puschRxChEstDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CH_EST_N_HOM_CFG dynamic descriptors
} puschRxChEstKernelArgs_t;

// Forward declaration for launch parameters
using puschRxChEstDynDescrVec_t   = std::array<puschRxChEstDynDescr_t, CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS>;
using puschRxChEstKernelArgsArr_t = std::array<puschRxChEstKernelArgs_t, CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS>;

// Class implementation of the channel estimation component
class puschRxChEst : public cuphyPuschRxChEst 
{
public:
    puschRxChEst()                    = default;
    ~puschRxChEst()                   = default;
    puschRxChEst(puschRxChEst const&) = delete;
    puschRxChEst& operator=(puschRxChEst const&) = delete;

    // initialize channel estimator object and static component descriptor
    void init(tensor_pair&             tFreqInterpCoefs,
              tensor_pair&             tFreqInterpCoefs4,
              tensor_pair&             tFreqInterpCoefsSmall,
              tensor_pair&             tShiftSeq,
              tensor_pair&             tShiftSeq4,
              tensor_pair&             tUnShiftSeq,
              tensor_pair&             tUnShiftSeq4,
              const uint32_t*          pSymStat,
              bool                     enableCpuToGpuDescrAsyncCpy,
              void**                   ppStatDescrsCpu,
              void**                   ppStatDescrsGpu,
              cudaStream_t             strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    // @todo: replace with new API structures once integrated
    cuphyStatus_t setup(cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                        cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                        uint16_t                              nUeGrps,
                        uint8_t                               enableDftSOfdm,
                        uint8_t*                              pPreEarlyHarqWaitKernelStatus_d,
                        uint8_t*                              pPostEarlyHarqWaitKernelStatus_d,
                        const uint16_t                        waitTimeOutPreEarlyHarqUs,
                        const uint16_t                        waitTimeOutPostEarlyHarqUs,
                        bool                                  enableCpuToGpuDescrAsyncCpy,
                        void**                                ppDynDescrsCpu,
                        void**                                ppDynDescrsGpu,
                        cuphyPuschRxChEstLaunchCfgs_t*        pLaunchCfgs,
                        uint8_t                               enableEarlyHarqProc,
                        cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPreEHQ,
                        cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPostEHQ,
                        cudaStream_t                          strm);

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    cuphyStatus_t batch(uint32_t                       chEstInstIdx,
               cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrms,
               uint16_t                       nUeGrps,
               uint8_t                        enableDftSOfdm,
               uint32_t&                      nHetCfgs,
               puschRxChEstDynDescrVec_t&     dynDescrVecCpu);
    void kernelSelectL2(uint16_t                        nBSAnts,
                        uint8_t                         nLayers,
                        uint8_t                         nDmrsSyms,
                        uint8_t                         nDmrsGridsPerPrb,
                        uint16_t                        nTotalDataPrb,
                        uint8_t                         Nh,
                        uint16_t                        nUeGrps,
                        uint8_t                         enableDftSOfdm,
                        cuphyDataType_t                 dataRxType,
                        cuphyDataType_t                 hEstType,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <typename TStorage, typename TDataRx, typename TCompute>
    void kernelSelectL1(uint16_t                        nBSAnts,
                        uint8_t                         nLayers,
                        uint8_t                         nDmrsSyms,
                        uint8_t                         nDmrsGridsPerPrb,
                        uint16_t                        nTotalDataPrb,
                        uint8_t                         Nh,
                        uint16_t                        nUeGrps,
                        uint8_t                         enableDftSOfdm,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <typename TStorage, 
              typename TDataRx, 
              typename TCompute, 
              uint32_t N_LAYERS, 
              uint32_t N_DMRS_GRIDS_PER_PRB, 
              uint32_t N_DMRS_SYMS>
    void kernelSelectL0(uint16_t                        nTotalDataPrb,
                        uint16_t                        nUeGrps,
                        uint32_t                        nRxAnt,
                        uint8_t                         enableDftSOfdm,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);


    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
              uint32_t N_DMRS_SYMS>
    void windowedChEst(uint16_t                        nTotalDataPrb,
                       uint16_t                        nUeGrps,
                       uint32_t                        nRxAnt,
                       uint8_t                         enabelDftSOfdm,
                       cuphyPuschRxChEstLaunchCfg_t&   launchCfg);


    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_PRBS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_SYMS>
    void smallChEst(uint16_t                        nUeGrps,
                    uint32_t                        nRxAnt,
                    uint8_t                         enabelDftSOfdm,
                    cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER>
    void
    computeKernelLaunchGeo(uint16_t nTotalDataPrb,
                           uint16_t nUeGrps,
                           uint32_t nRxAnt,
                           dim3&    gridDim,
                           dim3&    blockDim);

    // class state modifed by setup saved in data member
    puschRxChEstKernelArgsArr_t m_kernelArgsArr[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];

    // kernelSelectL0 -  5 templates (PRB cluster sizes)
    // kernelSelectL1 - 11 templates (Layers, DMRS symbols, DMRS grids)
    // Max # of heterogenous configs needed = 11 * 5 = 55 but capping CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS to 16
    // @todo: reduce the number of nRxAnt x nLayer and time-domain templates

    typedef struct _puschRxChEstHetCfg
    {
        CUfunction func;
        uint16_t   nMaxPrb; // Maximum number of PRBs across all UE groups
        uint16_t   nMaxRxAnt; // Maximum number of Rx Antenna across all UE groups
        uint16_t   nUeGrps;
    } puschRxChEstHetCfg_t;
    using puschRxChEstHetCfgArr_t = std::array<puschRxChEstHetCfg_t, CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS>;
    puschRxChEstHetCfgArr_t m_hetCfgsArr[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
};

} // namespace ch_est

#endif // !defined(CH_EST_HPP_INCLUDED_)
