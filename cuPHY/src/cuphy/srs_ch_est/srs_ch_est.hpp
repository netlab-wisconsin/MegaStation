/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(SRS_CH_EST_HPP_INCLUDED_)
#define SRS_CH_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"

// Implementation of the channel estimator interface exposed as an opaque data type to abstract out implementation
// details (channel estimator C++ class). The channel estimator is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphySrsChEst
{};

namespace srs_ch_est
{
static constexpr uint8_t N_MAX_SRS_SYMS = 4;

// Tensor parameters needed to access Channel estimator input/output tensors
template <size_t NDim>
struct srsChEstTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using srsChEstTensorPrm_t = srsChEstTensorPrm<NDim>;

// Channel estimator static descriptor
struct srsChEstStatDescr
{
    srsChEstTensorPrm_t<CUPHY_SRS_CH_EST_N_DIM_FREQ_INTERP_COEFS> tPrmFreqInterpCoefs; // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
};
typedef struct srsChEstStatDescr srsChEstStatDescr_t;

// Channel estimator dynamic descriptor
struct srsChEstDynDescr
{
    uint8_t  nIter;
    uint16_t nBSAnts;
    uint8_t  nLayers;
    uint16_t nPrb;
    uint16_t scsKHz;
    uint8_t  nCycShifts;
    uint8_t  nCombs;
    uint8_t  nSrsSyms;
    uint8_t  srsSymPos[N_MAX_SRS_SYMS];
    uint16_t nZc;
    uint8_t  zcSeqNum;
    float    delaySpreadSecs;

    srsChEstTensorPrm_t<CUPHY_SRS_CH_EST_N_DIM_DATA_RX> tPrmDataRx; // (NF, ND, N_BS_ANTS)
    srsChEstTensorPrm_t<CUPHY_SRS_CH_EST_N_DIM_H_EST>   tPrmHEst;   // (N_BS_ANTS, N_SRS_CH_EST, N_SRS_CYC_SHIFTS, N_SRS_COMBS)
    srsChEstTensorPrm_t<CUPHY_SRS_CH_EST_N_DIM_DBG>     tPrmDbg;
};
typedef struct srsChEstDynDescr srsChEstDynDescr_t;

typedef struct _srsChEstKernelLaunchGeo
{
    dim3 gridDim;
    dim3 blockDim;
} srsChEstKernelLaunchGeo_t;

// Channel estimator kernel arguments (supplied via descriptors)
typedef struct _srsChEstKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    // srsChEstStatDescr_t const* pStatDescr;
    srsChEstStatDescr_t* pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_SRS_CH_EST_N_HOM_CFG)
    srsChEstDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_SRS_CH_EST_N_HOM_CFG dynamic descriptors
} srsChEstKernelArgs_t;

// Forward declaration for launch parameters
struct srsChEstKernelLaunchPrms;
using srsChEstKernelLauncher_t = std::function<void(srsChEstKernelLaunchPrms&, cudaStream_t&)>;
using srsChEstDynDescrVec_t    = std::array<srsChEstDynDescr_t, CUPHY_SRS_CH_EST_N_HET_CFG>;
using srsChEstStrmVec_t        = std::array<cudaStream_t, CUPHY_SRS_CH_EST_N_HET_CFG>;
#if 0
using srsChEstKernelVec_t          = std::array<srsChEstKernelLauncher_t, CUPHY_SRS_CH_EST_N_HET_CFG>;
using srsChEstKernelLaunchGeoVec_t = std::array<srsChEstKernelLaunchGeo_t, CUPHY_SRS_CH_EST_N_HET_CFG>;
using srsChEstKernelArgsVec_t      = std::array<srsChEstKernelArgs_t, CUPHY_SRS_CH_EST_N_HET_CFG>;

using srsChEstStatDescrCpu_t   = cuphy::buffer<srsChEstStatDescr_t, cuphy::pinned_alloc>;
using srsChEstStatDescrGpu_t   = cuphy::buffer<srsChEstStatDescr_t, cuphy::device_alloc>;
using srsChEstDynDescrVecCpu_t = cuphy::buffer<srsChEstDynDescrVec_t, cuphy::pinned_alloc>;
using srsChEstDynDescrVecGpu_t = cuphy::buffer<srsChEstDynDescrVec_t, cuphy::device_alloc>;
#endif

// Channel estimation kernel launch parameters (kernel arguments, launch geometry, kernel launcher)
static constexpr uint32_t SRS_CH_EST_KERNEL_N_ARGS = 1; // channel estimation kernel has a single argument of type srsChEstKernelArgs_t
struct srsChEstKernelLaunchPrms
{
    srsChEstKernelArgs_t      args;
    srsChEstKernelLaunchGeo_t launchGeo;
    srsChEstKernelLauncher_t  launcher;
    // Graph arguments
    std::array<void*, SRS_CH_EST_KERNEL_N_ARGS> kernelArgs;
    void*                                       kernelFunc;
};
typedef struct srsChEstKernelLaunchPrms srsChEstKernelLaunchPrms_t;
using srsChEstKernelLaunchPrmsVec_t = std::array<srsChEstKernelLaunchPrms_t, CUPHY_SRS_CH_EST_N_HET_CFG>;

using srsChEstGraphNodePrmsVec_t = std::array<cuphyPuschRxFeGraphNodePrms_t, CUPHY_SRS_CH_EST_N_HET_CFG>;

// Class implementation of the channel estimation component
class srsChEst : public cuphySrsChEst {
public:
    srsChEst()                = default;
    ~srsChEst()               = default;
    srsChEst(srsChEst const&) = delete;
    srsChEst& operator=(srsChEst const&) = delete;

    // initialize channel estimator object and static component descriptor
    void init(tensor_pair&         tFreqInterpCoefs,
              bool                 enableCpuToGpuDescrAsyncCpy,
              srsChEstStatDescr_t& statDescrCpu,
              void*                pStatDescrGpu,
              cudaStream_t         strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(cuphySrsChEstDynPrms_t const* pDynPrms,
               tensor_pair&                  tDataRx,
               tensor_pair&                  tHEst,
               tensor_pair&                  tDbg,
               bool                          enableCpuToGpuDescrAsyncCpy,
               srsChEstDynDescrVec_t&        dynDescrVecCpu,
               void*                         pDynDescrsGpu,
               cudaStream_t                  strm);

    // run the component (launching kernels on supplied CUDA streams and providing parameters and arguments
    // setup previously)
    void run(srsChEstStrmVec_t& strmVec);

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    void copyTensorPair(tensor_pair& tensorPair, void** ppDstAddr, int* pDstStrideArr);

    void kernelSelectL1(bool                        enIter,
                        srsChEstDynDescr_t&         dynPrms,
                        tensor_pair&                tDataRx,
                        tensor_pair&                tH,
                        srsChEstKernelLaunchPrms_t& launchPrms);

    template <typename TStorage, typename TDataRx, typename TCompute>
    void kernelSelectL0(bool                        enIter,
                        srsChEstDynDescr_t&         dynPrms,
                        srsChEstKernelLaunchPrms_t& launchPrms);

    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_PRB_PER_THRD_BLK_ITER,
              uint32_t N_PRB_PER_PRB_GRP,
              uint32_t N_PRB_PER_SRS_CH_EST,
              uint32_t N_SRS_COMBS,
              uint32_t N_SRS_CYC_SHIFTS>
    void
    windowedSrsChEst(uint8_t                     nIter,
                     uint16_t                    nBSAnts,
                     uint16_t                    nPrb,
                     uint8_t                     nSrsSyms,
                     srsChEstKernelLaunchPrms_t& launchPrms);

    template <uint32_t N_PRB_PER_THRD_BLK_ITER,
              uint32_t N_PRB_PER_PRB_GRP,
              uint32_t N_PRB_PER_SRS_CH_EST,
              uint32_t N_SRS_COMBS,
              uint32_t N_SRS_CYC_SHIFTS>
    void
    computeKernelLaunchGeo(uint8_t                    nIter,
                           uint16_t                   nBSAnts,
                           uint16_t                   nPrb,
                           uint8_t                    nSrsSyms,
                           srsChEstKernelLaunchGeo_t& launchGeo);

    // class state modifed by setup saved in data member
    srsChEstKernelLaunchPrmsVec_t m_kernelLaunchPrmsVec;
};

} // namespace srs_ch_est

#endif // !defined(SRS_CH_EST_HPP_INCLUDED_)
