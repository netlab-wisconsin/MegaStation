/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CFO_TA_EST_HPP_INCLUDED_)
#define CFO_TA_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <vector>

#define USE_SPLIT_CFO_TA (CUPHY_ENABLE_SUB_SLOT_PROCESSING)

// Implementation of the CFO estimator interface exposed as an opaque data type to abstract out implementation
// details (carrier frequency offset estimator C++ class). The CFO estimator is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPuschRxCfoTaEst
{};

namespace cfo_ta_est
{

// Tensor parameters needed to access CFO estimator input/output tensors
template <size_t NDim>
struct puschRxCfoTaEstTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxCfoTaEstTensorPrm_t = puschRxCfoTaEstTensorPrm<NDim>;

// CFO estimator static descriptor
struct puschRxCfoTaEstStatDescr
{
};
typedef struct puschRxCfoTaEstStatDescr puschRxCfoTaEstStatDescr_t;


// CFO estimator dynamic descriptor
struct puschRxCfoTaEstDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
    uint16_t                 nUeGrps;
};
typedef struct puschRxCfoTaEstDynDescr puschRxCfoTaEstDynDescr_t;

// CFO estimator kernel arguments (supplied via descriptors)
typedef struct _puschRxCfoTaEstKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    puschRxCfoTaEstStatDescr_t* pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CFO_EST_N_HOM_CFG)
    puschRxCfoTaEstDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CFO_EST_N_HOM_CFG dynamic descriptors
} puschRxCfoTaEstKernelArgs_t;

// Forward declaration for launch parameters
using puschRxCfoTaEstDynDescrVec_t   = std::array<puschRxCfoTaEstDynDescr_t, CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS>;
using puschRxCfoTaEstKernelArgsArr_t = std::array<puschRxCfoTaEstKernelArgs_t, CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS>;

// Class implementation of the CFO estimation component
class puschRxCfoTaEst : public cuphyPuschRxCfoTaEst 
{
public:
    puschRxCfoTaEst()                    = default;
    ~puschRxCfoTaEst()                   = default;
    puschRxCfoTaEst(puschRxCfoTaEst const&) = delete;
    puschRxCfoTaEst& operator=(puschRxCfoTaEst const&) = delete;

    // initialize CFO estimator object and static component descriptor
    void init(bool                        enableCpuToGpuDescrAsyncCpy,
              puschRxCfoTaEstStatDescr_t& statDescrCpu,
              void*                       pStatDescrGpu,
              cudaStream_t                strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    // @todo: replace with new API structures once integrated
    cuphyStatus_t setup(cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                        cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                        uint16_t                          nUeGrps,
                        uint32_t                          nMaxPrb,
                        bool                              enableCpuToGpuDescrAsyncCpy,
                        puschRxCfoTaEstDynDescrVec_t&     dynDescrVecCpu,
                        void*                             pDynDescrsGpu,
                        cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                        cudaStream_t                      strm);


    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    void kernelSelectL2(uint16_t                         nBSAnts,
                        uint8_t                          nLayers,
                        uint8_t                          nDmrsAddlnPos,
                        uint16_t                         nMaxPrb,
                        uint16_t                         nUeGrps,
                        cuphyDataType_t                  hEstType,
                        cuphyDataType_t                  cfoEstType,
                        cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);

    template <typename TStorage, typename TDataRx, typename TCompute>
    void kernelSelectL1(uint16_t                         nBSAnts,
                        uint8_t                          nLayers,
                        uint8_t                          nDmrsAddlnPos,
                        uint16_t                         nMaxPrb,
                        uint16_t                         nUeGrps,
                        cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);

    template <typename TStorage, typename TDataRx, typename TCompute, uint32_t N_TIME_CH_EST>
    void kernelSelectL0(uint16_t                         nBSAnts,
                        uint8_t                          nLayers,
                        uint16_t                         nMaxPrb,
                        uint16_t                         nUeGrps,
                        cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);

    template <typename TStorageIn,
              typename TStorageOut,
              typename TCompute,
              uint32_t N_BS_ANTS,
              uint32_t N_LAYERS,
              uint32_t N_TIME_CH_EST>
    void cfoTaEstLowMimo(uint16_t                         nMaxPrb,
                         uint16_t                         nUeGrps,
                         cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);    

    template <uint32_t N_LAYERS,
              uint32_t THRD_GRP_TILE_SIZE,
              uint32_t N_THRD_GRP_TILES_PER_LAYER,
              uint32_t N_PRB_PER_THRD_BLK>
    void cfoTaEstLowMimoKernelLaunchGeo(uint16_t nMaxPrb,
                                        uint16_t nUeGrps,
                                        dim3&    gridDim,
                                        dim3&    blockDim);    

    // class state modifed by setup saved in data member
    puschRxCfoTaEstKernelArgsArr_t m_kernelArgsArr;
};

} // namespace cfo_ta_est

#endif // !defined(CFO_TA_EST_HPP_INCLUDED_)
