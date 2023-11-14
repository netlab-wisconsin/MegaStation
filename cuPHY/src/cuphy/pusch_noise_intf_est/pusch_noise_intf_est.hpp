/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(PUSCH_RX_NOISE_INTF_EST_HPP_INCLUDED_)
#define PUSCH_RX_NOISE_INTF_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <vector>

// Implementation of the PUSCH noise-interference estimator interface exposed as an opaque data type to abstract out implementation
// details (PUSCH noise-interference estimator C++ class). The PUSCH noise-interference estimator is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPuschRxNoiseIntfEst
{};

namespace pusch_noise_intf_est
{

//--------------------------------------------------------------------------------------------------------
// RSSI measurement

// Tensor parameters needed to access Interference noise measurment input/output tensors
template <size_t NDim>
struct puschRxNoiseIntfEstTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxNoiseIntfEstTensorPrm_t = puschRxNoiseIntfEstTensorPrm<NDim>;

// RSSI dynamic descriptor
struct puschRxNoiseIntfEstDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
    bool                     compCovFlag;         // Flag to enable computation of noise covariance matrix
    uint8_t                  subSlotStageIndex;   // Index for dmrs group processed per kernel call in sub-slot processing
};
typedef struct puschRxNoiseIntfEstDynDescr puschRxNoiseIntfEstDynDescr_t;

// RSSI measurement kernel arguments (supplied via descriptors)
typedef struct _puschRxNoiseIntfEstKernelArgs
{
    puschRxNoiseIntfEstDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_RSSI_MEAS_N_HOM_CFG dynamic descriptors
} puschRxNoiseIntfEstKernelArgs_t;

// Forward declaration for launch parameters
using puschRxNoiseIntfEstDynDescrVec_t = std::array<puschRxNoiseIntfEstDynDescr_t, CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS * N_MAX_SUB_SLOT_STAGES>;
using puschRxNoiseIntfEstKernelArgsArr_t = std::array<puschRxNoiseIntfEstKernelArgs_t, CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS * N_MAX_SUB_SLOT_STAGES>; 

//--------------------------------------------------------------------------------------------------------

// Class implementation of the PUSCH interference noise estimation component
class puschRxNoiseIntfEst : public cuphyPuschRxNoiseIntfEst 
{
public:
    puschRxNoiseIntfEst()                   = default;
    ~puschRxNoiseIntfEst()                  = default;
    puschRxNoiseIntfEst(puschRxNoiseIntfEst const&) = delete;
    puschRxNoiseIntfEst& operator=(puschRxNoiseIntfEst const&) = delete;

    //--------------------------------------------------------------------------------------------------------
    // RSSI estimation

    // setup object state and dynamic component descriptor in prepration towards execution

 cuphyStatus_t setup(cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                     cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                     uint16_t                              nUeGrps,
                     uint32_t		                           nMaxPrb,
                     uint8_t                               enableDftSOfdm,
                     uint8_t                               isEarlyHarq,
                     uint8_t                               enableCpuToGpuDescrAsyncCpy,
                     void*                                 pDynDescrsCpu,
                     void*                                 pDynDescrsGpu,
                     cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
                     cudaStream_t                          strm,
                     uint8_t                               subSlotStageIdx);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    template <typename TStorageIn,
              typename TDataRx,
              typename TStorageOut,
              typename TCompute>      
    void noiseIntfEst(uint16_t                             nMaxPrb,
                      uint16_t                             nRxAnt,
                      uint16_t                             nUeGrps,
                      uint8_t                              enableDftSOfdm,
                      uint8_t                              isEarlyHarq,
                      cuphyPuschRxNoiseIntfEstLaunchCfg_t& launchCfg);
   
    template<uint32_t N_PRB_PER_THRD_BLK, typename TCompute>  // N_PRB_PER_THRD_BLK # of PRBs processed per thread block
    void noiseIntfEstLaunchGeo(uint16_t nMaxPrb,
                               uint16_t nRxAnt,
                               uint16_t nUeGrps,
                               dim3&    gridDim,
                               dim3&    blockDim,
                               size_t&  sharedMemBytes);

    void kernelSelect(uint16_t                     nMaxPrb,
                      uint16_t                     nRxAnt,
                      uint16_t                     nUeGrps,
                      uint8_t                      enableDftSOfdm,
                      uint8_t                      isEarlyHarq,
                      cuphyDataType_t              chEstType,
                      cuphyDataType_t              dataRxType,
                      cuphyDataType_t              noiseIntfVarType,
                      cuphyDataType_t              lwInvType,
                      cuphyPuschRxNoiseIntfEstLaunchCfg_t& launchCfg);

    // class state modifed by setup saved in data member
    puschRxNoiseIntfEstKernelArgsArr_t m_noiseIntfKernelArgsArr[2];
};

} // namespace pusch_noise_intf_est

#endif // !defined(PUSCH_RX_NOISE_INTF_EST_HPP_INCLUDED_)
