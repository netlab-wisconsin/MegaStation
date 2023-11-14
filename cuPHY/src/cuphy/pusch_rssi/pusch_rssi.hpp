/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(PUSCH_RX_RSSI_HPP_INCLUDED_)
#define PUSCH_RX_RSSI_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <vector>

// Implementation of the PUSCH metrics interface exposed as an opaque data type to abstract out implementation
// details (PUSCH metrics C++ class). The PUSCH metrics is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPuschRxRssi
{};

namespace puschRx_rssi
{

//--------------------------------------------------------------------------------------------------------
// RSSI measurement

// Tensor parameters needed to access RSSI measurment input/output tensors
template <size_t NDim>
struct puschRxRssiTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxRssiTensorPrm_t = puschRxRssiTensorPrm<NDim>;

// RSSI dynamic descriptor
struct puschRxRssiDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
};
typedef struct puschRxRssiDynDescr puschRxRssiDynDescr_t;

// RSSI measurement kernel arguments (supplied via descriptors)
typedef struct _puschRxRssiKernelArgs
{
    puschRxRssiDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_RSSI_MEAS_N_HOM_CFG dynamic descriptors
} puschRxRssiKernelArgs_t;

// Forward declaration for launch parameters
using puschRxRssiDynDescrVec_t = std::array<puschRxRssiDynDescr_t, CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS>;
using puschRxRssiKernelArgsArr_t = std::array<puschRxRssiKernelArgs_t, CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS>;

//--------------------------------------------------------------------------------------------------------
// RSRP measurement

// Tensor parameters needed to access RSSI measurment input/output tensors
template <size_t NDim>
struct puschRxRsrpTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxRsrpTensorPrm_t = puschRxRsrpTensorPrm<NDim>;

// RSSI dynamic descriptor
struct puschRxRsrpDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
};
typedef struct puschRxRsrpDynDescr puschRxRsrpDynDescr_t;

// RSSI measurement kernel arguments (supplied via descriptors)
typedef struct _puschRxRsrpKernelArgs
{
    puschRxRsrpDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_RSRP_MEAS_N_HOM_CFG dynamic descriptors
} puschRxRsrpKernelArgs_t;

// Forward declaration for launch parameters
using puschRxRsrpDynDescrVec_t = std::array<puschRxRsrpDynDescr_t, CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS>;
using puschRxRsrpKernelArgsArr_t = std::array<puschRxRsrpKernelArgs_t, CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS>;
//--------------------------------------------------------------------------------------------------------


// Class implementation of the PUSCH metrics component
class puschRxRssi : public cuphyPuschRxRssi 
{
public:
    puschRxRssi()                   = default;
    ~puschRxRssi()                  = default;
    puschRxRssi(puschRxRssi const&) = delete;
    puschRxRssi& operator=(puschRxRssi const&) = delete;

    //--------------------------------------------------------------------------------------------------------
    // RSSI estimation

    // setup object state and dynamic component descriptor in prepration towards execution

    cuphyStatus_t setupRssiMeas(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                uint16_t                      nUeGrps,
                                uint32_t			          nMaxPrb,
                                uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                puschRxRssiDynDescrVec_t&     dynDescrVecCpu,
                                void*                         pDynDescrsGpu,
                                cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
                                cudaStream_t                  strm);

    cuphyStatus_t setupRsrpMeas(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                uint16_t                      nUeGrps,
                                uint32_t			          nMaxPrb,
                                uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                puschRxRsrpDynDescrVec_t&     dynDescrVecCpu,
                                void*                         pDynDescrsGpu,
                                cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
                                cudaStream_t                  strm);

    static void getDescrInfo(size_t& rssiDynDescrSizeBytes, size_t& rssiDynDescrAlignBytes, size_t& rsrpDynDescrSizeBytes, size_t& rsrpDynDescrAlignBytes);

private:
    template<typename TStorageIn,
             typename TStorageOut,
             typename TCompute>
    void rssiMeas(uint16_t                     nMaxPrb,
                  uint16_t                     nSymb,
                  uint16_t                     nRxAnt,
                  uint16_t                     nUeGrps,
                  cuphyPuschRxRssiLaunchCfg_t& launchCfg);    

    template<uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
             uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
             uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
    void rssiMeasLaunchGeo(uint16_t nMaxPrb, 
                           uint8_t  nSymb, 
                           uint16_t nRxAnt,
                           uint16_t nUeGrps,
                           dim3&    gridDim,
                           dim3&    blockDim);

    void rssiMeasKernelSelect(uint16_t                     nMaxPrb,
                              uint16_t                     symbCnt,
                              uint16_t                     nRxAnt,
                              uint16_t                     nUeGrps,
                              cuphyDataType_t              dataRxType,
                              cuphyDataType_t              rssiFullType,
                              cuphyDataType_t              rssiType,
                              cuphyPuschRxRssiLaunchCfg_t& launchCfg);

    template<typename TStorageIn,
             typename TStorageOut,
             typename TCompute>
    void rsrpMeas(uint16_t                     nMaxPrb,
                  uint16_t                     nRxAnt,
                  uint16_t                     nUeGrps,
                  cuphyPuschRxRsrpLaunchCfg_t& launchCfg);    

    template<typename TCompute,
             uint32_t THRD_GRP_TILE_SIZE,            // # of threads in a thread group tile
             uint32_t N_THRD_GRP_TILES_PER_THRD_BLK, // number of thread group tiles in the thread block
             uint32_t N_SC_PER_THRD_BLK_ITER,        // number of subcarriers processed per thread block iteration
             uint32_t N_ITER_PER_THRD_BLK>           // # of iterations in a thread block
    void rsrpMeasLaunchGeo(uint16_t         nMaxPrb, 
                           uint16_t         nRxAnt,
                           uint16_t         nUeGrps,
                           dim3&            gridDim,
                           dim3&            blockDim,
                           size_t&          sharedMemBytes);

    void rsrpMeasKernelSelect(uint16_t                     nMaxPrb,
                              uint16_t                     nRxAnt,
                              uint16_t                     nUeGrps,
                              cuphyDataType_t              hEstType,
                              cuphyDataType_t              rsrpType,
                              cuphyPuschRxRsrpLaunchCfg_t& launchCfg);

    // class state modifed by setup saved in data member
    puschRxRssiKernelArgsArr_t m_rssiKernelArgsArr;
    puschRxRsrpKernelArgsArr_t m_rsrpKernelArgsArr;
};

} // namespace puschRx_rssi

#endif // !defined(PUSCH_RX_RSSI_HPP_INCLUDED_)
