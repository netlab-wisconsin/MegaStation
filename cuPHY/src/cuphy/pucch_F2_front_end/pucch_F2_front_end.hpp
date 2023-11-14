/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"
#include "cuphy_api.h"
#include "tensor_desc.hpp"
#include <mutex>

#pragma once

static constexpr uint32_t F2_CG_SIZE        = 64;
static constexpr uint32_t F2_MAX_SYMS       = 2;
static constexpr uint32_t F2_MAX_PRBS       = 16;
static constexpr uint32_t F2_MAX_RX_ANTENNA = 4;
static constexpr uint32_t F2_UCIS_PER_BLOCK = 1;

static constexpr float    defF2DTXthreshold = -3.0;

// Implementation of the PUCCH F2 receiver interface exposed as an opaque data type to abstract out implementation
// details (PUCCH F2 receiver C++ class). The PUCCH F2 receiver is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPucchF2Rx
{};

struct pucchF2UciPrms
{
    uint8_t  freqHopFlag;
    uint16_t bwpStart;
    uint8_t  startSym;
    uint16_t startPrb;
    uint8_t  nSym;
    uint16_t secondHopPrb;
    uint8_t  prbSize;
    float    noiseVar;
    uint16_t uciOutputIdx;
    uint16_t E_seg1;
    float    DTXthreshold;
    uint16_t rnti;
    uint16_t dataScramblingId;
    uint16_t DmrsScramblingId;
    uint32_t randomSeqScrm[16];
    uint16_t cellIdx;
};
typedef struct pucchF2UciPrms pucchF2UciPrms_t;

// Pucch F2 receiver dynamic descriptor
struct pucchF2RxDynDescr
{
    pucchF2UciPrms_t     uciPrms[CUPHY_PUCCH_F2_MAX_UCI];          // parameters
    __half*              pDescramLLRaddrs[CUPHY_PUCCH_F2_MAX_UCI]; // address of output buffers
    cuphyPucchCellPrm_t* pCellPrms;                                // RX Antennas, slot num , hopping id and input slot buffer
    uint8_t*             pDTXflags;
    float*               pRssi;
    float*               pRsrp;
    float*               pSinr;
    float*               pInterf;
    float*               pNoiseVar;
    float*               pTaEst;
    uint16_t             numUcis;
};
typedef struct pucchF2RxDynDescr pucchF2RxDynDescr_t;

// PUCCH format 2 kernel arguments (supplied via descriptors)
typedef struct 
{
    pucchF2RxDynDescr_t*  pDynDescr;  
} pucchF2KernelArgs_t;

// Class implementation of PUCCH F2 reciver 
class pucchF2Rx : public cuphyPucchF2Rx 
{
public:
    pucchF2Rx(cudaStream_t strm);
    ~pucchF2Rx()                           = default;
    pucchF2Rx(pucchF2Rx const&)            = delete;
    pucchF2Rx& operator=(pucchF2Rx const&) = delete;

    void InitConstantMem(cudaStream_t strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(cuphyTensorPrm_t*          pDataRx,
               __half**                   pDescramLLRaddrs,
               uint8_t*                   pDTXflags,
               float*                     pSinr,
               float*                     pRssi,
               float*                     pRsrp,
               float*                     pInterf,
               float*                     pNoiseVar,
               float*                     pTaEst,
               uint16_t                   nCells,                          // number of cells
               uint16_t                   nF2Ucis,
               cuphyPucchUciPrm_t*        pF2UciPrms,
               cuphyPucchCellPrm_t*       pCmnCellPrms,                    // number of antennas, slot number, hopping idx and input slot buffer
               bool                       enableCpuToGpuDescrAsyncCpy,
               pucchF2RxDynDescr_t*       pCpuDynDesc,                     // pointer to descriptor in cpu
               void*                      pGpuDynDesc,                     // pointer to descriptor in gpu
               cuphyPucchF2RxLaunchCfg_t* pLaunchCfg,                      // pointer to launch configuration
               cudaStream_t               strm);                           // stream to perform copy

    void kernelSelect(uint16_t                   nUcis,
                      pucchF2RxDynDescr_t*       pCpuDynDesc, 
                      cuphyPucchF2RxLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);
    pucchF2KernelArgs_t m_kernelArgs;

    private:
      static bool isConstMemInited;
      static std::mutex m_mutexConstMemInit;
};