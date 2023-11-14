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

static constexpr uint32_t F3_CG_SIZE = 128; // Rounded up to next power of 2 from N_TONES_PER_PRB
static constexpr uint32_t F3_MAX_SYMS = 14;
static constexpr uint32_t F3_MAX_PRBS = CUPHY_PUCCH_F3_MAX_PRB;
static constexpr uint32_t MAX_DATA_SYMS_F3 = 12;
static constexpr uint32_t MAX_DMRS_SYMS_F3 = 4;
static constexpr uint32_t F3_UCIS_PER_BLOCK = 2;
static constexpr uint32_t F3_GROUPS_PER_BLOCK = 1;
static constexpr uint32_t F3_MAX_RX_ANTENNA = 4;
static constexpr uint32_t F3_DATA_FETCH_SCS = 4 * 12;

static constexpr float    defF3DTXthreshold = -3.0;

// Implementation of the PUCCH F3 receiver interface exposed as an opaque data type to abstract out implementation
// details (PUCCH F3 receiver C++ class). The PUCCH F3 receiver is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPucchF3Rx
{};

// Parameters for a single PUCCH F3 UCI
struct pucchF3UciPrms
{
    uint8_t  freqHopFlag;
    uint16_t bwpStart;
    uint8_t  startSym;
    uint16_t startPrb;
    uint8_t  nSym;
    uint8_t  groupHopFlag;
    uint8_t  sequenceHopFlag;
    uint16_t secondHopPrb;
    uint8_t  pi2Bpsk;
    uint8_t  prbSize;
    uint8_t  AddDmrsFlag;
    float    noiseVar;
    uint8_t  nSym_data;
    uint8_t  nSym_dmrs;
    uint8_t  SetSymData[12];
    uint8_t  SetSymDmrs[4];
    uint16_t uciOutputIdx;
    uint16_t E_tot;
    float    DTXthreshold;
    uint16_t cellIdx;
    uint16_t dataScramblingId;
    uint16_t rnti;
    uint16_t slotNum;
    uint16_t pucchHoppingId;
};
typedef struct pucchF3UciPrms pucchF3UciPrms_t;

// Pucch F3 receiver dynamic descriptor
struct pucchF3RxDynDescr
{
    pucchF3UciPrms_t     uciPrms[CUPHY_PUCCH_F3_MAX_UCI];          // parameters
    __half*              pDescramLLRaddrs[CUPHY_PUCCH_F3_MAX_UCI]; // address of output buffers
    cuphyPucchCellPrm_t* pCellPrms;                                // RX Antennas, slot num , hopping id and input slot buffer
    uint8_t*             pDTXflags;
    float*               pSinr;
    float*               pRssi;
    float*               pRsrp;
    float*               pInterf;
    float*               pNoiseVar;
    float*               pTaEst;
    uint16_t             numUcis;
};
typedef struct pucchF3RxDynDescr pucchF3RxDynDescr_t;

// PUCCH format 3 kernel arguments (supplied via descriptors)
typedef struct 
{
    pucchF3RxDynDescr_t*  pDynDescr;  
} pucchF3KernelArgs_t;

// Class implementation of PUCCH F3 reciver 
class pucchF3Rx : public cuphyPucchF3Rx 
{
public:
    pucchF3Rx(cudaStream_t strm);
    ~pucchF3Rx()                           = default;
    pucchF3Rx(pucchF3Rx const&)            = delete;
    pucchF3Rx& operator=(pucchF3Rx const&) = delete;

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
               uint16_t                   nCells,                         // number of cells
               uint16_t                   nF3Ucis,
               cuphyPucchUciPrm_t*        pF3UciPrms,
               cuphyPucchCellPrm_t*       pCmnCellPrms,                   // number of antennas, slot number, hopping idx and input slot buffer
               bool                       enableCpuToGpuDescrAsyncCpy,
               pucchF3RxDynDescr_t*       pCpuDynDesc,                     // pointer to descriptor in cpu
               void*                      pGpuDynDesc,                     // pointer to descriptor in gpu
               cuphyPucchF3RxLaunchCfg_t* pLaunchCfg,                      // pointer to launch configuration
               cudaStream_t               strm);                           // stream to perform copy

    void kernelSelect(uint16_t                   nUcis,
                      pucchF3RxDynDescr_t*       pCpuDynDesc, 
                      cuphyPucchF3RxLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);
    pucchF3KernelArgs_t m_kernelArgs;
    
private:
    static bool isConstMemInited;
    static std::mutex m_mutexConstMemInit;
};