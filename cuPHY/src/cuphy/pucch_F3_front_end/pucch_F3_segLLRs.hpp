/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

#pragma once

static constexpr uint8_t  F3_SEG_LLR_UCI_PER_Block = 1; // floor(1024/(CUPHY_PUCCH_F3_MAX_PRB*12))
static constexpr uint8_t  F3_SEG_LLR_THREAD_PER_UCI = 192; // 12*CUPHY_PUCCH_F3_MAX_PRB
static constexpr uint16_t F3_SEG_LLR_THREAD_PER_BLOCK = 192; // F3_SEG_LLR_THREAD_PER_UCI * F3_SEG_LLR_UCI_PER_Block
static constexpr uint16_t TEMP_LLR_ARR_MAX_SIZE = 4608; // CUPHY_PUCCH_F3_MAX_PRB*12*(14-12)[maximum number of data symbols]*2[QPSK]

struct cuphyPucchF3SegLLRs
{};

struct perUciPrms
{
    uint8_t nSym;
    uint8_t nSym_data;
    uint8_t nSym_dmrs;
    uint8_t Qm;

    uint16_t nSymUci;
    uint16_t E_seg1;
    uint16_t E_seg2;
};
typedef struct perUciPrms perUciPrms_t;

struct pucchF3SegLLRsDynDescr
{
    uint16_t numUcis;
    perUciPrms_t perUciPrmsArray[CUPHY_PUCCH_F3_MAX_UCI];
    
    __half* pInLLRaddrs[CUPHY_PUCCH_F3_MAX_UCI];
};
typedef struct pucchF3SegLLRsDynDescr pucchF3SegLLRsDynDescr_t;

struct pucchF3SegLLRsKernelArgs
{
    pucchF3SegLLRsDynDescr_t*  pDynDescr;
};
typedef struct pucchF3SegLLRsKernelArgs pucchF3SegLLRsKernelArgs_t;

class pucchF3SegLLRs : public cuphyPucchF3SegLLRs
{
public:
    pucchF3SegLLRs();
    ~pucchF3SegLLRs() = default;
    pucchF3SegLLRs(pucchF3SegLLRs const&)            = delete;
    pucchF3SegLLRs& operator=(pucchF3SegLLRs const&) = delete;

    void setup(uint16_t                             nF3Ucis,                 
               cuphyPucchUciPrm_t*                  pF3UciPrms,
               __half**                             pDescramLLRaddrs,
               pucchF3SegLLRsDynDescr_t*            pCpuDynDesc,
               void*                                pGpuDynDesc,
               bool                                 enableCpuToGpuDescrAsyncCpy,
               cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
               cudaStream_t                         strm);

    void kernelSelect(uint16_t                           nF3Ucis,
                      cuphyPucchF3SegLLRsLaunchCfg_t*    pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    pucchF3SegLLRsKernelArgs_t m_kernelArgs;
};