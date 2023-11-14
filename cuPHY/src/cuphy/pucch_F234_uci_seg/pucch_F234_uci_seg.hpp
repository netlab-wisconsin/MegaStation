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

#pragma once
static constexpr uint16_t F234_UCI_SEG_UCI_PER_Block    = 18;
static constexpr uint16_t F234_UCI_SEG_THREAD_PER_UCI   = 54; // ceil(1706/32), where 1706 is the maximum BitLengthHarq and BitLengthCsiPart1 size
static constexpr uint16_t F234_UCI_SEG_THREAD_PER_BLOCK = 972; // F234_UCI_SEG_UCI_PER_Block * F234_UCI_SEG_THREAD_PER_UCI

struct cuphyPucchF234UciSeg
{};

struct perUciPrmsF234UciSeg
{
    uint16_t bitLenHarq;
    uint16_t bitLenSr;
    uint16_t bitLenCsiPart1;

    uint32_t uciSeg1PayloadByteOffset;
    uint32_t harqPayloadByteOffset;
    uint32_t srPayloadByteOffset;
    uint32_t csi1PayloadByteOffset;
};
typedef struct perUciPrmsF234UciSeg perUciPrmsF234UciSeg_t;

struct pucchF234UciSegDynDescr
{
    uint16_t nF2Ucis;
    uint16_t nF3Ucis;

    uint8_t*  pUciPayloadsGpu;

    perUciPrmsF234UciSeg_t F2PerUciPrmsArray[CUPHY_PUCCH_F2_MAX_UCI];
    perUciPrmsF234UciSeg_t F3PerUciPrmsArray[CUPHY_PUCCH_F3_MAX_UCI];
};
typedef struct pucchF234UciSegDynDescr pucchF234UciSegDynDescr_t;

struct pucchF234UciSegKernelArgs
{
    pucchF234UciSegDynDescr_t*  pDynDescr;
};
typedef struct pucchF234UciSegKernelArgs pucchF234UciSegKernelArgs_t;

class pucchF234UciSeg : public cuphyPucchF234UciSeg
{
public:
    pucchF234UciSeg();
    ~pucchF234UciSeg() = default;
    pucchF234UciSeg(pucchF234UciSeg const&)            = delete;
    pucchF234UciSeg& operator=(pucchF234UciSeg const&) = delete;
    
    void setup(uint16_t                         nF2Ucis,
               uint16_t                         nF3Ucis,
               cuphyPucchUciPrm_t*              F2UciPrms,
               cuphyPucchUciPrm_t*              F3UciPrms,
               cuphyPucchF234OutOffsets_t*&     F2OutOffsetsCpu,
               cuphyPucchF234OutOffsets_t*&     F3OutOffsetsCpu,
               uint8_t*                         uciPayloadsGpu,
               pucchF234UciSegDynDescr_t*       pCpuDynDesc,
               void*                            pGpuDynDesc,
               bool                             enableCpuToGpuDescrAsyncCpy,
               cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg,
               cudaStream_t                     strm);
               
    void kernelSelect(uint16_t                         nF2Ucis,
                      uint16_t                         nF3Ucis,
                      cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    pucchF234UciSegKernelArgs_t m_kernelArgs;
};