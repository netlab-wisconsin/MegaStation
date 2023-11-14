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
#include "cuphy_internal.h"
#include "tensor_desc.hpp"
#define MAX_BITS_PER_RE 32

// Implementation of polSegDeRmDeItl interface exposed as an opaque data type to abstract out implementation
// details (polSegDeRmDeItl  C++ class). polSegDeRmDeItl is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyUciOnPuschSegLLRs0
{};


struct reGrid0_t 
{
    uint16_t nRes           = 0;
    uint16_t ReStride       = 0;
    uint32_t rmBufferOffset = 0;    
    uint8_t  gridOffset     = 0;
};

struct uciToUserMap_t
{
    uint16_t ueIdx;
    uint16_t ueGrpIdx;
};

struct perUciPrms0_t
{
    uint8_t nSym;  // number of symbols carrying UCI and/or SCH.

    reGrid0_t rvdHarqReGrids[MAX_ND_SUPPORTED];
    reGrid0_t harqReGrids[MAX_ND_SUPPORTED];
    reGrid0_t csi1ReGrids[MAX_ND_SUPPORTED];

    uint32_t descramOffsets[MAX_ND_SUPPORTED];
    bool     dmrsFlags[MAX_ND_SUPPORTED];
    uint32_t schRmBuffOffsets[MAX_ND_SUPPORTED];

    bool    harqPunctFlag;
    uint8_t harqSpx1Flag;
    uint8_t nBitsPerRe;
};


struct uciOnPuschSegLLRs0DynDescr_t{
    cuphyUciToSeg_t uciToSeg;

    perUciPrms0_t perUciPrmsArray[CUPHY_MAX_N_UCI_ON_PUSCH];

    // user indicies
    uciToUserMap_t uciToUserMap[CUPHY_MAX_N_UCI_ON_PUSCH];    //ToDo change to uint16_t*

    // pusch pipeline parameters
    PerTbParams*              pUePrmsGpu;
    cuphyPuschRxUeGrpPrms_t*  pUeGrpPrmsGpu;

    // input buffers
    tensor_ref_any<CUPHY_R_16F>   tEqOutLLRs[MAX_N_USER_GROUPS_SUPPORTED];
};


//  uciOnPuschSegLLRs0 kernel arguments (supplied via descriptors)
struct uciOnPuschSegLLRs0KernelArgs_t
{
    uciOnPuschSegLLRs0DynDescr_t*  pDynDescr;
};

class uciOnPuschSegLLRs0 : public cuphyUciOnPuschSegLLRs0
{
public:
    void setup(uint16_t                             nUciUes,
               uint16_t*                            pUciUeIdxs,
               PerTbParams*                         pTbPrmsCpu,
               PerTbParams*                         pTbPrmsGpu,
               uint16_t                             nUeGrps,
               cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu, 
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsGpu,
               cuphyUciToSeg_t                      uciToSeg,               
               uciOnPuschSegLLRs0DynDescr_t*        pCpuDynDesc,
               void*                                pGpuDynDesc,
               uint8_t                              enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs0LaunchCfg_t*  pLaunchCfg,
               cudaStream_t                         strm);
               




    // void kernelSelect(uint16_t                            nUciUes,
    //                   uint16_t*                           pUciUserIdxs,
    //                   PerTbParams*                        pTbPrmsCpu,
    //                   uint16_t*                           pNumPrbs,
    //                   uint8_t                             nPuschDataSym,
    //                   cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    uciOnPuschSegLLRs0KernelArgs_t m_kernelArgs;
};