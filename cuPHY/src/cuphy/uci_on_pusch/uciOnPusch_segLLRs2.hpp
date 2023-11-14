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
struct cuphyUciOnPuschSegLLRs2
{};


struct reGrid_t 
{
    uint16_t nRes           = 0;
    uint16_t ReStride       = 0;
    uint32_t rmBufferOffset = 0;
    uint8_t  gridOffset     = 0;  
};

struct harqAndCsi1RePrms_t
{
    uint8_t nSym;                       // number of symbols carrying UCI and/or SCH.
    bool    dmrsFlag[MAX_ND_SUPPORTED]; 

    uint32_t descramOffsets[MAX_ND_SUPPORTED];

    reGrid_t rvdHarqReGrids[MAX_ND_SUPPORTED];
    reGrid_t harqReGrids[MAX_ND_SUPPORTED];
    reGrid_t csi1ReGrids[MAX_ND_SUPPORTED];

    uint16_t nUnassignedResInSymbol[MAX_ND_SUPPORTED];
    uint32_t nUnassignedBitsInSymbol[MAX_ND_SUPPORTED];

    bool     harqPunctFlag;
    uint8_t  nBitsPerRe;
    uint32_t nBitsPerSym;
    uint32_t nResPerSym;
};

struct csi2ToUserMap_t
{
    uint16_t ueIdx;
    uint16_t ueGrpIdx;
};

struct uciOnPuschSegLLRs2DynDescr_t{
    harqAndCsi1RePrms_t harqAndCsi1RePrmsArray[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];

    // pusch pipeline parameters
    csi2ToUserMap_t           csi2ToUserMapArray[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
    PerTbParams*              pUePrmsGpu;
    cuphyPuschRxUeGrpPrms_t*  pUeGrpPrmsGpu;

    // input buffers
    tensor_ref_any<CUPHY_R_16F>   tEqOutLLRs[MAX_N_USER_GROUPS_SUPPORTED];
};






//  uciOnPuschSegLLRs2 kernel arguments (supplied via descriptors)
struct uciOnPuschSegLLRs2KernelArgs_t
{
    uciOnPuschSegLLRs2DynDescr_t*  pDynDescr;
};

class uciOnPuschSegLLRs2 : public cuphyUciOnPuschSegLLRs2
{
public:
    void setup(uint16_t                             nCsi2Ues,
               uint16_t*                            pCsi2UeIdxs,
               PerTbParams*                         pTbPrmsCpu,
               PerTbParams*                         pTbPrmsGpu,
               uint16_t                             nUeGrps,
               cuphyTensorPrm_t*                    pTensorPrmsEqOutLLRs,
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu,
               cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsGpu,               
               uciOnPuschSegLLRs2DynDescr_t*        pCpuDynDesc,
               void*                                pGpuDynDesc,
               uint8_t                              enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs2LaunchCfg_t*  pLaunchCfg,
               cudaStream_t                         strm);
               




    // void kernelSelect(uint16_t                            nUciUes,
    //                   uint16_t*                           pUciUserIdxs,
    //                   PerTbParams*                        pTbPrmsCpu,
    //                   uint16_t*                           pNumPrbs,
    //                   uint8_t                             nPuschDataSym,
    //                   cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    uciOnPuschSegLLRs2KernelArgs_t m_kernelArgs;
};