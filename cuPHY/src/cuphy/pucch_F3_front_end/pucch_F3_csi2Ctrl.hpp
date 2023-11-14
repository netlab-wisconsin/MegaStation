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
#include "tensor_desc.hpp"
#include "cuphy_api.h"

struct cuphyPucchF3Csi2Ctrl
{};

struct pucchF3Csi2ToBuffersMap_t 
{
    // Indicies used to look up parameters:
    uint16_t uciIdx;                    
    uint16_t statCellIdx; 

    // Offset for finding users decoded csi1Payload within pUciPayloads:
    uint32_t csi1PayloadByteOffset;    

    // Offset for where to store computed numCsi2Bits within pNumCsi2Bits:
    uint16_t numCsi2BitsOffset;
};

struct pucchF3Csi2CtrlDynDescr_t{

    uint16_t nCsi2Ucis;

    // Per CSI-P2 parameters:
    pucchF3Csi2ToBuffersMap_t   csi2ToBuffersMap[CUPHY_PUCCH_F3_MAX_UCI];

    // UCI payload buffer:
    uint8_t* pUciPayloads;

    // CSI-P2 size buffer:
    uint16_t* pNumCsi2Bits;

    // Parameter buffers:
    cuphyPucchCellStatPrm_t* pPucchCellStatPrms;
    cuphyPucchUciPrm_t*      pPerUciPrms;
    cuphyPolarCwPrm_t*       pPolCwPrms;
    cuphyPolarUciSegPrm_t*   pPolSegPrms;
    cuphySimplexCwPrm_t*     pSpxCwPrms;
    cuphyRmCwPrm_t*          pRmCwPrms;
};

//  pucchF3Csi2Ctrl kernel arguments (supplied via descriptors)
struct pucchF3Csi2CtrlKernelArgs_t
{
    pucchF3Csi2CtrlDynDescr_t*  pDynDescr;
};

class pucchF3Csi2Ctrl : public cuphyPucchF3Csi2Ctrl
{
public:
    pucchF3Csi2Ctrl();
    ~pucchF3Csi2Ctrl()                           = default;
    pucchF3Csi2Ctrl(pucchF3Csi2Ctrl const&)      = delete;
    pucchF3Csi2Ctrl& operator=(pucchF3Csi2Ctrl const&) = delete;

    void setup(uint16_t                             nCsi2Ucis,                 
               uint16_t*                            pCsi2UciIdxsCpu,
               cuphyPucchUciPrm_t*                  pUciPrmsCpu,                   
               cuphyPucchUciPrm_t*                  pUciPrmsGpu,
               cuphyPucchCellStatPrm_t*             pCellStatPrmsGpu,
               cuphyPucchF234OutOffsets_t*          pPucchF3OutOffsetsCpu,    
               uint8_t*                             pUciPayloadsGpu,              
               uint16_t*                            pNumCsi2BitsGpu,               
               cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,          
               cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,          
               cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,            
               cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,                  
               pucchF3Csi2CtrlDynDescr_t*           pCpuDynDesc,
               void*                                pGpuDynDesc,
               bool                                 enableCpuToGpuDescrAsyncCpy,
               cuphyPucchF3Csi2CtrlLaunchCfg_t*     pLaunchCfg,
               cudaStream_t                         strm);

    void kernelSelect(uint16_t                            nCsi2Ucis,
                      cuphyPucchF3Csi2CtrlLaunchCfg_t*    pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    pucchF3Csi2CtrlKernelArgs_t m_kernelArgs;
};