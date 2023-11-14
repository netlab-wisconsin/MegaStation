/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 #include "pucch_F3_csi2Ctrl.hpp"

 __global__ void pucchF3Csi2CtrlKernel(pucchF3Csi2CtrlDynDescr_t* pDesc)
 {
     // check for early exit
     int16_t  csi2Idx     = blockIdx.x * blockDim.x + threadIdx.x;
     if(csi2Idx >= pDesc->nCsi2Ucis){
         return;
     }

     // Buffer indicies:
     uint16_t uciIdx      = pDesc->csi2ToBuffersMap[csi2Idx].uciIdx;
     uint16_t statCellIdx = pDesc->csi2ToBuffersMap[csi2Idx].statCellIdx;

     // parameters:
     cuphyPucchCellStatPrm_t& cellStatPrms       = pDesc->pPucchCellStatPrms[statCellIdx];
     cuphyPucchUciPrm_t&      perUciPrms         = pDesc->pPerUciPrms[uciIdx];
     //cuphyPolarCwPrm_t&       polCwPrms          = pDesc->pPolCwPrms[csi2Idx];
     //cuphyPolarUciSegPrm_t&   polSegPrms         = pDesc->pPolSegPrms[csi2Idx];
     cuphySimplexCwPrm_t&     spxCwPrms          = pDesc->pSpxCwPrms[csi2Idx];
     cuphyRmCwPrm_t&          rmCwPrms           = pDesc->pRmCwPrms[csi2Idx];

     // pointer to decoded CSI-P1 bits:
     const uint8_t            rankBitOffset      = perUciPrms.rankBitOffset;
     const uint8_t            rankOffsetMultiple = rankBitOffset/8;
     uint8_t*                 pDecodedCsiPart1   = pDesc->pUciPayloads + pDesc->csi2ToBuffersMap[csi2Idx].csi1PayloadByteOffset + rankOffsetMultiple;
     int                      rankBitOffsetRel   = rankBitOffset - rankOffsetMultiple*8; 
     // pointer where to store numCsi2Bits
     uint16_t*                pNumCsi2Bits       = pDesc->pNumCsi2Bits + pDesc->csi2ToBuffersMap[csi2Idx].numCsi2BitsOffset;
     //const uint8_t            nCsiReports       
     const uint8_t            nRanksBits         = perUciPrms.nRanksBits;
     const uint8_t            N1                 = cellStatPrms.N1;
     const uint8_t            N2                 = cellStatPrms.N2;
     const uint8_t            nCsirsPorts        = cellStatPrms.nCsirsPorts;
     //const uint8_t            csiReportingBand   = cellStatPrms.csiReportingBand;
     //Only type1-SinglePanel codebook is supported
     //const uint8_t            codebookType       = cellStatPrms.codebookType;
     const uint8_t            codebookMode       = cellStatPrms.codebookMode;
     //const uint8_t            isCqi              = cellStatPrms.isCqi;
     //const uint8_t            isLi               = cellStatPrms.isLi;
     const uint32_t           Qm                 = perUciPrms.pi2Bpsk>0? 1:2;

     uint8_t rank = 0;
     uint8_t powFactor = 1;
     for (int i = 0; i < nRanksBits; i++) {
         if (rankBitOffsetRel + i == 8) {
             pDecodedCsiPart1 += 1;
             rankBitOffsetRel -= 8;
         }
         rank += ((*(pDecodedCsiPart1)>>(rankBitOffsetRel + i)) & 0x1) * powFactor;
         powFactor *= 2;
     }
     rank += 1;

     uint8_t O1 = 0;
     uint8_t O2 = 0;

     switch (nCsirsPorts) {
         case 4:
             O1 = 4;
             O2 = 1;
             break;
         case 8:
             if (N1 == 2 && N2 == 2) {
                 O1 = 4;
                 O2 = 4;
             } else if (N1 == 4 && N2 == 1) {
                 O1 = 4;
                 O2 = 1;
             }    
             break;
         case 12:
             if (N1 == 3 && N2 == 2) {
                 O1 = 4;
                 O2 = 4;
             } else if (N1 == 6 && N2 == 1) {
                 O1 = 4;
                 O2 = 1;  
             }
             break;
         case 16:
             if (N1 == 4 && N2 == 2) {
                 O1 = 4;
                 O2 = 4;
             } else if (N1 == 8 && N2 == 1) {
                 O1 = 4;
                 O2 = 1; 
             }
             break;
         case 24:
             if ((N1 == 4 && N2 == 3) || (N1 == 6 && N2 == 2)) {
                 O1 = 4;
                 O2 = 4;
             } else if (N1 == 12 && N2 == 1) {
                 O1 = 4;
                 O2 = 1;  
             }
             break;
         case 32:
             if ((N1 == 4 && N2 == 4) || (N1 == 8 && N2 == 2)) {
                 O1 = 4;
                 O2 = 4;
             } else if (N1 == 16 && N2 == 1) {
                 O1 = 4;
                 O2 = 1;  
             }
             break;
         default:;
     }

    *pNumCsi2Bits = 0;

     switch (rank) {
         case 1:
             if (nCsirsPorts > 2) {
                 if (N2 == 1) {
                     switch (codebookMode) {
                         case 1:                  
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 2;
                             break;
                         case 2:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + 4;
                             break;
                         default:;
                     }
                 } else if (N2 > 1) {
                     switch (codebookMode) {
                         case 1:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 2;
                             break;
                         case 2:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + ceil(log2f((N2 * O2)/2.0)) + 4;
                             break;
                         default:;
                     }
                 }
             } else if (nCsirsPorts == 2) {
                 *pNumCsi2Bits = *pNumCsi2Bits + 2;
             }
             break;
         case 2:
             if (nCsirsPorts == 2) {
                 *pNumCsi2Bits = *pNumCsi2Bits + 1;
             } else if (nCsirsPorts== 4) {
                 switch (codebookMode) {
                     case 1:
                         *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 2;
                         break;
                     case 2:
                         *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + 4;
                         break;
                     default:;
                 }
             } else {
                 if (N2 > 1) {
                     switch (codebookMode) {
                         case 1:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 3;
                             break;
                         case 2:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + ceil(log2f((N2 * O2)/2.0)) + 5;
                             break;
                         default:;
                     }
                 } else {
                     switch (codebookMode) {
                         case 1:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 3;
                             break;
                         case 2:
                             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + 5;
                             break;
                         default:;
                     }    
                 }
             }
             break;
         case 3:
         case 4:
             if (nCsirsPorts == 4) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 1;
             } else if (nCsirsPorts == 8 || nCsirsPorts == 12) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 3;
             } else if (nCsirsPorts > 16) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + ceil(log2f(N2 * O2)) + 3;
             }
             break;
         case 5:
         case 6:
             *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 1;
             break;
         case 7:
         case 8:
             if (N1 == 4 && N2 == 1) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f((N1 * O1)/2.0)) + ceil(log2f(N2 * O2)) + 1;
             } else if (N1 > 2 && N2 == 2) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f((N2 * O2)/2.0)) + 1;
             } else if ((N1 > 4 && N2 == 1) || (N1 == 2 && N2 == 2) || (N1 > 2 && N2 > 2)) {
                 *pNumCsi2Bits = *pNumCsi2Bits + ceil(log2f(N1 * O1)) + ceil(log2f(N2 * O2)) + 1;
             }
             break;
         default:;
     }

     perUciPrms.nBitsCsi2 = *pNumCsi2Bits;

     if(*pNumCsi2Bits == 0)
     {
        rmCwPrms.exitFlag  = 1;
        spxCwPrms.exitFlag = 1;

     }else if(*pNumCsi2Bits <= CUPHY_N_MAX_UCI_BITS_SIMPLEX) // use simplex decoder
     {
         rmCwPrms.exitFlag = 1;

         spxCwPrms.exitFlag     = 0;
         spxCwPrms.K            = *pNumCsi2Bits;
         spxCwPrms.nBitsPerQam  = Qm;
     } else if (*pNumCsi2Bits <= CUPHY_N_MAX_UCI_BITS_RM)
     {
         spxCwPrms.exitFlag = 1;

         rmCwPrms.exitFlag = 0;
         rmCwPrms.K        = *pNumCsi2Bits;
    }
 }

 void pucchF3Csi2Ctrl::kernelSelect(uint16_t                            nCsi2Ucis,
                                    cuphyPucchF3Csi2CtrlLaunchCfg_t*    pLaunchCfg)
{
// launch geometry
uint16_t nThreadBlocks  = div_round_up(nCsi2Ucis, static_cast<uint16_t>(1024));
uint16_t threadPerBlock = div_round_up(nCsi2Ucis, nThreadBlocks);
dim3 blockDim(threadPerBlock); 
dim3 gridDim(nThreadBlocks);  

// kernel (only one kernel option for now)
void* kernelFunc = reinterpret_cast<void*>(pucchF3Csi2CtrlKernel);
cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);

// populate kernel parameters
CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

kernelNodeParamsDriver.blockDimX = blockDim.x;
kernelNodeParamsDriver.blockDimY = blockDim.y;
kernelNodeParamsDriver.blockDimZ = blockDim.z;

kernelNodeParamsDriver.gridDimX = gridDim.x;
kernelNodeParamsDriver.gridDimY = gridDim.y;
kernelNodeParamsDriver.gridDimZ = gridDim.z;

kernelNodeParamsDriver.extra          = nullptr;
kernelNodeParamsDriver.sharedMemBytes = 0;
}

pucchF3Csi2Ctrl::pucchF3Csi2Ctrl()
{}

void pucchF3Csi2Ctrl::setup(uint16_t                             nCsi2Ucis,                 
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
                            cudaStream_t                         strm)
{
    pCpuDynDesc->nCsi2Ucis = nCsi2Ucis;

    // set pointers to buffers:
    pCpuDynDesc->pPucchCellStatPrms  =  pCellStatPrmsGpu;
    pCpuDynDesc->pPerUciPrms         =  pUciPrmsGpu;
    pCpuDynDesc->pPolCwPrms          =  pCsi2PolarCwPrmsGpu;
    pCpuDynDesc->pPolSegPrms         =  pCsi2PolarSegPrmsGpu;
    pCpuDynDesc->pSpxCwPrms          =  pCsi2SpxCwPrmsGpu;
    pCpuDynDesc->pRmCwPrms           =  pCsi2RmCwPrmsGpu;
    pCpuDynDesc->pUciPayloads        =  pUciPayloadsGpu;
    pCpuDynDesc->pNumCsi2Bits        =  pNumCsi2BitsGpu;

    // set mappings to buffers
    for(int csi2Idx = 0; csi2Idx < nCsi2Ucis; ++csi2Idx)
    {
        uint16_t uciIdx       = pCsi2UciIdxsCpu[csi2Idx];
        uint16_t statCellIdx  = pUciPrmsCpu[uciIdx].cellPrmStatIdx;

        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].uciIdx                = uciIdx;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].statCellIdx           = statCellIdx;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].csi1PayloadByteOffset = pPucchF3OutOffsetsCpu[uciIdx].csi1PayloadByteOffset;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].numCsi2BitsOffset     = pPucchF3OutOffsetsCpu[uciIdx].numCsi2BitsOffset;
    }

    // save pointer to GPU descriptor
   pucchF3Csi2CtrlKernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<pucchF3Csi2CtrlDynDescr_t*>(pGpuDynDesc);

   // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
       cudaMemcpyAsync(&pGpuDynDesc, &pCpuDynDesc, sizeof(pucchF3Csi2CtrlDynDescr_t), cudaMemcpyHostToDevice, strm);
   }

    // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nCsi2Ucis, pLaunchCfg);

   pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}

void pucchF3Csi2Ctrl::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(pucchF3Csi2CtrlDynDescr_t);
   dynDescrAlignBytes = alignof(pucchF3Csi2CtrlDynDescr_t);
}

