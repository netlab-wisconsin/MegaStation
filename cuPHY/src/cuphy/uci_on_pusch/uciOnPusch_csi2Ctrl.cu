/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 #include "uciOnPusch_csi2Ctrl.hpp"

static __device__ __constant__ float d_betaOffsetCsiMapping[19] = {1.125, 1.250, 1.375, 1.625, 1.750, 2.000, 2.250, 2.500, 2.875, 3.125, 3.500, 4.000, 5.000, 6.250, 8.000, 10.000, 12.625, 15.875, 20.000};

 __global__ void uciOnPuschCsi2CtrlKernel(uciOnPuschCsi2CtrlDynDescr_t* pDesc)
 {
    // check for early exit
    int16_t  csi2Idx     = blockIdx.x * blockDim.x + threadIdx.x;

    if(csi2Idx >= pDesc->nCsi2Ues){
        return;
    }
    
    // debug paramaters:
    uint16_t forcedNumCsi2Bits = pDesc->forcedNumCsi2Bits;
    
    // Buffer indicies:
    uint16_t ueIdx       = pDesc->csi2ToBuffersMap[csi2Idx].ueIdx;
    uint16_t statCellIdx = pDesc->csi2ToBuffersMap[csi2Idx].statCellIdx;

    // parameters:
    cuphyPuschCellStatPrm_t& cellStatPrms       = pDesc->pPuschCellStatPrms[statCellIdx];
    PerTbParams&             perTbPrms          = pDesc->pPerTbPrms[ueIdx];
    //cuphyPolarCwPrm_t&       polCwPrms          = pDesc->pPolCwPrms[csi2Idx];
    //cuphyPolarUciSegPrm_t&   polSegPrms         = pDesc->pPolSegPrms[csi2Idx];
    
    cuphySimplexCwPrm_t&     spxCwPrms          = pDesc->pSpxCwPrms[csi2Idx];
    cuphyRmCwPrm_t&          rmCwPrms           = pDesc->pRmCwPrms[csi2Idx];
    cuphyPolarCwPrm_t*       pPolCwPrms         = pDesc->pPolCwPrms;
    cuphyPolarUciSegPrm_t&   polSegPrms         = pDesc->pPolSegPrms[csi2Idx];

    const uint32_t           mScUciSum          = perTbPrms.mScUciSum;
    const uint32_t           codedBitsSum       = perTbPrms.codedBitsSum;
    const uint8_t            isDataPresent      = perTbPrms.isDataPresent;
    const uint8_t            betaOffsetCsi2     = perTbPrms.betaOffsetCsi2;
    const uint32_t           nBitsHarq          = perTbPrms.nBitsHarq;
    const uint32_t           qPrimeAck          = perTbPrms.qPrimeAck;
    const float              codeRate           = perTbPrms.codeRate;
    const float              betaOffsetPusch    = d_betaOffsetCsiMapping[betaOffsetCsi2];
    const uint32_t           Qm                 = perTbPrms.Qm;
    const float              alpha              = perTbPrms.alpha;
    const uint32_t           qPrimeCsi1         = perTbPrms.qPrimeCsi1;
    const uint32_t           Nl                 = perTbPrms.Nl;

    //const uint8_t nBitsAckRvd = (nBitsHarq <= 2) ? 2 : 0;

    uint32_t QPrimeAckCsi2 = 0;

    if (nBitsHarq >2) {
        QPrimeAckCsi2 = qPrimeAck;
    } else {
        QPrimeAckCsi2  = 0;
    }

    // pointer to decoded CSI-P1 bits:
    const uint8_t            rankBitOffset      = perTbPrms.rankBitOffset;
    const uint8_t            rankOffsetMultiple = rankBitOffset/8;
    uint8_t*                 pDecodedCsiPart1   = pDesc->pUciPayloads + pDesc->csi2ToBuffersMap[csi2Idx].csi1PayloadByteOffset + rankOffsetMultiple;
    int                      rankBitOffsetRel   = rankBitOffset - rankOffsetMultiple*8; 
    // pointer where to store numCsi2Bits
    uint16_t*                pNumCsi2Bits       = pDesc->pNumCsi2Bits + pDesc->csi2ToBuffersMap[csi2Idx].numCsi2BitsOffset;
    //const uint8_t            nCsiReports        = perTbPrms.nCsiReports;
    const uint8_t            nRanksBits         = perTbPrms.nRanksBits;
    const uint8_t            N1                 = cellStatPrms.N1;
    const uint8_t            N2                 = cellStatPrms.N2;
    const uint8_t            nCsirsPorts        = cellStatPrms.nCsirsPorts;
    //const uint8_t            csiReportingBand   = cellStatPrms.csiReportingBand;
    //Only type1-SinglePanel codebook is supported
    //const uint8_t            codebookType       = cellStatPrms.codebookType;
    const uint8_t            codebookMode       = cellStatPrms.codebookMode;
    //const uint8_t            isCqi              = cellStatPrms.isCqi;
    //const uint8_t            isLi               = cellStatPrms.isLi;

    uint8_t rank = 0;
    uint8_t powFactor = 1<<(nRanksBits-1);
 
    if(forcedNumCsi2Bits > 0)
    {
        *pNumCsi2Bits = forcedNumCsi2Bits;
    }else
    {
        for (int i = 0; i < nRanksBits; i++) {
            if (rankBitOffsetRel + i == 8) {
                pDecodedCsiPart1 += 1;
                rankBitOffsetRel -= 8;
            }
            rank += ((*(pDecodedCsiPart1)>>(rankBitOffsetRel + i)) & 0x1) * powFactor;
        powFactor = (powFactor>>1);
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
    }

    perTbPrms.nBitsCsi2 = *pNumCsi2Bits;

    if (*pNumCsi2Bits == 0) {
        perTbPrms.G_csi2 = 0;
    } else {
        uint8_t lCsi2 = 0;

        if (*pNumCsi2Bits <=11) {
            lCsi2 = 0;
        } else if (*pNumCsi2Bits >=12 && *pNumCsi2Bits <=19) {
            lCsi2 =6;
        } else {
            lCsi2 = 11;
        }
    
        float numerator = 0;
        float denominator = 0;
    
        if (isDataPresent) {
            numerator = ((*pNumCsi2Bits) + lCsi2) * betaOffsetPusch * mScUciSum;
            denominator = codedBitsSum;
        } else {
            numerator = ((*pNumCsi2Bits) + lCsi2)*betaOffsetPusch;
            denominator = codeRate*Qm;
        }
    
        uint32_t firstTermCsi2 = ceil (numerator/denominator);

        uint32_t qPrimeCsi2 = 0;

        if (isDataPresent) {
            qPrimeCsi2 = ceil(alpha*mScUciSum)-QPrimeAckCsi2-qPrimeCsi1;
            if (qPrimeCsi2 > firstTermCsi2) {
                qPrimeCsi2 = firstTermCsi2;
            }
        } else {
            qPrimeCsi2 = mScUciSum- QPrimeAckCsi2-qPrimeCsi1;
        }

        perTbPrms.G_csi2 = Nl*qPrimeCsi2*Qm;
        perTbPrms.G = perTbPrms.G_schAndCsi2 - perTbPrms.G_csi2;
    }

    __half* d_csi2LLRs = perTbPrms.d_schAndCsi2LLRs + perTbPrms.G;
    if(*pNumCsi2Bits == 0)
    {
        spxCwPrms.exitFlag = 1;
        rmCwPrms.exitFlag  = 1;

    }else if(*pNumCsi2Bits <= CUPHY_N_MAX_UCI_BITS_SIMPLEX) // use simplex decoder
    {
        rmCwPrms.exitFlag = 1;
        
        spxCwPrms.exitFlag     = 0;
        spxCwPrms.E            = perTbPrms.G_csi2;
        spxCwPrms.K            = *pNumCsi2Bits;
        spxCwPrms.nBitsPerQam  = Qm;
        spxCwPrms.d_LLRs       = d_csi2LLRs;
        spxCwPrms.en_DTXest    = spxCwPrms.en_DTXest;

    }else if(*pNumCsi2Bits <= CUPHY_N_MAX_UCI_BITS_RM)
    {
        spxCwPrms.exitFlag = 1;

        rmCwPrms.exitFlag = 0;
        rmCwPrms.E        = perTbPrms.G_csi2;
        rmCwPrms.K        = *pNumCsi2Bits;
        rmCwPrms.d_LLRs   = d_csi2LLRs;
        rmCwPrms.en_DTXest= rmCwPrms.en_DTXest;
    }
    else
    {
        spxCwPrms.exitFlag  = 1;
        rmCwPrms.exitFlag   = 1;

        uint32_t nInfoBits  = *pNumCsi2Bits;
        uint32_t nRmBits    = perTbPrms.G_csi2;

        polSegPrms.exitFlag = 0;
        polSegPrms.pUciSegLLRs = d_csi2LLRs;
        polSegPrms.E_seg       = nRmBits;

        // crc size (38.212 6.3.1.2.1)
        polSegPrms.nCrcBits = (nInfoBits <= 19) ? 6 : 11;

        // code block segmentation (38.212 6.3.1.3.1)
        // code block size         (38.212 5.2.1)
        if(((nInfoBits >= 360) && (nRmBits >= 1088)) || (nInfoBits >= 1013))
        {
            polSegPrms.nCbs           = 2;
            polSegPrms.K_cw           = div_round_up(nInfoBits, static_cast<uint32_t>(2)) + polSegPrms.nCrcBits;
            polSegPrms.E_cw           = nRmBits / 2;
            polSegPrms.zeroInsertFlag = nInfoBits % 2;
        }
        else
        {
            polSegPrms.nCbs           = 1;
            polSegPrms.K_cw           = nInfoBits + polSegPrms.nCrcBits;
            polSegPrms.E_cw           = nRmBits;
            polSegPrms.zeroInsertFlag = 0;
        }

        // encoded cb(s) size (38.212 5.3.1)
        uint32_t n_temp        = static_cast<uint32_t>(ceil(log2(static_cast<double>(polSegPrms.E_cw))) - 1);
        uint32_t two_to_n_temp = 1 << n_temp;

        uint32_t n_1;
        if((8 * polSegPrms.E_cw <= 9 * two_to_n_temp) && (16 * polSegPrms.K_cw <= 9 * polSegPrms.E_cw))
        {
            n_1 = n_temp;
        }
        else
        {
            n_1 = n_temp + 1;
        }

        uint32_t n_2   = static_cast<uint32_t>(ceil(log2(static_cast<double>(polSegPrms.K_cw) * 8)));
        uint32_t n_min = 5;
        uint32_t n_max = 10;

        polSegPrms.n_cw = max(min(min(n_1, n_2), n_max), n_min);
        polSegPrms.N_cw = 1 << polSegPrms.n_cw;

        // child cb(s)
        for(int i = 0; i < polSegPrms.nCbs; ++i)
        {
            cuphyPolarCwPrm_t& cwPrms = pPolCwPrms[2*csi2Idx + i];

            cwPrms.exitFlag = 0;
            cwPrms.N_cw     = polSegPrms.N_cw;
            cwPrms.pCwLLRs  = cwPrms.pCwTreeLLRs + cwPrms.N_cw;
            cwPrms.nCrcBits = polSegPrms.nCrcBits;
            cwPrms.A_cw     = polSegPrms.K_cw - polSegPrms.nCrcBits;

            if(polSegPrms.nCbs == 1)
            {
                cwPrms.pCbEst = cwPrms.pUciSegEst;
            }


            cwPrms.nCbsInUciSeg      = polSegPrms.nCbs;
            cwPrms.cbIdxWithinUciSeg = i;
            cwPrms.zeroInsertFlag    = polSegPrms.zeroInsertFlag;
        }
    }

    /*
    if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    {
       printf("\n\n CSI-P2 Control kernel running! \n");
       printf("\n\n csi2Idx = %d, ueIdx = %d, statCellIdx = %d \n\n", csi2Idx, ueIdx, statCellIdx);

       printf("\n Static cell parameters:");
       printf("\n nCsirsPorts      = %d", cellStatPrms.nCsirsPorts);
       printf("\n N1               = %d", cellStatPrms.N1);
       printf("\n N2               = %d", cellStatPrms.N2);
       printf("\n csiReportingBand = %d", cellStatPrms.csiReportingBand);
       printf("\n codebookType     = %d", cellStatPrms.codebookType);
       printf("\n codebookMode     = %d", cellStatPrms.codebookMode);
       printf("\n isCqi            = %d", cellStatPrms.isCqi);
       printf("\n isLi             = %d", cellStatPrms.isLi);
       printf("\n");

       printf("\n User parameters:");
       printf("\n nCsiReports      = %d", perTbPrms.nCsiReports);
       printf("\n rankBitOffset    = %d", perTbPrms.rankBitOffset);
       printf("\n nRanksBits       = %d", perTbPrms.nRanksBits);
       printf("\n");
    }*/

    // kernel computes: G (number of SCH rm bits), nBitsCsi2, G_csi2 (number of CSI-P2 ratematch bits)
    // Store as follows:
    // perTbPrms.G         = G;
    // perTbPrms.G_csi2    = G_csi2;
    // perTbPrms.nBitsCsi2 = nBitsCsi2;
    // &pNumCsi2Bits       = nBitsCsi2;
 }



void uciOnPuschCsi2Ctrl::kernelSelect(uint16_t                            nCsi2Ues,
                                      cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg)
{
    // launch geometry
    uint16_t nThreadBlocks  = div_round_up(nCsi2Ues, static_cast<uint16_t>(1024));
    uint16_t threadPerBlock = div_round_up(nCsi2Ues, nThreadBlocks);
    dim3 blockDim(threadPerBlock); 
    dim3 gridDim(nThreadBlocks);  

    // kernel (only one kernel option for now)
   void* kernelFunc = reinterpret_cast<void*>(uciOnPuschCsi2CtrlKernel);
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

 uciOnPuschCsi2Ctrl::uciOnPuschCsi2Ctrl()
 {}

 void uciOnPuschCsi2Ctrl::setup(uint16_t                             nCsi2Ues,                 
                                uint16_t*                            pCsi2UeIdxsCpu,
                                PerTbParams*                         pTbPrmsCpu,                   
                                PerTbParams*                         pTbPrmsGpu,
                                cuphyPuschRxUeGrpPrms_t*             pUeGrpPrmsCpu,
                                cuphyPuschCellStatPrm_t*             pCellStatPrmsGpu,
                                cuphyUciOnPuschOutOffsets_t*         pUciOnPuschOutOffsetsCpu,    
                                uint8_t*                             pUciPayloadsGpu,              
                                uint16_t*                            pNumCsi2BitsGpu,               
                                cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,          
                                cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,          
                                cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,            
                                cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,    
                                uint16_t                             forcedNumCsi2Bits,                             
                                uciOnPuschCsi2CtrlDynDescr_t*        pCpuDynDesc,
                                void*                                pGpuDynDesc,
                                bool                                 enableCpuToGpuDescrAsyncCpy,
                                cuphyUciOnPuschCsi2CtrlLaunchCfg_t*  pLaunchCfg,
                                cudaStream_t                         strm)
{

    pCpuDynDesc->nCsi2Ues          = nCsi2Ues;
    pCpuDynDesc->forcedNumCsi2Bits = forcedNumCsi2Bits;

    // set pointers to buffers:
    pCpuDynDesc->pPuschCellStatPrms  =  pCellStatPrmsGpu;
    pCpuDynDesc->pPerTbPrms          =  pTbPrmsGpu;
    pCpuDynDesc->pPolCwPrms          =  pCsi2PolarCwPrmsGpu;
    pCpuDynDesc->pPolSegPrms         =  pCsi2PolarSegPrmsGpu;
    pCpuDynDesc->pSpxCwPrms          =  pCsi2SpxCwPrmsGpu;
    pCpuDynDesc->pRmCwPrms           =  pCsi2RmCwPrmsGpu;
    pCpuDynDesc->pUciPayloads        =  pUciPayloadsGpu;
    pCpuDynDesc->pNumCsi2Bits        =  pNumCsi2BitsGpu;

    // set mappings to buffers
    for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
    {
        uint16_t ueIdx       = pCsi2UeIdxsCpu[csi2Idx];
        uint16_t ueGrpIdx    = pTbPrmsCpu[ueIdx].userGroupIndex;
        uint16_t statCellIdx = pUeGrpPrmsCpu[ueGrpIdx].statCellIdx;

        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].ueIdx                 = ueIdx;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].statCellIdx           = statCellIdx;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].csi1PayloadByteOffset = pUciOnPuschOutOffsetsCpu[ueIdx].csi1PayloadByteOffset;
        pCpuDynDesc->csi2ToBuffersMap[csi2Idx].numCsi2BitsOffset     = pUciOnPuschOutOffsetsCpu[ueIdx].numCsi2BitsOffset;
    }

   // save pointer to GPU descriptor
   uciOnPuschCsi2CtrlKernelArgs_t& kernelArgs = m_kernelArgs;
   kernelArgs.pDynDescr = reinterpret_cast<uciOnPuschCsi2CtrlDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(uciOnPuschCsi2CtrlDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nCsi2Ues, pLaunchCfg);

   pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 
}

 void uciOnPuschCsi2Ctrl::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
    dynDescrSizeBytes  = sizeof(uciOnPuschCsi2CtrlDynDescr_t);
    dynDescrAlignBytes = alignof(uciOnPuschCsi2CtrlDynDescr_t);
 }
