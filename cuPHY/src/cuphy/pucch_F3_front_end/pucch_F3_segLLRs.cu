/*
* Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "pucch_F3_segLLRs.hpp"

// constant tables in device memory
static __device__ __constant__ uint8_t uciSymInd[17][12] = {{0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                            {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                            {1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 2, 3, 5, 6, 0, 0, 0, 0, 0, 0, 0},
                                                            {0, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0},
                                                            {0, 2, 3, 4, 5, 7, 8, 0, 0, 0, 0, 0},
                                                            {0, 1, 3, 4, 5, 6, 8, 9, 0, 0, 0, 0},
                                                            {0, 2, 4, 5, 7, 9, 0, 0, 0, 0, 0, 0},
                                                            {0, 1, 3, 4, 5, 6, 8, 9, 10, 0, 0, 0},
                                                            {0, 2, 4, 5, 7, 8, 10, 0, 0, 0, 0, 0},
                                                            {0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 0, 0},
                                                            {0, 2, 3, 5, 6, 8, 9, 11, 0, 0, 0, 0},
                                                            {0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 0},
                                                            {0, 2, 3, 5, 6, 8, 9, 10, 12, 0, 0, 0},
                                                            {0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13},
                                                            {0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 0, 0}};

static __device__ __constant__ uint8_t SiUci_1[17][8] = {{0, 2, 0, 0, 0, 0, 0, 0},
                                                         {1, 3, 0, 0, 0, 0, 0, 0},
                                                         {1, 2, 4, 0, 0, 0, 0, 0},
                                                         {0, 2, 3, 5, 0, 0, 0, 0},
                                                         {0, 2, 3, 5, 0, 0, 0, 0},
                                                         {0, 2, 4, 6, 0, 0, 0, 0},
                                                         {0, 2, 5, 7, 0, 0, 0, 0},
                                                         {1, 3, 6, 8, 0, 0, 0, 0},
                                                         {0, 2, 4, 5, 7, 9, 0, 0},
                                                         {1, 3, 6, 8, 0, 0, 0, 0},
                                                         {0, 2, 4, 5, 7, 8, 10, 0},
                                                         {1, 3, 7, 9, 0, 0, 0, 0},
                                                         {0, 2, 3, 5, 6, 8, 9, 11},
                                                         {1, 3, 8, 10, 0, 0, 0, 0},
                                                         {0, 2, 3, 5, 6, 8, 10, 12},
                                                         {2, 4, 9, 11, 0, 0, 0, 0},
                                                         {0, 2, 4, 6, 7, 9, 11, 13}};
                                                         
static __device__ __constant__ uint8_t SiUci_2[11][4] = {{3, 0, 0, 0}, // nSym: 4, nSym_dmrs: 1
                                                         {6, 0, 0, 0}, // nSym: 7
                                                         {3, 7, 0, 0}, // nSym: 8
                                                         {3, 4, 8, 0}, // nSym: 9
                                                         {0, 4, 5, 9}, // nSym: 10, nSym_dmrs: 2
                                                         {0, 4, 5, 9}, // nSym: 11, nSym_dmrs: 2
                                                         {0, 4, 6, 10}, // nSym: 12, nSym_dmrs: 2
                                                         {0, 4, 7, 11}, // nSym: 13, nSym_dmrs: 2
                                                         {9, 0, 0, 0}, // nSym: 13, nSym_dmrs: 4
                                                         {1, 5, 8, 12}, // nSym: 14, nSym_dmrs: 2
                                                         {3, 10, 0, 0}}; // nSym: 14, nSym_dmrs: 4 

static __device__ __constant__ uint8_t SiUci_3[4][4] = {{10, 0, 0, 0}, // nSym: 11, nSym_dmrs: 2 
                                                        {5, 11, 0, 0}, // nSym: 12, nSym_dmrs: 2 
                                                        {5, 6, 12, 0}, // nSym: 13, nSym_dmrs: 2
                                                        {0, 6, 7, 13}}; // nSym: 14, nSym_dmrs: 2

__global__ void pucchF3SegLLRsKernel(pucchF3SegLLRsDynDescr_t* pDesc)
{
    const uint16_t uciIdxinBlock = threadIdx.x / F3_SEG_LLR_THREAD_PER_UCI;
    const uint16_t uciIdx = blockIdx.x * F3_SEG_LLR_UCI_PER_Block + uciIdxinBlock;
    const uint16_t subcarrIdx =  threadIdx.x - (threadIdx.x / F3_SEG_LLR_THREAD_PER_UCI) * F3_SEG_LLR_THREAD_PER_UCI;

    const uint16_t numUcis = pDesc->numUcis;
    const uint16_t nSymUci = pDesc->perUciPrmsArray[uciIdx].nSymUci;

    if (uciIdx >= numUcis) {
        return;
    }

    __shared__ __half descrmLLRSeq1[F3_SEG_LLR_UCI_PER_Block][TEMP_LLR_ARR_MAX_SIZE];
    __shared__ __half descrmLLRSeq2[F3_SEG_LLR_UCI_PER_Block][TEMP_LLR_ARR_MAX_SIZE];

    __half*        pInLLRaddrs   = pDesc->pInLLRaddrs[uciIdx];
    const uint8_t  nSym          = pDesc->perUciPrmsArray[uciIdx].nSym;
    const uint8_t  nPucchSymUci  = pDesc->perUciPrmsArray[uciIdx].nSym_data;
    const uint8_t  nSym_dmrs     = pDesc->perUciPrmsArray[uciIdx].nSym_dmrs;
    const uint8_t  Qm            = pDesc->perUciPrmsArray[uciIdx].Qm;
    const uint16_t E_seg1        = pDesc->perUciPrmsArray[uciIdx].E_seg1;
    const uint16_t E_seg2        = pDesc->perUciPrmsArray[uciIdx].E_seg2;

    if (subcarrIdx < nSymUci) {

    int uciSymInd_row = -1;
    int SiUci_1_row = -1;
    int SiUci_2_row = -1;
    int SiUci_3_row = -1;
    int NiUci_1 = 0;
    int NiUci_2 = 0;
    int NiUci_3 = 0;
    int sumNiUci[3] = {0};

    switch (nSym)
    {
        case 4:
            if (nSym_dmrs == 1) {
                uciSymInd_row = 0;
                SiUci_1_row   = 0;
                SiUci_2_row   = 0;
                NiUci_1       = 2;
                NiUci_2       = 1;
                sumNiUci[0]   = 2;
                sumNiUci[1]   = 3;
            } else { // nSym_dmrs == 2
                uciSymInd_row = 1;
                SiUci_1_row   = 1;
                NiUci_1       = 2;
                sumNiUci[0]   = 2;
            }
            break;
        case 5:
            uciSymInd_row = 2;
            SiUci_1_row   = 2;
            NiUci_1       = 3;
            sumNiUci[0]   = 3;
            break;
        case 6:
            uciSymInd_row = 3;
            SiUci_1_row   = 3;
            NiUci_1       = 4;
            sumNiUci[0]   = 4;
            break;
        case 7:
            uciSymInd_row = 4;
            SiUci_1_row   = 4;
            SiUci_2_row   = 1;
            NiUci_1       = 4;
            NiUci_2       = 1;
            sumNiUci[0]   = 4;
            sumNiUci[1]   = 5; 
            break;
        case 8:
            uciSymInd_row = 5;
            SiUci_1_row   = 5;
            SiUci_2_row   = 2;
            NiUci_1       = 4;
            NiUci_2       = 2;
            sumNiUci[0]   = 4;
            sumNiUci[1]   = 6;
            break;
        case 9:
            uciSymInd_row = 6;
            SiUci_1_row   = 6;
            SiUci_2_row   = 3;
            NiUci_1       = 4;
            NiUci_2       = 3;
            sumNiUci[0]   = 4;
            sumNiUci[1]   = 7;
            break;
        case 10:
            if (nSym_dmrs == 2) {
                uciSymInd_row = 7;
                SiUci_1_row   = 7;
                SiUci_2_row   = 4;
                NiUci_1       = 4;
                NiUci_2       = 4;
                sumNiUci[0]   = 4;
                sumNiUci[1]   = 8;
            } else { // nSym_dmrs == 4
                uciSymInd_row = 8;
                SiUci_1_row   = 8;
                NiUci_1       = 6;
                sumNiUci[0]   = 6;
            }
            break;
        case 11:
            if (nSym_dmrs == 2) {
                uciSymInd_row = 9;
                SiUci_1_row   = 9;
                SiUci_2_row   = 5;
                SiUci_3_row   = 0;
                NiUci_1       = 4;
                NiUci_2       = 4;
                NiUci_3       = 1;
                sumNiUci[0]   = 4;
                sumNiUci[1]   = 8;
                sumNiUci[2]   = 9;
            } else { // nSym_dmrs == 4
                uciSymInd_row = 10;
                SiUci_1_row   = 10;
                NiUci_1       = 7;
                sumNiUci[0]   = 7;
            }
            break;
        case 12:
            if (nSym_dmrs == 2) {
                uciSymInd_row = 11;
                SiUci_1_row   = 11;
                SiUci_2_row   = 6;
                SiUci_3_row   = 1;
                NiUci_1       = 4;
                NiUci_2       = 4;
                NiUci_3       = 2;
                sumNiUci[0]   = 4;
                sumNiUci[1]   = 8;
                sumNiUci[2]   = 10;
            } else { // nSym_dmrs == 4
                uciSymInd_row = 12;
                SiUci_1_row   = 12;
                NiUci_1       = 8;
                sumNiUci[0]   = 8;
            }
            break;
        case 13:
            if (nSym_dmrs == 2) {
                uciSymInd_row = 13;
                SiUci_1_row   = 13;
                SiUci_2_row   = 7;
                SiUci_3_row   = 2;
                NiUci_1       = 4;
                NiUci_2       = 4;
                NiUci_3       = 3;
                sumNiUci[0]   = 4;
                sumNiUci[1]   = 8;
                sumNiUci[2]   = 11;
            } else { // nSym_dmrs == 4
                uciSymInd_row = 14;
                SiUci_1_row   = 14;
                SiUci_2_row   = 8;
                NiUci_1       = 8;
                NiUci_2       = 1;
                sumNiUci[0]   = 8;
                sumNiUci[1]   = 9;
            }
            break;  
        case 14:
            if (nSym_dmrs == 2) {
                uciSymInd_row = 15;
                SiUci_1_row   = 15;
                SiUci_2_row   = 9;
                SiUci_3_row   = 3;
                NiUci_1       = 4;
                NiUci_2       = 4;
                NiUci_3       = 4;
                sumNiUci[0]   = 4;
                sumNiUci[1]   = 8;
                sumNiUci[2]   = 12;
            } else { // nSym_dmrs == 4
                uciSymInd_row = 16;
                SiUci_1_row   = 16;
                SiUci_2_row   = 10;
                NiUci_1       = 8;
                NiUci_2       = 2;
                sumNiUci[0]   = 8;
                sumNiUci[1]   = 10;
            }
            break;
    }

    uint8_t j = 0;

    while (sumNiUci[j]*nSymUci*Qm < E_seg1) {
        j++;
    }

    uint8_t comSiUciLessj[10] = {0};
    uint8_t comSiUciLessj_size = 0;

    if (j > 0) {
        for (int i = 0; i < NiUci_1; i++) {
            comSiUciLessj[i] = SiUci_1[SiUci_1_row][i];
        }
        comSiUciLessj_size = NiUci_1;
        if (j == 2) {
            for (int i = 0; i < NiUci_2; i++) {
                comSiUciLessj[i + NiUci_1] = SiUci_2[SiUci_2_row][i];
            }
            comSiUciLessj_size += NiUci_2;
        }
    }

    uint16_t nBarSymUci = 0;
    int M = 0;

    if (j == 1) {
        uint16_t temp = sumNiUci[0]*nSymUci*Qm;
        nBarSymUci = floor((E_seg1 - temp)/(NiUci_2*Qm));
        M = ((E_seg1 - temp)/Qm) % NiUci_2;
    } else if (j == 2) {
        uint16_t temp = sumNiUci[1]*nSymUci*Qm;
        nBarSymUci = floor((E_seg1 - temp)/(NiUci_3*Qm));
        M = ((E_seg1 - temp)/Qm) % NiUci_3;
    } else { // j == 0
        nBarSymUci = floor(E_seg1/(NiUci_1*Qm));
        M = (E_seg1/Qm) % NiUci_1;
    }

    uint16_t n1 = 0;
    uint16_t n2 = 0;

    for (uint8_t l = 0; l < nPucchSymUci; l++) {
        uint8_t sl = uciSymInd[uciSymInd_row][l];
        bool in_comSiUciLessj = false;
        for (int c = 0; c < comSiUciLessj_size; c++) {
            if (sl == comSiUciLessj[c]) {
                in_comSiUciLessj = true;
            }
        }

        if (in_comSiUciLessj) {
            for (int v = 0; v < Qm; v++) {
                descrmLLRSeq1[uciIdxinBlock][n1 + subcarrIdx*Qm + v] = pInLLRaddrs[l*nSymUci*Qm + subcarrIdx*Qm + v];
            }
            n1 += nSymUci*Qm;
        } else {
            bool in_SiUci_j = false;
            if (j == 0) {
                for (int c = 0; c < NiUci_1; c++) {
                    if (sl == SiUci_1[SiUci_1_row][c]) {
                        in_SiUci_j = true;
                    }
                }
            } else if (j == 1) {
                for (int c = 0; c < NiUci_2; c++) {
                    if (sl == SiUci_2[SiUci_2_row][c]) {
                        in_SiUci_j = true;
                    }
                }
            } else { // j == 2
                for (int c = 0; c < NiUci_3; c++) {
                    if (sl == SiUci_3[SiUci_3_row][c]) {
                        in_SiUci_j = true;
                    }
                }    
            }

            if (in_SiUci_j) {
                uint8_t gamma = 0;
                if (M > 0) {
                    gamma = 1;
                } else {
                    gamma = 0;
                }
                M--;
                if (subcarrIdx < nBarSymUci+gamma) {
                    for (int v = 0; v < Qm; v++) {
                        descrmLLRSeq1[uciIdxinBlock][n1 + subcarrIdx*Qm + v] = pInLLRaddrs[l*nSymUci*Qm + subcarrIdx*Qm + v];
                    }
                } else {
                    for (int v = 0; v < Qm; v++) {
                        descrmLLRSeq2[uciIdxinBlock][n2 + (subcarrIdx - nBarSymUci - gamma)*Qm + v] = pInLLRaddrs[l*nSymUci*Qm + subcarrIdx*Qm + v];
                    }
                }
                n1 += (nBarSymUci+gamma)*Qm;
                n2 += (nSymUci - nBarSymUci - gamma)*Qm;
            } else {
                for (int v = 0; v < Qm; v++) {
                    descrmLLRSeq2[uciIdxinBlock][n2 + subcarrIdx*Qm + v] = pInLLRaddrs[l*nSymUci*Qm + subcarrIdx*Qm + v];
                }
                n2 += nSymUci*Qm;
            }
        }
    }
    }
    __syncthreads();

    uint16_t round = E_seg1 / F3_SEG_LLR_THREAD_PER_UCI;
    if (E_seg1 - round * F3_SEG_LLR_THREAD_PER_UCI) {
        round++;
    }

    uint16_t index = 0;
    for (int r = 0; r < round; r++) {
        index = r * F3_SEG_LLR_THREAD_PER_UCI + subcarrIdx;
        if (index < E_seg1) {
            pInLLRaddrs[index] = descrmLLRSeq1[uciIdxinBlock][index];
        }
    }

    round = E_seg2 / F3_SEG_LLR_THREAD_PER_UCI;
    if (E_seg2 - round * F3_SEG_LLR_THREAD_PER_UCI) {
        round++;
    }
    for (int r = 0; r < round; r++) {
        index = r * F3_SEG_LLR_THREAD_PER_UCI + subcarrIdx;
        if (index < E_seg2) {
            pInLLRaddrs[E_seg1 + index] =  descrmLLRSeq2[uciIdxinBlock][index];
        }
    }

    __syncthreads();
}

pucchF3SegLLRs::pucchF3SegLLRs()
{}

void pucchF3SegLLRs::kernelSelect(uint16_t                           nF3Ucis,
                                  cuphyPucchF3SegLLRsLaunchCfg_t*    pLaunchCfg)
{
    // launch geometry
    uint16_t nThreadBlocks  = div_round_up(nF3Ucis, static_cast<uint16_t>(F3_SEG_LLR_UCI_PER_Block));
    uint16_t threadPerBlock = F3_SEG_LLR_THREAD_PER_BLOCK;
    dim3 blockDim(threadPerBlock); 
    dim3 gridDim(nThreadBlocks);  

    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(pucchF3SegLLRsKernel);
    cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc);

    // populate kernel parameters
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

    kernelNodeParamsDriver.blockDimX                = blockDim.x;
    kernelNodeParamsDriver.blockDimY                = blockDim.y;
    kernelNodeParamsDriver.blockDimZ                = blockDim.z;

    kernelNodeParamsDriver.gridDimX                 = gridDim.x;
    kernelNodeParamsDriver.gridDimY                 = gridDim.y;
    kernelNodeParamsDriver.gridDimZ                 = gridDim.z;

    kernelNodeParamsDriver.extra                    = nullptr;
    kernelNodeParamsDriver.sharedMemBytes           = 0;
}

void pucchF3SegLLRs::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(pucchF3SegLLRsDynDescr_t);
   dynDescrAlignBytes = alignof(pucchF3SegLLRsDynDescr_t);
}

void pucchF3SegLLRs::setup(uint16_t                             nF3Ucis,                 
                           cuphyPucchUciPrm_t*                  pF3UciPrms,
                           __half**                             pDescramLLRaddrs,
                           pucchF3SegLLRsDynDescr_t*            pCpuDynDesc,
                           void*                                pGpuDynDesc,
                           bool                                 enableCpuToGpuDescrAsyncCpy,
                           cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
                           cudaStream_t                         strm)
{
    pCpuDynDesc->numUcis = nF3Ucis;
    
    for(int uciIdx = 0; uciIdx < nF3Ucis; ++uciIdx)
    {
        pCpuDynDesc->pInLLRaddrs[uciIdx]                = pDescramLLRaddrs[uciIdx];
        pCpuDynDesc->perUciPrmsArray[uciIdx].nSym       = pF3UciPrms[uciIdx].nSym;
        pCpuDynDesc->perUciPrmsArray[uciIdx].nSym_data  = 0; // fix me
        pCpuDynDesc->perUciPrmsArray[uciIdx].nSym_dmrs  = 0; // fix me
        pCpuDynDesc->perUciPrmsArray[uciIdx].Qm         = pF3UciPrms[uciIdx].pi2Bpsk ? 1 : 2;
        pCpuDynDesc->perUciPrmsArray[uciIdx].nSymUci    = 12 * pF3UciPrms[uciIdx].prbSize;
        pCpuDynDesc->perUciPrmsArray[uciIdx].E_seg1     = 0; // fix me
        pCpuDynDesc->perUciPrmsArray[uciIdx].E_seg2     = 0; // fix me
    }

    pucchF3SegLLRsKernelArgs_t& kernelArgs = m_kernelArgs;
    kernelArgs.pDynDescr = reinterpret_cast<pucchF3SegLLRsDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
   if(enableCpuToGpuDescrAsyncCpy)
   {
    cudaMemcpyAsync(&pGpuDynDesc, &pCpuDynDesc, sizeof(pucchF3SegLLRsDynDescr_t), cudaMemcpyHostToDevice, strm);
   }

   // select kernel (includes launch geometry). Populate launchCfg.
   kernelSelect(nF3Ucis, pLaunchCfg);
   pLaunchCfg->kernelArgs[0] = &m_kernelArgs.pDynDescr;
   pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
}