/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "pucch_F234_uci_seg.hpp"

static __device__ __constant__ uint32_t bitMaskF234UciSeg[31] = {0x7FFFFFFF, 0x3FFFFFFF, 0x1FFFFFFF, 0x0FFFFFFF, 0x07FFFFFF, 0x03FFFFFF, 0x01FFFFFF, 0x00FFFFFF, 0x007FFFFF, 0x003FFFFF, 0x001FFFFF, 0x000FFFFF, 0x0007FFFF, 0x0003FFFF, 0x0001FFFF, 0x0000FFFF, 0x00007FFF, 0x00003FFF, 0x00001FFF, 0x00000FFF, 0x000007FF, 0x000003FF, 0x000001FF, 0x000000FF, 0x0000007F, 0x0000003F, 0x0000001F, 0x0000000F, 0x00000007, 0x00000003, 0x00000001};

__global__ void pucchF234UciSegKernel(pucchF234UciSegDynDescr_t* pDesc)
{
    const uint16_t uciIdxinBlock = threadIdx.x / F234_UCI_SEG_THREAD_PER_UCI;
    const uint16_t uciIdx        = blockIdx.x * F234_UCI_SEG_UCI_PER_Block + uciIdxinBlock;
    const uint16_t nF2Ucis       = pDesc->nF2Ucis;
    const uint16_t nF3Ucis       = pDesc->nF3Ucis;
    const uint16_t totNumUcis    = nF2Ucis + nF3Ucis;

    if (uciIdx >= totNumUcis) {
        return;
    }

    bool isF2UCI = true; // Currently only consider PF2 and PF3 UCIs: isF2UCI == false means the current thread processes a PF2 UCI
    const uint16_t F2UciIdx = uciIdx;
    uint16_t F3UciIdx;
    if (uciIdx >= nF2Ucis) {
        isF2UCI = false;
        F3UciIdx = uciIdx - nF2Ucis;
    }

    perUciPrmsF234UciSeg_t PerUciPrms;
    if (isF2UCI) {
        PerUciPrms = pDesc->F2PerUciPrmsArray[F2UciIdx];
    } else {
        PerUciPrms = pDesc->F3PerUciPrmsArray[F3UciIdx];
    }

    const uint16_t bitLenHarq               = PerUciPrms.bitLenHarq;
    const uint16_t bitLenSr                 = PerUciPrms.bitLenSr;
    const uint16_t bitLenCsiPart1           = PerUciPrms.bitLenCsiPart1;
    const uint32_t uciSeg1PayloadByteOffset = PerUciPrms.uciSeg1PayloadByteOffset;
    const uint32_t harqPayloadByteOffset    = PerUciPrms.harqPayloadByteOffset;
    const uint32_t srPayloadByteOffset      = PerUciPrms.srPayloadByteOffset;
    const uint32_t csi1PayloadByteOffset    = PerUciPrms.csi1PayloadByteOffset;

    const uint16_t uciWordIdx               = threadIdx.x - (threadIdx.x / F234_UCI_SEG_THREAD_PER_UCI) * F234_UCI_SEG_THREAD_PER_UCI; 
    void*          uciPayloadVoid           = static_cast<void*>(pDesc->pUciPayloadsGpu + uciSeg1PayloadByteOffset);
    uint32_t*      uciPayload               = static_cast<uint32_t*>(uciPayloadVoid);
    void*          harqPayloadVoid          = static_cast<void*>(pDesc->pUciPayloadsGpu + harqPayloadByteOffset);
    uint32_t*      harqPayload              = static_cast<uint32_t*>(harqPayloadVoid);
    void*          srPayloadVoid            = static_cast<void*>(pDesc->pUciPayloadsGpu + srPayloadByteOffset);
    uint32_t*      srPayload                = static_cast<uint32_t*>(srPayloadVoid);
    void*          csi1PayloadVoid          = static_cast<void*>(pDesc->pUciPayloadsGpu + csi1PayloadByteOffset);
    uint32_t*      csi1Payload              = static_cast<uint32_t*>(csi1PayloadVoid);

    int            numWords;
    if (bitLenHarq > 0) {
        numWords = div_round_up(bitLenHarq, static_cast<uint16_t>(32));

        if (uciWordIdx < numWords) {
            harqPayload[uciWordIdx] = uciPayload[uciWordIdx];
        
            int remainingNumBits = bitLenHarq - (bitLenHarq/32) * 32;
            if ((remainingNumBits > 0) && (uciWordIdx == (numWords - 1))) {
                harqPayload[uciWordIdx] = harqPayload[uciWordIdx] & bitMaskF234UciSeg[31-remainingNumBits];
            }
        }
    }
    if (bitLenSr > 0) {
        if (uciWordIdx == 0) {
            int firstWordIdx    = bitLenHarq/32;
            int firstWordBitIdx = bitLenHarq - firstWordIdx*32;
            if (firstWordBitIdx == 0) {
                srPayload[0] = uciPayload[firstWordIdx] & bitMaskF234UciSeg[31-bitLenSr];
            } else {
                int firstWordNumBits = 32 - firstWordBitIdx;
                if (bitLenSr <= firstWordNumBits) {
                    srPayload[0] = (uciPayload[firstWordIdx] >> firstWordBitIdx) & bitMaskF234UciSeg[31-bitLenSr];
                } else {
                    srPayload[0] = (uciPayload[firstWordIdx] >> firstWordBitIdx) & bitMaskF234UciSeg[31-firstWordNumBits];
                    uint32_t temp = uciPayload[firstWordIdx+1] & bitMaskF234UciSeg[31-(bitLenSr-firstWordNumBits)];
                    srPayload[0] += temp << firstWordNumBits;
                }
            }
        }
    }
    if (bitLenCsiPart1 > 0) {
        numWords = div_round_up(bitLenCsiPart1, static_cast<uint16_t>(32));
        if (uciWordIdx < numWords) {
            int firstWordIdx    = (bitLenHarq + bitLenSr)/32;
            int firstWordBitIdx = bitLenHarq + bitLenSr - firstWordIdx*32;
            if (firstWordBitIdx == 0) {
                csi1Payload[uciWordIdx] = uciPayload[uciWordIdx + firstWordIdx];

                int remainingNumBits = bitLenCsiPart1 - (bitLenCsiPart1/32) * 32;
                if ((remainingNumBits > 0) && (uciWordIdx == (numWords - 1))) {
                    csi1Payload[uciWordIdx] = csi1Payload[uciWordIdx] & bitMaskF234UciSeg[31-remainingNumBits];
                }
            } else {
                int firstWordNumBits = 32 - firstWordBitIdx;
                if (bitLenCsiPart1 <= firstWordNumBits) { // implies numWords == 1
                    csi1Payload[uciWordIdx] = (uciPayload[uciWordIdx + firstWordIdx] >> firstWordBitIdx) & bitMaskF234UciSeg[31-bitLenCsiPart1];
                } else {
                    csi1Payload[uciWordIdx] = (uciPayload[uciWordIdx + firstWordIdx] >> firstWordBitIdx) & bitMaskF234UciSeg[31-firstWordNumBits];
                    uint32_t temp = uciPayload[uciWordIdx + firstWordIdx + 1] & bitMaskF234UciSeg[31-firstWordBitIdx];
                    csi1Payload[uciWordIdx] += temp << firstWordNumBits;

                    int remainingNumBits = bitLenCsiPart1 - (bitLenCsiPart1/32) * 32;   
                    if ((remainingNumBits > 0) && (uciWordIdx == (numWords - 1))) {
                        csi1Payload[uciWordIdx] = csi1Payload[uciWordIdx] & bitMaskF234UciSeg[31-remainingNumBits];
                    }
                }
            }
        }  
    }
    __syncthreads();
}

pucchF234UciSeg::pucchF234UciSeg()
{}

void pucchF234UciSeg::kernelSelect(uint16_t                         nF2Ucis,
                                   uint16_t                         nF3Ucis,
                                   cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg)
{
// launch geometry
uint16_t nThreadBlocks  = div_round_up(static_cast<uint16_t>(nF2Ucis + nF3Ucis), static_cast<uint16_t>(F234_UCI_SEG_UCI_PER_Block));
uint16_t threadPerBlock = F234_UCI_SEG_THREAD_PER_BLOCK;
dim3 blockDim(threadPerBlock); 
dim3 gridDim(nThreadBlocks);  

// kernel (only one kernel option for now)
void* kernelFunc = reinterpret_cast<void*>(pucchF234UciSegKernel);
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

void pucchF234UciSeg::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
   dynDescrSizeBytes  = sizeof(pucchF234UciSegDynDescr_t);
   dynDescrAlignBytes = alignof(pucchF234UciSegDynDescr_t);
}

void pucchF234UciSeg::setup(uint16_t                         nF2Ucis,
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
                            cudaStream_t                     strm)
{
    pCpuDynDesc->nF2Ucis = nF2Ucis;
    pCpuDynDesc->nF3Ucis = nF3Ucis;

    pCpuDynDesc->pUciPayloadsGpu = uciPayloadsGpu;

    for(int uciIdx = 0; uciIdx < nF2Ucis; ++uciIdx)
    {
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].bitLenHarq               = F2UciPrms[uciIdx].bitLenHarq;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].bitLenSr                 = F2UciPrms[uciIdx].bitLenSr;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].bitLenCsiPart1           = F2UciPrms[uciIdx].bitLenCsiPart1;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].uciSeg1PayloadByteOffset = F2OutOffsetsCpu[uciIdx].uciSeg1PayloadByteOffset;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].harqPayloadByteOffset    = F2OutOffsetsCpu[uciIdx].harqPayloadByteOffset;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].srPayloadByteOffset      = F2OutOffsetsCpu[uciIdx].srPayloadByteOffset;
        pCpuDynDesc->F2PerUciPrmsArray[uciIdx].csi1PayloadByteOffset    = F2OutOffsetsCpu[uciIdx].csi1PayloadByteOffset;
    }

    for(int uciIdx = 0; uciIdx < nF3Ucis; ++uciIdx)
    {
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].bitLenHarq               = F3UciPrms[uciIdx].bitLenHarq;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].bitLenSr                 = F3UciPrms[uciIdx].bitLenSr;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].bitLenCsiPart1           = F3UciPrms[uciIdx].bitLenCsiPart1;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].uciSeg1PayloadByteOffset = F3OutOffsetsCpu[uciIdx].uciSeg1PayloadByteOffset;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].harqPayloadByteOffset    = F3OutOffsetsCpu[uciIdx].harqPayloadByteOffset;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].srPayloadByteOffset      = F3OutOffsetsCpu[uciIdx].srPayloadByteOffset;
        pCpuDynDesc->F3PerUciPrmsArray[uciIdx].csi1PayloadByteOffset    = F3OutOffsetsCpu[uciIdx].csi1PayloadByteOffset;
    }

    pucchF234UciSegKernelArgs_t& kernelArgs = m_kernelArgs;

    kernelArgs.pDynDescr = reinterpret_cast<pucchF234UciSegDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(&pGpuDynDesc, &pCpuDynDesc, sizeof(pucchF234UciSegDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect(nF2Ucis, nF3Ucis, pLaunchCfg);
    pLaunchCfg->kernelArgs[0] = &m_kernelArgs.pDynDescr;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
}