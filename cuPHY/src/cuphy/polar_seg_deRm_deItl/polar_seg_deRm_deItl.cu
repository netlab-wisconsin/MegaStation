/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 #include "polar_seg_deRm_deItl.hpp"

 #include <stdio.h>
 
 #include <functional>
 #include <cooperative_groups.h>
 
 static constexpr float CW_LLR_LOW_LIM  = -100.0f;
 static constexpr float CW_LLR_HIGH_LIM =  100.0f;
 
 using namespace cooperative_groups;
 
 //#define ENABLE_DEBUG
 
 namespace derm_deitl
 {
 // clang-format off
 static __device__ __constant__  uint32_t POLAR_SUB_BLK_ITL_32[] =
 {
     0,  1,	2,	4,	3,	5,	6,	7,
     8,	16,	9,	17,	10,	18,	11,	19,
     12,	20,	13,	21,	14,	22,	15,	23,
     24,	25,	26,	28,	27,	29,	30,	31,
 };
 // clang-format on
 
 // ToDo should it be moved as part of polSegDeRmDeItlDynDescr_t and computed on CPU?
 struct cuPolSegDeItlDesc
 {
     int32_t nItlMat;          // size of channel Itl matrix buffer
     int32_t nRowsRegion1;     // number of rows in region 1
     int32_t nColsRegion1;     // number of columns in region 1
     int32_t nBitsRegion1;     // number of bits in region 1
     int32_t nRowsRegion2;     // number of rows in region 2
     int32_t nBitsRegion1And2; // number of bits in region 1 and 2
 };
 
 __device__ __forceinline__ void compute_polDeItlDesc(cuPolSegDeItlDesc& deItl, uint32_t nTxBits)
 {
     int32_t& T            = deItl.nItlMat;
     int32_t& nRowsRegion1 = deItl.nRowsRegion1;
     int32_t& nColsRegion1 = deItl.nColsRegion1;
     int32_t& nBitsRegion1 = deItl.nBitsRegion1;
     int32_t& nRowsRegion2 = deItl.nRowsRegion2;
 
     // size of interleaved matrix buffer
     T = ceil((-1.f + sqrtf(static_cast<float>(1 + 8 * nTxBits))) / 2.f);
 
     // Region 1 boundaries
     float   b                 = static_cast<float>(-(1 + 2 * deItl.nItlMat));
     int32_t lastRmIdx         = nTxBits - 1;
     int32_t lastRowIdxRegion1 = static_cast<int32_t>(floor((-b - sqrtf(static_cast<float>(b * b - 8 * lastRmIdx))) / 2.f));
     int32_t lastColIdxRegion1 = lastRmIdx - lastRowIdxRegion1 * T + (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;
 
     // Region 1 sizes
     nBitsRegion1 = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
     nRowsRegion1 = lastRowIdxRegion1 + 1;
     nColsRegion1 = lastColIdxRegion1 + 1;
 
     // Region 2 sizes
     nRowsRegion2           = nRowsRegion1 - 1;
     int32_t nColsRegion2   = ((T - nRowsRegion2 + 1) - nColsRegion1);
     deItl.nBitsRegion1And2 = nBitsRegion1 + nColsRegion2 * nRowsRegion2;
 }
 
 __device__ __forceinline__ void computeColumnRowIndices(uint32_t chanItlIdx, uint32_t nTxBits, cuPolSegDeItlDesc& deItl, uint32_t& colIdx, uint32_t& rowIdx)
 {
     // First determine the index is within which region of interleaving matrix
     // based on region, row/col indices are computed
     if(chanItlIdx < deItl.nBitsRegion1)
     {
         colIdx = chanItlIdx / deItl.nRowsRegion1;
         rowIdx = chanItlIdx % deItl.nRowsRegion1;
     }
     else if(chanItlIdx < deItl.nBitsRegion1And2)
     {
         colIdx = (chanItlIdx - deItl.nBitsRegion1) / deItl.nRowsRegion2 + deItl.nColsRegion1;
         rowIdx = (chanItlIdx - deItl.nBitsRegion1) % deItl.nRowsRegion2;
     }
     else
     {
         uint32_t flippedIdx    = nTxBits - 1 - chanItlIdx;
         uint32_t flippedColIdx = floor((-1.f + sqrt(static_cast<float>(1 + 8 * flippedIdx))) / 2);
         uint32_t flippedRowIdx = flippedIdx - flippedColIdx * (flippedColIdx + 1) / 2;
         //
         colIdx = deItl.nItlMat - 1 - flippedColIdx;
         rowIdx = flippedColIdx - flippedRowIdx;
     }
 }
 
 static __global__ void
 polSegDeRmDeItlKernel(polSegDeRmDeItlDynDescr_t* pDynDescr)
 {
     const uint32_t UCI_SEG_IDX = blockIdx.x;
 
     uint32_t      nTxBits           = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].E_cw;
     uint16_t      nInfoBits            = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].K_cw;
     uint16_t      nCodedBits               = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].N_cw;
     //uint8_t       n                      = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].n_cw;
     uint8_t       nCbs                     = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].nCbs;
  //   const __half* uciSegLLRs               = pDynDescr->pUciSegLLRsAddrs[UCI_SEG_IDX];
     const __half* uciSegLLRs               = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].pUciSegLLRs;
    // __half*       cwLLRs            = pDynDescr->pCwLLRsAddrs[UCI_SEG_IDX];
    const uint8_t (&childCbIdxs)[2]        = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].childCbIdxs;
    const cuphyPolarCwPrm_t*  pPolarCwPrms = pDynDescr->pPolarCwPrms;
 
     __half* childCwLLRs[2];
     for(int i = 0; i < nCbs; ++i)
     {
        childCwLLRs[i] = pPolarCwPrms[childCbIdxs[i]].pCwLLRs;
     }
 
     thread_block const& thisThrdBlk  = this_thread_block();
     uint32_t            thrdIdxInBlk = thisThrdBlk.thread_rank();
 
     int16_t rmMethod; //  rate-matching method. 0 -> repetition, 1->puncturing, 2->shortening
     rmMethod = (nTxBits >= nCodedBits)         ? 0 :
                (16 * nInfoBits <= 7 * nTxBits) ? 1 :
                                                  2;
 
     cuPolSegDeItlDesc deItl;
     compute_polDeItlDesc(deItl, nTxBits);
 
     // initialize cwLLRs
     if(rmMethod == 2)
     {
         // Shortened codeword bits known by decoder to be 0, initialize their LLRs to large value
         for(uint32_t id = thrdIdxInBlk; id < nCbs * nCodedBits; id += thisThrdBlk.size())
         {
             uint32_t cbIdx = id / nCodedBits;
             uint32_t cwIdx = id % nCodedBits;
             childCwLLRs[cbIdx][cwIdx] = 20;
         }
     }
     if((rmMethod == 1) || (rmMethod == 0))
     {
         // Punctured bits unknown to decoder, initialize their LLRs to 0
         for(uint32_t id = thrdIdxInBlk; id < nCbs * nCodedBits; id += thisThrdBlk.size())
         {
             uint32_t cbIdx = id / nCodedBits;
             uint32_t cwIdx = id % nCodedBits;
             childCwLLRs[cbIdx][cwIdx] = 0;
         }
     }
 
     uint32_t colIdx;
     uint32_t rowIdx;
     uint32_t rmIdx;
     uint32_t nSubBlocks = nCodedBits >> 5; // number of sub-blocks in sub-block interleaving
 
     for(uint32_t cbIdx = 0; cbIdx < nCbs; cbIdx++)
     {
         for(uint32_t chanItlIdx = thrdIdxInBlk; chanItlIdx < nTxBits; chanItlIdx += thisThrdBlk.size())
         {
             computeColumnRowIndices(chanItlIdx, nTxBits, deItl, colIdx, rowIdx);
 #ifdef ENABLE_DEBUG
             printf("tid %d, colIdx %d, rowIdx %d\n", thrdIdxInBlk, colIdx, rowIdx);
 #endif
             // Compute rate-matched index
             rmIdx = colIdx + deItl.nItlMat * rowIdx - rowIdx * (rowIdx - 1) / 2;
             // Extract LLRs of polar encoded uci segment
             auto rmLLR = uciSegLLRs[nTxBits * cbIdx + chanItlIdx];
 #ifdef ENABLE_DEBUG
             printf("tid %d, rmLLR %f\n", thrdIdxInBlk, static_cast<float>(rmLLR));
 #endif
 
             // perform de-rate matching: write rmIdx LLR to cwLLR buffer
             if(rmMethod == 0)
             {
                 uint32_t subBlockItlCwIdx = rmIdx % nCodedBits;
                 uint32_t subBlockIdx      = subBlockItlCwIdx / nSubBlocks;
                 uint32_t cwIdx            = POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + (subBlockItlCwIdx % nSubBlocks);
 
                 // ToDo consider using shared memory and verify its impact on performance
                 atomicAdd(&childCwLLRs[cbIdx][cwIdx], rmLLR);
             }
             else if(rmMethod == 1)
             {
                 uint32_t subBlockItlCwIdx = nCodedBits - nTxBits + rmIdx;
                 uint32_t subBlockIdx      = subBlockItlCwIdx / nSubBlocks;
                 uint32_t cwIdx            = POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + (subBlockItlCwIdx % nSubBlocks);
 
                 childCwLLRs[cbIdx][cwIdx] = rmLLR;
             }
             else
             {
                 uint32_t& subBlockItlCwIdx = rmIdx;
                 uint32_t  subBlockIdx      = subBlockItlCwIdx / nSubBlocks;
                 uint32_t  cwIdx            = POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + (subBlockItlCwIdx % nSubBlocks);
 
                 childCwLLRs[cbIdx][cwIdx] = rmLLR;
             }
         }
     }
 
     __syncthreads();
     if(thrdIdxInBlk  < nCodedBits)
     {
         for(int cbIdx = 0; cbIdx < nCbs; ++cbIdx)
         {
             if(__hgt(childCwLLRs[cbIdx][thrdIdxInBlk], static_cast<__half>(CW_LLR_HIGH_LIM)))
             {
                 childCwLLRs[cbIdx][thrdIdxInBlk] = static_cast<__half>(CW_LLR_HIGH_LIM);
             }
         
             if(__hlt(childCwLLRs[cbIdx][thrdIdxInBlk], static_cast<__half>(CW_LLR_LOW_LIM)))
             {
                 childCwLLRs[cbIdx][thrdIdxInBlk] = static_cast<__half>(CW_LLR_LOW_LIM);
             }
         }
     }
 
 #ifdef ENABLE_DEBUG
     for(uint32_t cbIdx = 0; cbIdx < nCbs; cbIdx++)
     {
         if(thrdIdxInBlk < nCodedBits)
         {
             printf("cwIdx[%d] = %f\n", thrdIdxInBlk + cbIdx * nCodedBits, static_cast<float>(cwLLRs[cbIdx * nCodedBits + thrdIdxInBlk]));
         }
     }
 #endif
 
     //-----------------------------------------------------------------------------------------------------------
 #ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("\n polSegDeRmDeItlKernel running... \n");
     }
 
     if((0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("\n UCI segment %d has the following parameters: E = %d, K = %d, N = %d, n = %d, nCbs = %d\n", UCI_SEG_IDX, nTxBits, nInfoBits, nCodedBits, n, nCbs);
         printf(" Rate matching method %d, nRowsRegion1 %d, nColsRegion1 %d, nBitsRegion1And2 %d\n", rmMethod, deItl.nRowsRegion1, deItl.nColsRegion1, deItl.nBitsRegion1And2);
     }
 #endif
 }
 } // namespace derm_deitl
 
 void polSegDeRmDeItl::kernelSelect(uint16_t                         nPolUciSegs,
                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                    cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg)
 {
     // determine max encoded cb size:
     uint16_t max_N_cw = 0;
     for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
     {
         if(pPolUciSegPrmsCpu[segIdx].N_cw > max_N_cw)
             max_N_cw = pPolUciSegPrmsCpu[segIdx].N_cw;
     }
 
     // launch geometry (can change!)
     dim3 gridDim(nPolUciSegs);
     dim3 blockDim(max_N_cw);
 
     // kernel (only one kernel option for now)
     void* kernelFunc = reinterpret_cast<void*>(derm_deitl::polSegDeRmDeItlKernel);
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
 
 void polSegDeRmDeItl::setup(uint16_t                         nPolUciSegs,                 // number of polar UCI segments
                             uint16_t                         nPolCws,                     // number of polar codewords
                             const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,           // starting adreass of polar UCI segment parameters (CPU)
                             const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,           // starting adreass of polar UCI segment parameters (GPU)
                             const cuphyPolarCwPrm_t*         pPolCwPrmsCpu,               // starting address of polar codeword parameters (CPU)
                             const cuphyPolarCwPrm_t*         pPolCwPrmsGpu,               // starting address of polar codeword parameters (GPU)
                             __half**                         pUciSegLLRsAddrs,            // pointer to UCI segment LLRS (GPU)
                             __half**                         pCwLLRsAddrs,                // point to codeword LLRs (GPU)
                             polSegDeRmDeItlDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                             void*                            pGpuDynDesc,                 // pointer to descriptor in gpu
                             uint8_t                          enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
                             cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
                             cudaStream_t                     strm)                                            // stream to perform copy
 {
     // populate dynamic descriptor:
     pCpuDynDesc->pPolarUciSegPrms = pPolUciSegPrmsGpu;
     pCpuDynDesc->pPolarCwPrms     = pPolCwPrmsGpu;
     for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
     {
         pCpuDynDesc->pUciSegLLRsAddrs[segIdx] = pUciSegLLRsAddrs[segIdx];
     }
     for(uint16_t cwIdx = 0; cwIdx < nPolCws; ++cwIdx)
     {
         pCpuDynDesc->pCwLLRsAddrs[cwIdx] = pCwLLRsAddrs[cwIdx];
     }
 
     // save pointer to GPU descriptor
     polSegDeRmDeItlKernelArgs_t& kernelArgs = m_kernelArgs;
     kernelArgs.pDynDescr                    = reinterpret_cast<polSegDeRmDeItlDynDescr_t*>(pGpuDynDesc);
 
     // Optional descriptor copy to GPU memory
     if(enableCpuToGpuDescrAsyncCpy)
     {
         cudaMemcpyAsync(&pGpuDynDesc, &pCpuDynDesc, sizeof(polSegDeRmDeItlDynDescr_t), cudaMemcpyHostToDevice, strm);
     }
 
     // select kernel (includes launch geometry). Populate launchCfg.
     kernelSelect(nPolUciSegs, pPolUciSegPrmsCpu, pLaunchCfg);
     pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
     pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
 }
 
 void polSegDeRmDeItl::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
     dynDescrSizeBytes  = sizeof(polSegDeRmDeItlDynDescr_t);
     dynDescrAlignBytes = alignof(polSegDeRmDeItlDynDescr_t);
 }