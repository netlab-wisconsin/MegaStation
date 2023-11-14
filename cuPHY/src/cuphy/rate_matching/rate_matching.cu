/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

//#define LOOP 1
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <assert.h>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "rate_matching.hpp"

#include "descrambling.cuh"
#include "crc.hpp"
#include "descrambling.hpp"
#include "derate_matching_modulo.hpp"

using namespace cuphy_i;
using namespace descrambling;
using namespace crc;

__device__ inline int isnan_(float f) { return isnan(f); }
__device__ inline int isnan_(__half h) { return isnan(__half2float(h)); }

__device__ inline float rate_match_xor_sign(uint32_t seq, int bit_index, float llr_input)
{
    union u
    {
        float    f32;
        uint32_t u32;
    };
    u input, output;
    input.f32 = llr_input;
    // Extract the desired bit from the sequence and XOR with the input
    // float to (possibly) modify the sign bit of the input.
    output.u32 = (((seq << (31 - bit_index)) & 0x80000000) ^ input.u32);
    return output.f32;
}

__device__ inline __half rate_match_xor_sign(uint32_t seq, int bit_index, __half llr_input)
{
    // Shift the desired bit from the sequence to the sign position for
    // a half precision value (bit 15).
    uint32_t   half_sign_mask = (seq >> bit_index) << 15;
    __half_raw hraw           = llr_input;
    uint32_t   hraw32         = hraw.x;
    // XOR the sign mask with the original value to (possibly) modify
    // the sign bit of the input
    uint32_t out32 = (half_sign_mask & 0x00008000) ^ hraw32;
    hraw.x         = (unsigned short)out32;
    return __half(hraw);
}

template <typename T_IN, typename T_OUT>
__global__ void __launch_bounds__(1024, 1) de_rate_matching_global2(puschRxRateMatchDescr_t* pRmDesc)
{
    // PUSCH kernel descriptor
    puschRxRateMatchDescr_t& rmDesc = *pRmDesc;
    uint32_t fracCbIdx = blockIdx.x;
    uint32_t cbIdx = blockIdx.y;
    uint32_t tbIdx = blockIdx.z;
    uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];

    // Output tensor
    // @todo: rmDesc.out which holds an array of pointers to HARQ buffers (in GPU memory) lives in host pinned memory, 
    // check performance impact of accessing this memory 
    T_OUT* out = static_cast<T_OUT*>(rmDesc.out[ueIdx]);

    // Array of transport block parameters structs
    const PerTbParams* tbPrmsArray = rmDesc.tbPrmsArray;
    // code block index
    uint32_t r = cbIdx + tbPrmsArray[ueIdx].firstCodeBlockIndex;

    // Output code block stride
    uint32_t Ncb_padded = tbPrmsArray[ueIdx].Ncb_padded;
    uint32_t cbStartOffset = r * Ncb_padded;

    // Adjust for codeblock offset
    out += cbStartOffset;

    // Enable/Disable descrambling
    int descramblingOn = rmDesc.descramblingOn;
    // Input LLR tensor
    const T_IN* llr_vec_in = static_cast<const T_IN*>(rmDesc.llr_vec_in[tbIdx]);

    // Number of BBU antenna layers, KERNEL LEVEL parameter
    uint32_t nBBULayers = rmDesc.tbPrmsArray[ueIdx].nBBULayers;

    //******** The following parameters are invariant for all CTAs working on the same transport block*******/
    // They only vary along the y-dimension of the grid, namely across transport blocks
    

    // Output de-rate matched code block size excluding punctured bits
    uint32_t Ncb = tbPrmsArray[ueIdx].Ncb;

    // number of code blocks in transport block
    uint32_t C = tbPrmsArray[ueIdx].num_CBs;
    // base graph index
    uint32_t bg = tbPrmsArray[ueIdx].bg;
    // redundancy version
    uint32_t rv = tbPrmsArray[ueIdx].rv;
    // new data indicator
    uint32_t ndi = tbPrmsArray[ueIdx].ndi;
    // debug llr index buffer
    uint32_t* debug_d_derateCbsIndices = tbPrmsArray[ueIdx].debug_d_derateCbsIndices;

    // lifting factor
    uint32_t Zc = tbPrmsArray[ueIdx].Zc;
    // QAM modulation index
    uint32_t Qm = tbPrmsArray[ueIdx].Qm;
    // Number of layers occupied by transport block tbIdx
    uint32_t Nl = tbPrmsArray[ueIdx].Nl;

    /************/

    if(r < C)
    { // Only excutes code if thread is allocated a valid
        //  codeblock
        // (some threads will be idle)

        // Determine input rate matched block size E and start index codeBlockQAMStartIndex

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        // index at which the first LLR of code block r starts within transport block tbIdx
        uint32_t codeBlockQAMStartIndex;
        // Size (number of LLRS) of input rate-matched code block r
        uint32_t E;
        // Number of layers times modulation index: determines how many LLRs are read from each block of NBBULayers
        uint32_t TBLLRsPerNBBULayers = Nl * Qm;
        // total number of LLRS to be read for currrent transport block
        uint32_t totalNLLRsForTB = TBLLRsPerNBBULayers * C;

        // encodedSize is size (number of LLRs) of current transport block; q1 is number of NBBULayers blocks the transport block is spread over
        uint32_t q1 = tbPrmsArray[ueIdx].uciOnPuschFlag ? tbPrmsArray[ueIdx].G / TBLLRsPerNBBULayers : tbPrmsArray[ueIdx].encodedSize / TBLLRsPerNBBULayers; // exact division
        // number of NBBULayers blocks each code block is spread over
        uint32_t q = q1 / C;

        // This is straight from the spec: compute size E of each code block of current transport block
        uint32_t rr = C - (q1 - q * C) - 1;
        // smaller code blocks size
        uint32_t El = Nl * Qm * q;
        // larger code block size
        //uint32_t Eh = Nl * Qm * ((Ncb + totalNLLRsForTB - 1) / totalNLLRsForTB);
        uint32_t Eh = El + TBLLRsPerNBBULayers * (q * totalNLLRsForTB < tbPrmsArray[ueIdx].encodedSize);

        if(r <= rr)
        {
            E                      = El;
            codeBlockQAMStartIndex = r * El;
        }
        else
        {
            E                      = Eh;
            codeBlockQAMStartIndex = (rr + 1) * El + (r - rr - 1) * Eh;
        }

        // For incremental redundancy transmission: determine k0 based on rv and bg(base graph)
        // Uninitialized scalar variable this changes will treat rv 0 (detected if k0=0): no LLR combining, just write to memory; write
        uint32_t k0 = 0;
        if(bg == 1)
        {
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (17 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (33 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (56 * Ncb / (66 * Zc)) * Zc;
            }
        }
        else if(bg == 2)
        {
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (13 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (25 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (43 * Ncb / (50 * Zc)) * Zc;
            }
        }
        /************/
        // First code block LLR index for current CTA within transport block, used for descrambling sequence generation
        uint32_t t = codeBlockQAMStartIndex + fracCbIdx * blockDim.x * WORD_SIZE;

        // make sure each warp computes bits of the sequence that are used by that warp only
        uint32_t index = (threadIdx.x % 32) * 32 + (threadIdx.x / 32);

        // each thread in a warp computes a word of the descrambling sequence that will be used at step threadIdx.x, there are currently 32 steps
        uint32_t mySeq = gold32n(tbPrmsArray[ueIdx].cinit, t + (index * 32));

        //******** The following parameters are invariant for all CTAs working on the same transport block*******/

        uint32_t adjustedCodeBlockQAMStartIndex = (codeBlockQAMStartIndex / Qm) * QAM_STRIDE;
        // Number of filler bits
        uint32_t F = tbPrmsArray[ueIdx].F;
        uint32_t nPuncturedBits = 2 * Zc;

        // Number of systematic bits
        uint32_t K = tbPrmsArray[ueIdx].K;
        // Number of systematic bits in output code block excluding punctured bits
        uint32_t K_hat = K - nPuncturedBits;
        // Number of payload bits in output code block
        uint32_t Kd = K - nPuncturedBits - F;

        // nZpBitsPerCb is the total number of LLRs i.e. (2 * Zc) + E + F rounded up to a multiple of Zc (rounding needed by LDPC decoder)
        // (2 * Zc) - punctured LLRs
        // E        - rate matched LLRs
        // F        - Filler LLRs    
        // uint32_t nZpBitsPerCb = tbPrmsArray[tbIdx].nZpBitsPerCb;

        //number of LLRs belonging to other transport blocks to be skipped before getting to LLRS belonging to current transport block again
        /////////uint32_t cbStep = nBBULayers / Nl;
        // Deinterleave and fill output vector except filler bits

        // Make sure CTA is not reading beyond input code block size E

        int maxIndex = E + ((32 - (E % 32)) % 32);
        int maxIndexThisThrdBlk  = (fracCbIdx+1) * blockDim.x * 32;
        maxIndex = (maxIndexThisThrdBlk < maxIndex) ? maxIndexThisThrdBlk : maxIndex; //limit the maxIndex to the larger of per thread block max or global max

        // current thread LLR index
        uint32_t threadTBLLRindex = threadIdx.x;
        // used to broadcast de-scrambling sequence word to warp, currenty 32 lanes/steps are used
        uint32_t warpLane = 0;

        // if ((blockIdx.y == 0) && (blockIdx.z == 0) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
        // if ((blockIdx.z == 0) && (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0))
        // {
        //     printf("TB %u CB %u C %u ndi %u rv = %u Zc=%u k0=%u Kd=%u K=%u K_hat=%u F=%u E=%u Ncb=%u Ncb_padded=%u Qm=%u G=%u, encodedSize=%u\n",tbIdx,cbIdx,C,ndi,rv,Zc,k0,Kd,K,K_hat,F,E,Ncb,Ncb_padded,Qm, tbPrmsArray[ueIdx].G, tbPrmsArray[ueIdx].encodedSize);
        // }

        for(int tid = fracCbIdx * blockDim.x * 32 + threadIdx.x; tid < maxIndex; tid += blockDim.x)
        {
            T_OUT    llr;
            uint32_t j  = tid / Qm;
            uint32_t jl = j / Nl;
            uint32_t k  = tid - j * Qm;

	    // broadcast descrambling sequence word to entire warp
                uint32_t seq = __shfl_sync(0xFFFFFFFF, mySeq, warpLane);


            if(j < E / Qm)
            {
                if(tbPrmsArray[ueIdx].uciOnPuschFlag)
                    llr = static_cast<T_OUT>(llr_vec_in[codeBlockQAMStartIndex + tid]);

                else
                    // llr = static_cast<T_OUT>(llr_vec_in[adjustedCodeBlockQAMStartIndex * cbStep +
                    //                                     (k + (jl * nBBULayers + tbPrmsArray[tbIdx].layer_map_array[(j - jl * Nl)]) * QAM_STRIDE)]);
                    // use nBBULayers / Nl instead of cbStep for more than 1 cb; pass tc7337.
                    llr = static_cast<T_OUT>(llr_vec_in[(uint32_t)(adjustedCodeBlockQAMStartIndex * nBBULayers / Nl) +
                                                         (k + (jl * nBBULayers + tbPrmsArray[ueIdx].layer_map_array[(j - jl * Nl)]) * QAM_STRIDE)]);
            }
            if(j < E / Qm)
            {
                // change sign based on scrambling sequence bit

                if(descramblingOn && (!tbPrmsArray[ueIdx].uciOnPuschFlag))
                {
                    // Previous method generates NaNs when the llr value is inf,
                    // due to multiplicaion with 0. (This is likely to occur at
                    // high SNR conditions with the fp16 data type.)
                    //uint32_t s = (seq >> (t % WORD_SIZE)) & 1;
                    //uint32_t sn = (s + 1) & 0x1;
                    //T llr_old = -llr * static_cast<T>(s) + llr * static_cast<T>(sn);

                    llr = rate_match_xor_sign(seq, (threadTBLLRindex % WORD_SIZE), llr);
                    //if(static_cast<float>(llr) != static_cast<float>(llr_old)) { printf("Error: llr = %f, llr_check = %f\n", static_cast<float>(llr), static_cast<float>(llr_check));  }
                    //if(isnan_(static_cast<float>(llr_old))) { printf("llr_old = NaN: threadIdx.x = %u, s = %u, sn = %u, llr = %f, llr_old = %f\n", threadIdx.x, s, sn, static_cast<float>(llr), static_cast<float>(llr_check));
                }

                // Clamp the llr
                if(llr > static_cast<T_OUT>(10000.0))
                {
                    llr = static_cast<T_OUT>(10000.0);
                }
                else if(llr < static_cast<T_OUT>(-10000.0))
                {
                    llr = static_cast<T_OUT>(-10000.0);
                }

                uint32_t outIdx;
                int inIdx = k * E / Qm + j;
                outIdx = derate_match_fast_calc_modulo(inIdx, E, K_hat, Kd, F, k0, Ncb, Zc);

                // Within the output buffer, the Ncb circular buffer starts at offset 2*Zc (punctured bits)
                outIdx += nPuncturedBits;

                // ndi 1: no LLR combining, just write to memory;
                // ndi 0: LLR combining, use atomicAdd
                if(ndi)
                {
                    out[outIdx] = llr;
                    // if((12 == tbIdx) || (13 == tbIdx) || (14 == tbIdx) || (15 == tbIdx))  printf("derate-match: TB %02d out[%05d] %f inIdx %d E %d K_hat %d Kd %d F %d k0 %d Ncb %d Zc %d\n", tbIdx, outIdx, __half2float(llr), inIdx, E, K_hat, Kd, F, k0, Ncb, Zc);
                }
                else
                {
                    T_OUT prev_llr = out[outIdx];
                    llr += prev_llr;

                    // Clamp the llr
                    if(llr > static_cast<T_OUT>(10000.0))
                    {
                        llr = static_cast<T_OUT>(10000.0);
                    }
                    else if(llr < static_cast<T_OUT>(-10000.0))
                    {
                        llr = static_cast<T_OUT>(-10000.0);
                    }

                    // Write the updated LLR.  No need for atomic, different threads work on different outIdx.
                    out[outIdx] = llr;
                }
                if(debug_d_derateCbsIndices != nullptr)
                {
                    uint32_t debug_inIdx                  = codeBlockQAMStartIndex * Qm / QAM_STRIDE + k * E / Qm + j;
                    uint32_t debug_outIdx                 = outIdx - r * Ncb_padded;
                    debug_d_derateCbsIndices[debug_inIdx] = debug_outIdx;
                    //printf("c,outIdx,inIdx,llr,%05u,%05d,%05d,%f\n",tbIdx,debug_outIdx,debug_inIdx,__half2float(llr));
                }
            }
            warpLane++;
            threadTBLLRindex += blockDim.x;
        } 
        
        if(ndi)
        {
            // Use all thread blocks associated with this CB
            uint32_t nFracCbs = gridDim.x;
            uint32_t stride = nFracCbs*blockDim.x;
            uint32_t tid = (fracCbIdx*blockDim.x) + threadIdx.x;

            // Output buffer is of length Ncb_padded

            // 1. Circular buffer initilization (Ncb long circular buffer section of output buffer)
            // 1a. Write filler bits to circular buffer  
            for(uint32_t n = Kd + tid; n < Kd + F; n += stride)
            {
                // Note: Location of Filler bits is fixed to tail end of systematic bit section of 
                // circular buffer and is independent of k0
                uint32_t circBufIdx = n;
                out[nPuncturedBits + circBufIdx] = 10000.0;
                //uint32_t outIdx = 2*Zc + r * (Ncb + 2 * Zc) + n;
                //printf("c,outIdx,inIdx,llr,%05u,%05d,%05d,%f\n",tbIdx,outIdx,99998,10000.0);
            }
            
            // NOTE: EITHER 1b, 2a, 2b below needed OR at setup set RM output to zero
    
            // 1b. Write zeros into rest of Ncb long circular buffer
            for(uint32_t n = E + F + tid; n < Ncb; n += stride)
            {
                // Account for non-zero k0            
                int32_t circBufIdx = k0 + n;
                
                // Skip over filler bit region if circular buffer index lies in the filler bit region
                if((circBufIdx >= Kd) && (circBufIdx < Kd + F)) circBufIdx += F;

                // Per CUDA programmer's manual modulo operation is costly as it compiles to upto 20 instructions
                // Logic below to avoid integer modulo
                while(circBufIdx >= Ncb)
                {
                    circBufIdx -= Ncb;
                    if((circBufIdx >= Kd) && (circBufIdx < Kd + F)) circBufIdx += F;
                }                                    
                out[nPuncturedBits + circBufIdx] = 0;
            }

            // 2. Initialization of the rest of output buffer: section of length (Ncb_padded - Ncb)
            // 2a. Write zeros into punctured bits (first 2*Zc bits of Ncb_padded output buffer)
            for(uint32_t n = tid; n < nPuncturedBits; n += stride)
            {
                out[n] = 0;
            }

            // 2b. Write zeros into byte padding section of Ncb_padded output buffer
            for(uint32_t n = Ncb; n < Ncb_padded; n += stride)
            {
                out[n] = 0;
            }
        }    
    }
}

template <typename T_IN, typename T_OUT>
void deRateMatch2KernelLauncher(puschRxRateMatchLaunchPrms_t& kernelLaunchPrms,
                                cudaStream_t&                 strm)
{
    puschRxRateMatchLaunchGeo_t& launchGeo = kernelLaunchPrms.geo;
    de_rate_matching_global2<T_IN, T_OUT><<<launchGeo.gridDim, launchGeo.blockDim, launchGeo.shMemBytes, strm>>>(kernelLaunchPrms.args);
}

extern "C" {

// computes grid size and block size for rate matching kernel
// and stores them in output crcrDecodeDescriptor
}

void puschRxRateMatch::setup(uint16_t                          nSchUes,                     // number of users with sch data
                             uint16_t*                         pSchUserIdxsCpu,             // indicies of users with SCH data
                             const PerTbParams*                pTbPrmsCpu,                  // starting adress of transport block paramters (CPU)
                             const PerTbParams*                pTbPrmsGpu,                  // starting adress of transport block paramters (GPU)
                             cuphyTensorPrm_t*                 pTPrmRmIn,                   // starting adress of input LLR tensor parameters
                             cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                             void**                            ppRmOut,                     // array of rm outputs (GPU)
                             void*                             pCpuDesc,                    // pointer to descriptor in cpu
                             void*                             pGpuDesc,                    // pointer to descriptor in gpu
                             uint8_t                           enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
                             cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
                             cudaStream_t                      strm)                                             // stream to perform copy
{
    // setup CPU descriptor
    puschRxRateMatchDescr_t& desc    = *(static_cast<puschRxRateMatchDescr_t*>(pCpuDesc));
    uint16_t                 nUciUes = 0;

    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx      = pSchUserIdxsCpu[i];
        desc.schUserIdxs[i] = ueIdx;
        if(pTbPrmsCpu[ueIdx].uciOnPuschFlag)
        {
            desc.llr_vec_in[i] = pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
            nUciUes++;
        }
        else
        {
            uint32_t ueGrpIdx  = pTbPrmsCpu[ueIdx].userGroupIndex;
            if(pTbPrmsCpu[ueIdx].nDmrsCdmGrpsNoData==1)
            {
                desc.llr_vec_in[i] = pTPrmCdm1RmIn[ueGrpIdx].pAddr;
            }
            else
            {
                desc.llr_vec_in[i] = pTPrmRmIn[ueGrpIdx].pAddr;
            }
        }
    }
    desc.out            = ppRmOut;
    desc.tbPrmsArray    = pTbPrmsGpu;
    desc.descramblingOn = m_descramblingOn;

    // optional CPU->GPU copy
    if(enableCpuToGpuDescrAsyncCpy)
    {
        // added Unchecked return value
        CUDA_CHECK(cudaMemcpyAsync(pGpuDesc, pCpuDesc, sizeof(puschRxRateMatchDescr_t), cudaMemcpyHostToDevice, strm));
    }

    // Setup Launch Geometry
    uint32_t EMax = 0; // max number of encoded bits per CB
    uint32_t CMax = 0; // max number of CBs per TB
    for(uint32_t i = 0; i < nSchUes; ++i)
    {
        uint16_t ueIdx = pSchUserIdxsCpu[i];
        CMax           = CMax < pTbPrmsCpu[ueIdx].num_CBs ? pTbPrmsCpu[ueIdx].num_CBs : CMax;
        uint32_t Eh    = pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * ceilf(float(pTbPrmsCpu[ueIdx].encodedSize) / float(pTbPrmsCpu[ueIdx].Nl * pTbPrmsCpu[ueIdx].Qm * pTbPrmsCpu[ueIdx].num_CBs));
        EMax           = EMax < Eh ? Eh : EMax;
    }

    uint32_t threadBlkDim = 1024;

    dim3 gridDim((EMax + threadBlkDim * 32 - 1) / (threadBlkDim * 32), CMax, nSchUes);
    dim3 blockDim(threadBlkDim, 1, 1);
    // printf("gridDim(%d %d %d) blockDim(%d %d %d)\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

    pLaunchCfg->desc                                  = pGpuDesc;
    pLaunchCfg->kernelArgs[0]                         = &(pLaunchCfg->desc);
    pLaunchCfg->kernelNodeParamsDriver.gridDimX       = gridDim.x;
    pLaunchCfg->kernelNodeParamsDriver.gridDimY       = gridDim.y;
    pLaunchCfg->kernelNodeParamsDriver.gridDimZ       = gridDim.z;
    pLaunchCfg->kernelNodeParamsDriver.blockDimX      = blockDim.x;
    pLaunchCfg->kernelNodeParamsDriver.blockDimY      = blockDim.y;
    pLaunchCfg->kernelNodeParamsDriver.blockDimZ      = blockDim.z;
    pLaunchCfg->kernelNodeParamsDriver.func           = m_kernelFunc;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams   = &(pLaunchCfg->kernelArgs[0]);
    pLaunchCfg->kernelNodeParamsDriver.sharedMemBytes = 0;
    pLaunchCfg->kernelNodeParamsDriver.extra          = nullptr;
}

void puschRxRateMatch::init(int rmFPconfig,     // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
                            int descramblingOn) // enable/disable descrambling
{
    // Save configurations
    m_descramblingOn = descramblingOn;

    // Select Kernel
    switch(rmFPconfig)
    {
    case 0:
        CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, float>)));
        break;

    case 1:
        CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, float>)));
        break;

    case 2:
        CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<float, __half>)));
        break;

    case 3:
        CUDA_CHECK(cudaGetFuncBySymbol(&m_kernelFunc, reinterpret_cast<void*>(de_rate_matching_global2<__half, __half>)));
        break;
    default:
        break;
    }
}

void puschRxRateMatch::getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes)
{
    descrSizeBytes  = sizeof(puschRxRateMatchDescr_t);
    descrAlignBytes = alignof(puschRxRateMatchDescr_t);
}
