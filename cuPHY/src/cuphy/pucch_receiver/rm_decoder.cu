/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#if CUDART_VERSION >= 11000
#include <cooperative_groups/reduce.h>
#endif

#include <stdio.h>
#include "rm_decoder.hpp"
#include "rm_codebook.h"
#include "cuphy.hpp"

static constexpr float LLR_LOW_LIM  = -50.0f;
static constexpr float LLR_HIGH_LIM =  50.0f;


namespace cg = cooperative_groups;

const int MAX_TWO_TO_K = 2048;
static const unsigned int MAX_N = PUCCH_F2_RM_MAX_N;
const uint16_t RM_DECODER_TPB = 512;


template <typename T>
__device__ void findMaxBlock(T * __restrict__ sdata_in, T * __restrict__ sdata_out, int two_to_K, const cg::thread_block &cta)
{
    const unsigned int tid = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int k=cta.thread_rank(); k<two_to_K; k += blockDim.x)
    {
#if CUDART_VERSION >= 11000
        sdata_out[k] = cg::reduce(tile32, sdata_in[k], cg::greater<T>());
#else
        T m = sdata_in[k];
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            T m_tmp = __shfl_down_sync(0xffffffff, m, offset);
            m = m >= m_tmp ? m : m_tmp;
        }
        if (tile32.thread_rank() == 0)
        {
            sdata_out[k] = m;
        }
#endif
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        T m_max = 0.0;
        for (int k = 0; k < two_to_K; k += tile32.size()) {
            T m = sdata_out[k];
            if (m > m_max)
            {
                m_max = m;
            }
        }
        sdata_out[0] = m_max;
    }
    cg::sync(cta);
}




// __global__ void pucch_rm_decoder_0(const float* const* __restrict__ r_in, const float * __restrict__ codebook, uint32_t * __restrict__ x_out, int numCW, int *d_E, int *d_two_to_K)
// {
//     __shared__ float shmem[MAX_TWO_TO_K+MAX_N];
//     float *shmem_m = &shmem[0];
//     float *shmem_r = &shmem[MAX_TWO_TO_K];

//     // Grid-stride over numCW
//     for (int cw = blockIdx.x; cw < numCW; cw += gridDim.x)
//     {
//         int E = d_E[cw];
//         int N = min(E,MAX_N);

//         int two_to_K = d_two_to_K[cw];

//         // Load r_in into shmem using grid-stride de-rate-matching
//         if (threadIdx.x < MAX_N)
//         {
//             shmem_r[threadIdx.x] = 0;
//             for (int k=threadIdx.x; k<E; k += MAX_N)
//             {
//                 shmem_r[threadIdx.x] += r_in[cw][k];
//             }
//         }
//         __syncthreads();

//         // Grid-stride correlate
//         for (int codeIdx = threadIdx.x; codeIdx < two_to_K; codeIdx += blockDim.x)
//         {
//             float m = 0;
//             for (int n=0; n<N; n++)
//             {
//                 // Note: This is bad access pattern
//                 float c = codebook[codeIdx*MAX_N + n];
//                 m += shmem_r[n] * c;
//             }
//             shmem_m[codeIdx] = m;
//         }
//         __syncthreads();

//         // Find max using a single thread
//         if (threadIdx.x == 0)
//         {
//             int max_k = 0;
//             float max_m = 0;
//             for (int k=0; k<two_to_K; k++)
//             {
//                 if (shmem_m[k] > max_m)
//                 {
//                     max_m = shmem_m[k];
//                     max_k = k;
//                 }
//             }

//             x_out[cw] = max_k;
//         }
//     }
// }






// __global__ void pucch_rm_decoder_1(const float* const* __restrict__ r_in, const float * __restrict__ codebook, uint32_t * __restrict__ x_out, int numCW, int *d_E, int *d_two_to_K)
// {
//     __shared__ float shmem[MAX_TWO_TO_K+MAX_N];
//     float *shmem_m = &shmem[0];
//     float *shmem_r = &shmem[MAX_TWO_TO_K];

//     // Grid-stride over numCW
//     for (int cw = blockIdx.x; cw < numCW; cw += gridDim.x)
//     {
//         int E = d_E[cw];
//         int N = min(E,MAX_N);
//         int two_to_K = d_two_to_K[cw];

//         // Load r_in into shmem using grid-stride de-rate-matching
//         if (threadIdx.x < MAX_N)
//         {
//             shmem_r[threadIdx.x] = 0;
//             for (int k=threadIdx.x; k<E; k += MAX_N)
//             {
//                 shmem_r[threadIdx.x] += r_in[cw][k];
//             }
//         }
//         __syncthreads();

//         // Grid-stride correlate
//         for (int codeIdx = threadIdx.x; codeIdx < two_to_K; codeIdx += blockDim.x)
//         {
//             float m = 0;
//             for (int n=0; n<N; n++)
//             {
//                 float c = codebook[n*MAX_TWO_TO_K + codeIdx];
//                 m += shmem_r[n] * c;
//             }
//             shmem_m[codeIdx] = m;
//         }
//         __syncthreads();

//         // Find max using a single thread
//         if (threadIdx.x == 0)
//         {
//             int max_k = 0;
//             float max_m = 0;
//             for (int k=0; k<two_to_K; k++)
//             {
//                 if (shmem_m[k] > max_m)
//                 {
//                     max_m = shmem_m[k];
//                     max_k = k;
//                 }
//             }
//             x_out[cw] = max_k;
//         }
//     }
// }



// __global__ void pucch_rm_decoder_2(const float* const* __restrict__ r_in, const float * __restrict__ codebook, uint32_t * __restrict__ x_out, int numCW, int *d_E, int *d_two_to_K)
// {
//     __shared__ float shmem[MAX_TWO_TO_K*2+MAX_N];
//     float *shmem_m = &shmem[0];
//     float *shmem_reduce = &shmem[MAX_TWO_TO_K];
//     float *shmem_r = &shmem[MAX_TWO_TO_K*2];

//     // Grid-stride over numCW
//     for (int cw = blockIdx.x; cw < numCW; cw += gridDim.x)
//     {
//         int E = d_E[cw];
//         int N = min(E,MAX_N);
//         int two_to_K = d_two_to_K[cw];

//         // Load r_in into shmem using grid-stride de-rate-matching
//         if (threadIdx.x < MAX_N)
//         {
//             shmem_r[threadIdx.x] = 0;
//             for (int k=threadIdx.x; k<E; k += MAX_N)
//             {
//                 shmem_r[threadIdx.x] += r_in[cw][k];
//             }
//         }
//         __syncthreads();

//         // Grid-stride correlate
//         int codeIdx;
//         for (codeIdx = threadIdx.x; codeIdx < two_to_K; codeIdx += blockDim.x)
//         {
//             float m = 0;
//             for (int n=0; n<N; n++)
//             {
//                 float c = codebook[n*MAX_TWO_TO_K + codeIdx];
//                 m += shmem_r[n] * c;
//             }
//             shmem_m[codeIdx] = m;
//         }
//         // Zero remaining correlation scores
//         while (codeIdx < MAX_TWO_TO_K)
//         {
//             shmem_m[codeIdx] = 0;
//             codeIdx += blockDim.x;
//         }
//         __syncthreads();

//         cg::thread_block block = cg::this_thread_block();
//         findMaxBlock<float>(shmem_m,shmem_reduce,MAX_TWO_TO_K,block);

//         // Find the index of the max metric
//         // Note: Race condition if more than one codeword has the same score,
//         //       but in that case we would likely make an decoding error anyways.
//         for (int k=threadIdx.x; k<two_to_K; k+=blockDim.x)
//         {
//             if (shmem_m[k] == shmem_reduce[0])
//             {
//                 x_out[cw] = k;
//                 break;
//             }
//         }
//     }
// }


template <typename T>
__global__ void rm_decoder_3(rmDecoderDynDescr_t* pDynDescr)
{
    const uint32_t * __restrict__ codebook = pDynDescr->pRmCodebook;
    uint16_t  numCW = pDynDescr->nCws;
    cuphyRmCwPrm_t*  pCwPrmsGpu = pDynDescr->pCwPrmsGpu;

    __shared__ char shmem[sizeof(T)*MAX_TWO_TO_K*2+sizeof(uint32_t)*MAX_TWO_TO_K+sizeof(T)*MAX_N];
    T *shmem_m = reinterpret_cast<T*>(&shmem[0]);
    T *shmem_reduce = reinterpret_cast<T*>(&shmem[sizeof(T)*MAX_TWO_TO_K]);
    uint32_t *shmem_c = reinterpret_cast<uint32_t*>(&shmem[sizeof(T)*MAX_TWO_TO_K*2]);
    T *shmem_r = reinterpret_cast<T*>(&shmem[sizeof(T)*MAX_TWO_TO_K*2+sizeof(uint32_t)*MAX_TWO_TO_K]);

    __shared__ float shfloatmem[sizeof(float)*MAX_N];
    float *shmem_f = reinterpret_cast<float*>(&shfloatmem[0]);

    __shared__ float shfloatmem_sumsquare[sizeof(float)*MAX_N];
    float *shmem_f_sumsquare = reinterpret_cast<float*>(&shfloatmem_sumsquare[0]);

    // load codebook into shmem
    for (int codeIdx = threadIdx.x; codeIdx < MAX_TWO_TO_K; codeIdx += blockDim.x)
    {
        shmem_c[codeIdx] = codebook[codeIdx];
    }

    // Grid-stride over numCW
    for (int cw = blockIdx.x; cw < numCW; cw += gridDim.x)
    {
        if(pCwPrmsGpu[cw].exitFlag == 0)
        {
            uint32_t E = pCwPrmsGpu[cw].E;
            int N = min(E,MAX_N);
            uint16_t two_to_K = 1 << pCwPrmsGpu[cw].K;
            // Load LLRs into shmem using MAX_N-stride de-rate-matching
            if (threadIdx.x < MAX_N)
            {
                uint32_t maxNumRepitions = max(div_round_up(E, static_cast<uint32_t>(blockDim.x)), 10);
                T accum = 0;
                for (uint32_t k=threadIdx.x; k<E; k += MAX_N)
                {
//                    T clippedLLR = pCwPrmsGpu[cw].d_LLRs[k];
//                    if(clippedLLR < static_cast<T>(LLR_LOW_LIM))   clippedLLR = static_cast<T>(LLR_LOW_LIM);
//                    if(clippedLLR > static_cast<T>(LLR_HIGH_LIM))  clippedLLR = static_cast<T>(LLR_HIGH_LIM);
//
//                    accum += clippedLLR;
                      accum += (pCwPrmsGpu[cw].d_LLRs[k]/static_cast<T>(maxNumRepitions)); //scale to avoid accumulation overflow
                      if(accum < static_cast<T>(LLR_LOW_LIM))   accum = static_cast<T>(LLR_LOW_LIM);
                      if(accum > static_cast<T>(LLR_HIGH_LIM))  accum = static_cast<T>(LLR_HIGH_LIM);
                }
                shmem_r[threadIdx.x] = accum;
            }
            __syncthreads();

            // Block-stride correlate
            int codeIdx;
            for (codeIdx = threadIdx.x; codeIdx < two_to_K; codeIdx += blockDim.x)
            {
                T m = 0;
                uint32_t c = shmem_c[codeIdx];

                uint32_t n_shift = 1 << (MAX_N-N);
                for (int n=N-1; n>=0; n--)
                {
                    if (c & n_shift)
                    {
                        m -= shmem_r[n];
                    }
                    else
                    {
                        m += shmem_r[n];
                    }
                    n_shift *= 2;
                }
                shmem_m[codeIdx] = m;
            }
            // Zero remaining correlation scores using block-stride
            while (codeIdx < MAX_TWO_TO_K)
            {
                shmem_m[codeIdx] = 0;
                codeIdx += blockDim.x;
            }
            __syncthreads();

            cg::thread_block block = cg::this_thread_block();
            findMaxBlock<T>(shmem_m,shmem_reduce,MAX_TWO_TO_K,block);

            // Find the index of the max metric
            // Note: Race condition if more than one codeword has the same score,
            //       but in that case we would likely make an decoding error anyways.
            for (uint32_t k=threadIdx.x; k<two_to_K; k+=blockDim.x)
            {
                if (shmem_m[k] == shmem_reduce[0])
                {
                    *pCwPrmsGpu[cw].d_cbEst = k;
                    break;
                }
            }
            if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DTX_EN) == CUPHY_DTX_EN)
            {
                /////////////////////////////////////////////////////////////////////////////////
                // estimate DTX in pucchF2_rx_kernel of 5GModel                                //
                /////////////////////////////////////////////////////////////////////////////////
                if (threadIdx.x < MAX_N)
                {
                    float accumFloat = 0;
                    for (uint32_t k=threadIdx.x; k<E; k += MAX_N)
                    {
                        accumFloat += (float)pCwPrmsGpu[cw].d_LLRs[k];
                    }
                    shmem_f[threadIdx.x] = accumFloat;
                }
                __syncthreads();         
                if (threadIdx.x < MAX_N)
                {
                    float accumFloat = 0.0;
                    shmem_f_sumsquare[threadIdx.x] = 0.0;
                    for (uint32_t k=threadIdx.x; k<E; k += MAX_N)
                    {
                        //accum += pow((float)pCwPrmsGpu[cw].d_LLRs[k], 2);
                        accumFloat += ((float)pCwPrmsGpu[cw].d_LLRs[k] * (float)pCwPrmsGpu[cw].d_LLRs[k]);
                    }
                    shmem_f_sumsquare[threadIdx.x] = accumFloat;
                }
                __syncthreads();
                for(uint32_t k=threadIdx.x; k<two_to_K; k += blockDim.x)
                {
                    if(*pCwPrmsGpu[cw].d_cbEst == k)
                    {
                        uint8_t d_DTXEst;
                        if(*pCwPrmsGpu[cw].d_noiseVar<=CUPHY_NOISE_REGULARIZER) 
                        {
                            d_DTXEst = 1;
                            //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                            //printf("rm_decoder: noisevar = %f, d_cbEst = %d, d_DTXEst = %d\n", *pCwPrmsGpu[cw].d_noiseVar, *pCwPrmsGpu[cw].d_cbEst, *pCwPrmsGpu[cw].d_DTXEst);
                        }
                        else
                        {
                            float max_m = 0;
                            uint32_t c = shmem_c[k];
                            uint32_t n_shift = 1 << (MAX_N-N);
                            for (int n=N-1; n>=0; n--)
                            {
                                if (c & n_shift)
                                {
                                    max_m -= shmem_f[n];
                                }
                                else
                                {
                                    max_m += shmem_f[n];
                                }
                                n_shift *= 2;
                            }
                            float sumSquareFloat = 0.0;
                            for(uint32_t idx=0; idx<MAX_N; idx+=1)
                            {  
                                sumSquareFloat += shmem_f_sumsquare[idx];
                            }
                            //float confLevel = pow((float)max_m/(E*1.0), 2)/(sumsquare/(E*1.0));
                            float confLevelFactor = 1.0;
                            uint32_t Qm = pCwPrmsGpu[cw].Qm;
                            if(Qm == 4)
                            {
                                confLevelFactor = 1.5;
                            }
                            else if(Qm == 6)
                            {
                                confLevelFactor = 2.0;
                            }
                            else if(Qm == 8)
                            {
                                confLevelFactor = 3.0;
                            }
                            float confLevelFloat = confLevelFactor*((float)max_m/(E*1.0f))*((float)max_m/(E*1.0f))/(sumSquareFloat/(E*1.0f));
                            float confLevelThrFloat = (max(min(0.8f, CUPHY_DTX_THRESHOLD_ADJ_RM_DECODER*sqrtf(64.0f/E)*sqrt(pCwPrmsGpu[cw].K/4.0f)), 0.1f))*pCwPrmsGpu[cw].DTXthreshold;
                            d_DTXEst = 0;
                            //*pCwPrmsGpu[cw].d_DTXStatus = 4;      //no DTX
                            if(confLevelFloat < confLevelThrFloat)
                            {
                                d_DTXEst = 1;
                                //*pCwPrmsGpu[cw].d_DTXStatus = 3;  //DTX
                            }
                            //printf("rm_decoder: estmax(m) = %f, argmax(m) = %d, noisevar = %f, DTXthreshold = %f, confLevel = %f, confLevelThr = %f, rmDTX = %d\n", max_m, threadIdx.x, *pCwPrmsGpu[cw].d_noiseVar, pCwPrmsGpu[cw].DTXthreshold, confLevelFloat, confLevelThrFloat, *pCwPrmsGpu[cw].d_DTXEst);
                        }
                        if((pCwPrmsGpu[cw].en_DTXest&CUPHY_DET_EN) == CUPHY_DET_EN)
                        {
                            if(d_DTXEst)
                            {
                                *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_DTX;      //DTX
                                if((pCwPrmsGpu[cw].en_DTXest&CUPHY_PUCCH_DET_EN) == CUPHY_PUCCH_DET_EN)
                                {
                                    *pCwPrmsGpu[cw].d_DTXStatus1 = CUPHY_FAPI_DTX;
                                    *pCwPrmsGpu[cw].d_DTXStatus2 = CUPHY_FAPI_DTX;
                                }
                                //printf("rm_decoder: DTXStatus = %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                            }
                            else
                            {
                                *pCwPrmsGpu[cw].d_DTXStatus = CUPHY_FAPI_NO_DTX;      //no DTX
                                if((pCwPrmsGpu[cw].en_DTXest&CUPHY_PUCCH_DET_EN) == CUPHY_PUCCH_DET_EN)
                                {
                                    *pCwPrmsGpu[cw].d_DTXStatus1 = CUPHY_FAPI_NO_DTX;
                                    *pCwPrmsGpu[cw].d_DTXStatus2 = CUPHY_FAPI_NO_DTX;
                                }
                                //printf("rm_decoder: DTXStatus = %d\n", *pCwPrmsGpu[cw].d_DTXStatus);
                            }
                        }
                        //////pCwPrmsGpu[cw].en_DTXest = 0; 
                    }
                }   
                /////////////////////////////////////
                // end DTX estimation              //
                /////////////////////////////////////
            }
        }
    }
}



rmDecoder::rmDecoder(const cuphy_i::context& ctx) :
    deviceIndex_(ctx.index()),
    cc_(ctx.compute_cap()),
    sharedMemPerBlockOptin_(ctx.max_shmem_per_block_optin()),
    multiProcessorCount_(ctx.sm_count())
{
    pDstAddr_ = nullptr;
    pSrcAddrs_ = nullptr;
    mode_ = 0;
    numCW_ = 0;
    CW_PER_BLOCK_ = 4;
    d_E_ = nullptr;
    d_two_to_K_ = nullptr;
    d_codebook_f_ = nullptr;
    d_codebook_u32_ = nullptr;
}

void rmDecoder::init(void* pMemoryFootprint)
{
    // if (mode < 0 || mode > 3)
    // {
    // }

    // if (mode == 0 || mode == 1 || mode == 2)
    // {
    //     CUDA_CHECK(cudaMalloc(&d_codebook_f_, sizeof(float)*MAX_TWO_TO_K*MAX_N));
    //     if (mode == 0)
    //     {
    //         CUDA_CHECK(cudaMemcpy(d_codebook_f_, rm_codebook_0, sizeof(float)*MAX_TWO_TO_K*MAX_N, cudaMemcpyHostToDevice));
    //     }
    //     else
    //     {
    //         CUDA_CHECK(cudaMemcpy(d_codebook_f_, rm_codebook_1, sizeof(float)*MAX_TWO_TO_K*MAX_N, cudaMemcpyHostToDevice));
    //     }
    // }
    // else if (mode == 3)
    // {
        //TODO ideally the cudaMalloc should be swapped with a make_unique_device or some other helper, so footprint tracking can
        // happen automatically. Note this is the only component where memory allocation happens this way and not in the channel.
        CUDA_CHECK(cudaMalloc(&d_codebook_u32_, sizeof(uint32_t)*MAX_TWO_TO_K));
        if(pMemoryFootprint)
        {
           reinterpret_cast<cuphyMemoryFootprint*>(pMemoryFootprint)->addGpuAllocation(sizeof(uint32_t)*MAX_TWO_TO_K);
        }
        CUDA_CHECK(cudaMemcpy(d_codebook_u32_, rm_codebook_3, sizeof(uint32_t)*MAX_TWO_TO_K, cudaMemcpyHostToDevice));
    // }
    // else
    // {
    //     return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    // }

    // mode_ = mode;
    // return CUPHY_STATUS_SUCCESS;
}

// cuphyStatus_t rmDecoder::setup(void*                         pDstAddr,
//                                     const void* const*            pSrcAddrs,
//                                     cuphyDataType_t               srcType,
//                                     int                           numCW,
//                                     int                           CW_PER_BLOCK,
//                                     int                           *d_E,
//                                     int                           *d_two_to_K)
// {
//     // Copy input parameters to member variables
//     pDstAddr_ = pDstAddr;
//     pSrcAddrs_ = pSrcAddrs;
//     srcType_ = srcType;
//     numCW_ = numCW;
//     CW_PER_BLOCK_ = CW_PER_BLOCK;
//     d_E_ = d_E;
//     d_two_to_K_ = d_two_to_K;

//     return CUPHY_STATUS_SUCCESS;
// }

    void rmDecoder::setup(  uint16_t                   nCws,
                            cuphyRmCwPrm_t*            pCwPrmsGpu,
                            bool                       enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                            rmDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
                            void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                            cuphyRmDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                            cudaStream_t               strm)                        // stream to perform copy
{
    // populate dynamic descriptor:
    pCpuDynDesc->nCws        = nCws;
    pCpuDynDesc->pCwPrmsGpu  = pCwPrmsGpu;
    pCpuDynDesc->pRmCodebook = d_codebook_u32_;

    // save pointer to GPU descriptor
    rmDecoderKernelArgs_t& kernelArgs = m_kernelArgs;
    kernelArgs.pDynDescr              = reinterpret_cast<rmDecoderDynDescr_t*>(pGpuDynDesc);

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, sizeof(rmDecoderDynDescr_t), cudaMemcpyHostToDevice, strm);
    }

    // select kernel (includes launch geometry). Populate launchCfg.
    kernelSelect(nCws, pLaunchCfg);
    pLaunchCfg->kernelArgs[0]                       = &m_kernelArgs.pDynDescr;
    pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
}



rmDecoder::~rmDecoder()
{
    if (d_codebook_f_)   cudaFree(d_codebook_f_); // TODO allocation commented out; d_codebook_f_ always false.
    if (d_codebook_u32_) cudaFree(d_codebook_u32_);
}

void rmDecoder::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    dynDescrSizeBytes  = sizeof(rmDecoderDynDescr_t);
    dynDescrAlignBytes = alignof(rmDecoderDynDescr_t);
}

void rmDecoder::kernelSelect(uint16_t nCws, cuphyRmDecoderLaunchCfg_t* pLaunchCfg)
{
   // launch geometry
   uint16_t bpk = (nCws + CW_PER_BLOCK_ - 1) / CW_PER_BLOCK_;
   uint16_t tpb = RM_DECODER_TPB;

   dim3 gridDim(bpk);
   dim3 blockDim(tpb);

    // kernel (only one kernel option for now)
    void* kernelFunc = reinterpret_cast<void*>(rm_decoder_3<__half>);
    CUDA_CHECK(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));

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
