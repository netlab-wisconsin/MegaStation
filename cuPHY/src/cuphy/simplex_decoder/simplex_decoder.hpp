/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cuda.h>
#include "cuphy.h"

const int SIMPLEX_DECODER_MAX_E = 1536;

struct cuphySimplexDecoder
{
};

struct simplexDecoderDynDescr
{
    cuphySimplexCwPrm_t* pCwPrmsGpu;         
    uint16_t             nCws;
};
typedef struct simplexDecoderDynDescr simplexDecoderDynDescr_t;

// simplexDecoder kernel arguments (supplied via descriptors)
typedef struct
{
    simplexDecoderDynDescr_t* pDynDescr;
} simplexDecoderKernelArgs_t;

class SimplexDecoder : public cuphySimplexDecoder
{
public:
    SimplexDecoder();

    void setup(uint16_t                        nCws,
               cuphySimplexCwPrm_t*            pCwPrmsCpu, 
               cuphySimplexCwPrm_t*            pCwPrmsGpu, 
               bool                            enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
               simplexDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
               void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
               cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
               cudaStream_t                    strm);                       // stream to perform copy

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

private:
    uint8_t  CW_PER_BLOCK_;


    simplexDecoderKernelArgs_t m_kernelArgs;
    void kernelSelect(uint16_t  nCws, cuphySimplexDecoderLaunchCfg_t* pLaunchCfg);
};