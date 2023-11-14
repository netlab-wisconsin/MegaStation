/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"

// Implementation of polSegDeRmDeItl interface exposed as an opaque data type to abstract out implementation
// details (polSegDeRmDeItl  C++ class). polSegDeRmDeItl is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyPolSegDeRmDeItl
{};

struct polSegDeRmDeItlDynDescr
{
    __half**                     pUciSegLLRsAddrs;
    __half**                     pCwLLRsAddrs;
    const cuphyPolarUciSegPrm_t* pPolarUciSegPrms;
    const cuphyPolarCwPrm_t*     pPolarCwPrms;
};
typedef struct polSegDeRmDeItlDynDescr polSegDeRmDeItlDynDescr_t;

// polSegDeRmDeItl kernel arguments (supplied via descriptors)
typedef struct
{
    polSegDeRmDeItlDynDescr_t* pDynDescr;
} polSegDeRmDeItlKernelArgs_t;

class polSegDeRmDeItl : public cuphyPolSegDeRmDeItl {
public:
    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(uint16_t                         nPolUciSegs,                 // number of polar UCI segments
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
               cudaStream_t                     strm);                                           // stream to perform copy

    void kernelSelect(uint16_t                         nPolUciSegs,
                      const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                      cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    polSegDeRmDeItlKernelArgs_t m_kernelArgs;
};