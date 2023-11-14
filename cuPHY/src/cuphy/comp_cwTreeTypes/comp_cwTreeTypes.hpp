/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "../cuphy.h"

// Implementation of CompCwTreeTypes interface exposed as an opaque data type to abstract out implementation
// details (compCwTreeTypes  C++ class). compCwTreeTypes is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyCompCwTreeTypes
{};

struct compCwTypesDynDescr
{
    uint8_t**                    pCwTreeTypesAddrs;
    const cuphyPolarUciSegPrm_t* pPolarUciSegPrms;
};
typedef struct compCwTypesDynDescr compCwTreeTypesDynDescr_t;

// compCwTypes kernel arguments (supplied via descriptors)
typedef struct
{
    compCwTreeTypesDynDescr_t* pDynDescr;
} compCwTreeTypesKernelArgs_t;

// Class implementation of compCwTreeTypes
class compCwTreeTypes : public cuphyCompCwTreeTypes {
public:
    compCwTreeTypes()                       = default;
    ~compCwTreeTypes()                      = default;
    compCwTreeTypes(compCwTreeTypes const&) = delete;
    compCwTreeTypes& operator=(compCwTreeTypes const&) = delete;

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(uint16_t                         nPolUciSegs,                 // number of polar UCI segments
               const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,           // starting adreass of polar UCI segment parameters (CPU)
               const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,           // starting adreass of polar UCI segment parameters (GPU)
               uint8_t**                        pCwTreeTypesAddrs,           // pointer to cwTreeTypes addresses
               compCwTreeTypesDynDescr_t*       pCpuDynDesc,                 // pointer to dynamic descriptor in cpu
               void*                            pGpuDynDesc,                 // pointer to dynamic descriptor in gpu
               uint8_t                          enableCpuToGpuDescrAsyncCpy, // option to copy cpu descriptors from cpu to gpu
               cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,                  // pointer to rate matching launch configuration
               cudaStream_t                     strm);                                           // stream to perform copy

    void kernelSelect(uint16_t                         nPolUciSegs,
                      const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                      cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    compCwTreeTypesKernelArgs_t m_kernelArgs;
};
