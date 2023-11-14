/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <functional>
#include "cuphy.h"

struct puschRxRateMatchDescr
{
    const void*        llr_vec_in[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED]; // Rm input LLRs    //ToDo change to void**
    uint16_t           schUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
    void**             out;        // Rm output LLRs
    const PerTbParams* tbPrmsArray;
    int                descramblingOn;

};
typedef struct puschRxRateMatchDescr puschRxRateMatchDescr_t;


typedef struct _puschRxRateMatchLaunchGeo
{
    dim3      gridDim;
    dim3      blockDim;
    uint32_t  shMemBytes;
}puschRxRateMatchLaunchGeo_t;


struct puschRxRateMatchLaunchPrms;
using puschRxRateMatchKernelLauncher_t = std::function<void(puschRxRateMatchLaunchPrms&, cudaStream_t&)>;


struct puschRxRateMatchLaunchPrms
{
    puschRxRateMatchKernelLauncher_t launcher;
    puschRxRateMatchDescr_t*         args;
    puschRxRateMatchLaunchGeo_t      geo;

    // Graph
    void* kernelArgs;
    void* kernelFunc;
};
typedef struct puschRxRateMatchLaunchPrms puschRxRateMatchLaunchPrms_t;


class puschRxRateMatch : public cuphyPuschRxRateMatch {
public:
    puschRxRateMatch()                                   = default;
    ~puschRxRateMatch()                                  = default;
    puschRxRateMatch(puschRxRateMatch const&)            = delete;
    puschRxRateMatch& operator=(puschRxRateMatch const&) = delete;

    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);

    void init(int   rmFPconfig,          // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
              int   descramblingOn);   // enable/disable descrambling


    void setup( uint16_t                          nSchUes,                      // number of users with sch data
                uint16_t*                         pSchUserIdxsCpu,              // indicies of users with SCH data
                const PerTbParams*                pTbPrmsCpu,                   // starting adress of transport block paramters (CPU)
                const PerTbParams*                pTbPrmsGpu,                   // starting adress of transport block paramters (GPU)
                cuphyTensorPrm_t*                 pTPrmRmIn,                    // starting adress of input LLR tensor parameters
                cuphyTensorPrm_t*                 pTPrmCdm1RmIn, 
                void**                            ppRmOut,                      // array of rm outputs (GPU)
                void*                             pCpuDesc,                     // pointer to descriptor in cpu
                void*                             pGpuDesc,                     // pointer to descriptor in gpu
                uint8_t                           enableCpuToGpuDescrAsyncCpy,  // option to copy cpu descriptors from cpu to gpu
                cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,                   // pointer to rate matching launch configuration
                cudaStream_t                      strm);                        // stream to perform copy

private:
    // class state modifed by setup saved in data member.
    CUfunction m_kernelFunc;
    int        m_descramblingOn;
};
