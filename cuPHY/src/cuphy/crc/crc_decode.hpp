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

struct puschRxCrcDecodeDescr
{
    uint32_t*          pOutputCBCRCs;
    uint8_t*           pOutputTBs;
    const uint32_t*    pInputCodeBlocks;
    uint32_t*          pOutputTBCRCs;
    const PerTbParams* pTbPrmsArray;
    bool               reverseBytes;
    uint16_t           schUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
};
typedef struct puschRxCrcDecodeDescr puschRxCrcDecodeDescr_t;

class puschRxCrcDecode : public cuphyPuschRxCrcDecode {
public:
    puschRxCrcDecode()                                   = default;
    ~puschRxCrcDecode()                                  = default;
    puschRxCrcDecode(puschRxCrcDecode const&)            = delete;
    puschRxCrcDecode& operator=(puschRxCrcDecode const&) = delete;

    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);

    void init(int reverseBytes);

    void setup(uint16_t                          nSchUes,
               uint16_t*                         pSchUserIdxsCpu,
               uint32_t*                         pOutputCBCRCs,
               uint8_t*                          pOutputTBs,
               const uint32_t*                   pInputCodeBlocks,
               uint32_t*                         pOutputTBCRCs,
               const PerTbParams*                pTbPrmsCpu,
               const PerTbParams*                pTbPrmsGpu,
               void*                             pCpuDesc,                     
               void*                             pGpuDesc,                     
               uint8_t                           enableCpuToGpuDescrAsyncCpy, 
               cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
               cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
               cudaStream_t                      strm);                        

private:
    // class state modifed by setup saved in data member
    bool       m_reverseBytes;  // option to reverse order of bytes in each word before computing the CRC
    CUfunction m_cbCrcKernelFunc;
    CUfunction m_tbCrcKernelFunc;
};