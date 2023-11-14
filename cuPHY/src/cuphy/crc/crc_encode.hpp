/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


struct crcEncodeDescr
{
    uint32_t*     d_cbCRCs;
    uint32_t*     d_tbCRCs;
    const uint32_t* d_inputTransportBlocks;
    uint8_t*           d_codeBlocks;
    const PdschPerTbParams* d_tbPrmsArray;
    bool            reverseBytes;
    bool            codeBlocksOnly;
};
typedef struct crcEncodeDescr crcEncodeDescr_t;

struct prepareCrcEncodeDescr
{
    uint32_t                offset[PDSCH_MAX_UES_PER_CELL_GROUP]; //FIXME keep first field; used only for TBs in testing mode (TM)
    const uint32_t*         d_inputOrigTBs;
    uint32_t*               d_inputTBs;
    uint32_t*               d_inputTBsTM;
    const PdschPerTbParams* d_tbPrmsArray;
};
typedef struct prepareCrcEncodeDescr prepareCrcEncodeDescr_t;
