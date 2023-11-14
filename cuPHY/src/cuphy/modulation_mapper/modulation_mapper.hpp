/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "tensor_desc.hpp"

struct modulationDescr
{
    PdschDmrsParams* d_params;
    const uint32_t* modulation_input;
    const PdschPerTbParams* workspace;
    __half2* modulation_output;
    int max_bits_per_layer;
};
typedef struct modulationDescr modulationDescr_t;

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// symbol_modulate()
cuphyStatus_t symbol_modulate(const tensor_desc& tSym,
                              void*              pSym,
                              const tensor_desc& tBits,
                              const void*        pBits,
                              int                log2_QAM,
                              cudaStream_t       strm);
  
} // namespace cuphy_i
