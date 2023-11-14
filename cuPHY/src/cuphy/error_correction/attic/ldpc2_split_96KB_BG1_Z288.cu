/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include "ldpc2_split.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_half_96KB_BG1_Z288()
cuphyStatus_t decode_ldpc2_split_half_96KB_BG1_Z288(const LDPC_config&        cfg,
                                                    const LDPC_kernel_params& params,
                                                    const dim3&               grdDim,
                                                    const dim3&               blkDim,
                                                    cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    switch(cfg.mb)
    {
    default: break;
    }
    return s;
}

} // namespace ldpc2
