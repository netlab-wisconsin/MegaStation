/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ldpc2_split_cluster.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_half_96KB()
cuphyStatus_t decode_ldpc2_split_cluster_half_96KB(ldpc::decoder&            dec,
                                                   const LDPC_config&        cfg,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if(cfg.BG == 1)
    {
        // Shared memory requirements, assuming 2 bytes APP, 8 bytes cC2V:
        // SHMEM = Z * [(Kb + mb) * sizeof(APP) + mb * sizeof(cC2V)]
        // If Z = 128:
        // SHMEM(Z = 128) = 128*[(22 + 46)*2 + 46*8] = 63 * 1024
        // With a minimum of 64 KB shared memory, all APP and cC2V data
        // should fit in shared memory for Z <= 128.
        switch(cfg.Z)
        {
        //case 64:  s = decode_ldpc2_split_half_96KB_BG1_Z64 (dec, cfg, params, grdDim, blkDim, strm); break;
        //case 96:  s = decode_ldpc2_split_half_96KB_BG1_Z96 (dec, cfg, params, grdDim, blkDim, strm); break;
        //case 128: s = decode_ldpc2_split_half_96KB_BG1_Z128(dec, cfg, params, grdDim, blkDim, strm); break;
#if 0
        case 160: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z160(dec, cfg, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z192(dec, cfg, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z224(dec, cfg, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z256(dec, cfg, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z288(dec, cfg, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z320(dec, cfg, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z352(dec, cfg, params, grdDim, blkDim, strm); break;
#endif
        case 384: s = decode_ldpc2_split_cluster_half_96KB_BG1_Z384(dec, cfg, params, grdDim, blkDim, strm); break;
        default:                                                                                        break;
        }
    }
    else if(cfg.BG == 2)
    {
#if 0
        switch(cfg.Z)
        {
        case 64:  s = decode_ldpc2_split_half_96KB_BG2_Z64 (dec, cfg, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_split_half_96KB_BG2_Z96 (dec, cfg, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_split_half_96KB_BG2_Z128(dec, cfg, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_split_half_96KB_BG2_Z160(dec, cfg, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_split_half_96KB_BG2_Z192(dec, cfg, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_split_half_96KB_BG2_Z224(dec, cfg, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_split_half_96KB_BG2_Z256(dec, cfg, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_split_half_96KB_BG2_Z288(dec, cfg, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_split_half_96KB_BG2_Z320(dec, cfg, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_split_half_96KB_BG2_Z352(dec, cfg, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_split_half_96KB_BG2_Z384(dec, cfg, params, grdDim, blkDim, strm); break;
        default:                                                                                break;
        }
#endif
    }
    return s;
}

} // namespace ldpc2
