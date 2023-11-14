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

#include "ldpc2_shared_cluster.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_half_BG1_Z384()
cuphyStatus_t decode_ldpc2_shared_cluster_index_half_BG1_Z384(const LDPC_config&        cfg,
                                                              const LDPC_kernel_params& params,
                                                              const dim3&               grdDim,
                                                              const dim3&               blkDim,
                                                              cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 2
    constexpr int  BG = 1;
    constexpr int  Z  = 384;
    constexpr int  Kb = 22;
    
    typedef __half                                                                    T;
    typedef cC2V_index<__half, BG, sign_store_policy_src, sign_store_policy_split_src> cC2V_t;
    //typedef cC2V_index<__half, BG, sign_store_policy_dst, sign_store_policy_split_src> cC2V_t;

    // Shared memory requirements, assuming 2 bytes APP, 8 bytes cC2V:
    // SHMEM = Z * [(Kb + mb) * sizeof(APP) + mb * sizeof(cC2V)]
    // If Z = 384 and mb = 21:
    // SHMEM(Z = 384) = 384*[(22 + 21)*2 + 21*8] = 95.25 * 1024
    // With a maximum of 96 KB shared memory (Volta), all APP and cC2V data
    // can fit in shared memory up to 21 parity nodes
    
    switch(cfg.mb)
    {
    //case 4:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    //case 5:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    //case 6:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    //case 7:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    //case 8:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    //case 9:  s = launch_all_shared_cluster<__half, BG, Kb, Z,  9, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 10: s = launch_all_shared_cluster<__half, BG, Kb, Z, 10, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 11: s = launch_all_shared_cluster<__half, BG, Kb, Z, 11, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 12: s = launch_all_shared_cluster<__half, BG, Kb, Z, 12, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 13: s = launch_all_shared_cluster<__half, BG, Kb, Z, 13, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 14: s = launch_all_shared_cluster<__half, BG, Kb, Z, 14, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 15: s = launch_all_shared_cluster<__half, BG, Kb, Z, 15, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 16: s = launch_all_shared_cluster<__half, BG, Kb, Z, 16, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 17: s = launch_all_shared_cluster<__half, BG, Kb, Z, 17, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 18: s = launch_all_shared_cluster<__half, BG, Kb, Z, 18, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;            
    case 19: s = launch_all_shared_cluster<__half, BG, Kb, Z, 19, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    case 20: s = launch_all_shared_cluster<__half, BG, Kb, Z, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;            
    case 21: s = launch_all_shared_cluster<__half, BG, Kb, Z, 21, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
    default: break;
    }
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 2
    return s;
}

} // namespace ldpc2
