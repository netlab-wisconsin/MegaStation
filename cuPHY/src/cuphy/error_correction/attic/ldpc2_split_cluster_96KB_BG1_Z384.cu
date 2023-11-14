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

#include "ldpc2_split_cluster.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_half_96KB_BG1_Z384()
cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z384(ldpc::decoder&            dec,
                                                            const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            cudaStream_t              strm)
{
    constexpr int  BG        = 1;
    constexpr int  Z         = 384;
    constexpr int  Kb        = 22;
    
    typedef __half                                                           T;
    //typedef sign_store_policy_split_dst<__half, split_sign_update_bit_ops> sign_mgr_t;
    typedef sign_store_policy_split_src<__half, split_sign_update_bit_ops>   sign_mgr_t;
    typedef cC2V_index<__half, BG, sign_mgr_t, min_sum_update_half_0>        cC2V_t;

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    switch(cfg.mb)
    {
    case 22: s = launch_split_cluster<T, BG, Kb, Z, 21, 22, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
#if 0
    case 23:
    case 24:
    case 25:
    case 26:
        s = launch_split_cluster<T, BG, Kb, Z, 20, cC2V_t, app_loc_address>(dec, params, grdDim, blkDim, strm);
        break;
    case 27:
    case 28:
    case 29:
    case 30:
        s = launch_split_cluster<T, BG, Kb, Z, 19, cC2V_t, app_loc_address>(dec, params, grdDim, blkDim, strm);
        break;
    case 31:
    case 32:
    case 33:
    case 34:
        s = launch_split_cluster<T, BG, Kb, Z, 18, cC2V_t, app_loc_address>(dec, params, grdDim, blkDim, strm);
        break;
    case 35:
    case 36:
    case 37:
    case 38:
        s = launch_split_cluster<T, BG, Kb, Z, 17, cC2V_t, app_loc_address>(dec, params, grdDim, blkDim, strm);
        break;
    case 39:
    case 40:
    case 41:
    case 42:
        s = launch_split_cluster<T, BG, Kb, Z, 16, cC2V_t, app_loc_address>(dec, params, grdDim, blkDim, strm);
        break;
    case 43:
    case 44:
    case 45:
#endif
    case 32: s = launch_split_cluster<T, BG, Kb, Z, 18, 32, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 33: s = launch_split_cluster<T, BG, Kb, Z, 18, 33, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 34: s = launch_split_cluster<T, BG, Kb, Z, 18, 34, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 35: s = launch_split_cluster<T, BG, Kb, Z, 17, 35, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 36: s = launch_split_cluster<T, BG, Kb, Z, 17, 36, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 37: s = launch_split_cluster<T, BG, Kb, Z, 17, 37, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 38: s = launch_split_cluster<T, BG, Kb, Z, 17, 38, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 39: s = launch_split_cluster<T, BG, Kb, Z, 16, 39, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 40: s = launch_split_cluster<T, BG, Kb, Z, 16, 40, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 41: s = launch_split_cluster<T, BG, Kb, Z, 16, 41, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 42: s = launch_split_cluster<T, BG, Kb, Z, 16, 42, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 43: s = launch_split_cluster<T, BG, Kb, Z, 15, 43, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 44: s = launch_split_cluster<T, BG, Kb, Z, 15, 44, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 45: s = launch_split_cluster<T, BG, Kb, Z, 15, 45, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 46: s = launch_split_cluster<T, BG, Kb, Z, 15, 46, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    default:                                                                                                                break;
    }
    return s;
}

} // namespace ldpc2