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

#include "ldpc2_split_dynamic.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_dynamic_half_96KB_BG1_Z352()
cuphyStatus_t decode_ldpc2_split_dynamic_half_96KB_BG1_Z352(const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    constexpr int  BG        = 1;
    constexpr int  Z         = 352;
    constexpr int  Kb        = 22;
    
    typedef __half                                                                    T;
    typedef cC2V_index<__half, BG, sign_store_policy_src, sign_store_policy_split_src> cC2V_t;
    //typedef cC2V_index<__half, BG, sign_store_policy_dst, sign_store_policy_split_src> cC2V_t;
   
    switch(cfg.mb)
    {
    case 24:
    case 25:
        s = launch_split_dynamic<T, BG, Kb, Z, 23, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 26:
    case 27:
    case 28:
    case 29:
        s = launch_split_dynamic<T, BG, Kb, Z, 22, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 30:
    case 31:
    case 32:
    case 33:
        s = launch_split_dynamic<T, BG, Kb, Z, 21, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 34:
    case 35:
    case 36:
    case 37:
        s = launch_split_dynamic<T, BG, Kb, Z, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 38:
    case 39:
    case 40:
    case 41:
        s = launch_split_dynamic<T, BG, Kb, Z, 19, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 42:
    case 43:
    case 44:
    case 45:
        s = launch_split_dynamic<T, BG, Kb, Z, 18, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 46:
        s = launch_split_dynamic<T, BG, Kb, Z, 17, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    default:
        break;
    }
#endif // if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    return s;
}

} // namespace ldpc2