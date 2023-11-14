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
// decode_ldpc2_split_dynamic_half_96KB_BG1_Z256()
cuphyStatus_t decode_ldpc2_split_dynamic_half_96KB_BG1_Z256(const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
#if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    constexpr int  BG        = 1;
    constexpr int  Z         = 256;
    constexpr int  Kb        = 22;
    
    typedef __half                                                                    T;
    typedef cC2V_index<__half, BG, sign_store_policy_src, sign_store_policy_split_src> cC2V_t;
    //typedef cC2V_index<__half, BG, sign_store_policy_dst, sign_store_policy_split_src> cC2V_t;
    

    switch(cfg.mb)
    {
    case 35:
    case 36:
    case 37:
    case 38:
        s = launch_split_dynamic<T, BG, Kb, Z, 33, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 39:
    case 40:
    case 41:
    case 42:
        s = launch_split_dynamic<T, BG, Kb, Z, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    case 43:
    case 44:
    case 45:
    case 46:
        s = launch_split_dynamic<T, BG, Kb, Z, 31, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
        break;
    default:
        break;
    }
#endif // if CUPHY_LDPC_INCLUDE_ALL_ALGOS
    return s;
}

} // namespace ldpc2
