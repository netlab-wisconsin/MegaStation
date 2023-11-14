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

#include "ldpc2_c2v_x2.cuh"
#include "ldpc2_app_address_fp.cuh"
#include "ldpc2_shared.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_half_BG1_Z384()
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z384(ldpc::decoder&            dec,
                                                            const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            cudaStream_t              strm)
{
    cuphyStatus_t  s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    constexpr int  BG = 1;
    constexpr int  Z  = 384;
    constexpr int  Kb = 22;
    
    typedef __half2 T;
    typedef cC2V_index<T, BG, sign_mgr_pair_src<__half2>, unused> cC2V_t;

    // Maximum parity node based on 96 KiB shared mem configuration
    
    switch(cfg.mb)
    {
    case   4:  s = launch_all_shared_strided<T, BG, Kb, Z,  4, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   5:  s = launch_all_shared_strided<T, BG, Kb, Z,  5, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   6:  s = launch_all_shared_strided<T, BG, Kb, Z,  6, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   7:  s = launch_all_shared_strided<T, BG, Kb, Z,  7, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   8:  s = launch_all_shared_strided<T, BG, Kb, Z,  8, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   9:  s = launch_all_shared_strided<T, BG, Kb, Z,  9, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    default:                                                                                                                  break;
    }
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    return s;
}

} // namespace ldpc2
