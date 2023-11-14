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

#include "ldpc2_reg.cuh"
#include "ldpc2_sign.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_address_float_BG1_Z208()
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z208(ldpc::decoder&            dec,
                                                      const LDPC_config&        cfg,
                                                      const LDPC_kernel_params& params,
                                                      const dim3&               grdDim,
                                                      const dim3&               blkDim,
                                                      cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 2 && CUPHY_LDPC_INCLUDE_ALL_LIFTING
    constexpr int  BG = 1;
    constexpr int  Z  = 208;
    constexpr int  Kb = 22;

    typedef float                                                    T;
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Sign manager type
    typedef sign_store_policy_dst<float, sign_order_be<float>> sign_mgr_dst_be_t;
    typedef sign_store_policy_dst<float, sign_order_le<float>> sign_mgr_dst_le_t;
    typedef sign_store_policy_src<float, sign_order_be<float>> sign_mgr_src_be_t;
    typedef sign_store_policy_src<float, sign_order_le<float>> sign_mgr_src_le_t;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Check to Variable (C2V) message type
    // USE THIS for min1 delta
    typedef cC2V_address<T, BG, min1_policy_delta, sign_mgr_dst_be_t> cC2V_t;

    // USE THIS for min1 default
    //typedef cC2V_address<T, BG, min1_policy_default, sign_mgr_dst_be_t> cC2V_t;

    switch(cfg.mb)
    {
    case  4:  s = launch_register_kernel<T, BG, Kb, Z,  4, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  5:  s = launch_register_kernel<T, BG, Kb, Z,  5, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  6:  s = launch_register_kernel<T, BG, Kb, Z,  6, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  7:  s = launch_register_kernel<T, BG, Kb, Z,  7, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  8:  s = launch_register_kernel<T, BG, Kb, Z,  8, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case  9:  s = launch_register_kernel<T, BG, Kb, Z,  9, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 10:  s = launch_register_kernel<T, BG, Kb, Z, 10, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 11:  s = launch_register_kernel<T, BG, Kb, Z, 11, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    case 12:  s = launch_register_kernel<T, BG, Kb, Z, 12, cC2V_t, app_loc_address, 1>(dec, params, grdDim, blkDim, strm); break;
    default:                                                                                                          break;
    }
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 2 && CUPHY_LDPC_INCLUDE_ALL_LIFTING
    return s;
}

} // namespace ldpc2
