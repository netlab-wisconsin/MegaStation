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
#include "ldpc2_sign_split.cuh"
#include "ldpc2_sign.cuh"
#include "ldpc2_min_sum_update_half_0.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_half_BG2_Z384()
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z384(ldpc::decoder&             dec,
                                                   const LDPC_config&        cfg,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    constexpr int  BG = 2;
    constexpr int  Z  = 384;
    constexpr int  Kb = 10;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half                                                           T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_split_dst<__half, split_sign_update_bit_ops>   sign_mgr_t;
    //typedef sign_store_policy_split_src<__half, split_sign_update_bit_ops> sign_mgr_t;
    typedef cC2V_index<__half, BG, sign_mgr_t, min_sum_update_half_0>        cC2V_t;

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
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_float_BG2_Z384()
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z384(ldpc::decoder&            dec,
                                                    const LDPC_config&        cfg,
                                                    const LDPC_kernel_params& params,
                                                    const dim3&               grdDim,
                                                    const dim3&               blkDim,
                                                    cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    constexpr int  BG = 2;
    constexpr int  Z  = 384;
    constexpr int  Kb = 10;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef float                                                T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_dst<float, sign_order_be<float>>   sign_mgr_t;
    //typedef sign_store_policy_dst<float, sign_order_le<float>> sign_mgr_t;
    typedef cC2V_index<float, BG, sign_mgr_t, unused>            cC2V_t;

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
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1
    return s;
}

} // namespace ldpc2
