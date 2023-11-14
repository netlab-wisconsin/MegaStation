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
#include "ldpc2_app_address_fp.cuh"
#include "ldpc2_c2v_x2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_half_BG2_Z60()
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z60(ldpc::decoder&            dec,
                                                         const LDPC_config&        cfg,
                                                         const LDPC_kernel_params& params,
                                                         const dim3&               grdDim,
                                                         const dim3&               blkDim,
                                                         cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

#if CUPHY_LDPC_INCLUDE_LEVEL >= 1 && CUPHY_LDPC_INCLUDE_ALL_LIFTING
    constexpr int  BG = 2;
    constexpr int  Z  = 60;
    constexpr int  Kb = 9;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half2 T;
    // Check to Variable (C2V) message type
    typedef cC2V_index<T, BG, sign_mgr_pair_src<__half2>, unused> cC2V_t;

    switch(cfg.mb)
    {
    case   4:  s = launch_register_kernel<T, BG, Kb, Z,  4, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   5:  s = launch_register_kernel<T, BG, Kb, Z,  5, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   6:  s = launch_register_kernel<T, BG, Kb, Z,  6, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   7:  s = launch_register_kernel<T, BG, Kb, Z,  7, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   8:  s = launch_register_kernel<T, BG, Kb, Z,  8, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case   9:  s = launch_register_kernel<T, BG, Kb, Z,  9, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  10:  s = launch_register_kernel<T, BG, Kb, Z, 10, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  11:  s = launch_register_kernel<T, BG, Kb, Z, 11, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  12:  s = launch_register_kernel<T, BG, Kb, Z, 12, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  13:  s = launch_register_kernel<T, BG, Kb, Z, 13, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  14:  s = launch_register_kernel<T, BG, Kb, Z, 14, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  15:  s = launch_register_kernel<T, BG, Kb, Z, 15, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  16:  s = launch_register_kernel<T, BG, Kb, Z, 16, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  17:  s = launch_register_kernel<T, BG, Kb, Z, 17, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  18:  s = launch_register_kernel<T, BG, Kb, Z, 18, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  19:  s = launch_register_kernel<T, BG, Kb, Z, 19, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    case  20:  s = launch_register_kernel<T, BG, Kb, Z, 20, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
    default:                                                                                                                  break;
    }
#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1 && CUPHY_LDPC_INCLUDE_ALL_LIFTING
    return s;
}

} // namespace ldpc2
