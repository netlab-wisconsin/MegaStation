#!/usr/bin/env python
# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cu_file_bg1="""/*
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
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_half_BG1_Z{Z}()
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z{Z}(ldpc::decoder&            dec,
                                                   const LDPC_config&        cfg,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
{half_guard_begin}    constexpr int  BG = 1;
    constexpr int  Z  = {Z};
    constexpr int  Kb = 22;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half                                                           T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_split_dst<__half, split_sign_update_bit_ops>   sign_mgr_t;
    //typedef sign_store_policy_split_src<__half, split_sign_update_bit_ops> sign_mgr_t;
    typedef cC2V_index<__half, BG, sign_mgr_t, min_sum_update_half_0>        cC2V_t;

    switch(cfg.mb)
    {{
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
    }}
{half_guard_end}    return s;
}}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_float_BG1_Z{Z}()
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z{Z}(ldpc::decoder&            dec,
                                                   const LDPC_config&        cfg,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
{float_guard_begin}    constexpr int  BG = 1;
    constexpr int  Z  = {Z};
    constexpr int  Kb = 22;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef float                                                    T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_dst<float, sign_order_be<float>>   sign_mgr_t;
    //typedef sign_store_policy_dst<float, sign_order_le<float>> sign_mgr_t;
    typedef cC2V_index<float, BG, sign_mgr_t, unused>            cC2V_t;

    switch(cfg.mb)
    {{
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
    }}
{float_guard_end}    return s;
}}

}} // namespace ldpc2
"""

cu_file_bg2="""/*
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
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_half_BG2_Z{Z}()
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z{Z}(ldpc::decoder&             dec,
                                                   const LDPC_config&        cfg,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
{half_guard_begin}    constexpr int  BG = 2;
    constexpr int  Z  = {Z};
    constexpr int  Kb = {Kb};

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half                                                           T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_split_dst<__half, split_sign_update_bit_ops>   sign_mgr_t;
    //typedef sign_store_policy_split_src<__half, split_sign_update_bit_ops> sign_mgr_t;
    typedef cC2V_index<__half, BG, sign_mgr_t, min_sum_update_half_0>        cC2V_t;

    switch(cfg.mb)
    {{
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
    }}
{half_guard_end}    return s;
}}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_float_BG2_Z{Z}()
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z{Z}(ldpc::decoder&            dec,
                                                    const LDPC_config&        cfg,
                                                    const LDPC_kernel_params& params,
                                                    const dim3&               grdDim,
                                                    const dim3&               blkDim,
                                                    cudaStream_t              strm)
{{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
{float_guard_begin}    constexpr int  BG = 2;
    constexpr int  Z  = {Z};
    constexpr int  Kb = {Kb};

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef float                                                T;
    // Check to Variable (C2V) message type
    typedef sign_store_policy_dst<float, sign_order_be<float>>   sign_mgr_t;
    //typedef sign_store_policy_dst<float, sign_order_le<float>> sign_mgr_t;
    typedef cC2V_index<float, BG, sign_mgr_t, unused>            cC2V_t;

    switch(cfg.mb)
    {{
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
    }}
{float_guard_end}    return s;
}}

}} // namespace ldpc2
"""

Z_full = [2,  4,  8,  16, 32,   64, 128, 256, # 0
          3,  6, 12,  24, 48,   96, 192, 384, # 1
          5, 10, 20,  40, 80,  160, 320,      # 2
          7, 14, 28,  56, 112, 224,           # 3
          9, 18, 36,  72, 144, 288,           # 4
         11, 22, 44,  88, 176, 352,           # 5
         13, 26, 52, 104, 208,                # 6
         15, 30, 60, 120, 240]                # 7
#Z_current = [ 64,  96, 128, 160, 192, 224, 240, 256, 288, 320, 352, 384]
Z_current = [ 36,
              40,
              44,
              48,
              52,
              56,
              60,
              64,
              72,
              80,
              88,
              96,
              104,
              112,
              120,
              128,
              144,
              160,
              176,
              192,
              208,
              224,
              240,
              256,
              288,
              320,
              352,
              384]

# Compilation of these lifting sizes will be guarded by a #define to
# reduce compile time.
Z_optional = [36,
              40,
              44,
              48,
              52,
              56,
              60,
              72,
              80,
              88,
              104,
              112,
              120,
              144,
              176,
              208,
              240]

# We will use a preprocessor conditional around some kernel template
# instantiations.
# We currently have two reasons that we might not want to compile
# an LDPC implementation:
# 1.) The algorithm would not be chosen by the decoder algorithm selection
# 2.) The lifting size is not a common lifting size. (This would be a
#     temporary condition, until the number of test vectors increases to
#     exercise all lifting sizes.)
algo_guard_begin = '#if CUPHY_LDPC_INCLUDE_LEVEL >= 1\n'
algo_guard_end   = '#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1\n'
Z_guard_begin    = '#if CUPHY_LDPC_INCLUDE_ALL_LIFTING\n'
Z_guard_end      = '#endif // if CUPHY_LDPC_INCLUDE_ALL_LIFTING\n'
algo_Z_guard_begin = '#if CUPHY_LDPC_INCLUDE_LEVEL >= 1 && CUPHY_LDPC_INCLUDE_ALL_LIFTING\n'
algo_Z_guard_end   = '#endif // if CUPHY_LDPC_INCLUDE_LEVEL >= 1 && CUPHY_LDPC_INCLUDE_ALL_LIFTING\n'

#                    ALGO   Z
guard_dict_begin = {(False, False): '',
                    (False, True):  Z_guard_begin,
                    (True,  False): algo_guard_begin,
                    (True,  True):  algo_Z_guard_begin}
#                  ALGO   Z
guard_dict_end = {(False, False): '',
                  (False, True):  Z_guard_end,
                  (True,  False): algo_guard_end,
                  (True,  True):  algo_Z_guard_end}

for Z_val in Z_current:
    sub_dict = {'Z':                 str(Z_val),
                #                                      ALGO   Z
                'half_guard_begin':  guard_dict_begin[(True, Z_val in Z_optional)],
                'half_guard_end':    guard_dict_end  [(True, Z_val in Z_optional)],
                'float_guard_begin': guard_dict_begin[(True, Z_val in Z_optional)],
                'float_guard_end':   guard_dict_end  [(True, Z_val in Z_optional)]}
    formatted_file = cu_file_bg1.format(**sub_dict)
    fname = 'ldpc2_reg_index_BG1_Z%d.cu' % Z_val
    print('Writing %s' % fname)
    fOut = open(fname, 'w')
    fOut.write(formatted_file)

for Z_val in Z_current:
    # TODO: Fix for other Kb values
    Kb_val = 10 if (Z_val > 64) else 9
    sub_dict = {'Z':                 str(Z_val),
                'Kb':                str(Kb_val),
                #                                      ALGO   Z
                'half_guard_begin':  guard_dict_begin[(True, Z_val in Z_optional)],
                'half_guard_end':    guard_dict_end  [(True, Z_val in Z_optional)],
                'float_guard_begin': guard_dict_begin[(True, Z_val in Z_optional)],
                'float_guard_end':   guard_dict_end  [(True, Z_val in Z_optional)]}
    fname = 'ldpc2_reg_index_BG2_Z%d.cu' % Z_val
    formatted_file = cu_file_bg2.format(**sub_dict)
    print('Writing %s' % fname)
    fOut = open(fname, 'w')
    fOut.write(formatted_file)
