
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_REG_HPP_INCLUDED__)
#define LDPC2_REG_HPP_INCLUDED__

#include "ldpc2.hpp"

namespace ldpc2
{

/////////////////////////////////////////////////////////////////////////
// register C2V cache
// "index" min0 location 
// half precision (fp16) APP values
cuphyStatus_t decode_ldpc2_reg_index_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_half_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

/////////////////////////////////////////////////////////////////////////
// register C2V cache
// "index" min0 location
// alternate APP address calculation (using floating point instructions)
// half precision (fp16) APP values
cuphyStatus_t decode_ldpc2_reg_index_fp_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_half_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

/////////////////////////////////////////////////////////////////////////
// register C2V cache
// "index" min0 location
// alternate APP address calculation (using floating point instructions)
// half precision (fp16) APP values
// 2 codewords processes simultaneously (x2)
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

/////////////////////////////////////////////////////////////////////////
// register C2V cache
// "index" min0 location 
// single precision (fp32) APP values
cuphyStatus_t decode_ldpc2_reg_index_float(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_index_float_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

    
/////////////////////////////////////////////////////////////////////////
// register C2V cache
// "address" min0 location 
// single precision (fp32) APP values
cuphyStatus_t decode_ldpc2_reg_address_float(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z2  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z3  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z4  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z5  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z6  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z7  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z8  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z9  (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z10 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z11 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z12 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z13 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z14 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z15 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z16 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z18 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z20 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z22 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z24 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z26 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z28 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z30 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z32 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_reg_address_float_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_REG_HPP_INCLUDED__)
