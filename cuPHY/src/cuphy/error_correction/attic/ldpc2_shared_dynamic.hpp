
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SHARED_DYNAMIC_HPP_INCLUDED__)
#define LDPC2_SHARED_HPP_DYNAMIC_INCLUDED__

#include "ldpc2.hpp"

namespace ldpc2
{

cuphyStatus_t decode_ldpc2_shared_dynamic_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_dynamic_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_SHARED_DYNAMIC_HPP_INCLUDED__)
