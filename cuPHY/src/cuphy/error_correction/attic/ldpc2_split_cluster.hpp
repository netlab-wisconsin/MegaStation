/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SPLIT_CLUSTER_HPP_INCLUDED__)
#define LDPC2_SPLIT_CLUSTER_HPP_INCLUDED__

#include "ldpc2.hpp"

namespace ldpc2
{

cuphyStatus_t decode_ldpc2_split_cluster_half_96KB(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_split_cluster_half_96KB_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_SPLIT_CLUSTER_HPP_INCLUDED__)
