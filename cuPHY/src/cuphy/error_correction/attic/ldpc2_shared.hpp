
/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SHARED_HPP_INCLUDED__)
#define LDPC2_SHARED_HPP_INCLUDED__

#include "ldpc2.hpp"
#include "cuphy_context.hpp"

namespace ldpc2
{

inline 
bool decode_ldpc2_can_use_shared(const cuphy_i::context& ctx,
                                 const LDPC_config&      config)
{
    //-------------------------------------------------------------------
    // Determine the maximum amount of shared memory that the
    // current device can support
    int32_t device_shmem_max = ctx.max_shmem_per_block_optin();
    if(device_shmem_max <= 0)
    {
        return false;
    }
    int32_t Kmul      = (1 == config.BG) ? 22 : 10; 
    int32_t szElem    = (CUPHY_R_16F == config.type) ?
                        sizeof(data_type_traits<CUPHY_R_16F>::type) : 
                        sizeof(data_type_traits<CUPHY_R_32F>::type);
    //int32_t szC2V     = (CUPHY_R_16F == config.type) ?
    //                    sizeof(cC2V_storage<data_type_traits<CUPHY_R_16F>::type>) : 
    //                    sizeof(cC2V_storage<data_type_traits<CUPHY_R_32F>::type>);
    int32_t szC2V     = (CUPHY_R_16F == config.type) ? 8 : 12;
    int32_t shmem_req = config.Z * (((Kmul + config.mb) * szElem) + (config.mb * szC2V));
    return (shmem_req <= device_shmem_max);
}

cuphyStatus_t decode_ldpc2_shared_half(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_shared_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);


cuphyStatus_t decode_ldpc2_shared_index_fp_x2(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z36 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z40 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z44 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z48 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z52 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z56 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z60 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z64 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z72 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z80 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z88 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z96 (ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z104(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z112(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z120(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z128(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z144(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z160(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z176(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z192(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z208(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z224(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z240(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z256(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z288(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z320(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z352(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);
//cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z384(ldpc::decoder& dec, const LDPC_config& cfg, const LDPC_kernel_params& params, const dim3& grdDim, const dim3& blkDim, cudaStream_t strm);

} // namespace ldpc2

#endif // !defined(LDPC2_SHARED_HPP_INCLUDED__)
