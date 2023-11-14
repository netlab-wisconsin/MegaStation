/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ldpc2_shared.hpp"
#include <utility>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_half()
cuphyStatus_t decode_ldpc2_shared_index_half(ldpc::decoder&            dec,
                                             const LDPC_config&        config,
                                             const LDPC_kernel_params& params,
                                             const dim3&               grdDim,
                                             const dim3&               blkDim,
                                             cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if(config.BG == 1)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_shared_half_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_half_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_half_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_half_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_half_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_half_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_half_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_half_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_half_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_half_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_half_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                               break;
        }
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_half()
cuphyStatus_t decode_ldpc2_shared_index_fp_x2_half(ldpc::decoder&            dec,
                                                   const LDPC_config&        config,
                                                   const LDPC_kernel_params& params,
                                                   const dim3&               grdDim,
                                                   const dim3&               blkDim,
                                                   cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if(config.BG == 1)
    {
        switch(config.Z)
        {
        case 36:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z36 (dec, config, params, grdDim, blkDim, strm); break;
        case 40:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z40 (dec, config, params, grdDim, blkDim, strm); break;
        case 44:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z44 (dec, config, params, grdDim, blkDim, strm); break;
        case 48:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z48 (dec, config, params, grdDim, blkDim, strm); break;
        case 52:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z52 (dec, config, params, grdDim, blkDim, strm); break;
        case 56:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z56 (dec, config, params, grdDim, blkDim, strm); break;
        case 60:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z60 (dec, config, params, grdDim, blkDim, strm); break;
        case 64:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 72:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z72 (dec, config, params, grdDim, blkDim, strm); break;
        case 80:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z80 (dec, config, params, grdDim, blkDim, strm); break;
        case 88:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z88 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 104: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z104(dec, config, params, grdDim, blkDim, strm); break;
        case 112: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z112(dec, config, params, grdDim, blkDim, strm); break;
        case 120: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z120(dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 144: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z144(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 176: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z176(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 208: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z208(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 240: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z240(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_index_fp_x2_half_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                                break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        //case 64:  s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        //case 96:  s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        //case 128: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z128(dec, config, params, grdDim, blkDim, strm); break;
        //case 160: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z160(dec, config, params, grdDim, blkDim, strm); break;
        //case 192: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z192(dec, config, params, grdDim, blkDim, strm); break;
        //case 224: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z224(dec, config, params, grdDim, blkDim, strm); break;
        //case 256: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z256(dec, config, params, grdDim, blkDim, strm); break;
        //case 288: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z288(dec, config, params, grdDim, blkDim, strm); break;
        //case 320: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z320(dec, config, params, grdDim, blkDim, strm); break;
        //case 352: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z352(dec, config, params, grdDim, blkDim, strm); break;
        //case 384: s = decode_ldpc2_shared_index_fp_x2_half_BG2_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                             break;
        }
    }
    return s;
}

} // namespace ldpc2

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index()
cuphyStatus_t decode_ldpc2_shared_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_shared_index()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s;
    
    if(llrType == CUPHY_R_32F)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
    else if(llrType == CUPHY_R_16F)
    {
        // Convert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        s = decode_ldpc2_shared_index_half(dec,
                                           config,
                                           params,
                                           grdDim,
                                           blkDim,
                                           strm);
    }
    else
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2()
cuphyStatus_t decode_ldpc2_shared_index_fp_x2(decoder&               dec,
                                              LDPC_output_t&         tDst,
                                              const_tensor_pair&     tLLR,
                                              const LDPC_config&     config,
                                              float                  normalization,
                                              cuphyLDPCResults_t*    results,
                                              void*                  workspace,
                                              cuphyLDPCDiagnostic_t* diag,
                                              cudaStream_t           strm)
{
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    // Each block will process 2 codewords
    dim3 grdDim(div_round_up(config.num_codewords, 2));
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    if(llrType == CUPHY_R_16F)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Convert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        s = decode_ldpc2_shared_index_fp_x2_half(dec,
                                                 config,
                                                 params,
                                                 grdDim,
                                                 blkDim,
                                                 strm);
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ldpc2_shared_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg)
{
    return std::pair<bool, size_t>(true, 0);
}

//----------------------------------------------------------------------
// decode_ldpc2_shared_index_fp_x2_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_index_fp_x2_workspace_size(const decoder&     dec,
                                                                       const LDPC_config& cfg)
{
    return std::pair<bool, size_t>(true, 0);
}


} // namespace ldpc

