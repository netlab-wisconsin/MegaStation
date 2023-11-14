/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */



#include "ldpc2_shared_cluster.hpp"
#include "cuphy_internal.h"

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index_half()
cuphyStatus_t decode_ldpc2_shared_cluster_index_half(const LDPC_config&        config,
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
        case 64:  s = decode_ldpc2_shared_cluster_index_half_BG1_Z64 (config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_cluster_index_half_BG1_Z96 (config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_cluster_index_half_BG1_Z128(config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_cluster_index_half_BG1_Z160(config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_cluster_index_half_BG1_Z192(config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_cluster_index_half_BG1_Z224(config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_cluster_index_half_BG1_Z256(config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_cluster_index_half_BG1_Z288(config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_cluster_index_half_BG1_Z320(config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_cluster_index_half_BG1_Z352(config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_cluster_index_half_BG1_Z384(config, params, grdDim, blkDim, strm); break;
        default:                                                                                             break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_shared_cluster_index_half_BG2_Z64 (config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_shared_cluster_index_half_BG2_Z96 (config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_shared_cluster_index_half_BG2_Z128(config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_shared_cluster_index_half_BG2_Z160(config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_shared_cluster_index_half_BG2_Z192(config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_shared_cluster_index_half_BG2_Z224(config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_shared_cluster_index_half_BG2_Z256(config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_shared_cluster_index_half_BG2_Z288(config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_shared_cluster_index_half_BG2_Z320(config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_shared_cluster_index_half_BG2_Z352(config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_shared_cluster_index_half_BG2_Z384(config, params, grdDim, blkDim, strm); break;
        default:                                                                                             break;
        }
    }
    return s;
}

} // namespace ldpc2

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_cluster_index()
cuphyStatus_t decode_ldpc2_shared_cluster_index(decoder&               dec,
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
    dim3 grdDim(config.num_codewords);
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
        s = decode_ldpc2_shared_cluster_index_half(config,
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
// decode_ldpc2_shared_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_shared_cluster_index_workspace_size(const decoder&     dec,
                                                                         const LDPC_config& cfg)
{
    return std::pair<bool, size_t>(true, 0);
}

} // namespace ldpc
