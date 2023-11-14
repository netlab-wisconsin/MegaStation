/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */



#include "ldpc2_split_cluster.hpp"
#include "cuphy_internal.h"

using namespace ldpc2;

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_split_cluster_index()
cuphyStatus_t decode_ldpc2_split_cluster_index(decoder&               dec,
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
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Determine the maximum amount of shared memory that the
        // current device can support
        int32_t device_shmem_max = dec.max_shmem_per_block_optin();
        if(device_shmem_max <= 0)
        {
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        switch(device_shmem_max)
        {
        case (96*1024): s = decode_ldpc2_split_cluster_half_96KB(dec, config, params, grdDim, blkDim, strm); break; // Volta:  96 KiB max (opt-in)
        case (64*1024):                                                                                      break; // Turing: 64 KiB max (opt-in)
        default:                                                                                             break;
        }
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
// decode_ldpc2_split_cluster_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_split_cluster_index_workspace_size(const ldpc::decoder& dec, const LDPC_config& cfg)
{
    // For now, cC2V class in ldp2c.cuh loads and stores from global memory
    // using an offset that assumes ALL cC2Vs are in global memory, even
    // though some are in shared memory.
    // TODO: Modify the global load/store and reduce the amount of
    // allocation here.
    if(CUPHY_R_32F == cfg.type)
    {
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
    }
    else if(CUPHY_R_16F == cfg.type)
    {
        // Assumes all of workspace is used for cC2V messages (i.e. no APP values)
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int2));
    }
    else
    {
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
