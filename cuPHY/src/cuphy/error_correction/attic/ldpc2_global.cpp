/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ldpc2_global.hpp"
#include <utility>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_half()
cuphyStatus_t decode_ldpc2_global_index_half(ldpc::decoder&            dec,
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
        case 64:  s = decode_ldpc2_global_index_half_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_index_half_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_index_half_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_index_half_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_index_half_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_index_half_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_index_half_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_index_half_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_index_half_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_index_half_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_index_half_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                  break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_global_index_half_BG2_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_index_half_BG2_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_index_half_BG2_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_index_half_BG2_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_index_half_BG2_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_index_half_BG2_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_index_half_BG2_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_index_half_BG2_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_index_half_BG2_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_index_half_BG2_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_index_half_BG2_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                  break;
        }
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_float()
cuphyStatus_t decode_ldpc2_global_index_float(ldpc::decoder&            dec,
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
        case 36:  s = decode_ldpc2_global_index_float_BG1_Z36 (dec, config, params, grdDim, blkDim, strm); break;
        case 40:  s = decode_ldpc2_global_index_float_BG1_Z40 (dec, config, params, grdDim, blkDim, strm); break;
        case 44:  s = decode_ldpc2_global_index_float_BG1_Z44 (dec, config, params, grdDim, blkDim, strm); break;
        case 48:  s = decode_ldpc2_global_index_float_BG1_Z48 (dec, config, params, grdDim, blkDim, strm); break;
        case 52:  s = decode_ldpc2_global_index_float_BG1_Z52 (dec, config, params, grdDim, blkDim, strm); break;
        case 56:  s = decode_ldpc2_global_index_float_BG1_Z56 (dec, config, params, grdDim, blkDim, strm); break;
        case 60:  s = decode_ldpc2_global_index_float_BG1_Z60 (dec, config, params, grdDim, blkDim, strm); break;
        case 64:  s = decode_ldpc2_global_index_float_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 72:  s = decode_ldpc2_global_index_float_BG1_Z72 (dec, config, params, grdDim, blkDim, strm); break;
        case 80:  s = decode_ldpc2_global_index_float_BG1_Z80 (dec, config, params, grdDim, blkDim, strm); break;
        case 88:  s = decode_ldpc2_global_index_float_BG1_Z88 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_index_float_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 104: s = decode_ldpc2_global_index_float_BG1_Z104(dec, config, params, grdDim, blkDim, strm); break;
        case 112: s = decode_ldpc2_global_index_float_BG1_Z112(dec, config, params, grdDim, blkDim, strm); break;
        case 120: s = decode_ldpc2_global_index_float_BG1_Z120(dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_index_float_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 144: s = decode_ldpc2_global_index_float_BG1_Z144(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_index_float_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 176: s = decode_ldpc2_global_index_float_BG1_Z176(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_index_float_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_index_float_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 240: s = decode_ldpc2_global_index_float_BG1_Z240(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_index_float_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_index_float_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_index_float_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_index_float_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_index_float_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                   break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        case 36:  s = decode_ldpc2_global_index_float_BG2_Z36 (dec, config, params, grdDim, blkDim, strm); break;
        case 40:  s = decode_ldpc2_global_index_float_BG2_Z40 (dec, config, params, grdDim, blkDim, strm); break;
        case 44:  s = decode_ldpc2_global_index_float_BG2_Z44 (dec, config, params, grdDim, blkDim, strm); break;
        case 48:  s = decode_ldpc2_global_index_float_BG2_Z48 (dec, config, params, grdDim, blkDim, strm); break;
        case 52:  s = decode_ldpc2_global_index_float_BG2_Z52 (dec, config, params, grdDim, blkDim, strm); break;
        case 56:  s = decode_ldpc2_global_index_float_BG2_Z56 (dec, config, params, grdDim, blkDim, strm); break;
        case 60:  s = decode_ldpc2_global_index_float_BG2_Z60 (dec, config, params, grdDim, blkDim, strm); break;
        case 64:  s = decode_ldpc2_global_index_float_BG2_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 72:  s = decode_ldpc2_global_index_float_BG2_Z72 (dec, config, params, grdDim, blkDim, strm); break;
        case 80:  s = decode_ldpc2_global_index_float_BG2_Z80 (dec, config, params, grdDim, blkDim, strm); break;
        case 88:  s = decode_ldpc2_global_index_float_BG2_Z88 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_index_float_BG2_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 104: s = decode_ldpc2_global_index_float_BG2_Z104(dec, config, params, grdDim, blkDim, strm); break;
        case 112: s = decode_ldpc2_global_index_float_BG2_Z112(dec, config, params, grdDim, blkDim, strm); break;
        case 120: s = decode_ldpc2_global_index_float_BG2_Z120(dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_index_float_BG2_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 144: s = decode_ldpc2_global_index_float_BG2_Z144(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_index_float_BG2_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 176: s = decode_ldpc2_global_index_float_BG2_Z176(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_index_float_BG2_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_index_float_BG2_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 240: s = decode_ldpc2_global_index_float_BG2_Z240(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_index_float_BG2_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_index_float_BG2_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_index_float_BG2_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_index_float_BG2_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_index_float_BG2_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                   break;
        }
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address_float()
cuphyStatus_t decode_ldpc2_global_address_float(ldpc::decoder&            dec,
                                                const LDPC_config&        config,
                                                const LDPC_kernel_params& params,
                                                const dim3&               grdDim,
                                                const dim3&               blkDim,
                                                cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
#if 0
    if(config.BG == 1)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_global_address_float_BG1_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_address_float_BG1_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_address_float_BG1_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_address_float_BG1_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_address_float_BG1_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_address_float_BG1_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_address_float_BG1_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_address_float_BG1_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_address_float_BG1_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_address_float_BG1_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_address_float_BG1_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                        break;
        }
    }
    else if(config.BG == 2)
    {
        switch(config.Z)
        {
        case 64:  s = decode_ldpc2_global_address_float_BG2_Z64 (dec, config, params, grdDim, blkDim, strm); break;
        case 96:  s = decode_ldpc2_global_address_float_BG2_Z96 (dec, config, params, grdDim, blkDim, strm); break;
        case 128: s = decode_ldpc2_global_address_float_BG2_Z128(dec, config, params, grdDim, blkDim, strm); break;
        case 160: s = decode_ldpc2_global_address_float_BG2_Z160(dec, config, params, grdDim, blkDim, strm); break;
        case 192: s = decode_ldpc2_global_address_float_BG2_Z192(dec, config, params, grdDim, blkDim, strm); break;
        case 224: s = decode_ldpc2_global_address_float_BG2_Z224(dec, config, params, grdDim, blkDim, strm); break;
        case 256: s = decode_ldpc2_global_address_float_BG2_Z256(dec, config, params, grdDim, blkDim, strm); break;
        case 288: s = decode_ldpc2_global_address_float_BG2_Z288(dec, config, params, grdDim, blkDim, strm); break;
        case 320: s = decode_ldpc2_global_address_float_BG2_Z320(dec, config, params, grdDim, blkDim, strm); break;
        case 352: s = decode_ldpc2_global_address_float_BG2_Z352(dec, config, params, grdDim, blkDim, strm); break;
        case 384: s = decode_ldpc2_global_address_float_BG2_Z384(dec, config, params, grdDim, blkDim, strm); break;
        default:                                                                                        break;
        }
    }
#endif
    return s;
}

} // namespace ldpc2

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address()
cuphyStatus_t decode_ldpc2_global_address(decoder&               dec,
                                          LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          cuphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          cuphyLDPCDiagnostic_t* diag,
                                          cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_global_address()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    if(llrType == CUPHY_R_32F)
    {
        s = decode_ldpc2_global_address_float(dec,
                                              config,
                                              params,
                                              grdDim,
                                              blkDim,
                                              strm);
    }
    // No fp16 implementation right now...

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
// decode_ldpc2_global_index()
cuphyStatus_t decode_ldpc2_global_index(decoder&               dec,
                                        LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_global_index()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords);
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    
    if(llrType == CUPHY_R_32F)
    {
        s = decode_ldpc2_global_index_float(dec,
                                            config,
                                            params,
                                            grdDim,
                                            blkDim,
                                            strm);
    }
    else if(llrType == CUPHY_R_16F)
    {
        // Convert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        s = decode_ldpc2_global_index_half(dec,
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

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_address_workspace_size(const decoder&     dec,
                                                                   const LDPC_config& cfg)
{
    // TODO: Possibly adjust size of per-thread element
    return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
}

//----------------------------------------------------------------------
// decode_ldpc2_global_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_index_workspace_size(const decoder&     dec,
                                                                 const LDPC_config& cfg)
{
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

